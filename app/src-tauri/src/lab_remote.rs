// ── Hosted Lab analysis ──────────────────────────────────────────────────────
//
// Sends a 720p copy of the replay's video to the Spaghetti Showdown servers and fetches the result,
// in this order: check, transcode, upload, start, poll, result, clean up (docs/desktop-login.md,
// "Hosted run"). Written for runs that take minutes to hours:
// - The server decides where an analysis is charged; the app only reports it.
// - A started job is saved per replay (remote_job.json, ids only), so closing the app loses nothing.
// - The temp copy goes on every path (a guard), the uploaded file after the job ended. A delete that
//   can't finish goes to pending_deletes.json and is retried at startup and at sign-in.
// - The start POST is never aborted: an answer lost there would leave a charged job nobody knows of.

use crate::lab;
use crate::lab_api::{self as api, Answer, JobView, Quota, RemoteError, UploadAnswer};
use crate::lab_auth::{self, Auth};
use futures_util::StreamExt as _;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::Emitter;
use tokio::io::{AsyncBufReadExt, AsyncReadExt};
use tokio::sync::watch;

pub const MAX_SECONDS: f64 = 900.0;
pub const MAX_BYTES: u64 = 500 * 1024 * 1024;
pub const MIN_HEIGHT: u32 = 360;
pub const TARGET_HEIGHT: u32 = 720;
const POLL: Duration = Duration::from_secs(15);
const POLL_AFTER_ERROR: Duration = Duration::from_secs(30);
/// No bytes accepted for this long: the upload has stalled.
const UPLOAD_STALL: Duration = Duration::from_secs(120);
/// The answer only comes after the server copied the file to storage.
const UPLOAD_ANSWER: Duration = Duration::from_secs(30 * 60);
const START_RETRIES: usize = 6;
const START_NETWORK_RETRIES: usize = 3;
const CALL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Default, Deserialize)]
pub struct RemoteOptions {
    /// A result exists and the user agreed to spend another analysis
    pub again: Option<bool>,
    /// The video to send (a .spag session's extracted copy); else the replay's own video
    pub video_path: Option<String>,
}

/// One "lab-progress" event (same event as the local mode's, with more optional fields).
#[derive(Debug, Clone, Serialize, Default)]
pub struct Progress {
    pub replay_id: String,
    pub stage: String,
    pub line: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sent: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RunningInfo {
    pub replay_id: String,
    pub stage: String,
    pub position: Option<i64>,
    /// "before_charge", "queued" or "running": what a cancel would mean
    pub phase: String,
}

struct RemoteRun {
    info: RunningInfo,
    cancel: watch::Sender<bool>,
}

/// The hosted run going on (one at a time; the server allows one per user anyway).
static RUN: Mutex<Option<RemoteRun>> = Mutex::new(None);
static LAST_ERROR: Mutex<Option<(String, RemoteError)>> = Mutex::new(None);
static PENDING_LOCK: Mutex<()> = Mutex::new(());

pub fn running_replay() -> Option<String> {
    RUN.lock().unwrap().as_ref().map(|r| r.info.replay_id.clone())
}

pub fn lab_root_dir() -> PathBuf {
    lab::lab_root()
}

struct RunGuard;
impl Drop for RunGuard {
    fn drop(&mut self) {
        RUN.lock().unwrap().take();
    }
}

fn claim_run(id: &str) -> Result<(RunGuard, watch::Receiver<bool>), RemoteError> {
    let mut g = RUN.lock().unwrap();
    if g.is_some() || lab::local_run_active() {
        return Err(RemoteError::new("busy", "A lab analysis is already running."));
    }
    let (tx, rx) = watch::channel(false);
    *g = Some(RemoteRun {
        info: RunningInfo { replay_id: id.into(), stage: "check".into(), position: None, phase: "before_charge".into() },
        cancel: tx,
    });
    Ok((RunGuard, rx))
}

// ── pure pieces ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    pub seconds: f64,
    pub height: u32,
}

/// What `ffmpeg -i` printed: `Duration: 00:03:12.00` and the first video stream's `1280x720`. There
/// may be no ffprobe, and the DB's duration can be 0.
pub fn parse_probe(stderr: &str) -> Option<VideoInfo> {
    let d = stderr.split("Duration:").nth(1)?.trim_start();
    let stamp = d.split(',').next()?.trim();
    let mut parts = stamp.split(':');
    let (h, m, s) = (parts.next()?.parse::<f64>().ok()?, parts.next()?.parse::<f64>().ok()?, parts.next()?.parse::<f64>().ok()?);
    let line = stderr.lines().find(|l| l.contains("Stream #") && l.contains("Video:"))?;
    let height = line
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|tok| {
            let (w, h) = tok.split_once('x')?;
            let (w, h) = (w.parse::<u32>().ok()?, h.parse::<u32>().ok()?);
            (w >= 16 && h >= 16).then_some(h)
        })
        .next()?;
    Some(VideoInfo { seconds: h * 3600.0 + m * 60.0 + s, height })
}

/// Stop before any work: too long or too small to analyse.
pub fn check_limits(v: VideoInfo) -> Result<(), RemoteError> {
    if v.seconds > MAX_SECONDS {
        return Err(RemoteError::new(
            "too_long",
            format!("This video is {} minutes long; the limit is 15 minutes. If it's a whole VOD, cut it to one set first (the Split tool does this).", (v.seconds / 60.0).ceil()),
        ));
    }
    if v.height < MIN_HEIGHT {
        return Err(RemoteError::new("too_small", format!("This video is {}p; the analysis needs at least 360p.", v.height)));
    }
    Ok(())
}

pub fn too_big() -> RemoteError {
    RemoteError::new("too_big", "The video is over the 500 MB upload limit even at 720p. If it's a whole VOD, cut it to one set first (the Split tool does this).")
}

/// A re-encode only when it's over 720p or too large to send.
pub fn needs_encode(v: VideoInfo, size: u64) -> bool {
    v.height > TARGET_HEIGHT || size > MAX_BYTES
}

/// The ffmpeg arguments of the copy. Audio and metadata stay on this PC. `limit`: tests only.
pub fn copy_args(input: &str, output: &str, encode: bool, limit: Option<u32>) -> Vec<String> {
    let mut a: Vec<String> = ["-hide_banner", "-nostats", "-loglevel", "error", "-y", "-progress", "pipe:1"].iter().map(|s| s.to_string()).collect();
    if let Some(l) = limit {
        a.extend(["-t".to_string(), l.to_string()]);
    }
    a.extend(["-i".to_string(), input.to_string()]);
    a.extend(["-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1"].iter().map(|s| s.to_string()));
    if encode {
        a.extend(
            ["-vf", "scale=-2:'min(720,ih)',fps=30", "-c:v", "libx264", "-preset", "veryfast", "-b:v", "3M", "-maxrate", "3M", "-bufsize", "6M", "-pix_fmt", "yuv420p"]
                .iter()
                .map(|s| s.to_string()),
        );
    } else {
        a.extend(["-c:v", "copy"].iter().map(|s| s.to_string()));
    }
    a.extend(["-movflags", "+faststart", "-f", "mp4"].iter().map(|s| s.to_string()));
    a.push(output.to_string());
    a
}

pub enum StartOutcome {
    Started(JobView, Option<Quota>),
    Retry(u64),
    Fail(RemoteError),
}

/// The start POST's answer. Only `quota_exhausted` has an `error` code; the service's 429 and 503 have
/// `detail`, and the global limiter's 429 is plain text.
pub fn classify_start(a: &Answer) -> StartOutcome {
    let quota = a.quota();
    let with_quota = |mut e: RemoteError| {
        e.quota = quota.clone();
        e
    };
    match a.status {
        200 | 202 => match a.parse::<JobView>() {
            Some(j) => StartOutcome::Started(j, quota),
            None => StartOutcome::Fail(RemoteError::new("start_failed", "The server's answer could not be read.")),
        },
        409 if a.error() == Some("analysis_starting") => StartOutcome::Retry(a.retry_after.unwrap_or(5)),
        429 if a.error() == Some("quota_exhausted") => StartOutcome::Fail(with_quota(RemoteError::new("quota_exhausted", "You have no analyses left this week.")).retry(a.retry_after)),
        429 if a.detail().is_some() => StartOutcome::Fail(RemoteError::new("one_at_a_time", "You already have an analysis queued or running (on the website or another PC). Nothing was spent.").retry(a.retry_after)),
        429 => StartOutcome::Fail(RemoteError::new("rate_limited", "Too many requests right now. Please try again in a few minutes. Nothing was spent.").retry(a.retry_after)),
        503 if a.detail().is_some() => StartOutcome::Fail(RemoteError::new("queue_full", "The analysis queue is full. Please try again later. Nothing was spent.").retry(a.retry_after)),
        413 => StartOutcome::Fail(RemoteError::new("too_long", a.sentence().unwrap_or_else(|| "The video is over 15 minutes: cut it to one match or set.".into()))),
        400 | 404 | 415 => StartOutcome::Fail(RemoteError::new("start_failed", a.sentence().unwrap_or_else(|| "The server refused this video.".into()))),
        _ => StartOutcome::Fail(RemoteError::unavailable().retry(a.retry_after)),
    }
}

/// The upload's answer: the file's id, or why not. Limit errors have mixed shapes on the server.
pub fn classify_upload(a: &Answer) -> Result<String, RemoteError> {
    let message = a.json.get("message").and_then(|m| m.as_str()).map(api::scrub);
    match (a.status, a.error()) {
        (200, _) | (201, _) => a.parse::<UploadAnswer>().map(|u| u.data.file_id).ok_or_else(|| RemoteError::new("upload_failed", "The server's answer could not be read.")),
        (413, _) => Err(too_big()),
        (403, Some("banned_global")) => Err(RemoteError::new("banned", message.unwrap_or_else(|| "This account can't upload files.".into()))),
        (400, Some(e)) if e.contains("storage limit") => Err(RemoteError::new("storage_full", api::scrub(e))),
        (400, _) | (403, _) | (404, _) | (500, _) => Err(RemoteError::new("upload_failed", message.or_else(|| a.sentence()).unwrap_or_else(|| "The upload failed.".into()))),
        (429, _) => Err(RemoteError::new("rate_limited", "Too many requests right now. Please try again in a few minutes.").retry(a.retry_after)),
        _ => Err(RemoteError::unavailable().retry(a.retry_after)),
    }
}

// ── what survives a restart (never a token) ─────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SavedJob {
    #[serde(default)]
    pub file_id: Option<String>,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub player_user_id: Option<String>,
}

fn job_path(root: &Path, id: &str) -> PathBuf {
    root.join(id).join("remote_job.json")
}

fn read_job(root: &Path, id: &str) -> Option<SavedJob> {
    serde_json::from_str(&std::fs::read_to_string(job_path(root, id)).ok()?).ok()
}

fn save_job(root: &Path, id: &str, job: &SavedJob) {
    let p = job_path(root, id);
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Ok(t) = serde_json::to_string(job) {
        if let Err(e) = std::fs::write(&p, t) {
            eprintln!("[LAB] could not save the job: {}", e);
        }
    }
}

fn remove_job(root: &Path, id: &str) {
    let _ = std::fs::remove_file(job_path(root, id));
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PendingDelete {
    file_id: String,
    #[serde(default)]
    player_user_id: Option<String>,
}

fn pending_path(root: &Path) -> PathBuf {
    root.join("pending_deletes.json")
}

fn read_pending(root: &Path) -> Vec<PendingDelete> {
    std::fs::read_to_string(pending_path(root)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn write_pending(root: &Path, list: &[PendingDelete]) {
    let p = pending_path(root);
    if list.is_empty() {
        let _ = std::fs::remove_file(&p);
        return;
    }
    let _ = std::fs::create_dir_all(root);
    if let Ok(t) = serde_json::to_string(list) {
        let _ = std::fs::write(&p, t);
    }
}

fn add_pending(root: &Path, file_id: &str, player: Option<String>) {
    let _g = PENDING_LOCK.lock().unwrap();
    let mut list = read_pending(root);
    if !list.iter().any(|p| p.file_id == file_id) {
        list.push(PendingDelete { file_id: file_id.into(), player_user_id: player });
    }
    write_pending(root, &list);
}

/// True when there's nothing more to do for this file. A 403 (a file this token didn't upload) is a
/// logged warning, never a sign-out.
async fn try_delete_file(auth: &Arc<Auth>, file_id: &str) -> bool {
    match auth.send_simple(Method::DELETE, &api::path_file(file_id), CALL_TIMEOUT).await {
        Ok(a) if a.ok() || a.status == 404 => true,
        Ok(a) if a.status == 403 => {
            eprintln!("[LAB] warning: the server refused to delete an uploaded file (403)");
            true
        }
        Ok(a) => {
            eprintln!("[LAB] deleting the uploaded file answered {}; it will be tried again", a.status);
            false
        }
        Err(e) if e.code == "scope" => {
            eprintln!("[LAB] warning: deleting an uploaded file is outside this sign-in's scope (403 scope)");
            true
        }
        Err(e) => {
            eprintln!("[LAB] deleting the uploaded file failed ({}); it will be tried again", e.code);
            false
        }
    }
}

async fn delete_file(auth: &Arc<Auth>, root: &Path, file_id: &str) {
    if !try_delete_file(auth, file_id).await {
        add_pending(root, file_id, auth.player_id().await);
    }
}

/// Deletes that couldn't finish earlier, as the same player only.
pub async fn retry_pending_deletes(auth: &Arc<Auth>, root: &Path) {
    let Some(player) = auth.player_id().await else { return };
    let list = {
        let _g = PENDING_LOCK.lock().unwrap();
        read_pending(root)
    };
    let mut done = Vec::new();
    for p in &list {
        if p.player_user_id.as_deref() == Some(player.as_str()) && try_delete_file(auth, &p.file_id).await {
            done.push(p.file_id.clone());
        }
    }
    if !done.is_empty() {
        let _g = PENDING_LOCK.lock().unwrap();
        let left: Vec<_> = read_pending(root).into_iter().filter(|p| !done.contains(&p.file_id)).collect();
        write_pending(root, &left);
    }
}

// ── one run ─────────────────────────────────────────────────────────────────

type Emit = Arc<dyn Fn(&str, Value) + Send + Sync>;

#[derive(Clone)]
struct Ctx {
    auth: Arc<Auth>,
    root: PathBuf,
    id: String,
    emit: Emit,
    cancel: watch::Receiver<bool>,
    poll: Duration,
    poll_error: Duration,
}

fn set_run(id: &str, f: impl FnOnce(&mut RunningInfo)) {
    if let Some(r) = RUN.lock().unwrap().as_mut() {
        if r.info.replay_id == id {
            f(&mut r.info);
        }
    }
}

impl Ctx {
    fn progress(&self, p: Progress) {
        set_run(&self.id, |i| {
            i.stage = p.stage.clone();
            i.position = p.position;
        });
        (self.emit)("lab-progress", serde_json::to_value(Progress { replay_id: self.id.clone(), ..p }).unwrap_or(Value::Null));
    }

    fn stage(&self, stage: &str, line: &str) {
        self.progress(Progress { stage: stage.into(), line: line.into(), ..Default::default() });
    }

    fn quota(&self, q: &Quota) {
        (self.emit)("lab-quota", serde_json::to_value(q).unwrap_or(Value::Null));
    }

    fn is_cancelled(&self) -> bool {
        *self.cancel.borrow()
    }

    fn cancelled(&self, phase: &str) -> RemoteError {
        let mut e = RemoteError::new("cancelled", "Cancelled.");
        e.phase = Some(phase.into());
        e
    }

    /// Resolves when the user cancels.
    async fn wait_cancel(&self) {
        let mut rx = self.cancel.clone();
        let _ = rx.wait_for(|c| *c).await;
        // the sender is gone only when the run ended: never resolve for that
        if !*rx.borrow() {
            std::future::pending::<()>().await;
        }
    }

    /// Sleep, or wake up early on cancel.
    async fn nap(&self, d: Duration) {
        tokio::select! {
            _ = tokio::time::sleep(d) => {}
            _ = self.wait_cancel() => {}
        }
    }
}

fn ffmpeg_command(ffmpeg: &str) -> tokio::process::Command {
    let mut std_cmd = std::process::Command::new(ffmpeg);
    crate::hide_window(&mut std_cmd);
    let mut c = tokio::process::Command::from(std_cmd);
    c.kill_on_drop(true);
    c
}

async fn probe(ffmpeg: &str, video: &Path) -> Result<VideoInfo, RemoteError> {
    let mut c = ffmpeg_command(ffmpeg);
    c.args(["-hide_banner", "-i"]).arg(video).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
    let out = c.output().await.map_err(|e| RemoteError::new("no_video", format!("Could not run ffmpeg: {}", e)))?;
    let text = String::from_utf8_lossy(&out.stderr);
    parse_probe(&text).ok_or_else(|| RemoteError::new("no_video", "This file doesn't look like a readable video."))
}

/// Run ffmpeg on one copy: progress as `transcode` events, killed on cancel.
async fn run_copy(ctx: &Ctx, ffmpeg: &str, args: &[String], seconds: f64, report: bool) -> Result<bool, RemoteError> {
    let mut c = ffmpeg_command(ffmpeg);
    c.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = c.spawn().map_err(|e| RemoteError::new("no_video", format!("Could not run ffmpeg: {}", e)))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let work = async {
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        let mut last = -1.0;
        while let Ok(Some(l)) = lines.next_line().await {
            if !report {
                continue;
            }
            let us = l.strip_prefix("out_time_us=").or_else(|| l.strip_prefix("out_time_ms=")).and_then(|v| v.trim().parse::<f64>().ok());
            if let (Some(us), true) = (us, seconds > 0.0) {
                let pct = (us / 1e6 / seconds * 100.0).clamp(0.0, 100.0).floor();
                if pct != last {
                    last = pct;
                    ctx.progress(Progress { stage: "transcode".into(), line: format!("{}%", pct), percent: Some(pct), ..Default::default() });
                }
            }
        }
        child.wait().await
    };
    let ended = tokio::select! {
        r = work => Some(r.map(|s| s.success()).unwrap_or(false)),
        _ = ctx.wait_cancel() => None,
    };
    match ended {
        Some(ok) => Ok(ok),
        None => {
            // Gone before the temp folder is removed: Windows keeps a running process's output locked
            let _ = child.start_kill();
            let _ = child.wait().await;
            Err(ctx.cancelled("before_charge"))
        }
    }
}

/// The file to send, in the temp dir: a stream copy when it's small enough, else a 720p encode.
async fn prepare(ctx: &Ctx, ffmpeg: &str, video: &Path, info: VideoInfo, out: &Path) -> Result<u64, RemoteError> {
    let size = std::fs::metadata(video).map(|m| m.len()).map_err(|e| RemoteError::new("no_video", format!("Could not read the video: {}", e)))?;
    let (input, output) = (video.to_string_lossy().to_string(), out.to_string_lossy().to_string());
    let mut encode = needs_encode(info, size);
    let label = if encode { "Encoding" } else { "Copying" };
    ctx.progress(Progress { stage: "transcode".into(), line: label.into(), percent: Some(0.0), ..Default::default() });
    let mut ok = run_copy(ctx, ffmpeg, &copy_args(&input, &output, encode, None), info.seconds, encode).await?;
    if !ok && !encode {
        // a codec mp4 can't hold, for example
        encode = true;
        ok = run_copy(ctx, ffmpeg, &copy_args(&input, &output, encode, None), info.seconds, encode).await?;
    }
    if !ok {
        return Err(RemoteError::new("no_video", "Could not prepare the video for upload."));
    }
    let sent = std::fs::metadata(out).map(|m| m.len()).map_err(|e| RemoteError::new("no_video", e.to_string()))?;
    if sent > MAX_BYTES {
        return Err(too_big());
    }
    Ok(sent)
}

/// Streams from disk, never held in memory. Always named replay.mp4: the name becomes the facts'
/// title on the server, and player names must stay out of it.
async fn upload(ctx: &Ctx, file: &Path, total: u64) -> Result<String, RemoteError> {
    let sent = Arc::new(AtomicU64::new(0));
    let touched = Arc::new(Mutex::new(Instant::now()));
    let url = ctx.auth.url(api::PATH_UPLOAD);
    let send = {
        let (sent, touched) = (sent.clone(), touched.clone());
        ctx.auth.send(move |c| {
            let (url, file) = (url.clone(), file.to_path_buf());
            let (sent, touched, ctx) = (sent.clone(), touched.clone(), ctx.clone());
            async move {
                let f = tokio::fs::File::open(&file).await.map_err(|e| RemoteError::new("no_video", e.to_string()))?;
                sent.store(0, Ordering::SeqCst);
                let last_pct = Arc::new(AtomicU64::new(u64::MAX));
                let stream = futures_util::stream::unfold(f, |mut f| async move {
                    let mut buf = vec![0u8; 256 * 1024];
                    match f.read(&mut buf).await {
                        Ok(0) => None,
                        Ok(n) => {
                            buf.truncate(n);
                            Some((Ok::<Vec<u8>, std::io::Error>(buf), f))
                        }
                        Err(e) => Some((Err(e), f)),
                    }
                })
                .map(move |chunk| {
                    if let Ok(b) = &chunk {
                        let now = sent.fetch_add(b.len() as u64, Ordering::SeqCst) + b.len() as u64;
                        *touched.lock().unwrap() = Instant::now();
                        let pct = now * 100 / total.max(1);
                        if last_pct.swap(pct, Ordering::SeqCst) != pct {
                            let line = if now >= total { "Finishing the upload".to_string() } else { format!("{}%", pct) };
                            ctx.progress(Progress { stage: "upload".into(), line, percent: Some(pct as f64), sent: Some(now), total: Some(total), ..Default::default() });
                        }
                    }
                    chunk
                });
                let part = reqwest::multipart::Part::stream_with_length(reqwest::Body::wrap_stream(stream), total)
                    .file_name("replay.mp4")
                    .mime_str("video/mp4")
                    .map_err(|e| RemoteError::new("upload_failed", e.to_string()))?;
                Ok(c.post(url).multipart(reqwest::multipart::Form::new().part("file", part)))
            }
        })
    };
    let watchdog = async {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let idle = touched.lock().unwrap().elapsed();
            let limit = if sent.load(Ordering::SeqCst) >= total { UPLOAD_ANSWER } else { UPLOAD_STALL };
            if idle > limit {
                return;
            }
        }
    };
    // With every byte sent the server may already hold the file: wait for its id, so that new_run can
    // delete it. Before that, a cancel aborts the transfer.
    let cancel_mid_transfer = async {
        ctx.wait_cancel().await;
        if sent.load(Ordering::SeqCst) >= total {
            std::future::pending::<()>().await;
        }
    };
    let a = tokio::select! {
        r = send => r?,
        _ = watchdog => return Err(RemoteError::new("network", "The upload stalled. Check the connection and try again.")),
        _ = cancel_mid_transfer => return Err(ctx.cancelled("before_charge")),
    };
    classify_upload(&a)
}

/// POST the job. Runs to its answer even if the user cancels meanwhile; a second POST on the same file
/// never charges twice, so a lost answer can be retried.
async fn start_job(ctx: &Ctx, file_id: &str) -> Result<JobView, RemoteError> {
    ctx.stage("start", "Starting the analysis");
    // From here the server may charge: a cancel is no longer "nothing spent"
    set_run(&ctx.id, |i| i.phase = "queued".into());
    let url = ctx.auth.url(api::PATH_OPENERS);
    let body = json!({ "fileId": file_id });
    let (mut starting, mut network) = (0, 0);
    loop {
        let (u, b) = (url.clone(), body.clone());
        let a = ctx.auth.send(move |c| {
            let (u, b) = (u.clone(), b.clone());
            async move { Ok(c.post(u).json(&b).timeout(Duration::from_secs(120))) }
        });
        match a.await {
            Ok(a) => match classify_start(&a) {
                StartOutcome::Started(job, q) => {
                    if let Some(q) = q {
                        ctx.quota(&q);
                    }
                    return Ok(job);
                }
                StartOutcome::Retry(secs) if starting < START_RETRIES => {
                    starting += 1;
                    if ctx.is_cancelled() {
                        return Err(ctx.cancelled("queued"));
                    }
                    ctx.nap(Duration::from_secs(secs)).await;
                }
                StartOutcome::Retry(_) => {
                    return Err(RemoteError::new("start_failed", "The analysis is still starting. Please try again in a moment."));
                }
                StartOutcome::Fail(e) => return Err(e),
            },
            Err(e) if e.code == "network" && network < START_NETWORK_RETRIES => {
                network += 1;
                ctx.nap(Duration::from_secs(5)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Fetch, validate and store the result. The previous result stays unless the new one is valid.
async fn fetch_result(ctx: &Ctx, job_id: &str) -> Result<Value, RemoteError> {
    ctx.stage("result", "Getting the result");
    let path = api::path_job_result(job_id);
    let mut tries = 0;
    let a = loop {
        match ctx.auth.send_simple(Method::GET, &path, Duration::from_secs(300)).await {
            Ok(a) if a.status == 200 => break a,
            Ok(a) if a.status == 404 => return Err(RemoteError::new("expired", "The server no longer has this analysis.")),
            Err(e) if e.code == "signed_out" || e.code == "scope" => return Err(e),
            _ if tries < 4 => {
                tries += 1;
                ctx.nap(ctx.poll_error).await;
            }
            Ok(_) => return Err(RemoteError::new("bad_result", "The server didn't return the result.")),
            Err(e) => return Err(e),
        }
    };
    store_result(&ctx.root, &ctx.id, &a.text)
}

/// Validate first, then replace in one step.
pub fn store_result(root: &Path, id: &str, text: &str) -> Result<Value, RemoteError> {
    let value = lab::parse_lab_result(text).map_err(|e| RemoteError::new("bad_result", format!("The server's result can't be read by this version of the app: {}", e)))?;
    let path = lab::lab_result_path_in(root, id).map_err(|e| RemoteError::new("bad_result", e))?;
    let dir = path.parent().ok_or_else(|| RemoteError::new("bad_result", "No folder for the result"))?;
    std::fs::create_dir_all(dir).map_err(|e| RemoteError::new("bad_result", e.to_string()))?;
    let tmp = dir.join("lab_result.json.part");
    std::fs::write(&tmp, text).map_err(|e| RemoteError::new("bad_result", e.to_string()))?;
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        RemoteError::new("bad_result", e.to_string())
    })?;
    Ok(value)
}

fn job_failed(job: &JobView, quota: Option<Quota>) -> RemoteError {
    let mut e = RemoteError::new("job_failed", job.error.clone().unwrap_or_else(|| "The analysis failed.".into()));
    // True only for the call that gave it back; false says nothing either way (the UI shows the quota)
    e.refunded = job.analysis_refunded;
    e.quota = quota;
    e
}

/// DELETE the job, then see how it ended. Paid for and finished anyway: keep the result.
async fn cancel_job(ctx: &Ctx, job_id: &str, phase: &str) -> Result<Value, RemoteError> {
    let answer = ctx.auth.send_simple(Method::DELETE, &api::path_job(job_id), CALL_TIMEOUT).await;
    if let Ok(a) = &answer {
        if a.status == 409 {
            if let Ok(j) = ctx.auth.send_simple(Method::GET, &api::path_job(job_id), CALL_TIMEOUT).await {
                if let Some(job) = j.parse::<JobView>() {
                    match job.status.as_str() {
                        "done" => return fetch_result(ctx, job_id).await,
                        "failed" => return Err(job_failed(&job, None)),
                        _ => {}
                    }
                }
            }
        }
    }
    Err(ctx.cancelled(phase))
}

/// Poll until the job ends. No overall timeout: a job can wait in the queue for hours.
async fn follow_job(ctx: &Ctx, job_id: &str) -> Result<Value, RemoteError> {
    let mut phase = "queued".to_string();
    let mut last: Option<(String, Option<i64>)> = None;
    let path = api::path_job(job_id);
    loop {
        if ctx.is_cancelled() {
            return cancel_job(ctx, job_id, &phase).await;
        }
        let mut wait = ctx.poll;
        match ctx.auth.send_simple(Method::GET, &path, CALL_TIMEOUT).await {
            Ok(a) if a.status == 200 => {
                let Some(job) = a.parse::<JobView>() else {
                    ctx.nap(ctx.poll_error).await;
                    continue;
                };
                match job.status.as_str() {
                    "done" => return fetch_result(ctx, job_id).await,
                    "failed" => return Err(job_failed(&job, None)),
                    "cancelled" => return Err(ctx.cancelled(&phase)),
                    "queued" => phase = "queued".into(),
                    _ => phase = "running".into(),
                }
                let stage = if job.status == "queued" { "queued".to_string() } else { job.stage.clone().unwrap_or_else(|| "starting".into()) };
                set_run(&ctx.id, |i| i.phase = phase.clone());
                let now = (stage.clone(), job.position);
                if last.as_ref() != Some(&now) {
                    let line = match job.position {
                        Some(p) if job.status == "queued" => format!("position {}", p),
                        _ => stage.clone(),
                    };
                    ctx.progress(Progress { stage, line, position: job.position, ..Default::default() });
                    last = Some(now);
                }
            }
            Ok(a) if a.status == 404 => return Err(RemoteError::new("expired", "The server no longer has this analysis.")),
            Err(e) if e.code == "signed_out" || e.code == "scope" || e.code == "credential_store" => return Err(e),
            // Other answers and lost connections: keep polling, slower
            _ => wait = ctx.poll_error,
        }
        ctx.nap(wait).await;
    }
}

/// From a started (or saved) job to its end, and the clean-up. A sign-out keeps the job saved for
/// later: the file stays until the job can be collected.
async fn finish_job(ctx: &Ctx, saved: SavedJob, job_id: &str) -> Result<Value, RemoteError> {
    let out = follow_job(ctx, job_id).await;
    if matches!(&out, Err(e) if e.code == "signed_out" || e.code == "credential_store") {
        return out;
    }
    if let Some(f) = saved.file_id.as_deref() {
        delete_file(&ctx.auth, &ctx.root, f).await;
    }
    if matches!(&out, Err(e) if e.code == "network" || e.code == "unavailable") {
        // The job is done and paid for, only its result didn't arrive: collect it later
        save_job(&ctx.root, &ctx.id, &SavedJob { file_id: None, ..saved });
        return out;
    }
    remove_job(&ctx.root, &ctx.id);
    out
}

/// The whole run of a new analysis.
async fn new_run(ctx: &Ctx, ffmpeg: &str, video: &Path) -> Result<Value, RemoteError> {
    ctx.stage("check", "Checking the video");
    let status = lab_auth::service_status(&ctx.auth).await?;
    if !status.available {
        return Err(RemoteError::unavailable());
    }
    if let (Some(q), Some(m)) = (status.queued, status.max_queue) {
        if q >= m {
            return Err(RemoteError::new("queue_full", "The analysis queue is full. Please try again later."));
        }
    }
    let Some(q) = lab_auth::quota(&ctx.auth).await? else { return Err(RemoteError::unavailable()) };
    ctx.quota(&q);
    if q.free_remaining <= 0 && q.credits <= 0 {
        // Saves the upload; the server still decides and still answers quota_exhausted
        let mut e = RemoteError::new("quota_exhausted", "You have no analyses left this week.");
        e.quota = Some(q);
        return Err(e);
    }
    let info = probe(ffmpeg, video).await?;
    check_limits(info)?;

    // Removed on every path when this guard drops
    let tmp_root = ctx.root.join(".tmp");
    std::fs::create_dir_all(&tmp_root).map_err(|e| RemoteError::new("no_video", e.to_string()))?;
    let tmp = tempfile::Builder::new().prefix("upload").tempdir_in(&tmp_root).map_err(|e| RemoteError::new("no_video", e.to_string()))?;
    let copy = tmp.path().join("replay.mp4");
    let total = prepare(ctx, ffmpeg, video, info, &copy).await?;

    ctx.progress(Progress { stage: "upload".into(), line: "0%".into(), percent: Some(0.0), sent: Some(0), total: Some(total), ..Default::default() });
    let file_id = upload(ctx, &copy, total).await?;
    drop(tmp);
    let player = ctx.auth.player_id().await;
    let mut saved = SavedJob { file_id: Some(file_id.clone()), job_id: None, player_user_id: player };
    save_job(&ctx.root, &ctx.id, &saved);

    if ctx.is_cancelled() {
        delete_file(&ctx.auth, &ctx.root, &file_id).await;
        remove_job(&ctx.root, &ctx.id);
        return Err(ctx.cancelled("before_charge"));
    }

    let job = match start_job(ctx, &file_id).await {
        Ok(j) => j,
        Err(e) => {
            if e.code != "signed_out" {
                // Without a job: nothing to wait for. If an earlier POST did reach the server, that job
                // fails on the missing file and the analysis is given back.
                delete_file(&ctx.auth, &ctx.root, &file_id).await;
                remove_job(&ctx.root, &ctx.id);
            }
            return Err(e);
        }
    };
    saved.job_id = Some(job.id.clone());
    save_job(&ctx.root, &ctx.id, &saved);
    ctx.progress(Progress {
        stage: "start".into(),
        line: job.analysis_source.clone().unwrap_or_else(|| "reused".into()),
        source: job.analysis_source.clone(),
        ..Default::default()
    });
    let phase = if job.status == "queued" { "queued" } else { "running" };
    set_run(&ctx.id, |i| i.phase = phase.into());
    finish_job(ctx, saved, &job.id).await
}

/// What every ended run does last: a fresh quota (the server decides any give-back) on the error too.
async fn conclude(ctx: &Ctx, out: Result<Value, RemoteError>) -> Result<Value, RemoteError> {
    let mut out = out;
    let signed_out = matches!(&out, Err(e) if e.code == "signed_out");
    if !signed_out {
        if let Ok(Some(q)) = lab_auth::quota(&ctx.auth).await {
            ctx.quota(&q);
            if let Err(e) = &mut out {
                if e.quota.is_none() {
                    e.quota = Some(q);
                }
            }
        }
    }
    match &out {
        Ok(_) => ctx.stage("done", "done"),
        Err(e) => eprintln!("[LAB] hosted run ended: {}", e.code),
    }
    out
}

fn test_poll() -> Option<Duration> {
    if !api::test_mode() {
        return None;
    }
    std::env::var("SPAGHETTI_LAB_TEST_POLL_MS").ok().and_then(|v| v.parse().ok()).map(Duration::from_millis)
}

fn app_ctx(app: &tauri::AppHandle, auth: Arc<Auth>, id: &str, cancel: watch::Receiver<bool>) -> Ctx {
    let handle = app.clone();
    let poll = test_poll();
    Ctx {
        auth,
        root: lab::lab_root(),
        id: id.to_string(),
        emit: Arc::new(move |event, payload| {
            let _ = handle.emit(event, payload);
        }),
        cancel,
        poll: poll.unwrap_or(POLL),
        poll_error: poll.unwrap_or(POLL_AFTER_ERROR),
    }
}

// ── commands ────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn run_lab_analysis_remote(app: tauri::AppHandle, db_path: String, replay_id: String, options: Option<RemoteOptions>) -> Result<Value, RemoteError> {
    let options = options.unwrap_or_default();
    let id = lab::safe_id(&replay_id).map_err(|e| RemoteError::new("no_video", e))?;
    let auth = lab_auth::auth()?;
    if !auth.account().await.signed_in {
        return Err(RemoteError::signed_out());
    }
    let root = lab::lab_root();
    let me = auth.player_id().await;
    let mut saved = read_job(&root, &id);
    // Another player's job on this PC isn't ours to collect: its file is deleted when they sign in
    if let Some(other) = saved.as_ref().filter(|s| s.player_user_id != me) {
        if let Some(f) = other.file_id.as_deref() {
            add_pending(&root, f, other.player_user_id.clone());
        }
        remove_job(&root, &id);
        saved = None;
    }
    let has_job = saved.as_ref().map_or(false, |s| s.job_id.is_some());
    if !has_job && !options.again.unwrap_or(false) && lab::lab_result_path_in(&root, &id).map_or(false, |p| p.is_file()) {
        return Err(RemoteError::new("already_cached", "This replay already has a result. Analysing it again spends an analysis."));
    }
    let (_guard, cancel) = claim_run(&id)?;
    let ctx = app_ctx(&app, auth, &id, cancel);

    // A job already paid for: collect it instead of spending again
    if let Some(job_id) = saved.as_ref().and_then(|s| s.job_id.clone()) {
        set_run(&id, |i| i.phase = "queued".into());
        let out = finish_job(&ctx, saved.unwrap_or_default(), &job_id).await;
        return remember_error(&id, conclude(&ctx, out).await);
    }
    // An upload whose job never started: nothing will use that file
    if let Some(f) = saved.as_ref().and_then(|s| s.file_id.as_deref()) {
        delete_file(&ctx.auth, &ctx.root, f).await;
        remove_job(&ctx.root, &ctx.id);
    }
    let out = async {
        let video = match options.video_path.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) if Path::new(v).is_file() => PathBuf::from(v),
            Some(v) => return Err(RemoteError::new("no_video", format!("Video file not found: {}", v))),
            None => {
                let conn = rusqlite::Connection::open(&db_path).map_err(|e| RemoteError::new("no_video", e.to_string()))?;
                crate::resolve_video_for_replay(&conn, &replay_id).map_err(|e| RemoteError::new("no_video", e))?
            }
        };
        let ffmpeg = crate::ffmpeg_bin().map_err(|e| RemoteError::new("no_video", e))?;
        new_run(&ctx, &ffmpeg, &video).await
    }
    .await;
    remember_error(&id, conclude(&ctx, out).await)
}

/// The screen may have been closed meanwhile; it asks with `lab_remote_take_error` when it comes back.
fn remember_error(id: &str, out: Result<Value, RemoteError>) -> Result<Value, RemoteError> {
    if let Err(e) = &out {
        if e.code != "cancelled" {
            *LAST_ERROR.lock().unwrap() = Some((id.to_string(), e.clone()));
        }
    }
    out
}

#[derive(Serialize)]
pub struct CancelAnswer {
    pub phase: String,
}

/// Ask the run to stop; its command then ends with `cancelled`. Before the charge nothing was spent;
/// a queued job is given back; a running one is not (the UI warned).
#[tauri::command]
pub fn cancel_lab_analysis_remote(replay_id: String) -> Result<CancelAnswer, RemoteError> {
    let g = RUN.lock().unwrap();
    let run = g.as_ref().filter(|r| r.info.replay_id == replay_id).ok_or_else(|| RemoteError::new("cancelled", "No analysis of this replay is running."))?;
    let _ = run.cancel.send(true);
    Ok(CancelAnswer { phase: run.info.phase.clone() })
}

#[tauri::command]
pub fn lab_remote_running() -> Option<RunningInfo> {
    RUN.lock().unwrap().as_ref().map(|r| r.info.clone())
}

/// How the last background run of this replay ended, once (the screen may have been closed meanwhile).
#[tauri::command]
pub fn lab_remote_take_error(replay_id: String) -> Option<RemoteError> {
    let mut g = LAST_ERROR.lock().unwrap();
    if g.as_ref().map_or(false, |(id, _)| *id == replay_id) {
        return g.take().map(|(_, e)| e);
    }
    None
}

/// Leftovers of an earlier session: temp copies, deletes that failed, saved jobs. A job with an id is
/// collected; a file without a job is deleted, because the user didn't ask to spend again.
#[tauri::command]
pub async fn lab_remote_resume(app: tauri::AppHandle) -> Result<(), RemoteError> {
    let auth = lab_auth::auth()?;
    let root = lab::lab_root();
    if running_replay().is_none() {
        let _ = std::fs::remove_dir_all(root.join(".tmp"));
    }
    if !auth.account().await.signed_in {
        return Ok(());
    }
    tauri::async_runtime::spawn(async move { resume_all(app, auth, root).await });
    Ok(())
}

/// After a sign-in: deletes waiting for this player. The UI then calls `lab_remote_resume` for the
/// saved jobs.
pub fn after_sign_in(auth: Arc<Auth>) {
    tauri::async_runtime::spawn(async move { retry_pending_deletes(&auth, &lab::lab_root()).await });
}

fn saved_jobs(root: &Path) -> Vec<(String, SavedJob)> {
    let Ok(dirs) = std::fs::read_dir(root) else { return Vec::new() };
    dirs.filter_map(|d| {
        let name = d.ok()?.file_name().to_string_lossy().to_string();
        let job = read_job(root, &name)?;
        Some((name, job))
    })
    .collect()
}

async fn resume_all(app: tauri::AppHandle, auth: Arc<Auth>, root: PathBuf) {
    retry_pending_deletes(&auth, &root).await;
    let me = auth.player_id().await;
    for (id, job) in saved_jobs(&root) {
        // A live run owns its saved job, even before the job has an id
        if job.player_user_id != me || running_replay().as_deref() == Some(id.as_str()) {
            continue;
        }
        let Some(job_id) = job.job_id.clone() else {
            if let Some(f) = job.file_id.as_deref() {
                delete_file(&auth, &root, f).await;
            }
            remove_job(&root, &id);
            continue;
        };
        let Ok((_guard, cancel)) = claim_run(&id) else { continue };
        set_run(&id, |i| i.phase = "queued".into());
        let ctx = app_ctx(&app, auth.clone(), &id, cancel);
        let out = finish_job(&ctx, job, &job_id).await;
        let _ = remember_error(&id, conclude(&ctx, out).await);
        let _ = app.emit("lab-remote-ended", json!({ "replay_id": id }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab_auth::tests::{mock, signed_in_auth, token_json, Mock};

    fn run<T>(f: impl std::future::Future<Output = T>) -> T {
        tauri::async_runtime::block_on(f)
    }

    const PROBE_720: &str = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'x.mp4':\n  Metadata:\n    major_brand     : isom\n  Duration: 00:03:12.04, start: 0.000000, bitrate: 1640 kb/s\n  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1280x720 [SAR 1:1 DAR 16:9], 1500 kb/s, 30 fps, 30 tbr\n  Stream #0:1: Audio: aac, 48000 Hz, stereo\nAt least one output file must be specified";

    #[test]
    fn lab_probe_reads_duration_and_height() {
        let v = parse_probe(PROBE_720).unwrap();
        assert!((v.seconds - 192.04).abs() < 0.001);
        assert_eq!(v.height, 720);
        // codec tags with an x in them aren't a size
        let hex = "  Duration: 00:00:10.00, start\n  Stream #0:0: Video: h264 (avc1 / 0x31637661), yuv420p, 1920x1080, 60 fps";
        assert_eq!(parse_probe(hex).unwrap().height, 1080);
        assert!(parse_probe("not a video").is_none());
        assert!(parse_probe("  Duration: N/A, bitrate\n").is_none());
    }

    #[test]
    fn lab_limits_are_checked_with_clear_errors() {
        assert!(check_limits(VideoInfo { seconds: 900.0, height: 720 }).is_ok());
        let long = check_limits(VideoInfo { seconds: 7200.0, height: 1080 }).unwrap_err();
        assert_eq!(long.code, "too_long");
        assert!(long.message.contains("cut it to one set"));
        assert_eq!(check_limits(VideoInfo { seconds: 60.0, height: 240 }).unwrap_err().code, "too_small");
        assert!(too_big().message.contains("cut it to one set"));
    }

    #[test]
    fn lab_encode_only_when_needed() {
        let v = |h| VideoInfo { seconds: 60.0, height: h };
        assert!(!needs_encode(v(720), MAX_BYTES));
        assert!(needs_encode(v(1080), 1000));
        assert!(needs_encode(v(720), MAX_BYTES + 1));
        assert!(!needs_encode(v(480), 1000));
    }

    #[test]
    fn lab_copy_never_keeps_audio_or_metadata_and_uses_libx264() {
        let enc = copy_args("in.mp4", "out.mp4", true, None).join(" ");
        assert!(enc.contains("-an") && enc.contains("-map_metadata -1"));
        assert!(enc.contains("libx264") && enc.contains("-b:v 3M") && enc.contains("fps=30") && enc.contains("min(720,ih)"));
        assert!(!enc.contains("nvenc"));
        let copy = copy_args("in.mp4", "out.mp4", false, None).join(" ");
        assert!(copy.contains("-c:v copy") && copy.contains("-an") && copy.contains("-map_metadata -1"));
        assert!(copy.ends_with("out.mp4"));
    }

    /// The real encode on two seconds of the small e2e clip; skipped when the clip isn't there.
    #[test]
    fn lab_encode_two_seconds_of_the_e2e_clip() {
        let clip = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../e2e/.cache/e2e_set.mp4");
        let Ok(ffmpeg) = crate::ffmpeg_bin() else { return };
        if !clip.is_file() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("o.mp4");
        let st = std::process::Command::new(&ffmpeg)
            .args(copy_args(&clip.to_string_lossy(), &out.to_string_lossy(), true, Some(2)))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(st.success());
        let probe = std::process::Command::new(&ffmpeg).arg("-hide_banner").arg("-i").arg(&out).output().unwrap();
        let text = String::from_utf8_lossy(&probe.stderr).to_string();
        let v = parse_probe(&text).unwrap();
        assert_eq!(v.height, 720);
        assert!(text.contains("30 fps"));
        assert!(!text.contains("Audio:"));
    }

    fn outcome(status: u16, retry: Option<u64>, body: &str) -> StartOutcome {
        classify_start(&Answer::new(status, retry, body.into()))
    }
    fn fail_code(o: StartOutcome) -> String {
        match o {
            StartOutcome::Fail(e) => e.code,
            StartOutcome::Started(..) => "started".into(),
            StartOutcome::Retry(_) => "retry".into(),
        }
    }

    #[test]
    fn lab_start_answers_are_told_apart() {
        let q = r#""free_remaining":0,"free_per_week":2,"credits":0,"week_resets_at":"2026-11-09T00:00:00+01:00""#;
        let job = format!(r#"{{"id":"j1","status":"queued","stage":null,"position":3,"analysis_source":"free",{q}}}"#);
        match outcome(202, None, &job) {
            StartOutcome::Started(j, quota) => {
                assert_eq!(j.analysis_source.as_deref(), Some("free"));
                assert_eq!(j.position, Some(3));
                assert_eq!(quota.unwrap().free_per_week, 2);
            }
            _ => panic!(),
        }
        match outcome(200, None, r#"{"id":"j1","status":"done","reused":true}"#) {
            StartOutcome::Started(j, _) => assert_eq!(j.analysis_source, None),
            _ => panic!(),
        }
        match outcome(429, Some(9000), &format!(r#"{{"error":"quota_exhausted",{q}}}"#)) {
            StartOutcome::Fail(e) => {
                assert_eq!(e.code, "quota_exhausted");
                assert_eq!(e.retry_after, Some(9000));
                assert!(e.quota.is_some());
            }
            _ => panic!(),
        }
        assert!(matches!(outcome(409, Some(5), &format!(r#"{{"error":"analysis_starting",{q}}}"#)), StartOutcome::Retry(5)));
        assert_eq!(fail_code(outcome(429, Some(60), r#"{"detail":"you already have an analysis queued or running"}"#)), "one_at_a_time");
        assert_eq!(fail_code(outcome(429, None, "Too many requests from this IP")), "rate_limited");
        assert_eq!(fail_code(outcome(503, Some(300), r#"{"detail":"the analysis queue is full, try again later"}"#)), "queue_full");
        assert_eq!(fail_code(outcome(503, None, r#"{"error":"Opener analysis is not available"}"#)), "unavailable");
        assert_eq!(fail_code(outcome(413, None, r#"{"error":"The video is over 15 min: cut it to one match or set"}"#)), "too_long");
        assert_eq!(fail_code(outcome(502, None, "<html>")), "unavailable");
        assert_eq!(fail_code(outcome(404, None, r#"{"error":"File not found"}"#)), "start_failed");
    }

    #[test]
    fn lab_upload_answers_are_told_apart() {
        let a = |s, b: &str| classify_upload(&Answer::new(s, None, b.into()));
        assert_eq!(a(200, r#"{"success":true,"data":{"file_id":"f1","compression_status":"skipped"}}"#).unwrap(), "f1");
        assert_eq!(a(413, "<html>too large</html>").unwrap_err().code, "too_big");
        assert_eq!(a(413, r#"{"error":"file_too_large","message":"File exceeds the 500MB limit."}"#).unwrap_err().code, "too_big");
        let full = a(400, r#"{"error":"Your storage limit reached (5GB). Delete some files first."}"#).unwrap_err();
        assert_eq!(full.code, "storage_full");
        assert!(full.message.contains("5GB"));
        assert_eq!(a(403, r#"{"error":"banned_global","message":"Banned."}"#).unwrap_err().code, "banned");
        assert_eq!(a(500, r#"{"error":"Failed to upload video"}"#).unwrap_err().code, "upload_failed");
        assert_eq!(a(502, "<html>").unwrap_err().code, "unavailable");
    }

    fn ctx_for(m: &Mock, root: &Path, id: &str) -> (Ctx, watch::Sender<bool>, Arc<Mutex<Vec<(String, Value)>>>) {
        let (auth, _) = signed_in_auth(&m.url);
        let (tx, rx) = watch::channel(false);
        let events: Arc<Mutex<Vec<(String, Value)>>> = Default::default();
        let e2 = events.clone();
        let ctx = Ctx {
            auth,
            root: root.to_path_buf(),
            id: id.into(),
            emit: Arc::new(move |e, v| e2.lock().unwrap().push((e.to_string(), v))),
            cancel: rx,
            poll: Duration::from_millis(10),
            poll_error: Duration::from_millis(10),
        };
        (ctx, tx, events)
    }

    fn fixture() -> String {
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../e2e/fixtures/lab_result.json")).unwrap()
    }

    #[test]
    fn lab_upload_streams_a_named_video_part_with_a_known_length() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let file = tmp.path().join("Jason_vs_Enero.mp4");
            std::fs::write(&file, vec![7u8; 600_000]).unwrap();
            let seen: Arc<Mutex<Vec<String>>> = Default::default();
            let s2 = seen.clone();
            let m = mock(move |_, p, b, _| {
                if p.contains("token") {
                    return (200, token_json(1));
                }
                s2.lock().unwrap().push(b.chars().take(300).collect());
                assert!(b.len() > 600_000, "the whole file arrived");
                (200, r#"{"success":true,"data":{"file_id":"f1"}}"#.into())
            });
            let (ctx, _tx, events) = ctx_for(&m, tmp.path(), "r1");
            assert_eq!(upload(&ctx, &file, 600_000).await.unwrap(), "f1");
            let head = seen.lock().unwrap()[0].clone();
            assert!(head.contains("name=\"file\""));
            assert!(head.contains("filename=\"replay.mp4\""), "never the local file name");
            assert!(head.contains("video/mp4"));
            let ev = events.lock().unwrap();
            let last = ev.iter().rev().find(|(e, _)| e == "lab-progress").unwrap();
            assert_eq!(last.1["percent"], 100.0);
            assert_eq!(last.1["line"], "Finishing the upload");
        });
    }

    #[test]
    fn lab_cancel_aborts_the_upload_wait() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let file = tmp.path().join("v.mp4");
            std::fs::write(&file, vec![0u8; 1000]).unwrap();
            let m = mock(|_, p, _, _| {
                if p.contains("token") {
                    return (200, token_json(1));
                }
                std::thread::sleep(Duration::from_secs(20));
                (200, "{}".into())
            });
            let (ctx, tx, _) = ctx_for(&m, tmp.path(), "r1");
            tx.send(true).unwrap();
            let started = Instant::now();
            assert_eq!(upload(&ctx, &file, 1000).await.unwrap_err().code, "cancelled");
            assert!(started.elapsed() < Duration::from_secs(5));
        });
    }

    /// A job through its stages: result stored, stage events, the file deleted, the job forgotten.
    #[test]
    fn lab_a_job_ends_with_a_stored_result_and_a_deleted_file() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let polls = Arc::new(Mutex::new(0));
            let p2 = polls.clone();
            let m = mock(move |method, p, _, _| {
                if p.contains("token") {
                    return (200, token_json(1));
                }
                if method == "DELETE" {
                    assert_eq!(p, "/api/user-files/f1");
                    assert!(*p2.lock().unwrap() >= 3, "the file must wait for the job to end");
                    return (200, r#"{"success":true}"#.into());
                }
                if p.ends_with("/result") {
                    return (200, fixture());
                }
                let mut n = p2.lock().unwrap();
                *n += 1;
                match *n {
                    1 => (200, r#"{"id":"j1","status":"queued","stage":null,"position":2}"#.into()),
                    2 => (200, r#"{"id":"j1","status":"running","stage":"hud"}"#.into()),
                    _ => (200, r#"{"id":"j1","status":"done","stage":"done"}"#.into()),
                }
            });
            let (ctx, _tx, events) = ctx_for(&m, tmp.path(), "r1");
            let saved = SavedJob { file_id: Some("f1".into()), job_id: Some("j1".into()), player_user_id: Some("u-1".into()) };
            save_job(tmp.path(), "r1", &saved);
            let v = finish_job(&ctx, saved, "j1").await.unwrap();
            assert!(v["openers"].is_array());
            assert!(tmp.path().join("r1/lab_result.json").is_file());
            assert!(!tmp.path().join("r1/lab_result.json.part").exists());
            assert!(read_job(tmp.path(), "r1").is_none());
            let ev = events.lock().unwrap();
            let stages: Vec<String> = ev.iter().map(|(_, v)| v["stage"].as_str().unwrap_or("").to_string()).collect();
            assert!(stages.contains(&"queued".to_string()) && stages.contains(&"hud".to_string()) && stages.contains(&"result".to_string()));
            assert!(ev.iter().any(|(_, v)| v["position"] == 2));
            assert!(m.calls.lock().unwrap().iter().any(|c| c.0 == "DELETE"));
        });
    }

    #[test]
    fn lab_a_bad_result_keeps_the_old_one() {
        let tmp = tempfile::tempdir().unwrap();
        store_result(tmp.path(), "r1", &fixture()).unwrap();
        let e = store_result(tmp.path(), "r1", r#"{"openers":[]}"#).unwrap_err();
        assert_eq!(e.code, "bad_result");
        let kept = std::fs::read_to_string(tmp.path().join("r1/lab_result.json")).unwrap();
        assert!(lab::parse_lab_result(&kept).is_ok());
    }

    #[test]
    fn lab_a_failed_job_reports_the_refund_only_when_the_server_says_so() {
        run(async {
            for (body, expect) in [
                (r#"{"id":"j","status":"failed","error":"not a readable video","analysis_refunded":true}"#, Some(true)),
                (r#"{"id":"j","status":"failed","analysis_refunded":false}"#, Some(false)),
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let m = mock(move |method, p, _, _| {
                    if p.contains("token") {
                        (200, token_json(1))
                    } else if method == "DELETE" {
                        (200, "{}".into())
                    } else {
                        (200, body.to_string())
                    }
                });
                let (ctx, _tx, _) = ctx_for(&m, tmp.path(), "r1");
                let saved = SavedJob { file_id: Some("f1".into()), job_id: Some("j".into()), player_user_id: None };
                let e = finish_job(&ctx, saved, "j").await.unwrap_err();
                assert_eq!(e.code, "job_failed");
                assert_eq!(e.refunded, expect);
                assert!(!e.message.is_empty());
                assert!(m.calls.lock().unwrap().iter().any(|c| c.0 == "DELETE" && c.1 == "/api/user-files/f1"));
            }
        });
    }

    #[test]
    fn lab_cancel_deletes_the_job_then_the_file() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|method, p, _, _| {
                if p.contains("token") {
                    (200, token_json(1))
                } else if method == "DELETE" {
                    (200, r#"{"cancelled":true}"#.into())
                } else {
                    (200, r#"{"id":"j","status":"running","stage":"hud"}"#.into())
                }
            });
            let (ctx, tx, _) = ctx_for(&m, tmp.path(), "r1");
            tx.send(true).unwrap();
            let saved = SavedJob { file_id: Some("f1".into()), job_id: Some("j".into()), player_user_id: None };
            assert_eq!(finish_job(&ctx, saved, "j").await.unwrap_err().code, "cancelled");
            let calls = m.calls.lock().unwrap();
            let deletes: Vec<&str> = calls.iter().filter(|c| c.0 == "DELETE").map(|c| c.1.as_str()).collect();
            assert_eq!(deletes, vec!["/api/spaghetti-lab/openers/j", "/api/user-files/f1"]);
        });
    }

    #[test]
    fn lab_a_cancel_too_late_keeps_the_paid_result() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|method, p, _, _| {
                if p.contains("token") {
                    (200, token_json(1))
                } else if method == "DELETE" && p.contains("openers") {
                    (409, r#"{"detail":"the job has already ended"}"#.into())
                } else if method == "DELETE" {
                    (200, "{}".into())
                } else if p.ends_with("/result") {
                    (200, fixture())
                } else {
                    (200, r#"{"id":"j","status":"done","stage":"done"}"#.into())
                }
            });
            let (ctx, tx, _) = ctx_for(&m, tmp.path(), "r1");
            tx.send(true).unwrap();
            let saved = SavedJob { file_id: Some("f1".into()), job_id: Some("j".into()), player_user_id: None };
            assert!(finish_job(&ctx, saved, "j").await.is_ok());
            assert!(tmp.path().join("r1/lab_result.json").is_file());
        });
    }

    #[test]
    fn lab_a_sign_out_while_polling_keeps_the_job_saved() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|_, p, _, _| {
                if p.contains("token") {
                    (400, r#"{"error":"invalid_grant"}"#.into())
                } else {
                    (401, r#"{"error":"invalid_token"}"#.into())
                }
            });
            let (ctx, _tx, _) = ctx_for(&m, tmp.path(), "r1");
            let saved = SavedJob { file_id: Some("f1".into()), job_id: Some("j".into()), player_user_id: None };
            save_job(tmp.path(), "r1", &saved);
            assert_eq!(finish_job(&ctx, saved.clone(), "j").await.unwrap_err().code, "signed_out");
            assert_eq!(read_job(tmp.path(), "r1"), Some(saved));
        });
    }

    #[test]
    fn lab_a_404_job_is_expired_and_errors_keep_polling() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let n = Arc::new(Mutex::new(0));
            let m = mock(move |method, p, _, _| {
                if p.contains("token") {
                    return (200, token_json(1));
                }
                if method == "DELETE" {
                    return (200, "{}".into());
                }
                let mut n = n.lock().unwrap();
                *n += 1;
                match *n {
                    1 => (502, "<html>".into()),
                    2 => (429, "Too many requests".into()),
                    3 => (503, r#"{"error":"unavailable"}"#.into()),
                    _ => (404, r#"{"error":"Analysis not found"}"#.into()),
                }
            });
            let (ctx, _tx, _) = ctx_for(&m, tmp.path(), "r1");
            let saved = SavedJob { file_id: Some("f1".into()), job_id: Some("j".into()), player_user_id: None };
            assert_eq!(finish_job(&ctx, saved, "j").await.unwrap_err().code, "expired");
        });
    }

    #[test]
    fn lab_a_403_on_file_delete_is_a_warning_and_a_failure_is_retried_later() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|_, p, _, _| if p.contains("token") { (200, token_json(1)) } else { (403, r#"{"error":"scope"}"#.into()) });
            let (auth, store) = signed_in_auth(&m.url);
            delete_file(&auth, tmp.path(), "f1").await;
            assert!(read_pending(tmp.path()).is_empty());
            use crate::lab_auth::TokenStore as _;
            assert!(store.get("refresh_token").unwrap().is_some(), "not a sign-out");

            let down = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let d2 = down.clone();
            let m = mock(move |_, p, _, _| {
                if p.contains("token") {
                    (200, token_json(1))
                } else if d2.load(Ordering::SeqCst) {
                    (502, "<html>".into())
                } else {
                    (200, "{}".into())
                }
            });
            let (auth, _) = signed_in_auth(&m.url);
            delete_file(&auth, tmp.path(), "f2").await;
            let list = read_pending(tmp.path());
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].player_user_id.as_deref(), Some("u-1"));
            let text = std::fs::read_to_string(pending_path(tmp.path())).unwrap();
            assert!(!text.contains("sslab_"));
            retry_pending_deletes(&auth, tmp.path()).await;
            assert_eq!(read_pending(tmp.path()).len(), 1, "still down: kept");
            down.store(false, Ordering::SeqCst);
            retry_pending_deletes(&auth, tmp.path()).await;
            assert!(read_pending(tmp.path()).is_empty());
            assert!(!pending_path(tmp.path()).exists());
        });
    }

    #[test]
    fn lab_a_result_lost_to_the_network_stays_collectable_without_the_file() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|method, p, _, _| {
                if p.contains("token") {
                    (200, token_json(1))
                } else if method == "DELETE" {
                    (200, "{}".into())
                } else if p.ends_with("/result") {
                    (503, r#"{"error":"unavailable"}"#.into())
                } else {
                    (200, r#"{"id":"j","status":"done","stage":"done"}"#.into())
                }
            });
            let (ctx, _tx, _) = ctx_for(&m, tmp.path(), "r1");
            let saved = SavedJob { file_id: Some("f1".into()), job_id: Some("j".into()), player_user_id: Some("u-1".into()) };
            assert_eq!(finish_job(&ctx, saved, "j").await.unwrap_err().code, "unavailable");
            let left = read_job(tmp.path(), "r1").unwrap();
            assert_eq!(left.file_id, None, "the upload is deleted");
            assert_eq!(left.job_id.as_deref(), Some("j"));
            assert!(m.calls.lock().unwrap().iter().any(|c| c.0 == "DELETE" && c.1 == "/api/user-files/f1"));
        });
    }

    /// Cancelling with every byte sent waits for the file's id: it's the only way to delete that file.
    #[test]
    fn lab_a_cancel_after_the_last_byte_waits_for_the_file_id() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let file = tmp.path().join("v.mp4");
            std::fs::write(&file, vec![0u8; 1000]).unwrap();
            let m = mock(|_, p, _, _| {
                if p.contains("token") {
                    return (200, token_json(1));
                }
                std::thread::sleep(Duration::from_millis(1500));
                (200, r#"{"success":true,"data":{"file_id":"f9"}}"#.into())
            });
            let (ctx, tx, _) = ctx_for(&m, tmp.path(), "r1");
            let c2 = ctx.clone();
            let file2 = file.clone();
            let up = tauri::async_runtime::spawn(async move { upload(&c2, &file2, 1000).await });
            tokio::time::sleep(Duration::from_millis(600)).await;
            tx.send(true).unwrap();
            assert_eq!(up.await.unwrap().unwrap(), "f9");
        });
    }

    /// Manual, on the dev server only (app/e2e/live/dev-auth-check.mjs passes a fresh code): the real upload
    /// and the file delete, with the small e2e clip. Never starts an analysis, so nothing is charged.
    /// Prints status codes and counts, never a token or an id.
    #[test]
    #[ignore]
    fn lab_live_dev_upload() {
        run(async {
            let base = api::check_base(&std::env::var("LAB_LIVE_API").expect("LAB_LIVE_API"), false).unwrap();
            assert!(base.starts_with("https://dev."), "this check runs on dev only");
            let code = std::env::var("LAB_LIVE_CODE").expect("LAB_LIVE_CODE");
            let verifier = std::env::var("LAB_LIVE_VERIFIER").expect("LAB_LIVE_VERIFIER");
            let clip = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../e2e/.cache/e2e_set.mp4");
            let total = std::fs::metadata(&clip).expect("the e2e clip").len();

            let store = Arc::new(crate::lab_auth::MemoryStore::default());
            let auth = Auth::new(base, store);
            auth.exchange(&code, &verifier).await.expect("exchange");
            let before = lab_auth::quota(&auth).await.unwrap().expect("quota");
            println!("quota before: free={} credits={}", before.free_remaining, before.credits);

            let tmp = tempfile::tempdir().unwrap();
            let (tx, rx) = watch::channel(false);
            let events: Arc<Mutex<Vec<(String, Value)>>> = Default::default();
            let e2 = events.clone();
            let ctx = Ctx {
                auth: auth.clone(),
                root: tmp.path().to_path_buf(),
                id: "live".into(),
                emit: Arc::new(move |e, v| e2.lock().unwrap().push((e.to_string(), v))),
                cancel: rx,
                poll: POLL,
                poll_error: POLL_AFTER_ERROR,
            };
            let started = Instant::now();
            let file_id = upload(&ctx, &clip, total).await.expect("upload");
            println!("upload: ok, {} bytes in {:.1} s, {} progress events", total, started.elapsed().as_secs_f64(), events.lock().unwrap().len());

            delete_file(&auth, tmp.path(), &file_id).await;
            println!("file delete: pending left = {}", read_pending(tmp.path()).len());
            assert!(read_pending(tmp.path()).is_empty());
            // A second delete: 404, which also counts as done
            let again = auth.send_simple(Method::DELETE, &api::path_file(&file_id), CALL_TIMEOUT).await.unwrap();
            println!("second file delete: {}", again.status);

            let after = lab_auth::quota(&auth).await.unwrap().expect("quota");
            println!("quota after: free={} credits={}", after.free_remaining, after.credits);
            assert_eq!((before.free_remaining, before.credits), (after.free_remaining, after.credits), "an upload alone spends nothing");
            auth.sign_out().await;
            drop(tx);
        });
    }

    #[test]
    fn lab_saved_files_hold_no_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        save_job(tmp.path(), "r1", &SavedJob { file_id: Some("f".into()), job_id: Some("j".into()), player_user_id: Some("u".into()) });
        let t = std::fs::read_to_string(job_path(tmp.path(), "r1")).unwrap();
        assert!(!t.contains("sslab_") && !t.contains("token"));
        assert_eq!(saved_jobs(tmp.path()).len(), 1);
    }

    #[test]
    fn lab_a_cancelled_run_removes_its_temp_copy() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|_, p, _, _| {
                if p.contains("token") {
                    (200, token_json(1))
                } else if p.ends_with("/status") {
                    (200, r#"{"available":true,"queued":0,"max_queue":8}"#.into())
                } else {
                    (200, r#"{"free_remaining":1,"free_per_week":2,"credits":0,"week_resets_at":"x"}"#.into())
                }
            });
            let (ctx, tx, _) = ctx_for(&m, tmp.path(), "r1");
            let clip = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../e2e/.cache/e2e_set.mp4");
            let Ok(ffmpeg) = crate::ffmpeg_bin() else { return };
            if !clip.is_file() {
                return;
            }
            tx.send(true).unwrap();
            assert_eq!(new_run(&ctx, &ffmpeg, &clip).await.unwrap_err().code, "cancelled");
            let left = std::fs::read_dir(tmp.path().join(".tmp")).unwrap().count();
            assert_eq!(left, 0, "temp copy removed");
        });
    }

    #[test]
    fn lab_nothing_left_stops_before_any_upload() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|_, p, _, _| {
                if p.contains("token") {
                    (200, token_json(1))
                } else if p.ends_with("/status") {
                    (200, r#"{"available":true,"queued":0,"max_queue":8}"#.into())
                } else {
                    (200, r#"{"free_remaining":0,"free_per_week":2,"credits":0,"week_resets_at":"x"}"#.into())
                }
            });
            let (ctx, _tx, _) = ctx_for(&m, tmp.path(), "r1");
            let e = new_run(&ctx, "ffmpeg-not-needed", Path::new("nope.mp4")).await.unwrap_err();
            assert_eq!(e.code, "quota_exhausted");
            assert!(e.quota.is_some());
            assert!(!m.calls.lock().unwrap().iter().any(|c| c.1.contains("upload")));
        });
    }

    #[test]
    fn lab_service_off_stops_first_and_keeps_the_session() {
        run(async {
            let tmp = tempfile::tempdir().unwrap();
            let m = mock(|_, _, _, _| (503, r#"{"error":"Opener analysis is not available"}"#.into()));
            let (ctx, _tx, _) = ctx_for(&m, tmp.path(), "r1");
            let e = new_run(&ctx, "ffmpeg-not-needed", Path::new("nope.mp4")).await.unwrap_err();
            assert_eq!(e.code, "unavailable");
            assert!(ctx.auth.account().await.signed_in);
        });
    }
}
