// ── Lab analysis on this PC (only with the `local-models` feature) ──────────
//
// Our own build (`npm run tauri:local`, app/build-local.ps1) runs the research models here instead of on the
// server: no sign-in, nothing spent. The builds we publish are compiled without this file, so they can't
// start tools/ggst_logger/model/lab_analyze.py (scripts/check_published_binary.py checks the binary).
//
// lab_analyze.py produces openers, the move transcript, the state streams and the match facts: the
// result the UI reads (lab.rs).
//
// AI summary: opt-in per run (`llm: "claude"` -> `--llm claude`). The script reads ANTHROPIC_API_KEY
// from the environment it inherits; the app never reads, stores or logs the key.
//
// Progress: the script's `stage: <name> ...` stdout lines go to the UI as "lab-progress" events
// ({replay_id, stage, line}); every other line is only logged. Cancelling kills the process tree.

use super::{default_output_paths, find_project_root, require_python, resolve_video_for_replay};
use crate::lab::{kill_tree, lab_root, parse_lab_result, safe_id};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Read as _};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::Emitter;

/// Errors that mean "the lab isn't set up on this PC" start with this, so the UI can show the setup
/// hint instead of a failure.
pub const NOT_INSTALLED: &str = "Lab analysis is not installed";

#[derive(Debug, Deserialize, Default, Clone)]
pub struct LabOptions {
    /// "auto" (default: only with a GPU), "on" or "off"
    pub transcript: Option<String>,
    /// Lighter models (faster on CPU, a little less accurate)
    pub light: Option<bool>,
    /// "auto" / "cuda" / "cpu"; the script decides when absent
    pub device: Option<String>,
    /// The video to read (a .spag session's extracted copy); else the replay's own video
    pub video_path: Option<String>,
    /// Where the models are (the app's setting); else SPAGHETTI_LAB_MODELS / GGST_MODEL_DIR / the
    /// research data folder
    pub models_dir: Option<String>,
    /// The match title in the facts and prompts (the replay's name); the replay id when absent
    pub title: Option<String>,
    /// "claude": also an AI summary + coaching notes from the facts (`--llm claude`; the script needs
    /// ANTHROPIC_API_KEY in the environment, which the app passes through untouched and never reads);
    /// "none": never. Absent: the script's default (GGST_LANG_PROVIDER, else none).
    pub llm: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct LabProgress {
    pub replay_id: String,
    /// The stage's name (the first word after `stage:`), e.g. "hud", "openers", "transcript", "facts"
    pub stage: String,
    /// The whole line after `stage:`
    pub line: String,
}

struct LabRun {
    replay_id: String,
    pid: u32,
    cancelled: Arc<AtomicBool>,
}

/// The running lab analysis (one at a time: it can use the whole GPU).
static LAB_RUN: Mutex<Option<LabRun>> = Mutex::new(None);

/// The replay a local run is working on. A statement of its own: the guard must be gone before RUN is
/// taken (see run_lab).
pub fn running_replay() -> Option<String> {
    LAB_RUN.lock().unwrap().as_ref().map(|r| r.replay_id.clone())
}

/// The research data folder's model dir (`GGST_DATA` or G:\Projects\replanal-data, then
/// ggst_dataset/model), as the research scripts find it (tools/ggst_logger/model/common.py).
fn research_model_dir() -> PathBuf {
    let data = std::env::var_os("GGST_DATA").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"G:\Projects\replanal-data"));
    data.join("ggst_dataset").join("model")
}

/// The models folder this build finds by itself (shown as the Lab panel's default): SPAGHETTI_LAB_MODELS,
/// GGST_MODEL_DIR, then the research data folder; the first that exists.
pub fn default_models_dir() -> Option<PathBuf> {
    models_dir(None).ok()
}

/// Where the models are: the setting, SPAGHETTI_LAB_MODELS, GGST_MODEL_DIR, then the research data
/// folder; the first that exists.
fn models_dir(setting: Option<&str>) -> Result<PathBuf, String> {
    let mut tried: Vec<PathBuf> = Vec::new();
    let setting = setting.map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from);
    let candidates = setting
        .into_iter()
        .chain(std::env::var_os("SPAGHETTI_LAB_MODELS").map(PathBuf::from))
        .chain(std::env::var_os("GGST_MODEL_DIR").map(PathBuf::from))
        .chain(std::iter::once(research_model_dir()));
    for c in candidates {
        if c.is_dir() {
            return Ok(c);
        }
        tried.push(c);
    }
    Err(format!(
        "{}: the models folder was not found (tried {}). Set the models folder in the Lab panel, or \
         SPAGHETTI_LAB_MODELS, to the folder with the trained models.",
        NOT_INSTALLED,
        tried.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    ))
}

/// tools/ggst_logger/model/lab_analyze.py in the project.
fn lab_script() -> Result<PathBuf, String> {
    let root = find_project_root().map_err(|e| format!("{}: {}", NOT_INSTALLED, e))?;
    let p = root.join("tools").join("ggst_logger").join("model").join("lab_analyze.py");
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!("{}: {} is missing (this build has no research tools).", NOT_INSTALLED, p.display()))
    }
}

/// The CLI arguments for lab_analyze.py (after the script).
pub fn lab_args(video: &str, out: &str, analysis: Option<&str>, o: &LabOptions) -> Result<Vec<String>, String> {
    let mut a: Vec<String> = vec![video.into(), "--out".into(), out.into()];
    let tr = o.transcript.as_deref().unwrap_or("auto");
    if !matches!(tr, "auto" | "on" | "off") {
        return Err(format!("unknown transcript mode '{}'", tr));
    }
    a.push("--transcript".into());
    a.push(tr.into());
    if o.light.unwrap_or(false) {
        a.push("--light".into());
    }
    if let Some(d) = o.device.as_deref().map(str::trim).filter(|d| !d.is_empty() && *d != "auto") {
        if !matches!(d, "cuda" | "cpu") {
            return Err(format!("unknown device '{}'", d));
        }
        a.push("--device".into());
        a.push(d.into());
    }
    if let Some(l) = o.llm.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
        if !matches!(l, "none" | "claude") {
            return Err(format!("unknown llm provider '{}'", l));
        }
        a.push("--llm".into());
        a.push(l.into());
    }
    if let Some(an) = analysis {
        a.push("--analysis".into());
        a.push(an.into());
    }
    if let Some(t) = o.title.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        a.push("--title".into());
        a.push(t.into());
    }
    Ok(a)
}

/// `stage: openers 2/5` -> ("openers", "openers 2/5"); other lines -> None.
pub fn parse_stage(line: &str) -> Option<(String, String)> {
    let rest = line.trim().strip_prefix("stage:")?.trim();
    let name = rest.split(|c: char| c.is_whitespace() || c == ':').next().unwrap_or("").to_string();
    if name.is_empty() {
        return None;
    }
    Some((name, rest.to_string()))
}

#[tauri::command]
pub async fn run_lab_analysis(
    app_handle: tauri::AppHandle,
    db_path: String,
    replay_id: String,
    options: Option<LabOptions>,
) -> Result<serde_json::Value, String> {
    let options = options.unwrap_or_default();
    // Minutes on a GPU, longer on a CPU: keep it off the async runtime's workers.
    tauri::async_runtime::spawn_blocking(move || run_lab(app_handle, db_path, replay_id, options))
        .await
        .map_err(|e| e.to_string())?
}

fn run_lab(app_handle: tauri::AppHandle, db_path: String, replay_id: String, options: LabOptions) -> Result<serde_json::Value, String> {
    let id = safe_id(&replay_id)?;
    // The e2e tests load a fixture result; they must never start the models (GPU)
    if std::env::var_os("SPAGHETTI_LAB_NO_MODELS").is_some() {
        return Err("Running the lab models is disabled (SPAGHETTI_LAB_NO_MODELS).".into());
    }
    // One lock at a time: claim_run takes RUN and then LAB_RUN, so holding LAB_RUN here could deadlock
    if running_replay().is_some() || crate::lab_remote::running_replay().is_some() {
        return Err("A lab analysis is already running.".into());
    }
    let script = lab_script()?;
    let models = models_dir(options.models_dir.as_deref())?;
    let video = match options.video_path.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        Some(v) if Path::new(v).is_file() => PathBuf::from(v),
        Some(v) => return Err(format!("Video file not found: {}", v)),
        None => {
            let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
            resolve_video_for_replay(&conn, &replay_id)?
        }
    };
    let python = require_python()?;
    // The models need PyTorch, which the CV pipeline doesn't
    let mut check = python.command();
    check.args(["-c", "import torch"]).stdout(Stdio::null()).stderr(Stdio::null());
    if !check.status().map(|s| s.success()).unwrap_or(false) {
        return Err(format!(
            "{}: {} has no PyTorch. Install it (`pip install torch`, with CUDA for the transcript) or set \
             SPAGHETTI_LAB_PYTHON to an interpreter that has it.",
            NOT_INSTALLED, python
        ));
    }

    let dir = lab_root().join(&id);
    let work = dir.join("run");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    // The CV pipeline's HUD readings of this replay (frames/<id>_frames.parquet), so the script
    // doesn't run the pipeline again. The shared output dir holds every replay's parquet and the
    // script takes the first one it finds, so it gets a folder with only this one.
    let analysis = default_output_paths()
        .ok()
        .map(|(out, _)| out.join("frames").join(format!("{}_frames.parquet", id)))
        .filter(|pq| pq.is_file())
        .and_then(|pq| {
            let an = work.join("analysis");
            let dst = an.join("frames").join(pq.file_name()?);
            std::fs::create_dir_all(dst.parent()?).ok()?;
            std::fs::copy(&pq, &dst).ok()?;
            Some(an.to_string_lossy().to_string())
        });
    let options = LabOptions { title: options.title.clone().or_else(|| Some(id.clone())), ..options };
    let args = lab_args(&video.to_string_lossy(), &work.to_string_lossy(), analysis.as_deref(), &options)?;
    let project_root = find_project_root()?;

    let mut cmd = python.command();
    cmd.env("PYTHONPATH", project_root.to_str().unwrap_or("."))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("GGST_MODEL_DIR", &models)
        .arg(&script)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Ok(ff) = super::ffmpeg_bin() {
        cmd.env("GGST_FFMPEG", ff);
    }
    eprintln!("[LAB] Spawning: {} {} {}", python, script.display(), args.join(" "));
    let mut child = cmd.spawn().map_err(|e| format!("Failed to run Python: {}", e))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    *LAB_RUN.lock().unwrap() = Some(LabRun { replay_id: id.clone(), pid: child.id(), cancelled: cancelled.clone() });

    let stderr = child.stderr.take().unwrap();
    let err_reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::BufReader::new(stderr).read_to_string(&mut s);
        s
    });
    let mut tail: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    for line in std::io::BufReader::new(child.stdout.take().unwrap()).lines().map_while(Result::ok) {
        if let Some((stage, text)) = parse_stage(&line) {
            let _ = app_handle.emit("lab-progress", LabProgress { replay_id: id.clone(), stage, line: text });
        } else {
            eprintln!("[LAB] {}", line);
        }
        tail.push_back(line);
        if tail.len() > 30 {
            tail.pop_front();
        }
    }
    let status = child.wait();
    *LAB_RUN.lock().unwrap() = None;
    let stderr = err_reader.join().unwrap_or_default();
    if cancelled.load(Ordering::SeqCst) {
        let _ = std::fs::remove_dir_all(&work);
        return Err("Cancelled".into());
    }
    let status = status.map_err(|e| e.to_string())?;
    if !status.success() {
        if !stderr.is_empty() {
            eprintln!("[LAB] stderr:\n{}", stderr);
        }
        let last: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
        let last = last[last.len().saturating_sub(6)..].join("\n");
        let out = tail.into_iter().collect::<Vec<_>>().join("\n");
        // A missing module is a setup problem, not a failure of the analysis
        if stderr.contains("ModuleNotFoundError") {
            return Err(format!("{}: a Python package is missing.\n{}", NOT_INSTALLED, last));
        }
        return Err(format!("Lab analysis failed (exit {}):\n{}\n{}", status.code().unwrap_or(-1), out, last));
    }
    let produced = work.join("lab_result.json");
    let text = std::fs::read_to_string(&produced).map_err(|e| format!("The lab analysis wrote no result ({}): {}", produced.display(), e))?;
    let value = parse_lab_result(&text)?;
    // Keep the run's other files (states, logs) next to the result
    let final_path = dir.join("lab_result.json");
    std::fs::write(&final_path, &text).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&produced);
    let _ = std::fs::remove_dir_all(dir.join("last_run"));
    let _ = std::fs::rename(&work, dir.join("last_run"));
    Ok(value)
}

/// Stop the running lab analysis (of `replay_id` when given). The run's command then fails with
/// "Cancelled" and the previous result, if any, stays.
#[tauri::command]
pub fn cancel_lab_analysis(replay_id: Option<String>) -> Result<(), String> {
    let guard = LAB_RUN.lock().unwrap();
    let run = guard.as_ref().ok_or("No lab analysis is running.")?;
    if let Some(r) = replay_id {
        if r != run.replay_id {
            return Err("The running lab analysis is of another replay.".into());
        }
    }
    run.cancelled.store(true, Ordering::SeqCst);
    kill_tree(run.pid);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_and_stages() {
        let o = LabOptions::default();
        assert_eq!(lab_args("v.mp4", "out", None, &o).unwrap(), ["v.mp4", "--out", "out", "--transcript", "auto"]);
        let o = LabOptions {
            transcript: Some("off".into()),
            light: Some(true),
            device: Some("cpu".into()),
            title: Some("Jason vs Enero".into()),
            ..Default::default()
        };
        assert_eq!(
            lab_args("v.mp4", "out", Some("an"), &o).unwrap(),
            ["v.mp4", "--out", "out", "--transcript", "off", "--light", "--device", "cpu", "--analysis", "an", "--title", "Jason vs Enero"]
        );
        assert!(lab_args("v", "o", None, &LabOptions { transcript: Some("x".into()), ..Default::default() }).is_err());
        // the AI summary is opt-in: off is an explicit `--llm none`, on is `--llm claude`
        let ai = |l: &str| lab_args("v.mp4", "out", None, &LabOptions { llm: Some(l.into()), ..Default::default() });
        assert_eq!(ai("claude").unwrap(), ["v.mp4", "--out", "out", "--transcript", "auto", "--llm", "claude"]);
        assert_eq!(ai("none").unwrap(), ["v.mp4", "--out", "out", "--transcript", "auto", "--llm", "none"]);
        assert_eq!(ai(" ").unwrap(), ["v.mp4", "--out", "out", "--transcript", "auto"]);
        assert!(ai("openai").is_err());
        assert!(ai("claude --x").is_err());
        assert_eq!(parse_stage("stage: openers 2/5"), Some(("openers".into(), "openers 2/5".into())));
        assert_eq!(parse_stage("  stage:hud"), Some(("hud".into(), "hud".into())));
        assert_eq!(parse_stage("stage: transcript: 40%"), Some(("transcript".into(), "transcript: 40%".into())));
        assert_eq!(parse_stage("loading model"), None);
        assert_eq!(parse_stage("stage:"), None);
    }

    #[test]
    fn cancel_without_a_run_is_an_error() {
        if LAB_RUN.lock().unwrap().is_none() {
            assert!(cancel_lab_analysis(None).is_err());
        }
    }
}
