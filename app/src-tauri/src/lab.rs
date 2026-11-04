// ── Lab analysis (research models) ──────────────────────────────────────────
//
// Runs the research models on a replay's video (tools/ggst_logger/model/lab_analyze.py, see
// tools/ggst_logger/model/LAB_RESULT.md): openers, the move transcript, the state streams and the
// match facts. The result, `lab_result.json` (schema "spaghetti-lab-result/1"), is cached per replay
// in the app data dir (`<data dir>/SpaghettiLab/lab/<replay_id>/lab_result.json`), outside the
// analysis DB: no schema change, so .spag / .spagz files and the web app are unaffected.
//
// Progress: the script's `stage: <name> ...` stdout lines go to the UI as "lab-progress" events
// ({replay_id, stage, line}); every other line is only logged. Cancelling kills the process tree.

use super::{default_output_paths, find_project_root, hide_window, local_data_dir, require_python, resolve_video_for_replay};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Read as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::Emitter;

/// The schema family the app reads; a different major version is refused.
pub const LAB_SCHEMA: &str = "spaghetti-lab-result/1";
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

/// A replay id usable as one folder name (no separators, no `..`).
fn safe_id(replay_id: &str) -> Result<String, String> {
    let id = replay_id.trim();
    if id.is_empty() || id == "." || id == ".." || id.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()) {
        return Err(format!("Invalid replay id '{}'", replay_id));
    }
    Ok(id.to_string())
}

/// `<data dir>/SpaghettiLab/lab`
fn lab_root() -> PathBuf {
    local_data_dir().join("SpaghettiLab").join("lab")
}

/// The cached result of one replay (it may not exist).
pub fn lab_result_path_in(root: &Path, replay_id: &str) -> Result<PathBuf, String> {
    Ok(root.join(safe_id(replay_id)?).join("lab_result.json"))
}

/// Parse and check a lab_result.json: an object with `schema` "spaghetti-lab-result/1[.x]".
pub fn parse_lab_result(text: &str) -> Result<serde_json::Value, String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("The lab result is not valid JSON: {}", e))?;
    let schema = v.get("schema").and_then(|s| s.as_str()).ok_or("The lab result has no schema")?;
    let ok = schema == LAB_SCHEMA || schema.strip_prefix(LAB_SCHEMA).map_or(false, |rest| rest.starts_with('.'));
    if !ok {
        return Err(format!("Unsupported lab result '{}' (this app reads {})", schema, LAB_SCHEMA));
    }
    if !v.get("openers").map_or(false, |o| o.is_array()) {
        return Err("The lab result has no openers list".into());
    }
    Ok(v)
}

fn read_cached(root: &Path, replay_id: &str) -> Result<Option<serde_json::Value>, String> {
    let p = lab_result_path_in(root, replay_id)?;
    if !p.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
    parse_lab_result(&text).map(Some)
}

/// The research data folder's model dir (`GGST_DATA` or G:\Projects\replanal-data, then
/// ggst_dataset/model), as the research scripts find it (tools/ggst_logger/model/common.py).
fn research_model_dir() -> PathBuf {
    let data = std::env::var_os("GGST_DATA").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"G:\Projects\replanal-data"));
    data.join("ggst_dataset").join("model")
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

/// Kill the process and its children (DataLoader workers, ffmpeg).
fn kill_tree(pid: u32) {
    if cfg!(target_os = "windows") {
        let mut c = Command::new("taskkill");
        c.args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null());
        let _ = hide_window(&mut c).status();
    } else {
        let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
    }
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
    if LAB_RUN.lock().unwrap().is_some() {
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

/// The replay whose lab analysis is running, if any (the UI picks the progress back up on reopen).
#[tauri::command]
pub fn lab_analysis_running() -> Option<String> {
    LAB_RUN.lock().unwrap().as_ref().map(|r| r.replay_id.clone())
}

/// The cached lab result of a replay, or null when it was never run.
#[tauri::command]
pub fn get_lab_result(replay_id: String) -> Result<Option<serde_json::Value>, String> {
    read_cached(&lab_root(), &replay_id)
}

/// Forget a replay's lab result (and the last run's files).
#[tauri::command]
pub fn delete_lab_result(replay_id: String) -> Result<(), String> {
    let id = safe_id(&replay_id)?;
    if LAB_RUN.lock().unwrap().as_ref().map_or(false, |r| r.replay_id == id) {
        return Err("The lab analysis of this replay is running; cancel it first.".into());
    }
    let dir = lab_root().join(id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Agent A's fixture (tools/ggst_logger/model/fixtures), else the app's e2e copy.
    fn fixture() -> String {
        let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo = here.join("..").join("..");
        for p in [
            repo.join("tools/ggst_logger/model/fixtures/lab_result_sample.json"),
            repo.join("app/e2e/fixtures/lab_result.json"),
        ] {
            if let Ok(t) = std::fs::read_to_string(&p) {
                return t;
            }
        }
        panic!("no lab_result fixture found");
    }

    #[test]
    fn fixture_parses() {
        let v = parse_lab_result(&fixture()).unwrap();
        assert!(v["schema"].as_str().unwrap().starts_with(LAB_SCHEMA));
        let openers = v["openers"].as_array().unwrap();
        assert!(!openers.is_empty());
        for o in openers {
            assert!(o["time_s"].is_number());
            assert!(matches!(o["attacker"].as_str(), Some("P1") | Some("P2")));
        }
        assert!(v["characters"]["P1"].is_string());
        assert!(v["capabilities"].is_object());
        if !v["transcript"].is_null() {
            assert!(v["transcript"]["events"].is_array());
        }
    }

    #[test]
    fn schema_is_checked() {
        assert!(parse_lab_result("not json").is_err());
        assert!(parse_lab_result(r#"{"openers": []}"#).is_err());
        assert!(parse_lab_result(r#"{"schema": "spaghetti-lab-result/2", "openers": []}"#).is_err());
        assert!(parse_lab_result(r#"{"schema": "spaghetti-lab-result/10", "openers": []}"#).is_err());
        assert!(parse_lab_result(r#"{"schema": "spaghetti-lab-result/1", "openers": {}}"#).is_err());
        assert!(parse_lab_result(r#"{"schema": "spaghetti-lab-result/1", "openers": []}"#).is_ok());
        assert!(parse_lab_result(r#"{"schema": "spaghetti-lab-result/1.1", "openers": []}"#).is_ok());
    }

    #[test]
    fn cache_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert_eq!(read_cached(root, "set_a").unwrap(), None);
        let p = lab_result_path_in(root, "set_a").unwrap();
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, fixture()).unwrap();
        let v = read_cached(root, "set_a").unwrap().unwrap();
        assert!(v["openers"].is_array());
        std::fs::write(&p, "{").unwrap();
        assert!(read_cached(root, "set_a").is_err());
        assert!(lab_result_path_in(root, "../x").is_err());
        assert!(lab_result_path_in(root, "a/b").is_err());
        assert!(lab_result_path_in(root, "").is_err());
    }

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
