// ── Lab analysis: what every build has ──────────────────────────────────────
//
// The result of the research models on a replay's video: `lab_result.json` (schema
// "spaghetti-lab-result/1", tools/ggst_logger/model/LAB_RESULT.md), cached per replay in the app data dir
// (`<data dir>/SpaghettiLab/lab/<replay_id>/lab_result.json`), outside the analysis DB, so .spag / .spagz
// files and the web app are unaffected. It comes from the hosted analysis (lab_remote.rs) or, only in our own
// build with the `local-models` feature, from the models on this PC (lab_local.rs).

use super::{hide_window, local_data_dir};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The schema family the app reads; a different major version is refused.
pub const LAB_SCHEMA: &str = "spaghetti-lab-result/1";

/// The replay a local run is working on. Only the `local-models` build has local runs.
pub fn local_run_replay() -> Option<String> {
    #[cfg(feature = "local-models")]
    {
        crate::lab_local::running_replay()
    }
    #[cfg(not(feature = "local-models"))]
    {
        None
    }
}

/// A local run is going on (the hosted mode shares the one-at-a-time rule).
pub fn local_run_active() -> bool {
    local_run_replay().is_some()
}

/// A replay id usable as one folder name (no separators, no `..`).
pub fn safe_id(replay_id: &str) -> Result<String, String> {
    let id = replay_id.trim();
    if id.is_empty() || id == "." || id == ".." || id.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()) {
        return Err(format!("Invalid replay id '{}'", replay_id));
    }
    Ok(id.to_string())
}

/// `<data dir>/SpaghettiLab/lab`
pub fn lab_root() -> PathBuf {
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

/// A paid result this version couldn't read, kept next to the cache (lab_remote::store_result).
pub const REJECTED_RESULT: &str = "lab_result.rejected.json";

pub(crate) fn read_cached(root: &Path, replay_id: &str) -> Result<Option<serde_json::Value>, String> {
    let p = lab_result_path_in(root, replay_id)?;
    // After an app update that reads it, a refused result becomes the result it always was
    let rejected = p.with_file_name(REJECTED_RESULT);
    if std::fs::read_to_string(&rejected).map_or(false, |t| parse_lab_result(&t).is_ok()) {
        let _ = std::fs::rename(&rejected, &p);
    }
    if !p.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
    parse_lab_result(&text).map(Some)
}

/// Kill the process and its children (DataLoader workers, ffmpeg).
pub(crate) fn kill_tree(pid: u32) {
    if cfg!(target_os = "windows") {
        let mut c = Command::new("taskkill");
        c.args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null());
        let _ = hide_window(&mut c).status();
    } else {
        let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
    }
}

/// The replay whose lab analysis is running, if any (the UI picks the progress back up on reopen).
#[tauri::command]
pub fn lab_analysis_running() -> Option<String> {
    local_run_replay().or_else(crate::lab_remote::running_replay)
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
    if local_run_replay().as_deref() == Some(id.as_str()) {
        return Err("The lab analysis of this replay is running; cancel it first.".into());
    }
    if crate::lab_remote::running_replay().as_deref() == Some(id.as_str()) {
        return Err("The lab analysis of this replay is running; cancel it first.".into());
    }
    let dir = lab_root().join(id);
    if !dir.exists() {
        return Ok(());
    }
    // A saved hosted job stays: it's already paid for, and its uploaded file still has to be deleted
    if dir.join("remote_job.json").is_file() {
        let _ = std::fs::remove_file(dir.join("lab_result.json"));
        let _ = std::fs::remove_file(dir.join(REJECTED_RESULT));
        let _ = std::fs::remove_dir_all(dir.join("last_run"));
        return Ok(());
    }
    std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
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
    fn ai_summary_is_kept() {
        // optional field (LAB_RESULT.md "ai_summary"): passed to the UI as it is
        let mut v = parse_lab_result(&fixture()).unwrap();
        v["ai_summary"] = serde_json::json!({"provider": "claude", "model": "claude-opus-5-5", "langs": {
            "en": {"text": "## Summary\nP1 won.", "grounded": true, "errors": [], "warnings": []},
            "it": {"text": null, "grounded": false, "errors": ["timestamp 9:99.9 not in the facts"], "warnings": []}}});
        let back = parse_lab_result(&v.to_string()).unwrap();
        assert_eq!(back["ai_summary"]["langs"]["en"]["grounded"], true);
        assert!(back["ai_summary"]["langs"]["it"]["text"].is_null());
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

    /// The published build: no local mode, whatever the environment says.
    #[cfg(not(feature = "local-models"))]
    #[test]
    fn published_build_has_no_local_mode() {
        std::env::set_var("SPAGHETTI_LAB_MODELS", "C:/models");
        assert!(!crate::lab_auth::lab_config().unwrap().local_models);
        assert!(crate::lab_auth::lab_config().unwrap().models_dir.is_none());
        std::env::remove_var("SPAGHETTI_LAB_MODELS");
        assert!(!local_run_active());
        assert_eq!(local_run_replay(), None);
    }

    #[cfg(feature = "local-models")]
    #[test]
    fn the_local_build_offers_the_local_mode() {
        assert!(crate::lab_auth::lab_config().unwrap().local_models);
    }
}
