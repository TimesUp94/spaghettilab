use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Read as _, Write as _};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::Emitter;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Apply CREATE_NO_WINDOW on Windows so spawned processes don't show a console.
#[cfg(target_os = "windows")]
fn hide_window(cmd: &mut Command) -> &mut Command {
    cmd.creation_flags(CREATE_NO_WINDOW)
}

#[cfg(not(target_os = "windows"))]
fn hide_window(cmd: &mut Command) -> &mut Command {
    cmd
}

/// A Python interpreter to spawn: the program plus any arguments that go
/// before the script (`py -3` on Windows).
#[derive(Debug, Clone, PartialEq)]
struct PythonCmd {
    program: String,
    args: Vec<String>,
}

impl PythonCmd {
    fn new(program: impl Into<String>, args: &[&str]) -> Self {
        PythonCmd { program: program.into(), args: args.iter().map(|a| a.to_string()).collect() }
    }

    /// A `Command` for this interpreter (console hidden on Windows), ready
    /// for the script's own arguments.
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        hide_window(&mut cmd);
        cmd.args(&self.args);
        cmd
    }
}

impl std::fmt::Display for PythonCmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.program)?;
        for a in &self.args {
            write!(f, " {}", a)?;
        }
        Ok(())
    }
}

/// Folders a GUI app doesn't get on PATH but where Python and ffmpeg are
/// usually installed: on macOS an app started from Finder gets only
/// /usr/bin:/bin:..., where `python3` is the Xcode stub without OpenCV.
fn extra_tool_dirs() -> Vec<PathBuf> {
    if cfg!(target_os = "macos") {
        vec![
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/Library/Frameworks/Python.framework/Versions/Current/bin"),
        ]
    } else if cfg!(target_os = "linux") {
        vec![PathBuf::from("/usr/local/bin"), PathBuf::from("/usr/bin")]
    } else {
        Vec::new()
    }
}

/// Python interpreters to try, best first: an explicitly configured one
/// (`SPAGHETTI_LAB_PYTHON`), the project's `.venv` (dev checkout), a venv in the
/// app's data dir (installed app), then the system Python: the `py -3`
/// launcher on Windows (the python.org installer doesn't add `python` to
/// PATH by default, and a bare `python` is often the Store alias), Homebrew /
/// python.org `python3` on macOS, then `python3` / `python` from PATH.
fn python_candidates() -> Vec<PythonCmd> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("SPAGHETTI_LAB_PYTHON") {
        if !p.trim().is_empty() {
            out.push(PythonCmd::new(p, &[]));
        }
    }
    let venv_python = |venv: PathBuf| -> PathBuf {
        if cfg!(target_os = "windows") {
            venv.join("Scripts").join("python.exe")
        } else {
            venv.join("bin").join("python")
        }
    };
    let mut venvs = Vec::new();
    if let Ok(root) = find_project_root() {
        venvs.push(root.join(".venv"));
        venvs.push(root.join("venv"));
    }
    if let Some(d) = dirs::data_local_dir() {
        venvs.push(d.join("SpaghettiLab").join("venv"));
    }
    for v in venvs {
        let p = venv_python(v);
        if p.is_file() {
            out.push(PythonCmd::new(p.to_string_lossy(), &[]));
        }
    }
    if cfg!(target_os = "windows") {
        out.push(PythonCmd::new("py", &["-3"]));
    } else {
        for d in extra_tool_dirs() {
            let p = d.join("python3");
            if p.is_file() {
                out.push(PythonCmd::new(p.to_string_lossy(), &[]));
            }
        }
        out.push(PythonCmd::new("python3", &[]));
    }
    out.push(PythonCmd::new("python", &[]));
    out
}

/// The interpreter that can run the CV pipeline (imports cv2 and numpy),
/// or None.  A bare `python` on Windows is often the Store alias or an
/// unrelated install without OpenCV, so each candidate is checked.  Only a
/// success is cached: after the user installs the missing packages, the
/// next call finds them without restarting the app.
fn resolve_python() -> Option<PythonCmd> {
    use std::sync::Mutex;
    static PYTHON: Mutex<Option<PythonCmd>> = Mutex::new(None);
    if let Some(p) = PYTHON.lock().unwrap().clone() {
        return Some(p);
    }
    let candidates = python_candidates();
    for c in &candidates {
        let ok = c
            .command()
            .args(["-c", "import cv2, numpy"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            eprintln!("[PYTHON] using {}", c);
            *PYTHON.lock().unwrap() = Some(c.clone());
            return Some(c.clone());
        }
    }
    eprintln!("[PYTHON] no interpreter with cv2 among {:?}", candidates);
    None
}

/// The Python interpreter for helpers that don't need OpenCV (yt-dlp,
/// imageio-ffmpeg): the pipeline's one when found, else the first candidate.
fn python_cmd() -> PythonCmd {
    resolve_python()
        .or_else(|| python_candidates().into_iter().next())
        .unwrap_or_else(|| PythonCmd::new("python", &[]))
}

/// The Python interpreter, or an actionable error when none can run the
/// pipeline.
fn require_python() -> Result<PythonCmd, String> {
    resolve_python().ok_or_else(|| {
        format!(
            "No Python with OpenCV found (tried {}). Install Python 3.10+ and run \
             `pip install -e \".[ocr]\"` in the Spaghetti Lab folder (or `pip install \
             opencv-python numpy pyyaml pyarrow easyocr imageio-ffmpeg`), or set \
             SPAGHETTI_LAB_PYTHON to an interpreter that has them, then try again.",
            python_candidates().iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ")
        )
    })
}

/// Folders that may hold bundled resources: `resources/` next to the exe
/// (NSIS) and, on macOS, the app bundle's `Contents/Resources` (the exe is
/// in `Contents/MacOS`).
fn exe_resource_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf())) {
        out.push(d.join("resources"));
        if let Some(contents) = d.parent() {
            out.push(contents.join("Resources"));
        }
    }
    out
}

/// Folders that may hold a bundled ffmpeg: the bundle's resources dir, next
/// to the exe, and the project root (dev).
fn bundled_tool_dirs() -> Vec<PathBuf> {
    let mut dirs_out: Vec<PathBuf> = exe_resource_dirs().into_iter().map(|r| r.join("ffmpeg")).collect();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs_out.push(d.join("ffmpeg"));
            dirs_out.push(d.to_path_buf());
        }
    }
    if let Ok(root) = find_project_root() {
        dirs_out.push(root.join("ffmpeg"));
        dirs_out.push(root.join("resources").join("ffmpeg"));
    }
    dirs_out
}

fn exe_name(name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.exe", name)
    } else {
        name.to_string()
    }
}

/// `name` on PATH, or in the install folders a GUI app doesn't get on PATH.
fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain(extra_tool_dirs())
        .map(|d| d.join(exe_name(name)))
        .find(|p| p.is_file())
}

/// Resolve ffmpeg: `SPAGHETTI_FFMPEG`, PATH, the Python env's imageio-ffmpeg,
/// then a bundled copy.  Only a success is cached, so an ffmpeg installed
/// while the app runs is picked up on the next try.
fn ffmpeg_bin() -> Result<String, String> {
    use std::sync::Mutex;
    static FFMPEG: Mutex<Option<String>> = Mutex::new(None);
    if let Some(p) = FFMPEG.lock().unwrap().clone() {
        return Ok(p);
    }
    let find = || -> Option<String> {
        if let Ok(p) = std::env::var("SPAGHETTI_FFMPEG") {
            if PathBuf::from(&p).is_file() {
                return Some(p);
            }
        }
        if let Some(p) = find_on_path("ffmpeg") {
            return Some(p.to_string_lossy().to_string());
        }
        if let Ok(out) = python_cmd()
            .command()
            .args(["-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"])
            .stderr(Stdio::null())
            .output()
        {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if out.status.success() && PathBuf::from(&p).is_file() {
                return Some(p);
            }
        }
        bundled_tool_dirs()
            .into_iter()
            .map(|d| d.join(exe_name("ffmpeg")))
            .find(|p| p.is_file())
            .map(|p| p.to_string_lossy().to_string())
    };
    let found = find().ok_or_else(|| {
        "ffmpeg not found. Install ffmpeg and add it to PATH, or run \
         `pip install imageio-ffmpeg` for the Python Spaghetti Lab uses, \
         or set SPAGHETTI_FFMPEG to the ffmpeg executable, then try again."
            .to_string()
    })?;
    *FFMPEG.lock().unwrap() = Some(found.clone());
    Ok(found)
}

#[cfg(test)]
mod tool_resolution_tests {
    use super::*;

    #[test]
    fn project_venv_is_preferred_over_bare_python() {
        let c = python_candidates();
        assert_eq!(c.last().map(|p| p.program.as_str()), Some("python"));
        if cfg!(target_os = "windows") {
            let py = c.iter().position(|p| p == &PythonCmd::new("py", &["-3"])).unwrap();
            assert_eq!(py, c.len() - 2, "the py launcher comes right before a bare python");
        }
        let root = find_project_root().unwrap();
        if root.join(".venv").is_dir() {
            let first_venv = c.iter().position(|p| p.program.contains(".venv")).unwrap();
            let bare = c.iter().position(|p| p.program == "python").unwrap();
            assert!(first_venv < bare);
            // The dev venv has OpenCV, so it is the interpreter we spawn
            assert!(require_python().unwrap().program.contains(".venv"));
        }
    }

    #[test]
    fn python_cmd_puts_its_args_before_the_script() {
        let c = PythonCmd::new("py", &["-3"]);
        let mut cmd = c.command();
        cmd.arg("script.py");
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().to_string()).collect();
        assert_eq!(args, ["-3", "script.py"]);
        assert_eq!(c.to_string(), "py -3");
    }

    #[test]
    fn ffmpeg_resolves_or_explains() {
        match ffmpeg_bin() {
            Ok(p) => assert!(PathBuf::from(p).is_file()),
            Err(e) => assert!(e.contains("imageio-ffmpeg")),
        }
    }
}

// ── Data models ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct Replay {
    pub replay_id: String,
    pub video_path: String,
    pub duration_ms: f64,
    pub frame_count: i64,
}

#[derive(Debug, Serialize, Clone)]
pub struct FrameDataPoint {
    pub timestamp_ms: f64,
    pub p1_health_pct: Option<f64>,
    pub p2_health_pct: Option<f64>,
    pub p1_tension_pct: Option<f64>,
    pub p2_tension_pct: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct DamageEvent {
    pub event_id: i64,
    pub replay_id: String,
    pub timestamp_ms: f64,
    pub frame_start: i64,
    pub frame_end: i64,
    pub target_side: i64,
    pub damage_pct: f64,
    pub pre_health_pct: f64,
    pub post_health_pct: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct RoundResult {
    pub replay_id: String,
    pub round_index: usize,
    pub round_start_ms: f64,
    pub round_end_ms: f64,
    pub winner: String,
    pub winner_final_hp: f64,
    pub loser_final_hp: f64,
    pub winner_min_hp: f64,
    pub max_deficit: f64,
    pub deficit_timestamp_ms: f64,
    pub is_comeback: bool,
    pub is_match_start: bool,
    pub winner_confident: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct MatchStats {
    pub replay_id: String,
    pub total_rounds: usize,
    pub p1_round_wins: usize,
    pub p2_round_wins: usize,
    pub total_damage_events: usize,
    pub p1_damage_taken: f64,
    pub p2_damage_taken: f64,
    pub p1_biggest_hit: f64,
    pub p2_biggest_hit: f64,
    pub avg_round_duration_s: f64,
    pub longest_round_s: f64,
    pub shortest_round_s: f64,
    pub comeback_count: usize,
    pub close_rounds: usize,
    pub avg_winner_final_hp: f64,
    pub duration_s: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct Highlight {
    pub kind: String,         // "comeback", "close_round", "big_damage", "perfect"
    pub label: String,
    pub timestamp_ms: f64,
    pub end_ms: f64,
    pub details: String,
    pub severity: f64,        // 0-1, for sorting
}

#[derive(Debug, Serialize, Clone)]
pub struct Note {
    pub note_id: i64,
    pub replay_id: String,
    pub timestamp_ms: f64,
    pub text: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct Drawing {
    pub drawing_id: i64,
    pub replay_id: String,
    pub timestamp_ms: f64,
    pub strokes_json: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct AnalysisStatus {
    pub running: bool,
    pub progress_lines: Vec<String>,
    pub error: Option<String>,
    pub db_path: Option<String>,
}

// ── Constants for round detection ────────────────────────────────────────────

const HP_CEILING: f64 = 0.875;
const MIN_ROUND_FRAMES: usize = 450;  // ~15 seconds at 30fps
/// Shortest leading partial round kept when it ends in a KO (~8 s at 30/s).
const LEAD_PARTIAL_MIN_FRAMES: usize = 240;
const COMEBACK_DEFICIT: f64 = 0.35;
const DEFICIT_SMOOTH_WINDOW: usize = 91;  // ~3s heavy smoothing for deficit tracking
const TIMER_SMOOTH_WINDOW: usize = 61;  // ~2s window for timer noise filtering
const HP_RESET_WINDOW: usize = 90;  // ~3s window for HP reset detection

// ── Signal processing helpers ────────────────────────────────────────────────

fn rolling_median(arr: &[f64], window: usize) -> Vec<f64> {
    let half = window / 2;
    let min_valid = (window / 3).max(3); // require at least 1/3 of window to be valid data
    let n = arr.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = if i + half + 1 <= n { i + half + 1 } else { n };
        let mut valid: Vec<f64> = arr[lo..hi].iter().copied().filter(|v| !v.is_nan()).collect();
        if valid.len() >= min_valid {
            valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[i] = valid[valid.len() / 2];
        }
    }
    out
}

/// Sample-count windows in the round detection are written for 30 samples per
/// second (what the pipeline produces: it samples every round(fps/30)-th video
/// frame).  For data sampled at another rate they are scaled by this factor,
/// so each window keeps its length in seconds.  Exactly 1.0 within 5% of
/// 30/s, so 30-sample/s data gives the same windows as before.
fn sample_rate_scale(ts: &[f64]) -> f64 {
    let mut d: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|&x| x > 0.0).collect();
    if d.is_empty() {
        return 1.0;
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rate = 1000.0 / d[d.len() / 2];
    let scale = rate / 30.0;
    if (scale - 1.0).abs() < 0.05 { 1.0 } else { scale }
}

/// A window of `frames` samples at 30/s, in samples at the data's rate.
fn scale_frames(frames: usize, scale: f64) -> usize {
    if scale == 1.0 {
        frames
    } else {
        ((frames as f64) * scale).round().max(1.0) as usize
    }
}

fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Forward-fill None values (carry last known value forward).
fn forward_fill_i32(arr: &[Option<i32>]) -> Vec<Option<i32>> {
    let mut out = arr.to_vec();
    let mut last: Option<i32> = None;
    for v in out.iter_mut() {
        if v.is_some() {
            last = *v;
        } else {
            *v = last;
        }
    }
    out
}

/// Rolling median for optional i32 values.
fn rolling_median_i32(arr: &[Option<i32>], window: usize) -> Vec<Option<i32>> {
    let half = window / 2;
    let n = arr.len();
    let mut out = vec![None; n];
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = if i + half + 1 <= n { i + half + 1 } else { n };
        let mut vals: Vec<i32> = arr[lo..hi].iter().filter_map(|v| *v).collect();
        if !vals.is_empty() {
            vals.sort();
            out[i] = Some(vals[vals.len() / 2]);
        }
    }
    out
}

/// Round-start timer readings need temporal support: the round timer shows
/// each value for about a second, so a real reading has at least
/// TIMER_SUPPORT other readings within ±TIMER_SUPPORT_MS of the same or an
/// adjacent value.  A single OCR misread of a start value (e.g. 97 on a super
/// flash in the middle of a round) does not, and is dropped: forward-filled
/// across the HUD gap that follows, it would fake a round start.  Only
/// readings >= TIMER_SUPPORT_MIN are checked (an isolated low reading cannot
/// start a round).  `ts` is in milliseconds and sorted.
const TIMER_SUPPORT: usize = 2;
const TIMER_SUPPORT_MS: f64 = 1000.0;
const TIMER_SUPPORT_MIN: i32 = 88;

fn supported_timer(timer: &[Option<i32>], ts: &[f64]) -> Vec<Option<i32>> {
    let seen: Vec<usize> = (0..timer.len()).filter(|&i| timer[i].is_some()).collect();
    let mut out = vec![None; timer.len()];
    let mut lo = 0usize;
    for &i in seen.iter() {
        while ts[seen[lo]] < ts[i] - TIMER_SUPPORT_MS {
            lo += 1;
        }
        let v = timer[i].unwrap();
        if v < TIMER_SUPPORT_MIN {
            out[i] = timer[i];
            continue;
        }
        let mut support = 0usize;
        for &j in seen[lo..].iter() {
            if ts[j] > ts[i] + TIMER_SUPPORT_MS {
                break;
            }
            if j != i && (timer[j].unwrap() - v).abs() <= 1 {
                support += 1;
            }
        }
        if support >= TIMER_SUPPORT {
            out[i] = timer[i];
        }
    }
    out
}

// ── Round detection using timer + HP resets + gaps ──────────────────────────
//
// GGS round detection strategy (three complementary signals):
// 1. TIMER: Timer counts down from ~99 each round. Smoothed with 61-frame
//    window to filter OCR noise. Jump from <=70 to >=88 = new round.
// 2. HP RESETS: Both players' rolling median HP (90-frame window) goes
//    from <0.50 to >0.85 = new round boundary.
// 3. GAME BREAKS: Data gaps >5s = character select / game boundary.
//
// A typical GGS round lasts ~30 seconds. Rounds >45s are recursively
// split at internal gaps or HP resets.

fn detect_rounds_for_replay(
    conn: &Connection,
    replay_id: &str,
) -> Result<Vec<RoundResult>, String> {
    // Read ALL frame data (including NULL HP for gap detection)
    let has_timer_col = conn
        .prepare("SELECT timer_value FROM frame_data LIMIT 1")
        .is_ok();
    let has_rounds_won_col = conn
        .prepare("SELECT p1_rounds_won FROM frame_data LIMIT 1")
        .is_ok();

    // (timestamp_ms, p1_hp, p2_hp, timer, p1_rounds_won, p2_rounds_won)
    let rows: Vec<(f64, Option<f64>, Option<f64>, Option<i32>, Option<i32>, Option<i32>)> = if has_timer_col && has_rounds_won_col {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct, timer_value, p1_rounds_won, p2_rounds_won
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let result: Vec<_> = stmt
            .query_map([replay_id], |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, Option<f64>>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                    row.get::<_, Option<i32>>(3)?,
                    row.get::<_, Option<i32>>(4)?,
                    row.get::<_, Option<i32>>(5)?,
                ))
            })
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    } else if has_timer_col {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct, timer_value
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let result: Vec<_> = stmt
            .query_map([replay_id], |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, Option<f64>>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                    row.get::<_, Option<i32>>(3)?,
                    None::<i32>,
                    None::<i32>,
                ))
            })
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let result: Vec<_> = stmt
            .query_map([replay_id], |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, Option<f64>>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                    None::<i32>,
                    None::<i32>,
                    None::<i32>,
                ))
            })
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    };

    let n = rows.len();
    if n < 2 {
        return Ok(vec![]);
    }

    let ts: Vec<f64> = rows.iter().map(|r| r.0).collect();
    // Windows below are in samples at 30/s; `fr` converts them to the data's
    // sample rate (see sample_rate_scale).
    let rate_scale = sample_rate_scale(&ts);
    let fr = |frames: usize| -> usize { scale_frames(frames, rate_scale) };
    let min_round_frames = fr(MIN_ROUND_FRAMES);
    if n < min_round_frames {
        return Ok(vec![]);
    }
    let p1_raw: Vec<f64> = rows
        .iter()
        .map(|r| r.1.unwrap_or(f64::NAN))
        .collect();
    let p2_raw: Vec<f64> = rows
        .iter()
        .map(|r| r.2.unwrap_or(f64::NAN))
        .collect();
    let timer_raw: Vec<Option<i32>> = rows.iter().map(|r| r.3).collect();
    let p1_rounds_won: Vec<Option<i32>> = rows.iter().map(|r| r.4).collect();
    let p2_rounds_won: Vec<Option<i32>> = rows.iter().map(|r| r.5).collect();

    // Normalize HP by ceiling
    let mut p1_norm: Vec<f64> = p1_raw
        .iter()
        .map(|v| if v.is_nan() { f64::NAN } else { clip(v / HP_CEILING, 0.0, 1.0) })
        .collect();
    let mut p2_norm: Vec<f64> = p2_raw
        .iter()
        .map(|v| if v.is_nan() { f64::NAN } else { clip(v / HP_CEILING, 0.0, 1.0) })
        .collect();

    // ── Data cleaning: filter junk HP from non-gameplay screens ──
    // If both players' HP stays below 0.15 for > 5 seconds (150 frames),
    // it's a lobby/menu screen misidentified as gameplay. Null out those frames.
    // Threshold is intentionally low (0.15) because both players CAN legitimately
    // be at low health in close GGS rounds. Only catch clear non-gameplay.
    let mut junk_count = 0usize;
    {
        let junk_thresh = 0.15;
        let junk_window = fr(150); // ~5 seconds
        let p1_junk = rolling_median(&p1_norm, junk_window);
        let p2_junk = rolling_median(&p2_norm, junk_window);
        // A timer reading nearby means we're in a round: both players can
        // genuinely sit near KO for seconds (e.g. 5% vs 3% at round end).
        // Menus/lobbies never show the round timer.
        let mut timer_seen_at: Vec<usize> = Vec::new();
        for i in 0..n {
            if timer_raw[i].is_some() {
                timer_seen_at.push(i);
            }
        }
        // Looks further back than ahead: the end of a round (the last combo
        // and the KO) often plays with the timer hidden by a sprite or super.
        let timer_near = |i: usize| -> bool {
            let lo = i.saturating_sub(junk_window);
            let hi = i + junk_window / 2;
            let k = timer_seen_at.partition_point(|&t| t < lo);
            k < timer_seen_at.len() && timer_seen_at[k] <= hi
        };
        for i in 0..n {
            if !p1_junk[i].is_nan() && !p2_junk[i].is_nan()
                && p1_junk[i] < junk_thresh && p2_junk[i] < junk_thresh
                && !timer_near(i)
            {
                // Debug: log first few junk regions
                if junk_count < 5 || (junk_count % 200 == 0) {
                    eprintln!("[JUNK-DBG] {} NaN-ing i={} t={:.1}s p1_junk={:.3} p2_junk={:.3} p1_raw={:.3} p2_raw={:.3}",
                        replay_id, i, ts[i]/1000.0, p1_junk[i], p2_junk[i], p1_norm[i], p2_norm[i]);
                }
                junk_count += 1;
                p1_norm[i] = f64::NAN;
                p2_norm[i] = f64::NAN;
            }
        }
    }
    if junk_count > 0 {
        eprintln!("[JUNK-DBG] {} total junk frames NaN-ed: {}", replay_id, junk_count);
    }

    // ── Non-gameplay detection: no timer for 20+ seconds ──
    // The round timer is the most reliable gameplay indicator. Round counter
    // hearts can produce false positives (lobby UI has similar shapes that
    // read as 0/0). If the timer is absent for 20+ seconds, we're not in
    // gameplay (character select, lobby, replay browser, loading, etc.).
    // NaN-out HP data in these regions so they don't contaminate round
    // detection, and let the game-break detector handle boundaries.
    let nongame_threshold = fr(600); // ~20 seconds
    let mut nongame_regions: Vec<(usize, usize)> = Vec::new();
    {
        let mut run_start: Option<usize> = None;
        for i in 0..n {
            let has_timer = timer_raw[i].is_some();

            if !has_timer {
                if run_start.is_none() {
                    run_start = Some(i);
                }
            } else {
                if let Some(rs) = run_start {
                    let run_len = i - rs;
                    if run_len >= nongame_threshold {
                        nongame_regions.push((rs, i));
                    }
                    run_start = None;
                }
            }
        }
        // Handle run extending to end of data
        if let Some(rs) = run_start {
            let run_len = n - rs;
            if run_len >= nongame_threshold {
                nongame_regions.push((rs, n));
            }
        }
    }
    if !nongame_regions.is_empty() {
        let mut nongame_nulled = 0usize;
        for &(start, end) in &nongame_regions {
            eprintln!("[NONGAME-DBG] {} non-gameplay region: frames {}-{} t={:.1}s-{:.1}s ({:.1}s)",
                replay_id, start, end, ts[start]/1000.0, ts[end.min(n-1)]/1000.0,
                (ts[end.min(n-1)] - ts[start])/1000.0);
            for j in start..end {
                if !p1_norm[j].is_nan() || !p2_norm[j].is_nan() {
                    p1_norm[j] = f64::NAN;
                    p2_norm[j] = f64::NAN;
                    nongame_nulled += 1;
                }
            }
        }
        eprintln!("[NONGAME-DBG] {} total non-gameplay frames NaN-ed: {} across {} regions",
            replay_id, nongame_nulled, nongame_regions.len());
    }

    // Smooth HP for round metrics and winner detection (window=31 ≈ 1s)
    // Must be wide enough to filter OCR/scene-detection noise spikes
    let p1_smooth = rolling_median(&p1_norm, fr(31));
    let p2_smooth = rolling_median(&p2_norm, fr(31));

    // Heavy smoothing for deficit/comeback tracking (window=91 ≈ 3s)
    // HP extraction noise can create false deficits with lighter smoothing
    let p1_heavy = rolling_median(&p1_norm, fr(DEFICIT_SMOOTH_WINDOW));
    let p2_heavy = rolling_median(&p2_norm, fr(DEFICIT_SMOOTH_WINDOW));

    // ── Signal 1: Timer-based round starts ──
    // Drop isolated timer readings first (see supported_timer): forward-fill
    // would otherwise stretch one misread across a HUD gap (a super
    // cinematic) and the rolling median would take it as the timer value.
    // Then forward-fill + wide-window smooth to filter OCR noise, and detect jumps.
    let timer_sup = supported_timer(&timer_raw, &ts);
    let timer_ff = forward_fill_i32(&timer_sup);
    let timer_smooth = rolling_median_i32(&timer_ff, fr(TIMER_SMOOTH_WINDOW));

    let mut timer_starts: Vec<usize> = Vec::new();
    let mut lowest_since_start: i32 = 99;
    for i in 0..n {
        if let Some(t) = timer_smooth[i] {
            lowest_since_start = lowest_since_start.min(t);
            // Timer jumped from low (<=70) to high (>=88) = potential new round
            if t >= 88 && lowest_since_start <= 70 {
                if timer_starts.last().map_or(true, |&last| i - last > min_round_frames) {
                    // Validate: smoothed timer must reach ≥90 within 90 frames.
                    // Filters wallbreak/super flash OCR noise (peaks briefly at ~88).
                    // Real round starts show timer ≥90 consistently (OCR reads 90-96
                    // for the "99" display, a timer-reader template mismatch).
                    let check_end = (i + fr(90)).min(n);
                    let confirmed = (i..check_end)
                        .any(|j| timer_smooth[j].map_or(false, |tv| tv >= 90));
                    // Validate: a real round start has both bars refilled.
                    // Timer OCR can misread mid-round digits as ~90 (e.g. 57
                    // after a wallbreak read as 92); if the bars clearly stay
                    // low right after the jump, the "new round" is a misread
                    // and would split the round before its KO.
                    let post_min: Vec<f64> = (i..check_end)
                        .filter(|&j| !p1_norm[j].is_nan() && !p2_norm[j].is_nan())
                        .map(|j| p1_norm[j].min(p2_norm[j]))
                        .collect();
                    let hp_stayed_low = post_min.len() >= fr(15) && {
                        let mut sorted = post_min.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted[sorted.len() / 2] < 0.50
                    };
                    if confirmed && hp_stayed_low {
                        eprintln!("[ROUND-DBG] timer_start REJECTED (HP did not reset) at i={} t={:.1}s (smooth_timer={}, lowest={})",
                            i, ts[i]/1000.0, t, lowest_since_start);
                    } else if confirmed {
                        eprintln!("[ROUND-DBG] timer_start at i={} t={:.1}s (smooth_timer={}, lowest={})",
                            i, ts[i]/1000.0, t, lowest_since_start);
                        timer_starts.push(i);
                        lowest_since_start = t;
                    } else {
                        eprintln!("[ROUND-DBG] timer_start REJECTED (no >=90 confirm) at i={} t={:.1}s (smooth_timer={}, lowest={})",
                            i, ts[i]/1000.0, t, lowest_since_start);
                    }
                    // else: false timer jump — don't push, don't reset lowest
                }
            }
        }
    }

    // ── Signal 2: HP reset detection ──
    // When both players' rolling median HP goes from low (<0.50) to high (>0.85),
    // a new round has started.
    let p1_rm = rolling_median(&p1_norm, fr(HP_RESET_WINDOW));
    let p2_rm = rolling_median(&p2_norm, fr(HP_RESET_WINDOW));
    let mut hp_resets: Vec<usize> = Vec::new();
    let mut min_hp_since_reset: f64 = 1.0;
    for i in 0..n {
        if !p1_rm[i].is_nan() && !p2_rm[i].is_nan() {
            let current_min = p1_rm[i].min(p2_rm[i]);
            min_hp_since_reset = min_hp_since_reset.min(current_min);
            if p1_rm[i] > 0.85 && p2_rm[i] > 0.85 && min_hp_since_reset < 0.50 {
                if hp_resets.last().map_or(true, |&last| i - last > min_round_frames) {
                    eprintln!("[ROUND-DBG] hp_reset at i={} t={:.1}s (p1_rm={:.3}, p2_rm={:.3}, min_since={:.3})",
                        i, ts[i]/1000.0, p1_rm[i], p2_rm[i], min_hp_since_reset);
                    hp_resets.push(i);
                    min_hp_since_reset = current_min;
                }
            }
        }
    }

    // ── Signal 2b: Gap-based HP reset detection ──
    // Detects round resets across NaN gaps that the rolling-median HP reset
    // misses.  When both players' health drops to low values, then a NaN gap
    // occurs (KO animation / round transition), and health returns to high
    // (both > 0.85), that's unambiguously a new round.
    let mut gap_hp_resets: Vec<usize> = Vec::new();
    {
        let mut last_both_valid_idx: Option<usize> = None;
        let mut last_p1_val: f64 = f64::NAN;
        let mut last_p2_val: f64 = f64::NAN;
        for i in 0..n {
            if !p1_norm[i].is_nan() && !p2_norm[i].is_nan() {
                if let Some(prev) = last_both_valid_idx {
                    let gap_frames = i - prev;
                    // Gap >= 30 frames (~1s), at least one player was low before,
                    // and both are high now → round reset.
                    // For short gaps (< 60 frames / ~2s), require at least one player
                    // to be near-KO (< 0.25) before the gap. This filters wallbreaks
                    // where both players have moderate health (e.g. P1=0.40, P2=0.75)
                    // and the brief stage-transition VFX causes a short NaN gap + 100%
                    // flash, but the round continues.
                    let pre_gap_min = last_p1_val.min(last_p2_val);
                    if gap_frames >= fr(30)
                        && (last_p1_val < 0.50 || last_p2_val < 0.50)
                        && p1_norm[i] > 0.85 && p2_norm[i] > 0.85
                        && (gap_frames >= fr(60) || pre_gap_min < 0.25)
                    {
                        if gap_hp_resets.last().map_or(true, |&last| i - last > min_round_frames) {
                            eprintln!("[ROUND-DBG] gap_hp_reset at i={} t={:.1}s (gap={}f, pre_p1={:.3}, pre_p2={:.3})",
                                i, ts[i]/1000.0, gap_frames, last_p1_val, last_p2_val);
                            gap_hp_resets.push(i);
                        }
                    } else if gap_frames >= fr(30) && gap_frames < fr(60)
                        && (last_p1_val < 0.50 || last_p2_val < 0.50)
                        && p1_norm[i] > 0.85 && p2_norm[i] > 0.85
                        && pre_gap_min >= 0.25
                    {
                        eprintln!("[ROUND-DBG] gap_hp_reset REJECTED (wallbreak, short gap + moderate pre-HP) at i={} t={:.1}s (gap={}f, pre_p1={:.3}, pre_p2={:.3}, min={:.3})",
                            i, ts[i]/1000.0, gap_frames, last_p1_val, last_p2_val, pre_gap_min);
                    }
                }
                last_both_valid_idx = Some(i);
                last_p1_val = p1_norm[i];
                last_p2_val = p2_norm[i];
            }
        }
    }

    // ── Signal 3: Game break detection (gaps > 5s) ──
    // Track gaps in health data for EITHER player.  When one player is at
    // very low health (bar too small to read), the other player's bar is
    // still valid and prevents false game breaks mid-round.
    let mut game_breaks: Vec<usize> = Vec::new();
    let mut prev_valid_idx: Option<usize> = None;
    for i in 0..n {
        if !p1_norm[i].is_nan() || !p2_norm[i].is_nan() {
            if let Some(pv) = prev_valid_idx {
                let gap_ms = ts[i] - ts[pv];
                if gap_ms > 5000.0 {
                    eprintln!("[ROUND-DBG] game_break at i={} t={:.1}s (gap={:.1}s)",
                        i, ts[i]/1000.0, gap_ms/1000.0);
                    game_breaks.push(i);
                }
            }
            prev_valid_idx = Some(i);
        }
    }

    // ── Merge all boundary signals ──
    // Only true game breaks (>5s data gaps) are exempt from the wallbreak timer
    // filter. Gap-based HP resets (shorter gaps from super/burst VFX occlusion)
    // must still pass the timer check — if the timer didn't reset to ≥90, the
    // "reset" was just VFX covering the health bars, not a real round boundary.
    let game_break_set: std::collections::HashSet<usize> = game_breaks.iter()
        .copied().collect();

    let mut all_indices: Vec<usize> = Vec::new();
    all_indices.extend_from_slice(&timer_starts);
    all_indices.extend_from_slice(&hp_resets);
    all_indices.extend_from_slice(&gap_hp_resets);
    all_indices.extend_from_slice(&game_breaks);
    all_indices.sort();
    all_indices.dedup();

    eprintln!("[ROUND-DBG] {} — pre-filter: {} timer_starts, {} hp_resets, {} gap_hp_resets, {} game_breaks",
        replay_id, timer_starts.len(), hp_resets.len(), gap_hp_resets.len(), game_breaks.len());
    eprintln!("[ROUND-DBG] all_indices before wallbreak filter: {:?}",
        all_indices.iter().map(|&i| format!("{}({:.1}s)", i, ts[i]/1000.0)).collect::<Vec<_>>());

    // ── Wallbreak filter: real round starts always have timer ≈90+ ──
    // Apply BEFORE dedup so that false boundaries (wallbreak/super flash) don't
    // shadow valid nearby boundaries during the 10-second dedup window.
    // Game breaks are exempt — large data gaps are unambiguous boundaries.
    all_indices.retain(|&b| {
        if game_break_set.contains(&b) {
            return true; // game breaks always pass
        }
        let check_end = (b + fr(90)).min(n);
        let pass = (b..check_end).any(|j| timer_smooth[j].map_or(false, |t| t >= 90));
        if !pass {
            eprintln!("[ROUND-DBG] wallbreak filter REMOVED i={} t={:.1}s (max smooth timer in 90 frames: {:?})",
                b, ts[b]/1000.0,
                (b..check_end).filter_map(|j| timer_smooth[j]).max());
        }
        pass
    });

    // Deduplicate within 10 seconds
    let mut deduped: Vec<usize> = Vec::new();
    for &idx in &all_indices {
        if deduped.last().map_or(true, |&last| ts[idx] - ts[last] > 10000.0) {
            deduped.push(idx);
        }
    }

    // Add first valid data frame as start if needed
    // (first-valid doesn't need timer validation — could be mid-round)
    let first_valid = (0..n).find(|&i| !p1_norm[i].is_nan() || !p2_norm[i].is_nan());
    // A video can open mid-round, or with its first round start unreadable
    // (e.g. the stream shows the game scaled down for the first seconds).
    // Such a leading partial round is shorter than MIN_ROUND_FRAMES; keep it
    // only when it ends in a KO (one side's cleaned HP at 0, the other alive)
    // reached after at least LEAD_PARTIAL_MIN_FRAMES of play.
    let mut lead_partial: Option<usize> = None;
    if let Some(fv) = first_valid {
        let end = deduped.first().copied().unwrap_or(n);
        let ends_in_ko = end > fv && {
            // The first KO must come late in the span and after >= ~8 s of
            // play: a video that opens on the last seconds of a round (KO,
            // win pose) is not a round worth listing.
            let tail_lo = (fv + (end - fv) * 6 / 10).max(fv + fr(LEAD_PARTIAL_MIN_FRAMES));
            let is_ko = |j: usize, a: &[f64], b: &[f64]| {
                !a[j].is_nan() && a[j] <= 0.005 && !b[j].is_nan() && b[j] > 0.02
            };
            match (fv..end).find(|&j| is_ko(j, &p1_norm, &p2_norm) || is_ko(j, &p2_norm, &p1_norm)) {
                Some(k) => k >= tail_lo,
                None => false,
            }
        };
        if deduped.is_empty() || (deduped[0] > fv && deduped[0] - fv > min_round_frames) {
            deduped.insert(0, fv);
            if ends_in_ko { lead_partial = Some(fv); }
        } else if deduped[0] > fv && deduped[0] - fv >= fr(LEAD_PARTIAL_MIN_FRAMES) && ends_in_ko {
            eprintln!("[ROUND-DBG] {} leading partial round kept at {:.1}s (ends in a KO)",
                replay_id, ts[fv] / 1000.0);
            deduped.insert(0, fv);
            lead_partial = Some(fv);
        }
    }

    eprintln!("[ROUND-DBG] {} — final boundaries: {:?}",
        replay_id, deduped.iter().map(|&i| format!("{}({:.1}s)", i, ts[i]/1000.0)).collect::<Vec<_>>());

    // ── Build rounds from boundaries ──
    // (match starts are set from the round-counter hearts once the rounds
    // are known, see below)
    let mut round_starts: Vec<(usize, f64, bool)> = Vec::new();
    for &idx in deduped.iter() {
        round_starts.push((idx, ts[idx], false));
    }

    // Helper: find KO moment and determine winner for a span.
    // Uses smoothed HP. Handles two GGS edge cases:
    //   1. Last round of match: bars disappear (HP goes NaN before KO visible)
    //   2. Wallbreak: HP shows combo damage but doesn't visually reach zero
    // Strategy: KO moment detection → fallback to average HP in last 25% if ambiguous
    let find_winner_ko = |s: usize, e: usize| -> Option<(String, f64, f64, bool)> {
        let span = e - s;
        let search_start = s + (span as f64 * 0.4) as usize;

        // Detect mid-round wallbreaks: find the last frame where both players
        // reset to high HP after at least one was low (wallbreak stage transition).
        // Post-wallbreak, both bars flash ~1.0 — pre-wallbreak low-HP frames
        // must NOT contaminate KO detection or min-HP calculations.
        //
        // IMPORTANT: Round-end KO animations can also briefly flash both bars
        // to ~1.0 before transitioning. To distinguish real wallbreaks from
        // KO animations, require substantial valid data AFTER the reset
        // (real wallbreaks have continued combat; KO animations go to NaN quickly).
        let effective_search_start = {
            let mut eff = search_start;
            let mut saw_low_hp = false;
            for j in search_start..e {
                if !p1_norm[j].is_nan() && !p2_norm[j].is_nan() {
                    if p1_norm[j] < 0.50 || p2_norm[j] < 0.50 {
                        saw_low_hp = true;
                    }
                    if saw_low_hp && p1_norm[j] > 0.85 && p2_norm[j] > 0.85 {
                        // Before accepting as wallbreak, check if either player
                        // was truly dead (< 0.02) in the recent ~6s (180 frames).
                        // A dead player's "recovery" during KO animation is NOT
                        // a wallbreak — it's a post-KO artifact.
                        let look_back = fr(180).min(j.saturating_sub(search_start));
                        let had_ko = (j.saturating_sub(look_back)..j).any(|k|
                            (!p1_norm[k].is_nan() && p1_norm[k] < 0.02) ||
                            (!p2_norm[k].is_nan() && p2_norm[k] < 0.02)
                        );
                        if had_ko {
                            eprintln!("[WINNER-DBG] wallbreak candidate REJECTED at frame {} t={:.1}s (s={}, e={}): player was near-dead recently (KO animation)",
                                j, ts[j] / 1000.0, s, e);
                        } else {
                            // Also verify there's enough valid data after the
                            // reset (real wallbreaks have continued gameplay).
                            let post_valid = (j+1..e)
                                .filter(|&k| !p1_norm[k].is_nan() || !p2_norm[k].is_nan())
                                .count();
                            if post_valid >= fr(20) {
                                eprintln!("[WINNER-DBG] mid-round wallbreak detected at frame {} t={:.1}s (s={}, e={}), {} post-valid frames, advancing search start",
                                    j, ts[j] / 1000.0, s, e, post_valid);
                                eff = j;
                                saw_low_hp = false;
                            } else {
                                eprintln!("[WINNER-DBG] wallbreak candidate REJECTED at frame {} t={:.1}s (s={}, e={}): only {} post-valid frames",
                                    j, ts[j] / 1000.0, s, e, post_valid);
                            }
                        }
                    }
                }
            }
            eff
        };

        // Terminal-state check. The Python cleaner makes HP non-increasing
        // within a round and reads a KO as exactly 0, so each side's minimum
        // over the round is its final HP. A side that reached 0 while the
        // other didn't is the loser; otherwise the lower final HP loses.
        // (Do NOT count "frames below 0.15": a survivor can sit at 3% for
        // tens of seconds while the opponent is KO'd at the very end.)
        {
            let mut p1_min = f64::INFINITY;
            let mut p2_min = f64::INFINITY;
            for j in effective_search_start..e {
                if !p1_norm[j].is_nan() { p1_min = p1_min.min(p1_norm[j]); }
                if !p2_norm[j].is_nan() { p2_min = p2_min.min(p2_norm[j]); }
            }
            if p1_min.is_finite() && p2_min.is_finite() {
                const KO_HP: f64 = 0.01;
                let p1_ko = p1_min <= KO_HP;
                let p2_ko = p2_min <= KO_HP;
                let diff = (p1_min - p2_min).abs();
                eprintln!("[WINNER-DBG] {} terminal s={} e={} p1_min={:.3} p2_min={:.3}",
                    replay_id, s, e, p1_min, p2_min);
                if p1_ko != p2_ko {
                    let winner = if p1_ko { "P2" } else { "P1" };
                    eprintln!("[WINNER-DBG]   → {} wins via single KO", winner);
                    return Some((winner.to_string(), p1_min, p2_min, true));
                }
                if !p1_ko && !p2_ko && diff >= 0.03 {
                    let winner = if p1_min > p2_min { "P1" } else { "P2" };
                    eprintln!("[WINNER-DBG]   → {} wins via lower final HP (diff={:.3})", winner, diff);
                    return Some((winner.to_string(), p1_min, p2_min, diff >= 0.10));
                }
                // Both at 0 (double KO / KO animation artifacts) or too close
                // to call: fall through to the KO-moment heuristics below.
            }
        }

        // HP pair to report when the KO-count rules below pick a winner: the
        // winner's HP when the loser first hit 0, and the loser's minimum.
        let ko_hp = |p1_loses: bool| -> (f64, f64) {
            let (lose, win) = if p1_loses { (&p1_norm, &p2_norm) } else { (&p2_norm, &p1_norm) };
            let ko = (effective_search_start..e).find(|&j| !lose[j].is_nan() && lose[j] <= 0.02);
            let upto = ko.map(|k| k + 1).unwrap_or(e);
            let win_hp = (effective_search_start..upto).rev().find(|&j| !win[j].is_nan()).map(|j| win[j]).unwrap_or(1.0);
            let lose_hp = (effective_search_start..e).filter(|&j| !lose[j].is_nan()).map(|j| lose[j]).fold(f64::INFINITY, f64::min);
            let lose_hp = if lose_hp.is_finite() { lose_hp } else { 0.0 };
            if p1_loses { (lose_hp, win_hp) } else { (win_hp, lose_hp) }
        };

        // Early check: count raw frames with HP < 0.15 in last 60%.
        // Filter post-KO garbage: only count if the OTHER player is NOT at full HP.
        // After a KO, the winning player's bar resets to ~1.0 while the loser's
        // shows residual low values — these aren't real combat frames.
        let p1_ko_count = (effective_search_start..e)
            .filter(|&j| !p1_norm[j].is_nan() && p1_norm[j] < 0.15
                      && !p2_norm[j].is_nan() && p2_norm[j] < 0.90)
            .count();
        let p2_ko_count = (effective_search_start..e)
            .filter(|&j| !p2_norm[j].is_nan() && p2_norm[j] < 0.15
                      && !p1_norm[j].is_nan() && p1_norm[j] < 0.90)
            .count();
        eprintln!("[WINNER-DBG] {} find_winner_ko s={} e={} t={:.1}s-{:.1}s eff_start={} p1_ko={} p2_ko={}",
            replay_id, s, e, ts[s] / 1000.0, ts[e.min(n - 1)] / 1000.0, effective_search_start, p1_ko_count, p2_ko_count);
        if p1_ko_count >= 3 && p1_ko_count >= p2_ko_count * 3 + 1 {
            eprintln!("[WINNER-DBG]   → P2 wins via p1_ko_count");
            let (a, b) = ko_hp(true);
            return Some(("P2".to_string(), a, b, true));
        }
        if p2_ko_count >= 3 && p2_ko_count >= p1_ko_count * 3 + 1 {
            eprintln!("[WINNER-DBG]   → P1 wins via p2_ko_count");
            let (a, b) = ko_hp(false);
            return Some(("P1".to_string(), a, b, true));
        }
        // Both players have KO frames but neither dominates — could be KO
        // animation artifact OR wallbreak/comeback. When the two KO moments
        // are close together (≤90 frames / 3s), it's a KO animation and the
        // first player to reach zero is the real loser. When far apart (>90),
        // it's a wallbreak/comeback — let the KO clustering handle it.
        if p1_ko_count >= 3 && p2_ko_count >= 3 {
            let first_p1 = (effective_search_start..e).find(|&j| !p1_norm[j].is_nan() && p1_norm[j] < 0.05);
            let first_p2 = (effective_search_start..e).find(|&j| !p2_norm[j].is_nan() && p2_norm[j] < 0.05);
            if let (Some(fp1), Some(fp2)) = (first_p1, first_p2) {
                let gap = if fp1 > fp2 { fp1 - fp2 } else { fp2 - fp1 };
                if gap <= fr(90) {
                    // Only use first-to-zero when it's a one-sided KO
                    // (other player > 0.15 HP). If both are near KO at the
                    // earlier moment, it's ambiguous — check the later zero.
                    let (earlier, later, earlier_is_p1) = if fp1 < fp2 {
                        (fp1, fp2, true)
                    } else {
                        (fp2, fp1, false)
                    };
                    let other_at_earlier = if earlier_is_p1 { p2_norm[earlier] } else { p1_norm[earlier] };
                    let other_at_later = if earlier_is_p1 { p1_norm[later] } else { p2_norm[later] };
                    if other_at_earlier > 0.15 {
                        // Earlier zero is one-sided — earlier player died
                        if earlier_is_p1 {
                            let (a, b) = ko_hp(true);
            return Some(("P2".to_string(), a, b, true));
                        } else {
                            let (a, b) = ko_hp(false);
            return Some(("P1".to_string(), a, b, true));
                        }
                    } else if other_at_later > 0.15 {
                        // Later zero is one-sided — later player died
                        if earlier_is_p1 {
                            let (a, b) = ko_hp(false);
            return Some(("P1".to_string(), a, b, true));
                        } else {
                            let (a, b) = ko_hp(true);
            return Some(("P2".to_string(), a, b, true));
                        }
                    }
                }
            }
        }

        // Search post-wallbreak portion of the round for the frame where min(p1, p2) is lowest
        let mut best_min_hp: f64 = 2.0;
        let mut best_p1: f64 = 0.5;
        let mut best_p2: f64 = 0.5;
        let mut valid_frames: Vec<(usize, f64, f64)> = Vec::new();  // (frame_idx, p1, p2)

        for j in effective_search_start..e {
            if !p1_smooth[j].is_nan() && !p2_smooth[j].is_nan() {
                valid_frames.push((j, p1_smooth[j], p2_smooth[j]));
                let min_hp = p1_smooth[j].min(p2_smooth[j]);
                if min_hp < best_min_hp {
                    best_min_hp = min_hp;
                    best_p1 = p1_smooth[j];
                    best_p2 = p2_smooth[j];
                }
            }
        }

        if valid_frames.is_empty() {
            return None;
        }

        // Collect frames near the KO threshold, then split into contiguous
        // temporal clusters. Use the LAST cluster since KO happens at round end.
        // This avoids mixing disconnected mid-round dips with the actual KO.
        let ko_frames: Vec<(usize, f64, f64)> = valid_frames
            .iter()
            .copied()
            .filter(|&(_, p1, p2)| p1.min(p2) < best_min_hp + 0.10)
            .collect();

        let (ep1, ep2) = if !ko_frames.is_empty() {
            // Split into clusters separated by > 60 frames (~2s)
            let mut clusters: Vec<Vec<(usize, f64, f64)>> = vec![vec![ko_frames[0]]];
            for &item in &ko_frames[1..] {
                let last_cluster = clusters.last().unwrap();
                if item.0 - last_cluster.last().unwrap().0 > fr(60) {
                    clusters.push(vec![item]);
                } else {
                    clusters.last_mut().unwrap().push(item);
                }
            }

            // Use the last cluster (closest to round end = actual KO)
            let last_cluster = clusters.last().unwrap();
            let take_count = last_cluster.len().min(fr(30));
            let ko_use = &last_cluster[last_cluster.len() - take_count..];
            let mut p1_vals: Vec<f64> = ko_use.iter().map(|f| f.1).collect();
            let mut p2_vals: Vec<f64> = ko_use.iter().map(|f| f.2).collect();
            p1_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p2_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (p1_vals[p1_vals.len() / 2], p2_vals[p2_vals.len() / 2])
        } else {
            (best_p1, best_p2)
        };

        let hp_diff = (ep1 - ep2).abs();

        eprintln!("[WINNER-DBG]   cluster: ep1={:.3} ep2={:.3} hp_diff={:.3}", ep1, ep2, hp_diff);
        // If KO moment is clear (large HP gap), use it directly
        if hp_diff >= 0.15 {
            let winner = if ep1 >= ep2 { "P1".to_string() } else { "P2".to_string() };
            eprintln!("[WINNER-DBG]   → {} wins via cluster hp_diff", winner);
            // ep1/ep2 are smoothed medians and can pick up the next round's
            // full bars; when the loser has a KO frame, report the cleaned HP
            // at that moment instead (as the KO-count rules do).
            let p1_loses = winner == "P2";
            let lose = if p1_loses { &p1_norm } else { &p2_norm };
            if (effective_search_start..e).any(|j| !lose[j].is_nan() && lose[j] <= 0.02) {
                let (a, b) = ko_hp(p1_loses);
                return Some((winner, a, b, true));
            }
            return Some((winner, ep1, ep2, true));
        }

        // Ambiguous KO moment — fallback strategies:

        // Strategy A: Find last valid HP readings before data gaps out
        // (handles last-round-of-match where bars disappear)
        let last_quarter_start = (s + (span as f64 * 0.75) as usize).max(effective_search_start);
        let mut last_valid_p1: Option<f64> = None;
        let mut last_valid_p2: Option<f64> = None;
        for j in (last_quarter_start..e).rev() {
            if last_valid_p1.is_none() && !p1_smooth[j].is_nan() {
                last_valid_p1 = Some(p1_smooth[j]);
            }
            if last_valid_p2.is_none() && !p2_smooth[j].is_nan() {
                last_valid_p2 = Some(p2_smooth[j]);
            }
            if last_valid_p1.is_some() && last_valid_p2.is_some() {
                break;
            }
        }

        // Find minimum HP each player reaches post-wallbreak (smoothed)
        let mut p1_min_last = 2.0f64;
        let mut p2_min_last = 2.0f64;
        for j in effective_search_start..e {
            if !p1_smooth[j].is_nan() { p1_min_last = p1_min_last.min(p1_smooth[j]); }
            if !p2_smooth[j].is_nan() { p2_min_last = p2_min_last.min(p2_smooth[j]); }
        }

        // Combine signals: whoever reached lower minimum HP is the loser
        let mut min_signal = p2_min_last - p1_min_last;  // positive = P1 went lower = P2 wins

        // Detect transition artifact: both min values in [0.40, 0.55] with
        // tiny difference = "loading screen" where both bars show ~50%.
        if p1_min_last >= 0.40 && p1_min_last <= 0.55
            && p2_min_last >= 0.40 && p2_min_last <= 0.55
            && min_signal.abs() < 0.05
        {
            min_signal = 0.0;
        }

        // Strategy B: Average HP in last 25% of the round.
        // Exclude transition artifacts where both bars drop simultaneously
        // below 0.60 (loading screen pattern).
        let mut p1_avg_vals: Vec<f64> = Vec::new();
        let mut p2_avg_vals: Vec<f64> = Vec::new();
        for j in last_quarter_start..e {
            if !p1_smooth[j].is_nan() && !p2_smooth[j].is_nan() {
                if p1_smooth[j] < 0.60 && p2_smooth[j] < 0.60 {
                    continue;
                }
                p1_avg_vals.push(p1_smooth[j]);
                p2_avg_vals.push(p2_smooth[j]);
            }
        }

        let avg_signal = if p1_avg_vals.len() >= fr(10) && p2_avg_vals.len() >= fr(10) {
            let p1_avg: f64 = p1_avg_vals.iter().sum::<f64>() / p1_avg_vals.len() as f64;
            let p2_avg: f64 = p2_avg_vals.iter().sum::<f64>() / p2_avg_vals.len() as f64;
            p2_avg - p1_avg  // positive = P1 lower avg = P2 wins
        } else if let (Some(lp1), Some(lp2)) = (last_valid_p1, last_valid_p2) {
            lp2 - lp1  // positive = P1 lower = P2 wins
        } else {
            ep2 - ep1  // fall back to KO moment
        };

        // Dynamic weighting: when both players' min HP is similar, the
        // difference is noise (transition artifacts, OCR jitter). Use
        // avg_signal which captures the post-round result screen HP.
        let hp_range = (p1_min_last - p2_min_last).abs();
        let min_weight = (hp_range / 0.20).min(0.8);
        let avg_weight = 1.0 - min_weight;
        let mut combined = min_signal * min_weight + avg_signal * avg_weight;
        eprintln!("[WINNER-DBG]   fallback: p1_min={:.3} p2_min={:.3} min_sig={:.3} avg_sig={:.3} combined={:.3}",
            p1_min_last, p2_min_last, min_signal, avg_signal, combined);

        // Track confidence before tiebreaker — if the signal was already clear
        // (>= 0.10), we're confident. Tiebreaker is a best-guess for display
        // but should NOT be marked as confident since it can be wrong
        // (e.g. next round starting quickly pollutes post-round visibility).
        let confident = combined.abs() >= 0.10;

        // Tiebreaker: asymmetric data presence beyond round end.
        // In GGS, the winner's HP bar stays visible on the result screen while
        // the loser's disappears. If one player has many more visible frames
        // after the round ends, they're likely the winner. This is a best-guess
        // for ambiguous rounds — the user can override via the UI.
        if combined.abs() < 0.10 {
            let scan_end = (e + fr(300)).min(n);
            let p1_post = (e..scan_end).filter(|&k| !p1_norm[k].is_nan()).count();
            let p2_post = (e..scan_end).filter(|&k| !p2_norm[k].is_nan()).count();
            if p2_post > p1_post + fr(10) {
                eprintln!("[WINNER-DBG]   tiebreaker(post-round): P2 visible {} frames vs P1 {} after round end → P2 (uncertain)", p2_post, p1_post);
                combined = 0.10;
            } else if p1_post > p2_post + fr(10) {
                eprintln!("[WINNER-DBG]   tiebreaker(post-round): P1 visible {} frames vs P2 {} after round end → P1 (uncertain)", p1_post, p2_post);
                combined = -0.10;
            }
        }
        let (winner, _fp1, _fp2) = if combined > 0.0 {
            ("P2".to_string(), p1_min_last.min(2.0), p2_min_last.min(2.0))
        } else {
            ("P1".to_string(), p1_min_last.min(2.0), p2_min_last.min(2.0))
        };
        if !confident {
            eprintln!("[WINNER-DBG]   ⚠ UNCERTAIN winner={} combined={:.3}", winner, combined);
        }

        // Use the better HP values for reporting
        let report_p1 = if p1_min_last < 2.0 { p1_min_last } else { ep1 };
        let report_p2 = if p2_min_last < 2.0 { p2_min_last } else { ep2 };
        Some((winner, report_p1, report_p2, confident))
    };

    // Helper: recursively split long rounds at gaps or HP resets.
    // All split candidates must be confirmed by timer or HP reset evidence.
    fn split_long_round(
        s_idx: usize,
        e_idx: usize,
        ts: &[f64],
        p1_norm: &[f64],
        p2_norm: &[f64],
        p1_smooth: &[f64],
        p2_smooth: &[f64],
        timer_smooth: &[Option<i32>],
        depth: usize,
        scale: f64,
    ) -> Vec<(usize, usize)> {
        let n = timer_smooth.len();
        let fr = |frames: usize| -> usize { scale_frames(frames, scale) };
        let dur = (ts[e_idx.min(ts.len() - 1)] - ts[s_idx]) / 1000.0;
        if dur <= 45.0 || depth > 3 {
            return vec![(s_idx, e_idx)];
        }

        // Timer confirmation: smoothed timer must reach ≥90 within 210 frames (7s).
        let timer_ok = |idx: usize| -> bool {
            let check_end = (idx + fr(210)).min(n);
            (idx..check_end).any(|j| timer_smooth[j].map_or(false, |t| t >= 90))
        };

        // Alternative confirmation for wallbreak transitions: KO evidence before
        // the gap + timer ≥90 nearby + both HP reset to >0.90 after.
        let round_reset_ok = |idx: usize| -> bool {
            // Check for KO evidence before the gap (within preceding 450 frames / 15s)
            let ko_start = idx.saturating_sub(fr(450));
            let p1_ko = (ko_start..idx)
                .filter(|&j| !p1_norm[j].is_nan() && p1_norm[j] < 0.10)
                .count();
            let p2_ko = (ko_start..idx)
                .filter(|&j| !p2_norm[j].is_nan() && p2_norm[j] < 0.10)
                .count();
            if p1_ko < 3 && p2_ko < 3 {
                return false;
            }
            // Timer must be ≥90 nearby
            let timer_near = (idx + fr(120)).min(n);
            let has_timer = (idx..timer_near)
                .any(|j| timer_smooth[j].map_or(false, |t| t >= 90));
            if !has_timer {
                return false;
            }
            // Both smoothed HP must reset to >0.90 after the gap
            let hp_check = (idx + fr(300)).min(n);
            (idx..hp_check).any(|j| {
                !p1_smooth[j].is_nan() && !p2_smooth[j].is_nan()
                    && p1_smooth[j] > 0.90 && p2_smooth[j] > 0.90
            })
        };

        let split_confirmed = |idx: usize| -> bool {
            timer_ok(idx) || round_reset_ok(idx)
        };

        let mut best_split: Option<usize> = None;
        let mut best_score: f64 = 0.0;

        // Strategy 1: largest data gap > 1.5s (timer or round reset must confirm)
        let mut last_v: Option<usize> = None;
        for i in s_idx..e_idx {
            if !p1_norm[i].is_nan() {
                if let Some(lv) = last_v {
                    let g = (ts[i] - ts[lv]) / 1000.0;
                    if g > 1.5 && g > best_score {
                        let mid_t = ts[i] / 1000.0;
                        let start_t = ts[s_idx] / 1000.0;
                        let end_t = ts[e_idx.min(ts.len() - 1)] / 1000.0;
                        let dur1 = mid_t - start_t;
                        let dur2 = end_t - mid_t;
                        if dur1 >= 12.0 && dur2 >= 12.0 && split_confirmed(i) {
                            best_split = Some(i);
                            best_score = g;
                        }
                    }
                }
                last_v = Some(i);
            }
        }

        // Strategy 2: HP reset within the long round (timer must confirm)
        let local_len = e_idx - s_idx;
        if local_len > fr(60) {
            let half = fr(30);
            let mut min_hp_local: f64 = 1.0;
            for j in 0..local_len {
                let gi = s_idx + j;
                let lo = if j >= half { gi - half } else { s_idx };
                let hi = (gi + half + 1).min(e_idx);
                let mut p1_chunk: Vec<f64> = (lo..hi)
                    .filter(|&k| !p1_norm[k].is_nan())
                    .map(|k| p1_norm[k])
                    .collect();
                let mut p2_chunk: Vec<f64> = (lo..hi)
                    .filter(|&k| !p2_norm[k].is_nan())
                    .map(|k| p2_norm[k])
                    .collect();
                if p1_chunk.len() >= fr(10) && p2_chunk.len() >= fr(10) {
                    p1_chunk.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    p2_chunk.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p1_med = p1_chunk[p1_chunk.len() / 2];
                    let p2_med = p2_chunk[p2_chunk.len() / 2];
                    let current_min = p1_med.min(p2_med);
                    min_hp_local = min_hp_local.min(current_min);

                    if p1_med > 0.75 && p2_med > 0.75 && min_hp_local < 0.55 {
                        let mid_t = ts[gi] / 1000.0;
                        let start_t = ts[s_idx] / 1000.0;
                        let end_t = ts[e_idx.min(ts.len() - 1)] / 1000.0;
                        let dur1 = mid_t - start_t;
                        let dur2 = end_t - mid_t;
                        if dur1 >= 12.0 && dur2 >= 12.0 && timer_ok(gi) {
                            let balance = 1.0 - (dur1 - dur2).abs() / dur;
                            let hp_drop = 1.0 - min_hp_local;
                            let score = balance * 0.3 + hp_drop * 0.7;
                            if score > best_score {
                                best_split = Some(gi);
                                best_score = score;
                            }
                        }
                        min_hp_local = current_min;
                    }
                }
            }
        }

        if let Some(split_idx) = best_split {
            let mut result = split_long_round(s_idx, split_idx, ts, p1_norm, p2_norm, p1_smooth, p2_smooth, timer_smooth, depth + 1, scale);
            result.extend(split_long_round(split_idx, e_idx, ts, p1_norm, p2_norm, p1_smooth, p2_smooth, timer_smooth, depth + 1, scale));
            return result;
        }

        vec![(s_idx, e_idx)]
    }

    // ── Build initial round spans ──
    struct RoundSpan {
        s_idx: usize,
        e_idx: usize,
        is_match_start: bool,
    }

    let mut initial_spans: Vec<RoundSpan> = Vec::new();
    for ri in 0..round_starts.len() {
        let (s_idx, _, is_ms) = round_starts[ri];
        let e_idx = if ri + 1 < round_starts.len() {
            round_starts[ri + 1].0
        } else {
            n - 1
        };
        // End the round shortly after its last HP reading, not at the next
        // round's start or the end of the video: the loading screen (or the
        // rest of a stream after the last set) is not part of the round, and
        // a long tail of it would fail the valid-ratio check below.
        const END_PAD_FRAMES: usize = 30;
        // A short block of readings after a long HUD gap, with no timer, is the
        // next round's intro (empty bars before the HUD fills), not this round.
        const TAIL_BLOCK_MAX: usize = 60;
        const TAIL_GAP_MIN: usize = 90;
        let has_hp = |j: usize| !p1_norm[j].is_nan() || !p2_norm[j].is_nan();
        let mut last_hp = (s_idx..e_idx).rev().find(|&j| has_hp(j));
        if let Some(last) = last_hp {
            let mut b = last;
            while b > s_idx && (b.saturating_sub(fr(15))..b).any(|j| j >= s_idx && has_hp(j)) {
                b -= 1;
            }
            let prev = (s_idx..b).rev().find(|&j| has_hp(j));
            if let Some(prev) = prev {
                if last - b < fr(TAIL_BLOCK_MAX) && b - prev >= fr(TAIL_GAP_MIN)
                    && (b..=last).all(|j| timer_raw[j].is_none())
                {
                    last_hp = Some(prev);
                }
            }
        }
        let e_idx = match last_hp {
            Some(last) => (last + 1 + fr(END_PAD_FRAMES)).min(e_idx),
            None => e_idx,
        };

        let total_frames = e_idx - s_idx;
        let min_len = if lead_partial == Some(s_idx) { fr(LEAD_PARTIAL_MIN_FRAMES) } else { min_round_frames };
        if total_frames < min_len {
            eprintln!("[ROUND-DBG] span {:.1}s-{:.1}s DROPPED: too short ({} < {})",
                ts[s_idx]/1000.0, ts[e_idx.min(n-1)]/1000.0, total_frames, min_len);
            continue;
        }

        // Valid data ratio check — accept frames where at least one player
        // has health data.  When one player is at very low health (<50 pixels),
        // their bar is too small to read but the other player's bar is still valid.
        let valid_count = (s_idx..e_idx)
            .filter(|&j| !p1_norm[j].is_nan() || !p2_norm[j].is_nan())
            .count();
        // Diagnostic: count NaN breakdown for dropped spans
        if total_frames > 0 && (valid_count as f64) / (total_frames as f64) < 0.30 {
            let p1_nan = (s_idx..e_idx).filter(|&j| p1_norm[j].is_nan()).count();
            let p2_nan = (s_idx..e_idx).filter(|&j| p2_norm[j].is_nan()).count();
            let both_nan = (s_idx..e_idx).filter(|&j| p1_norm[j].is_nan() && p2_norm[j].is_nan()).count();
            let p1_raw_nan = (s_idx..e_idx).filter(|&j| p1_raw[j].is_nan()).count();
            let p2_raw_nan = (s_idx..e_idx).filter(|&j| p2_raw[j].is_nan()).count();
            // Sample values at 25%, 50%, 75% through span
            let samples: Vec<String> = [0.25, 0.50, 0.75].iter().map(|&frac| {
                let si = s_idx + (total_frames as f64 * frac) as usize;
                format!("{:.1}s:p1={:.3}/p2={:.3}", ts[si]/1000.0,
                    p1_norm[si], p2_norm[si])
            }).collect();
            eprintln!("[ROUND-DBG] span {:.1}s-{:.1}s DROPPED: low valid ratio ({}/{} = {:.2}) \
                p1_nan={} p2_nan={} both_nan={} p1_raw_nan={} p2_raw_nan={} samples=[{}]",
                ts[s_idx]/1000.0, ts[e_idx.min(n-1)]/1000.0, valid_count, total_frames,
                valid_count as f64 / total_frames as f64,
                p1_nan, p2_nan, both_nan, p1_raw_nan, p2_raw_nan,
                samples.join(", "));
            continue;
        }

        eprintln!("[ROUND-DBG] span KEPT: {:.1}s-{:.1}s ({} frames, {}/{} valid, match_start={})",
            ts[s_idx]/1000.0, ts[e_idx.min(n-1)]/1000.0, total_frames, valid_count, total_frames, is_ms);
        initial_spans.push(RoundSpan { s_idx, e_idx, is_match_start: is_ms });
    }

    // ── Split long rounds and build results ──
    let mut results: Vec<RoundResult> = Vec::new();
    let mut round_counter: usize = 0;

    for span in &initial_spans {
        let dur_s = (ts[span.e_idx.min(n - 1)] - ts[span.s_idx]) / 1000.0;
        let sub_spans = if dur_s > 45.0 {
            split_long_round(span.s_idx, span.e_idx, &ts, &p1_norm, &p2_norm, &p1_smooth, &p2_smooth, &timer_smooth, 0, rate_scale)
        } else {
            vec![(span.s_idx, span.e_idx)]
        };

        for (sub_i, (s_idx, e_idx)) in sub_spans.iter().enumerate() {
            let (s_idx, e_idx) = (*s_idx, *e_idx);
            // Only the first sub-span inherits the match_start flag
            let is_match_start = sub_i == 0 && span.is_match_start;

            // Trim span to first/last frame with valid HP data.
            // This excludes dead time (character select, menus, transitions)
            // that appears as NULL HP gaps at the start/end of a round boundary.
            let trimmed_start = (s_idx..e_idx)
                .find(|&j| !p1_norm[j].is_nan() && !p2_norm[j].is_nan());
            let trimmed_end = (s_idx..e_idx).rev()
                .find(|&j| !p1_norm[j].is_nan() && !p2_norm[j].is_nan());
            let (s_idx, e_idx) = match (trimmed_start, trimmed_end) {
                (Some(s), Some(e)) => (s, e + 1),
                _ => continue, // no valid data in this span
            };

            let s_ms = ts[s_idx];
            let e_ms = ts[e_idx.min(n - 1)];
            let sub_dur = (e_ms - s_ms) / 1000.0;

            let min_dur = if sub_i == 0 && lead_partial == Some(span.s_idx) { 8.0 } else { 12.0 };
            if sub_dur < min_dur {
                continue;
            }

            // Valid data check for sub-span
            let valid_count = (s_idx..e_idx)
                .filter(|&j| !p1_norm[j].is_nan() && !p2_norm[j].is_nan())
                .count();
            if valid_count == 0 {
                continue;
            }

            // Determine winner using KO-moment approach
            let (winner, ep1, ep2, winner_confident) = match find_winner_ko(s_idx, e_idx) {
                Some(v) => v,
                None => continue,
            };

            // ── Round metrics ──
            let (w_smooth, l_smooth) = if winner == "P2" {
                (&p2_smooth, &p1_smooth)
            } else {
                (&p1_smooth, &p2_smooth)
            };

            let mut w_valid: Vec<f64> = Vec::new();
            let mut l_valid: Vec<f64> = Vec::new();
            let mut ts_valid: Vec<f64> = Vec::new();

            for i in s_idx..e_idx {
                if !w_smooth[i].is_nan() && !l_smooth[i].is_nan() {
                    w_valid.push(w_smooth[i]);
                    l_valid.push(l_smooth[i]);
                    ts_valid.push(ts[i]);
                }
            }

            if w_valid.is_empty() {
                continue;
            }

            let winner_final_hp = if winner == "P2" { ep2 } else { ep1 };
            let loser_final_hp = if winner == "P2" { ep1 } else { ep2 };
            let winner_min_hp = w_valid.iter().cloned().fold(f64::INFINITY, f64::min);

            // Use heavy smoothing for deficit tracking to filter HP noise
            let (w_heavy, l_heavy) = if winner == "P2" {
                (&p2_heavy, &p1_heavy)
            } else {
                (&p1_heavy, &p2_heavy)
            };

            // Skip first 15% of round to avoid HP reset transition artifacts
            let deficit_start = s_idx + (e_idx - s_idx) * 15 / 100;
            let mut max_deficit: f64 = 0.0;
            let mut deficit_ts: f64 = ts[s_idx];
            for i in deficit_start..e_idx {
                if !w_heavy[i].is_nan() && !l_heavy[i].is_nan() {
                    let deficit = l_heavy[i] - w_heavy[i];
                    if deficit > max_deficit {
                        max_deficit = deficit;
                        deficit_ts = ts[i];
                    }
                }
            }

            let is_comeback = max_deficit >= COMEBACK_DEFICIT;

            results.push(RoundResult {
                replay_id: replay_id.to_string(),
                round_index: round_counter,
                round_start_ms: s_ms,
                round_end_ms: e_ms,
                winner,
                winner_final_hp,
                loser_final_hp,
                winner_min_hp,
                max_deficit,
                deficit_timestamp_ms: deficit_ts,
                is_comeback,
                is_match_start,
                winner_confident,
            });
            round_counter += 1;
        }
    }

    // ── Round-counter hearts: match starts and winner check ─────────────
    // The two hearts on each side of the timer are part of the game's HUD
    // (present in every GGST video, not only our stream overlay; only the
    // name/score banner above them is overlay and is never read).  A side
    // loses a heart for each round the opponent wins in the current match;
    // p1_rounds_won = hearts P2 lost.  The HUD updates them at the start of
    // the next round, so the state read early in a round is the result of
    // the rounds BEFORE it: the hearts read between rounds (or just before a
    // round starts) still show the previous state.  All windows are in
    // milliseconds.  The hearts never add or move round boundaries: rounds
    // come only from the HP/timer resets and data gaps above.
    const HEARTS_FROM_MS: f64 = 500.0;
    const HEARTS_TO_MS: f64 = 10000.0;
    const HEARTS_MIN_PCT: usize = 80;
    // A round start follows a HUD gap: >= HUD_GAP_MS without a timer reading
    // (KO / win pose / loading / character select), looked for up to
    // HUD_GAP_LOOKBACK_MS before the round's first HP reading ...
    const HUD_GAP_MS: f64 = 1000.0;
    const HUD_GAP_LOOKBACK_MS: f64 = 15000.0;
    // (the timer can appear a few seconds after the first HP reading)
    const HUD_GAP_AFTER_MS: f64 = 5000.0;
    // ... and the timer shows its intro value (99) right after that gap.
    const TIMER_INTRO_MIN: i32 = 97;
    const TIMER_INTRO_WINDOW_MS: f64 = 5000.0;
    // A hearts state needs >= ~1 s of identical reads (both sides readable).
    let hearts_min_reads = fr(30);
    let hearts_state = |s_ms: f64, e_ms: f64| -> Option<(i32, i32)> {
        let a = ts.partition_point(|&t| t < s_ms + HEARTS_FROM_MS);
        let b = ts.partition_point(|&t| t < (s_ms + HEARTS_TO_MS).min(e_ms));
        let mut counts: std::collections::HashMap<(i32, i32), usize> = std::collections::HashMap::new();
        let mut total = 0usize;
        for j in a..b.max(a) {
            if let (Some(x), Some(y)) = (p1_rounds_won[j], p2_rounds_won[j]) {
                *counts.entry((x, y)).or_insert(0) += 1;
                total += 1;
            }
        }
        let (best, count) = counts.into_iter().max_by_key(|&(_, c)| c)?;
        if count >= hearts_min_reads && count * 100 >= total * HEARTS_MIN_PCT {
            Some(best)
        } else {
            None
        }
    };
    // Indices of the first timer reading after each HUD gap around a round
    // start (latest first).  The start of the video counts as a gap.
    let hud_gap_ends = |s_ms: f64| -> Vec<usize> {
        let lo = ts.partition_point(|&t| t < s_ms - HUD_GAP_LOOKBACK_MS);
        let hi = ts.partition_point(|&t| t < s_ms + HUD_GAP_AFTER_MS);
        let mut ends = Vec::new();
        let mut next_read: Option<usize> = None;
        for j in (lo..hi).rev() {
            if timer_raw[j].is_none() {
                continue;
            }
            if let Some(r) = next_read {
                if ts[r] - ts[j] >= HUD_GAP_MS {
                    ends.push(r);
                }
            }
            next_read = Some(j);
        }
        // The first reading in the window: a gap end if nothing was read for
        // HUD_GAP_MS before it, or the video starts there.
        if let Some(r) = next_read {
            if ts[r] - ts[lo] >= HUD_GAP_MS || (0..lo).all(|j| timer_raw[j].is_none()) {
                ends.push(r);
            }
        }
        ends
    };
    let timer_intro_after = |g: usize| -> bool {
        let b = ts.partition_point(|&t| t <= ts[g] + TIMER_INTRO_WINDOW_MS);
        (g..b).any(|j| timer_sup[j].map_or(false, |v| v >= TIMER_INTRO_MIN))
    };
    let states: Vec<Option<(i32, i32)>> = results
        .iter()
        .map(|r| hearts_state(r.round_start_ms, r.round_end_ms))
        .collect();
    let intros: Vec<bool> = results
        .iter()
        .map(|r| hud_gap_ends(r.round_start_ms).into_iter().any(|g| timer_intro_after(g)))
        .collect();
    // Hearts state after round `p`: the latest state read at or before `p`
    // (since the last match start), or 0/0 at the last match start, plus the
    // HP winners of the rounds since.
    let state_after = |p: usize, last_ms: Option<usize>, res: &[RoundResult]| -> Option<(i32, i32)> {
        let floor = last_ms.unwrap_or(0);
        let (q, mut st) = match (floor..=p).rev().find(|&q| states[q].is_some()) {
            Some(q) => (q, states[q].unwrap()),
            None => (last_ms?, (0, 0)),
        };
        for r in &res[q..=p] {
            if r.winner == "P1" { st.0 += 1 } else { st.1 += 1 }
        }
        Some(st)
    };
    let mut last_match_start: Option<usize> = None;
    for ri in 0..results.len() {
        // The first round of a match: after a HUD gap the timer starts at
        // its intro value, and both sides' hearts read full (0 lost) for
        // >= 1 s, after a round in which some were lost (the old hearts
        // reader reads 0/0 nearly always).  When this round's hearts are
        // unreadable, the previous round must have ended a first-to-2 match
        // (its hearts state plus its HP winner).
        let ms = intros[ri] && match states[ri] {
            Some(st) => st == (0, 0) && (ri == 0 || states[ri - 1] != Some((0, 0))),
            None => ri > 0 && state_after(ri - 1, last_match_start, &results)
                .map_or(false, |(a, b)| a.max(b) >= 2),
        };
        if ms && states[ri].is_none() {
            eprintln!("[HEARTS-DBG] round {} t={:.1}s: hearts unreadable, match start from the previous round's result",
                ri, results[ri].round_start_ms / 1000.0);
        }
        results[ri].is_match_start = ms;
        if ms {
            last_match_start = Some(ri);
        }
    }
    // Winner check from the hearts: compare this round's state with the next
    // round's.  One side up by one: that side won.  The next round back at
    // 0/0: the match ended here, won by the side that
    // already had a round.  The hearts are a secondary signal: they confirm
    // an uncertain HP winner or break its tie (it stays uncertain), and a
    // disagreement with a confident HP winner only flags the round as
    // uncertain.  HP and timer stay primary.
    for ri in 0..results.len() {
        let pre = states[ri];
        let post = if ri + 1 < results.len() { states[ri + 1] } else { None };
        let (Some(pre), Some(post)) = (pre, post) else {
            eprintln!("[HEARTS-DBG] round {} t={:.1}s-{:.1}s: no hearts state (pre={:?} post={:?}) hp_winner={}",
                ri, results[ri].round_start_ms / 1000.0, results[ri].round_end_ms / 1000.0, pre, post, results[ri].winner);
            continue;
        };
        let hearts_winner = if post == (0, 0) && pre != (0, 0) {
            match pre {
                (1, 0) => Some("P1"),
                (0, 1) => Some("P2"),
                _ => None, // 1-1: the deciding round, either side
            }
        } else if post.0 == pre.0 + 1 && post.1 == pre.1 {
            Some("P1")
        } else if post.1 == pre.1 + 1 && post.0 == pre.0 {
            Some("P2")
        } else {
            None
        };
        eprintln!("[HEARTS-DBG] round {} t={:.1}s-{:.1}s: pre={:?} post={:?} hearts_winner={:?} hp_winner={} confident={} match_start={}",
            ri, results[ri].round_start_ms / 1000.0, results[ri].round_end_ms / 1000.0, pre, post,
            hearts_winner, results[ri].winner, results[ri].winner_confident, results[ri].is_match_start);
        let Some(hw) = hearts_winner else { continue };
        if results[ri].winner == hw {
            if !results[ri].winner_confident {
                eprintln!("[HEARTS-DBG]   → hearts confirm uncertain round {} winner {}", ri, hw);
                results[ri].winner_confident = true;
            }
            continue;
        }
        if results[ri].winner_confident {
            eprintln!("[HEARTS-DBG]   → hearts say {} but HP says {}; keeping HP, marking uncertain", hw, results[ri].winner);
            results[ri].winner_confident = false;
            continue;
        }
        eprintln!("[HEARTS-DBG]   → tie-break: uncertain round {} winner {} → {} (hearts), stays uncertain",
            ri, results[ri].winner, hw);
        results[ri].winner = hw.to_string();
        let tmp = results[ri].winner_final_hp;
        results[ri].winner_final_hp = results[ri].loser_final_hp;
        results[ri].loser_final_hp = tmp;
    }

    // A round cut off by the end of the video has no visible ending: the
    // loser never reaches 0, or only in the last moments (the KO and slash
    // take a few seconds), so the winner is a guess. Ask the user.
    const UNFINISHED_MS: f64 = 2000.0;
    let video_end_ms = ts[n - 1];
    for r in results.iter_mut() {
        if !r.winner_confident || video_end_ms - r.round_end_ms >= UNFINISHED_MS {
            continue;
        }
        let loser = if r.winner == "P1" { &p2_norm } else { &p1_norm };
        let ko_ms = (0..n)
            .filter(|&j| ts[j] >= r.round_start_ms && ts[j] <= r.round_end_ms)
            .find(|&j| !loser[j].is_nan() && loser[j] <= 0.01)
            .map(|j| ts[j]);
        if ko_ms.map_or(true, |t| video_end_ms - t < UNFINISHED_MS) {
            eprintln!("[ROUND-DBG] {} round {} cut off by the end of the video (KO at {:?}); marking uncertain",
                replay_id, r.round_index, ko_ms.map(|t| t / 1000.0));
            r.winner_confident = false;
        }
    }

    Ok(results)
}

fn compute_match_stats(
    replay: &Replay,
    rounds: &[RoundResult],
    damage_events: &[DamageEvent],
) -> MatchStats {
    let p1_wins = rounds.iter().filter(|r| r.winner == "P1").count();
    let p2_wins = rounds.iter().filter(|r| r.winner == "P2").count();

    let mut p1_dmg_taken = 0.0;
    let mut p2_dmg_taken = 0.0;
    let mut p1_biggest = 0.0f64;
    let mut p2_biggest = 0.0f64;

    for e in damage_events {
        if e.target_side == 1 {
            p1_dmg_taken += e.damage_pct;
            p1_biggest = p1_biggest.max(e.damage_pct);
        } else {
            p2_dmg_taken += e.damage_pct;
            p2_biggest = p2_biggest.max(e.damage_pct);
        }
    }

    let durations: Vec<f64> = rounds
        .iter()
        .map(|r| (r.round_end_ms - r.round_start_ms) / 1000.0)
        .collect();
    let avg_dur = if durations.is_empty() {
        0.0
    } else {
        durations.iter().sum::<f64>() / durations.len() as f64
    };
    let longest = durations.iter().cloned().fold(0.0f64, f64::max);
    let shortest = durations
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let shortest = if shortest.is_infinite() {
        0.0
    } else {
        shortest
    };

    let comeback_count = rounds.iter().filter(|r| r.is_comeback).count();
    let close_rounds = rounds.iter().filter(|r| r.winner_final_hp < 0.20).count();
    let avg_winner_hp = if rounds.is_empty() {
        0.0
    } else {
        rounds.iter().map(|r| r.winner_final_hp).sum::<f64>() / rounds.len() as f64
    };

    MatchStats {
        replay_id: replay.replay_id.clone(),
        total_rounds: rounds.len(),
        p1_round_wins: p1_wins,
        p2_round_wins: p2_wins,
        total_damage_events: damage_events.len(),
        p1_damage_taken: p1_dmg_taken,
        p2_damage_taken: p2_dmg_taken,
        p1_biggest_hit: p1_biggest,
        p2_biggest_hit: p2_biggest,
        avg_round_duration_s: avg_dur,
        longest_round_s: longest,
        shortest_round_s: shortest,
        comeback_count,
        close_rounds,
        avg_winner_final_hp: avg_winner_hp,
        duration_s: replay.duration_ms / 1000.0,
    }
}

fn generate_highlights(
    rounds: &[RoundResult],
    damage_events: &[DamageEvent],
) -> Vec<Highlight> {
    let mut highlights: Vec<Highlight> = Vec::new();

    // Comeback highlights
    for r in rounds {
        if r.is_comeback {
            highlights.push(Highlight {
                kind: "comeback".to_string(),
                label: format!(
                    "{} comeback ({:.0}% deficit)",
                    r.winner,
                    r.max_deficit * 100.0
                ),
                timestamp_ms: r.round_start_ms,
                end_ms: r.round_end_ms,
                details: format!(
                    "{} was down {:.0}% HP but won with {:.0}% remaining. Min HP: {:.0}%",
                    r.winner,
                    r.max_deficit * 100.0,
                    r.winner_final_hp * 100.0,
                    r.winner_min_hp * 100.0
                ),
                severity: r.max_deficit,
            });
        }
    }

    // Close rounds (winner < 20% HP)
    for r in rounds {
        if r.winner_final_hp < 0.20 && !r.is_comeback {
            highlights.push(Highlight {
                kind: "close_round".to_string(),
                label: format!(
                    "Close round - {} wins at {:.0}% HP",
                    r.winner,
                    r.winner_final_hp * 100.0
                ),
                timestamp_ms: r.round_start_ms,
                end_ms: r.round_end_ms,
                details: format!(
                    "{} barely won with only {:.0}% HP remaining",
                    r.winner,
                    r.winner_final_hp * 100.0
                ),
                severity: 1.0 - r.winner_final_hp,
            });
        }
    }

    // Big damage events (> 30%)
    for e in damage_events {
        if e.damage_pct > 0.30 {
            let target = if e.target_side == 1 { "P1" } else { "P2" };
            highlights.push(Highlight {
                kind: "big_damage".to_string(),
                label: format!("{:.0}% damage to {}", e.damage_pct * 100.0, target),
                timestamp_ms: e.timestamp_ms,
                end_ms: e.timestamp_ms + 2000.0,
                details: format!(
                    "{} took {:.0}% damage ({:.0}% -> {:.0}%)",
                    target,
                    e.damage_pct * 100.0,
                    e.pre_health_pct * 100.0,
                    e.post_health_pct * 100.0
                ),
                severity: e.damage_pct,
            });
        }
    }

    // Perfect rounds (winner ends > 90%)
    for r in rounds {
        if r.winner_final_hp > 0.90 {
            highlights.push(Highlight {
                kind: "perfect".to_string(),
                label: format!("Near-perfect by {}", r.winner),
                timestamp_ms: r.round_start_ms,
                end_ms: r.round_end_ms,
                details: format!(
                    "{} won with {:.0}% HP remaining (min: {:.0}%)",
                    r.winner,
                    r.winner_final_hp * 100.0,
                    r.winner_min_hp * 100.0
                ),
                severity: r.winner_final_hp,
            });
        }
    }

    highlights.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    highlights
}

// ── Tauri commands ───────────────────────────────────────────────────────────

#[tauri::command]
fn get_replays(db_path: String) -> Result<Vec<Replay>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT replay_id, video_path, duration_ms, frame_count FROM replays ORDER BY replay_id")
        .map_err(|e| e.to_string())?;

    let replays: Vec<Replay> = stmt
        .query_map([], |row| {
            Ok(Replay {
                replay_id: row.get(0)?,
                video_path: row.get(1)?,
                duration_ms: row.get(2)?,
                frame_count: row.get(3)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(replays)
}

#[tauri::command]
fn get_frame_data(db_path: String, replay_id: String) -> Result<Vec<FrameDataPoint>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Check if tension columns exist
    let has_tension = conn
        .prepare("SELECT p1_tension_pct FROM frame_data LIMIT 1")
        .is_ok();

    let points: Vec<FrameDataPoint> = if has_tension {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct, p1_tension_pct, p2_tension_pct
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt.query_map([&replay_id], |row| {
                Ok(FrameDataPoint {
                    timestamp_ms: row.get(0)?,
                    p1_health_pct: row.get(1)?,
                    p2_health_pct: row.get(2)?,
                    p1_tension_pct: row.get(3)?,
                    p2_tension_pct: row.get(4)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.filter_map(|r| r.ok()).collect()
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt.query_map([&replay_id], |row| {
                Ok(FrameDataPoint {
                    timestamp_ms: row.get(0)?,
                    p1_health_pct: row.get(1)?,
                    p2_health_pct: row.get(2)?,
                    p1_tension_pct: None,
                    p2_tension_pct: None,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.filter_map(|r| r.ok()).collect()
    };

    Ok(points)
}

#[tauri::command]
fn get_damage_events(db_path: String, replay_id: String) -> Result<Vec<DamageEvent>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT event_id, replay_id, timestamp_ms, frame_start, frame_end,
                    target_side, damage_pct, pre_health_pct, post_health_pct
             FROM damage_events
             WHERE replay_id = ?
             ORDER BY timestamp_ms",
        )
        .map_err(|e| e.to_string())?;

    let events: Vec<DamageEvent> = stmt
        .query_map([&replay_id], |row| {
            Ok(DamageEvent {
                event_id: row.get(0)?,
                replay_id: row.get(1)?,
                timestamp_ms: row.get(2)?,
                frame_start: row.get(3)?,
                frame_end: row.get(4)?,
                target_side: row.get(5)?,
                damage_pct: row.get(6)?,
                pre_health_pct: row.get(7)?,
                post_health_pct: row.get(8)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(events)
}

#[tauri::command]
fn get_rounds(db_path: String, replay_id: String) -> Result<Vec<RoundResult>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let mut rounds = detect_rounds_for_replay(&conn, &replay_id)?;

    // Apply winner overrides from the database
    ensure_winner_overrides_table(&conn)?;
    let mut stmt = conn.prepare(
        "SELECT round_index, winner FROM winner_overrides WHERE replay_id = ?"
    ).map_err(|e| e.to_string())?;
    let overrides: Vec<(usize, String)> = stmt.query_map([&replay_id], |row| {
        Ok((row.get::<_, i64>(0)? as usize, row.get::<_, String>(1)?))
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();

    for (idx, new_winner) in overrides {
        if let Some(round) = rounds.iter_mut().find(|r| r.round_index == idx) {
            if round.winner != new_winner {
                std::mem::swap(&mut round.winner_final_hp, &mut round.loser_final_hp);
                round.winner = new_winner;
            }
            round.winner_confident = true;
        }
    }

    Ok(rounds)
}

#[tauri::command]
fn get_match_stats(db_path: String, replay_id: String) -> Result<MatchStats, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Get replay info
    let replay: Replay = conn
        .query_row(
            "SELECT replay_id, video_path, duration_ms, frame_count FROM replays WHERE replay_id = ?",
            [&replay_id],
            |row| {
                Ok(Replay {
                    replay_id: row.get(0)?,
                    video_path: row.get(1)?,
                    duration_ms: row.get(2)?,
                    frame_count: row.get(3)?,
                })
            },
        )
        .map_err(|e| e.to_string())?;

    let rounds = detect_rounds_for_replay(&conn, &replay_id)?;

    // Get damage events
    let mut stmt = conn
        .prepare(
            "SELECT event_id, replay_id, timestamp_ms, frame_start, frame_end,
                    target_side, damage_pct, pre_health_pct, post_health_pct
             FROM damage_events WHERE replay_id = ? ORDER BY timestamp_ms",
        )
        .map_err(|e| e.to_string())?;

    let events: Vec<DamageEvent> = stmt
        .query_map([&replay_id], |row| {
            Ok(DamageEvent {
                event_id: row.get(0)?,
                replay_id: row.get(1)?,
                timestamp_ms: row.get(2)?,
                frame_start: row.get(3)?,
                frame_end: row.get(4)?,
                target_side: row.get(5)?,
                damage_pct: row.get(6)?,
                pre_health_pct: row.get(7)?,
                post_health_pct: row.get(8)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(compute_match_stats(&replay, &rounds, &events))
}

#[tauri::command]
fn get_highlights(db_path: String, replay_id: String) -> Result<Vec<Highlight>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let rounds = detect_rounds_for_replay(&conn, &replay_id)?;

    let mut stmt = conn
        .prepare(
            "SELECT event_id, replay_id, timestamp_ms, frame_start, frame_end,
                    target_side, damage_pct, pre_health_pct, post_health_pct
             FROM damage_events WHERE replay_id = ? ORDER BY timestamp_ms",
        )
        .map_err(|e| e.to_string())?;

    let events: Vec<DamageEvent> = stmt
        .query_map([&replay_id], |row| {
            Ok(DamageEvent {
                event_id: row.get(0)?,
                replay_id: row.get(1)?,
                timestamp_ms: row.get(2)?,
                frame_start: row.get(3)?,
                frame_end: row.get(4)?,
                target_side: row.get(5)?,
                damage_pct: row.get(6)?,
                pre_health_pct: row.get(7)?,
                post_health_pct: row.get(8)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(generate_highlights(&rounds, &events))
}

/// Find project root: try CARGO_MANIFEST_DIR (dev), then walk up from exe.
/// Find the root directory containing scripts/analyze_replay.py.
///
/// Search order:
/// 1. CARGO_MANIFEST_DIR parent (dev builds)
/// 2. Walk up from exe (dev / portable)
/// 3. Resource directory next to exe (installed via NSIS)
fn find_project_root() -> Result<PathBuf, String> {
    let marker = |d: &PathBuf| d.join("scripts").join("analyze_replay.py").exists();

    // 1. Dev: walk up from Cargo manifest dir (app/src-tauri -> app -> repo root)
    let mut manifest_dir: Option<PathBuf> = Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    while let Some(d) = manifest_dir {
        if marker(&d) {
            return Ok(d);
        }
        manifest_dir = d.parent().map(|p| p.to_path_buf());
    }

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let exe_dir = exe.parent().map(|p| p.to_path_buf())
        .ok_or_else(|| "Cannot determine exe directory".to_string())?;

    // 2. Walk up from exe (dev / portable)
    let mut dir = Some(exe_dir.clone());
    while let Some(d) = dir {
        if marker(&d) {
            return Ok(d);
        }
        dir = d.parent().map(|p| p.to_path_buf());
    }

    // 3. Installed: resources are bundled next to the exe in a resources/ subdir,
    //    or directly next to the exe (NSIS places them in the install dir).
    //    Tauri NSIS puts resources at <install_dir>/resources/
    //    On macOS they are in the app bundle's Contents/Resources.
    for resource_dir in exe_resource_dirs() {
        if marker(&resource_dir) {
            return Ok(resource_dir);
        }
    }

    // 4. Also check exe_dir itself (some bundle layouts)
    if marker(&exe_dir) {
        return Ok(exe_dir);
    }

    Err(format!(
        "Cannot find project root with scripts/analyze_replay.py. Searched near: {}",
        exe_dir.display()
    ))
}

/// Return the default output directory and DB path.
///
/// For installed apps, output goes to AppData/Local/SpaghettiLab/output.
/// For dev, output goes to <project_root>/output.
fn default_output_paths() -> Result<(PathBuf, PathBuf), String> {
    // Try dev location first
    if let Ok(root) = find_project_root() {
        // Check if this looks like a dev environment (has pyproject.toml or .git)
        if root.join("pyproject.toml").exists() || root.join(".git").exists() {
            let output_dir = root.join("output");
            let db_path = output_dir.join("analysis.db");
            return Ok((output_dir, db_path));
        }
    }

    // Installed: use AppData/Local
    let app_data = dirs::data_local_dir()
        .unwrap_or_else(|| dirs_fallback());
    let output_dir = app_data.join("SpaghettiLab").join("output");
    let db_path = output_dir.join("analysis.db");
    Ok((output_dir, db_path))
}

fn dirs_fallback() -> PathBuf {
    // Fallback: put data next to exe
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

#[tauri::command]
fn get_default_db_path() -> Result<String, String> {
    let (_, db_path) = default_output_paths()?;
    Ok(db_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn analyze_video(
    video_path: String,
    _sample_every: Option<u32>,
) -> Result<String, String> {
    let project_root = find_project_root()?;
    let (output_dir, db_path) = default_output_paths()?;

    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

    let script = project_root.join("scripts").join("analyze_replay.py");
    let config = project_root.join("config").join("default.yaml");

    let mut cmd = require_python()?.command();
    // Set PYTHONPATH so `import replanal` works from installed location
    cmd.env("PYTHONPATH", project_root.to_str().unwrap())
        .arg(script.to_str().unwrap())
        .arg(&video_path)
        .arg("--config")
        .arg(config.to_str().unwrap())
        .arg("--output")
        .arg(output_dir.to_str().unwrap());

    let output = cmd.output().map_err(|e| format!("Failed to run Python: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        return Err(format!("Analysis failed:\n{}\n{}", stdout, stderr));
    }

    Ok(db_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn reanalyze_replay(
    app_handle: tauri::AppHandle,
    db_path: String,
    replay_id: String,
) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let video_path = resolve_video_for_replay(&conn, &replay_id)?;
    drop(conn);

    let project_root = find_project_root()?;
    let (output_dir, _) = default_output_paths()?;

    let script = project_root.join("scripts").join("analyze_replay.py");
    let config = project_root.join("config").join("default.yaml");

    let mut cmd = require_python()?.command();
    cmd.env("PYTHONPATH", project_root.to_str().unwrap())
        .arg(script.to_str().unwrap())
        .arg(video_path.to_str().unwrap())
        .arg("--config")
        .arg(config.to_str().unwrap())
        .arg("--output")
        .arg(output_dir.to_str().unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to run Python: {}", e))?;
    let stdout = child.stdout.take().unwrap();

    for line in std::io::BufReader::new(stdout).lines() {
        if let Ok(line) = line {
            // Forward lines that contain frame progress
            if line.contains("Processed") || line.contains("Analyzing") || line.contains("Detected") {
                let _ = app_handle.emit("reanalyze-progress", &line);
            }
        }
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("Reanalysis failed. Check console for errors.".into());
    }

    Ok(())
}

#[tauri::command]
async fn export_clip(
    video_path: String,
    start_ms: f64,
    end_ms: f64,
    output_path: String,
) -> Result<(), String> {
    let start_s = start_ms / 1000.0;
    let duration_s = (end_ms - start_ms) / 1000.0;

    let mut ffmpeg_cmd = Command::new(ffmpeg_bin()?);
    hide_window(&mut ffmpeg_cmd);
    let output = ffmpeg_cmd
        .args([
            "-y",
            "-ss",
            &format!("{:.3}", start_s),
            "-i",
            &video_path,
            "-t",
            &format!("{:.3}", duration_s),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            &output_path,
        ])
        .output()
        .map_err(|e| format!("Failed to run ffmpeg: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg error: {}", stderr));
    }
    Ok(())
}

#[tauri::command]
fn resolve_video_path(db_path: String, replay_id: String) -> Result<String, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let path: String = conn
        .query_row(
            "SELECT video_path FROM replays WHERE replay_id = ?",
            [&replay_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;

    let p = PathBuf::from(&path);

    // If already absolute and exists, return as-is
    if p.is_absolute() && p.exists() {
        return Ok(path);
    }

    // Try resolving relative to project root
    if let Ok(root) = find_project_root() {
        let resolved = root.join(&p);
        if resolved.exists() {
            return Ok(resolved.to_string_lossy().to_string());
        }
    }

    // Return the raw path — frontend will handle "not found"
    Ok(path)
}

// ── Winner overrides ─────────────────────────────────────────────────────────

fn ensure_winner_overrides_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS winner_overrides (
            replay_id   TEXT NOT NULL,
            round_index INTEGER NOT NULL,
            winner      TEXT NOT NULL,
            PRIMARY KEY (replay_id, round_index)
        );"
    ).map_err(|e| e.to_string())
}

#[tauri::command]
fn set_round_winner(db_path: String, replay_id: String, round_index: usize, winner: String) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_winner_overrides_table(&conn)?;
    conn.execute(
        "INSERT OR REPLACE INTO winner_overrides (replay_id, round_index, winner) VALUES (?, ?, ?)",
        rusqlite::params![&replay_id, round_index as i64, &winner],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

// ── Notes ────────────────────────────────────────────────────────────────────

fn ensure_notes_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS notes (
            note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id  TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            text       TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );"
    ).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_notes(db_path: String, replay_id: String) -> Result<Vec<Note>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    let mut stmt = conn.prepare(
        "SELECT note_id, replay_id, timestamp_ms, text, created_at
         FROM notes WHERE replay_id = ? ORDER BY timestamp_ms"
    ).map_err(|e| e.to_string())?;
    let notes = stmt.query_map([&replay_id], |row| {
        Ok(Note {
            note_id: row.get(0)?,
            replay_id: row.get(1)?,
            timestamp_ms: row.get(2)?,
            text: row.get(3)?,
            created_at: row.get(4)?,
        })
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();
    Ok(notes)
}

#[tauri::command]
fn add_note(db_path: String, replay_id: String, timestamp_ms: f64, text: String) -> Result<Note, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    conn.execute(
        "INSERT INTO notes (replay_id, timestamp_ms, text) VALUES (?, ?, ?)",
        rusqlite::params![&replay_id, timestamp_ms, &text],
    ).map_err(|e| e.to_string())?;
    let note_id = conn.last_insert_rowid();
    let note = conn.query_row(
        "SELECT note_id, replay_id, timestamp_ms, text, created_at FROM notes WHERE note_id = ?",
        [note_id],
        |row| Ok(Note {
            note_id: row.get(0)?,
            replay_id: row.get(1)?,
            timestamp_ms: row.get(2)?,
            text: row.get(3)?,
            created_at: row.get(4)?,
        })
    ).map_err(|e| e.to_string())?;
    Ok(note)
}

#[tauri::command]
fn update_note(db_path: String, note_id: i64, text: String, timestamp_ms: Option<f64>) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    match timestamp_ms {
        Some(ts) => conn.execute(
            "UPDATE notes SET text = ?, timestamp_ms = ? WHERE note_id = ?",
            rusqlite::params![&text, ts, note_id],
        ),
        None => conn.execute(
            "UPDATE notes SET text = ? WHERE note_id = ?",
            rusqlite::params![&text, note_id],
        ),
    }.map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn delete_note(db_path: String, note_id: i64) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    conn.execute("DELETE FROM notes WHERE note_id = ?", [note_id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ── Drawings ─────────────────────────────────────────────────────────────────

fn ensure_drawings_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS drawings (
            drawing_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id    TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            strokes_json TEXT NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(replay_id, timestamp_ms)
        );"
    ).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_drawings(db_path: String, replay_id: String) -> Result<Vec<Drawing>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_drawings_table(&conn)?;
    let mut stmt = conn.prepare(
        "SELECT drawing_id, replay_id, timestamp_ms, strokes_json, created_at
         FROM drawings WHERE replay_id = ? ORDER BY timestamp_ms"
    ).map_err(|e| e.to_string())?;
    let drawings = stmt.query_map([&replay_id], |row| {
        Ok(Drawing {
            drawing_id: row.get(0)?,
            replay_id: row.get(1)?,
            timestamp_ms: row.get(2)?,
            strokes_json: row.get(3)?,
            created_at: row.get(4)?,
        })
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();
    Ok(drawings)
}

#[tauri::command]
fn save_drawing(db_path: String, replay_id: String, timestamp_ms: f64, strokes_json: String) -> Result<Option<Drawing>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_drawings_table(&conn)?;

    // If strokes are empty, delete the drawing at this timestamp
    if strokes_json == "[]" {
        conn.execute(
            "DELETE FROM drawings WHERE replay_id = ? AND timestamp_ms = ?",
            rusqlite::params![&replay_id, timestamp_ms],
        ).map_err(|e| e.to_string())?;
        return Ok(None);
    }

    // Upsert: INSERT OR REPLACE on the unique(replay_id, timestamp_ms) constraint
    conn.execute(
        "INSERT INTO drawings (replay_id, timestamp_ms, strokes_json)
         VALUES (?, ?, ?)
         ON CONFLICT(replay_id, timestamp_ms)
         DO UPDATE SET strokes_json = excluded.strokes_json",
        rusqlite::params![&replay_id, timestamp_ms, &strokes_json],
    ).map_err(|e| e.to_string())?;

    let drawing = conn.query_row(
        "SELECT drawing_id, replay_id, timestamp_ms, strokes_json, created_at
         FROM drawings WHERE replay_id = ? AND timestamp_ms = ?",
        rusqlite::params![&replay_id, timestamp_ms],
        |row| Ok(Drawing {
            drawing_id: row.get(0)?,
            replay_id: row.get(1)?,
            timestamp_ms: row.get(2)?,
            strokes_json: row.get(3)?,
            created_at: row.get(4)?,
        })
    ).map_err(|e| e.to_string())?;
    Ok(Some(drawing))
}

#[tauri::command]
fn delete_drawing(db_path: String, drawing_id: i64) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_drawings_table(&conn)?;
    conn.execute("DELETE FROM drawings WHERE drawing_id = ?", [drawing_id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ── Questionnaire answers ────────────────────────────────────────────────────
// Free-form review checklist attached to each replay. Keys are namespaced by
// template id (e.g. "ggst.roundstart.main_options"). Reserved meta keys are
// prefixed with an underscore (e.g. "_meta.selected_template").

fn ensure_questionnaire_answers_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS questionnaire_answers (
            replay_id    TEXT NOT NULL,
            question_key TEXT NOT NULL,
            answer_text  TEXT,
            updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (replay_id, question_key)
        );"
    ).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_questionnaire_answers(db_path: String, replay_id: String) -> Result<std::collections::HashMap<String, String>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_schema_version(&conn)?;
    ensure_questionnaire_answers_table(&conn)?;
    let mut stmt = conn.prepare(
        "SELECT question_key, COALESCE(answer_text, '') FROM questionnaire_answers WHERE replay_id = ?"
    ).map_err(|e| e.to_string())?;
    let rows = stmt.query_map([&replay_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }).map_err(|e| e.to_string())?;
    let mut map = std::collections::HashMap::new();
    for r in rows {
        if let Ok((k, v)) = r { map.insert(k, v); }
    }
    Ok(map)
}

#[tauri::command]
fn set_questionnaire_answer(db_path: String, replay_id: String, question_key: String, answer_text: String) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_schema_version(&conn)?;
    ensure_questionnaire_answers_table(&conn)?;
    if answer_text.trim().is_empty() {
        conn.execute(
            "DELETE FROM questionnaire_answers WHERE replay_id = ? AND question_key = ?",
            rusqlite::params![&replay_id, &question_key],
        ).map_err(|e| e.to_string())?;
    } else {
        conn.execute(
            "INSERT INTO questionnaire_answers (replay_id, question_key, answer_text, updated_at)
             VALUES (?, ?, ?, datetime('now'))
             ON CONFLICT(replay_id, question_key)
             DO UPDATE SET answer_text = excluded.answer_text, updated_at = excluded.updated_at",
            rusqlite::params![&replay_id, &question_key, &answer_text],
        ).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── Schema versioning ────────────────────────────────────────────────────────
// `PRAGMA user_version` lives in the SQLite header. We use it to tell whether
// a .spag/.spagz file predates a non-additive schema change and migrate it
// in place on open. The web app and the desktop app must agree on the
// version number so files round-trip cleanly between them.
//
// Version history:
//   0 → 1 : questionnaire_answers keys gained a template-id prefix
//           ("roundstart.main_options" → "ggst.roundstart.main_options").

const SCHEMA_VERSION: u32 = 1;

fn ensure_schema_version(conn: &Connection) -> Result<(), String> {
    ensure_questionnaire_answers_table(conn)?;
    let current: u32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
        .map(|n| n as u32)
        .unwrap_or(0);

    if current < SCHEMA_VERSION {
        // v0 → v1: prefix legacy unprefixed questionnaire keys with "ggst.".
        // Meta keys (leading underscore) and already-prefixed keys are untouched.
        if current < 1 {
            let _ = conn.execute(
                "UPDATE questionnaire_answers
                 SET question_key = 'ggst.' || question_key
                 WHERE question_key NOT LIKE '_%'
                   AND instr(question_key, '.') > 0
                   AND question_key NOT LIKE 'ggst.%'",
                [],
            );
        }
        conn.execute_batch(&format!("PRAGMA user_version = {}", SCHEMA_VERSION))
            .map_err(|e| e.to_string())?;
    } else if current > SCHEMA_VERSION {
        eprintln!(
            "[spaghetti-lab] WARNING: opened a file with schema v{} but this build only understands up to v{}. \
             Newer fields may not be preserved.",
            current, SCHEMA_VERSION
        );
    }
    Ok(())
}

// ── VOD Splitter ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RoiRect {
    pub y1: u32,
    pub y2: u32,
    pub x1: u32,
    pub x2: u32,
}

impl RoiRect {
    fn to_arg(&self) -> String {
        format!("{},{},{},{}", self.y1, self.y2, self.x1, self.x2)
    }
}

#[derive(Debug, Deserialize)]
pub struct RoiQuad {
    pub tl: [u32; 2],  // top-left [x, y]
    pub tr: [u32; 2],  // top-right
    pub br: [u32; 2],  // bottom-right
    pub bl: [u32; 2],  // bottom-left
}

impl RoiQuad {
    fn to_arg(&self) -> String {
        format!("{},{},{},{},{},{},{},{}",
            self.tl[0], self.tl[1], self.tr[0], self.tr[1],
            self.br[0], self.br[1], self.bl[0], self.bl[1])
    }
}

#[derive(Debug, Deserialize)]
pub struct VodRoiConfig {
    pub p1_tension: RoiRect,
    pub p2_tension: RoiRect,
    pub timer: RoiRect,
    pub p1_name: RoiQuad,
    pub p2_name: RoiQuad,
}

#[derive(Debug, Serialize, Clone)]
pub struct DetectedSetInfo {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
    pub gameplay_duration_secs: f64,
    pub game_count: u32,
    pub p1_name: String,
    pub p2_name: String,
}

#[tauri::command]
async fn download_vod(
    app_handle: tauri::AppHandle,
    url: String,
    output_dir: String,
) -> Result<String, String> {
    // Ensure yt-dlp is installed (auto-install if missing)
    let python = python_cmd();
    let check = python
        .command()
        .args(["-m", "yt_dlp", "--version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let needs_install = match check {
        Ok(s) => !s.success(),
        Err(_) => true,
    };

    if needs_install {
        let _ = app_handle.emit("vod-download-progress", "Installing yt-dlp...");
        let mut pip_cmd = python.command();
        let pip_status = pip_cmd
            .args(["-m", "pip", "install", "yt-dlp"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Failed to run pip: {}", e))?;

        if !pip_status.success() {
            return Err("Failed to install yt-dlp. Run manually: pip install yt-dlp".to_string());
        }
    }

    // Use python -m yt_dlp to download the VOD as mp4
    // -u forces unbuffered stdout/stderr so progress lines arrive in real time
    let mut cmd = python.command();
    cmd.arg("-u").args(["-m", "yt_dlp"])
        .arg(&url)
        .arg("-f").arg("best[ext=mp4]/best")
        .arg("--merge-output-format").arg("mp4")
        .arg("-o").arg(format!("{}/%(title)s.%(ext)s", output_dir))
        .arg("--newline")  // one progress line per update
        .arg("--no-colors")
        .arg("--print").arg("after_move:filepath")  // print final path after download
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to run yt-dlp: {}", e))?;

    let stderr = child.stderr.take().unwrap();
    let stdout = child.stdout.take().unwrap();

    // Stream stderr progress to frontend
    let handle = app_handle.clone();
    let stderr_thread = std::thread::spawn(move || {
        let reader = std::io::BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                let _ = handle.emit("vod-download-progress", &line);
            }
        }
    });

    // Capture stdout (final file path printed by --print after_move:filepath)
    let stdout_thread = std::thread::spawn(move || {
        let reader = std::io::BufReader::new(stdout);
        let mut last_line = String::new();
        for line in reader.lines() {
            if let Ok(line) = line {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    last_line = trimmed;
                }
            }
        }
        last_line
    });

    let status = child.wait().map_err(|e| format!("yt-dlp process error: {}", e))?;
    stderr_thread.join().ok();
    let final_path = stdout_thread.join().map_err(|_| "Failed to read yt-dlp output".to_string())?;

    if !status.success() {
        return Err(format!("yt-dlp exited with code {}", status.code().unwrap_or(-1)));
    }

    if final_path.is_empty() {
        // Fallback: find the most recently created mp4 in output_dir
        let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
        if let Ok(entries) = std::fs::read_dir(&output_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "mp4").unwrap_or(false) {
                    if let Ok(meta) = path.metadata() {
                        if let Ok(modified) = meta.modified() {
                            if newest.as_ref().map_or(true, |(t, _)| modified > *t) {
                                newest = Some((modified, path));
                            }
                        }
                    }
                }
            }
        }
        match newest {
            Some((_, path)) => Ok(path.to_string_lossy().to_string()),
            None => Err("Download completed but could not find output file".to_string()),
        }
    } else {
        Ok(final_path)
    }
}

#[tauri::command]
async fn extract_preview_frame(
    video_path: String,
    timestamp_secs: f64,
) -> Result<String, String> {
    let (output_dir, _) = default_output_paths()?;
    let preview_path = output_dir.join(".vod_preview.png");
    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

    let mut ffmpeg_cmd = Command::new(ffmpeg_bin()?);
    hide_window(&mut ffmpeg_cmd);
    let output = ffmpeg_cmd
        .args([
            "-y",
            "-ss", &format!("{:.3}", timestamp_secs),
            "-i", &video_path,
            "-frames:v", "1",
            "-q:v", "2",
            "-v", "quiet",
            preview_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("Failed to run ffmpeg: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg error: {}", stderr));
    }

    Ok(preview_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn scan_vod(
    app_handle: tauri::AppHandle,
    video_path: String,
    roi_config: VodRoiConfig,
) -> Result<Vec<DetectedSetInfo>, String> {
    let project_root = find_project_root()?;
    let script = project_root.join("scripts").join("split_vod.py");
    let python = require_python()?;
    let ffmpeg = ffmpeg_bin()?;

    let mut cmd = python.command();
    cmd.env("PYTHONPATH", project_root.to_str().unwrap())
        // The script resolves ffmpeg the same way; pass ours so both agree
        .env("SPAGHETTI_FFMPEG", &ffmpeg)
        .arg(script.to_str().unwrap())
        .arg(&video_path)
        .arg("--preview")
        .arg("--json-output")
        .arg("--p1-tension").arg(roi_config.p1_tension.to_arg())
        .arg("--p2-tension").arg(roi_config.p2_tension.to_arg())
        .arg("--timer-roi").arg(roi_config.timer.to_arg())
        .arg("--p1-name-quad").arg(roi_config.p1_name.to_arg())
        .arg("--p2-name-quad").arg(roi_config.p2_name.to_arg())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    eprintln!("[VOD-SCAN] Spawning: {} {} {} --preview --json-output ...", python, script.display(), &video_path);

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn python: {}", e))?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Read stderr in a separate thread to avoid deadlock
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        use std::io::Read;
        let mut reader = std::io::BufReader::new(stderr);
        let _ = reader.read_to_string(&mut buf);
        buf
    });

    let mut json_result: Option<String> = None;
    let mut all_stdout = Vec::new();

    for line in std::io::BufReader::new(stdout).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.starts_with("PROGRESS:") || line.starts_with("WARNING:") {
            let _ = app_handle.emit("vod-scan-progress", &line);
        } else if line.starts_with("JSON_RESULT:") {
            json_result = Some(line["JSON_RESULT:".len()..].to_string());
        }
        all_stdout.push(line);
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    let stderr_output = stderr_thread.join().unwrap_or_default();

    if !stderr_output.is_empty() {
        eprintln!("[VOD-SCAN] Python stderr:\n{}", stderr_output);
    }

    if !status.success() {
        let last_err = stderr_output.lines().rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("unknown error");
        return Err(format!("VOD scan failed: {}", last_err));
    }

    let json_str = json_result.ok_or("No JSON result from split_vod.py")?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let sets = parsed["sets"]
        .as_array()
        .ok_or("Invalid JSON: missing sets array")?
        .iter()
        .map(|s| DetectedSetInfo {
            index: s["index"].as_u64().unwrap_or(0) as u32,
            start_secs: s["start_secs"].as_f64().unwrap_or(0.0),
            end_secs: s["end_secs"].as_f64().unwrap_or(0.0),
            gameplay_duration_secs: s["gameplay_duration_secs"].as_f64().unwrap_or(0.0),
            game_count: s["game_count"].as_u64().unwrap_or(0) as u32,
            p1_name: s["p1_name"].as_str().unwrap_or("").to_string(),
            p2_name: s["p2_name"].as_str().unwrap_or("").to_string(),
        })
        .collect();

    Ok(sets)
}

#[derive(Debug, Deserialize)]
pub struct CutSetRequest {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
    pub p1_name: Option<String>,
    pub p2_name: Option<String>,
}

#[tauri::command]
async fn cut_vod_sets(
    app_handle: tauri::AppHandle,
    video_path: String,
    sets: Vec<CutSetRequest>,
    output_dir: String,
) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;
    let ffmpeg = ffmpeg_bin()?;
    let mut paths: Vec<String> = Vec::new();
    // Same padding as split_vod.py. A set's end is the last second with the
    // game's HUD, which fades at the match-deciding KO: 20 s keeps the KO
    // well inside the file (round detection treats a KO in the last 2 s as
    // unfinished) and the win screen after it. Stream copy starts at the
    // keyframe before `start`, so files may begin a few seconds earlier.
    let padding_before = 10.0;
    let padding_after = 20.0;

    for (i, set) in sets.iter().enumerate() {
        let start = (set.start_secs - padding_before).max(0.0);
        let duration = (set.end_secs + padding_after) - start;
        let filename = match (&set.p1_name, &set.p2_name) {
            (Some(p1), Some(p2)) if !p1.is_empty() && !p2.is_empty() => {
                let safe = |s: &str| -> String {
                    s.chars().filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect()
                };
                format!("{:02}_{}_vs_{}.mp4", set.index, safe(p1), safe(p2))
            }
            _ => format!("set_{:02}.mp4", set.index),
        };
        let out_path = PathBuf::from(&output_dir).join(&filename);

        let mut ffmpeg_cmd = Command::new(&ffmpeg);
        hide_window(&mut ffmpeg_cmd);
        let output = ffmpeg_cmd
            .args([
                "-y",
                "-ss", &format!("{:.3}", start),
                "-i", &video_path,
                "-t", &format!("{:.3}", duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-v", "warning",
                out_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("ffmpeg error: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to cut set {}: {}", set.index, stderr));
        }

        paths.push(out_path.to_string_lossy().to_string());
        let _ = app_handle.emit("vod-cut-progress", format!("{}/{}", i + 1, sets.len()));
    }

    Ok(paths)
}

// ── .spag file format ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct SpagSession {
    pub db_path: String,
    pub video_path: String,
    pub spag_path: String,
    pub replay_id: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct SpagzSession {
    pub db_path: String,
    pub video_hint: String,
    pub video_url: String,
    pub spagz_path: String,
    pub replay_id: String,
}

/// Resolve the video path for a replay (reusable helper).
fn resolve_video_for_replay(conn: &Connection, replay_id: &str) -> Result<PathBuf, String> {
    let raw_path: String = conn
        .query_row(
            "SELECT video_path FROM replays WHERE replay_id = ?",
            [replay_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;

    let p = PathBuf::from(&raw_path);
    if p.is_absolute() && p.exists() {
        return Ok(p);
    }
    if let Ok(root) = find_project_root() {
        let resolved = root.join(&p);
        if resolved.exists() {
            return Ok(resolved);
        }
    }
    Err(format!("Video file not found: {}", raw_path))
}

#[tauri::command]
async fn export_spag(
    db_path: String,
    replay_id: String,
    output_path: String,
) -> Result<(), String> {
    let src_conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Resolve the video file
    let video_path = resolve_video_for_replay(&src_conn, &replay_id)?;

    // Create temp DB with just this replay's data using ATTACH DATABASE
    let tmp_db = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    let tmp_db_path = tmp_db.path().to_path_buf();
    {
        // Use the source connection and attach the destination
        let tmp_str = tmp_db_path.to_string_lossy().replace('\\', "/");
        src_conn.execute(&format!("ATTACH DATABASE '{}' AS dst", tmp_str), [])
            .map_err(|e| e.to_string())?;

        // Create tables in destination
        src_conn.execute_batch(
            "CREATE TABLE dst.replays (
                replay_id TEXT PRIMARY KEY,
                video_path TEXT,
                duration_ms REAL,
                frame_count INTEGER
            );
            CREATE TABLE dst.frame_data (
                replay_id TEXT,
                frame_number INTEGER,
                timestamp_ms REAL,
                p1_health_pct REAL,
                p2_health_pct REAL,
                timer_value INTEGER,
                p1_rounds_won INTEGER,
                p2_rounds_won INTEGER,
                p1_tension_pct REAL,
                p2_tension_pct REAL
            );
            CREATE TABLE dst.damage_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                replay_id TEXT,
                timestamp_ms REAL,
                frame_start INTEGER,
                frame_end INTEGER,
                target_side INTEGER,
                damage_pct REAL,
                pre_health_pct REAL,
                post_health_pct REAL
            );
            CREATE TABLE dst.notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                replay_id TEXT NOT NULL,
                timestamp_ms REAL NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE dst.drawings (
                drawing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                replay_id TEXT NOT NULL,
                timestamp_ms REAL NOT NULL,
                strokes_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(replay_id, timestamp_ms)
            );
            CREATE TABLE dst.questionnaire_answers (
                replay_id    TEXT NOT NULL,
                question_key TEXT NOT NULL,
                answer_text  TEXT,
                updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (replay_id, question_key)
            );"
        ).map_err(|e| e.to_string())?;

        // Copy data with video_path rewritten
        src_conn.execute(
            "INSERT INTO dst.replays SELECT replay_id, 'video.mp4', duration_ms, frame_count FROM main.replays WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        src_conn.execute(
            "INSERT INTO dst.frame_data SELECT * FROM main.frame_data WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        src_conn.execute(
            "INSERT INTO dst.damage_events SELECT * FROM main.damage_events WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        ensure_notes_table(&src_conn)?;
        src_conn.execute(
            "INSERT INTO dst.notes SELECT * FROM main.notes WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        ensure_drawings_table(&src_conn)?;
        src_conn.execute(
            "INSERT INTO dst.drawings SELECT * FROM main.drawings WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        // Copy questionnaire_answers — keeps review-checklist data intact
        ensure_questionnaire_answers_table(&src_conn)?;
        src_conn.execute(
            "INSERT INTO dst.questionnaire_answers SELECT * FROM main.questionnaire_answers WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        // Copy winner_overrides
        src_conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS dst.winner_overrides (
                replay_id   TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                winner      TEXT NOT NULL,
                PRIMARY KEY (replay_id, round_index)
            );"
        ).map_err(|e| e.to_string())?;
        ensure_winner_overrides_table(&src_conn)?;
        src_conn.execute(
            "INSERT INTO dst.winner_overrides SELECT * FROM main.winner_overrides WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        // Stamp destination with current schema version
        src_conn.execute_batch(&format!("PRAGMA dst.user_version = {}", SCHEMA_VERSION))
            .map_err(|e| e.to_string())?;

        src_conn.execute("DETACH DATABASE dst", []).map_err(|e| e.to_string())?;
    }

    // Build the ZIP
    let out_file = std::fs::File::create(&output_path)
        .map_err(|e| format!("Cannot create {}: {}", output_path, e))?;
    let mut zip = zip::ZipWriter::new(out_file);

    // Add replay.db (compressed)
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("replay.db", options).map_err(|e| e.to_string())?;
    let db_bytes = std::fs::read(&tmp_db_path).map_err(|e| e.to_string())?;
    zip.write_all(&db_bytes).map_err(|e| e.to_string())?;

    // Add video.mp4 (stored, no re-compression)
    let stored = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
    zip.start_file("video.mp4", stored).map_err(|e| e.to_string())?;
    let mut vf = std::fs::File::open(&video_path)
        .map_err(|e| format!("Cannot open video: {}", e))?;
    let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8MB buffer
    loop {
        let n = vf.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        zip.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn open_spag(spag_path: String) -> Result<SpagSession, String> {
    let spag = PathBuf::from(&spag_path);
    if !spag.exists() {
        return Err(format!("File not found: {}", spag_path));
    }

    // Deterministic extraction dir based on file path
    let app_data = dirs::data_local_dir()
        .unwrap_or_else(|| dirs_fallback());
    let hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        spag_path.hash(&mut h);
        h.finish()
    };
    let session_dir = app_data.join("SpaghettiLab").join("spag_sessions").join(format!("{:016x}", hash));
    std::fs::create_dir_all(&session_dir).map_err(|e| e.to_string())?;

    let db_dest = session_dir.join("replay.db");
    let video_dest = session_dir.join("video.mp4");

    // Extract (always re-extract to get fresh data)
    let file = std::fs::File::open(&spag).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;

    // Extract replay.db
    {
        let mut entry = archive.by_name("replay.db").map_err(|e| e.to_string())?;
        let mut out = std::fs::File::create(&db_dest).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
    }

    // Extract video.mp4
    {
        let mut entry = archive.by_name("video.mp4").map_err(|e| e.to_string())?;
        let mut out = std::fs::File::create(&video_dest).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
    }

    // Get replay_id from extracted DB, running any pending schema migrations
    let conn = Connection::open(&db_dest).map_err(|e| e.to_string())?;
    ensure_schema_version(&conn)?;
    let replay_id: String = conn
        .query_row("SELECT replay_id FROM replays LIMIT 1", [], |row| row.get(0))
        .map_err(|e| e.to_string())?;

    Ok(SpagSession {
        db_path: db_dest.to_string_lossy().to_string(),
        video_path: video_dest.to_string_lossy().to_string(),
        spag_path,
        replay_id,
    })
}

#[tauri::command]
async fn save_spag(spag_path: String, db_path: String) -> Result<(), String> {
    let spag = PathBuf::from(&spag_path);

    // Read the original .spag to get the video
    let orig_file = std::fs::File::open(&spag).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(orig_file).map_err(|e| e.to_string())?;

    // Write to a temp file next to the original, then rename
    let tmp_path = spag.with_extension("spag.tmp");
    {
        let out_file = std::fs::File::create(&tmp_path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipWriter::new(out_file);

        // Write updated DB
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("replay.db", options).map_err(|e| e.to_string())?;
        let db_bytes = std::fs::read(&db_path).map_err(|e| e.to_string())?;
        zip.write_all(&db_bytes).map_err(|e| e.to_string())?;

        // Copy video from original archive
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file("video.mp4", stored).map_err(|e| e.to_string())?;
        let mut video_entry = archive.by_name("video.mp4").map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        loop {
            let n = video_entry.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 { break; }
            zip.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        }

        zip.finish().map_err(|e| e.to_string())?;
    }

    // Atomic rename
    std::fs::rename(&tmp_path, &spag).map_err(|e| e.to_string())?;
    Ok(())
}

// ── .spagz file format (lightweight, no video) ──────────────────────────────

/// Helper: copy a single replay's data (all tables) into an attached "dst" database.
/// Caller must ATTACH the destination as "dst" beforehand and DETACH after.
fn copy_replay_tables_to_dst(src_conn: &Connection, replay_id: &str, rewrite_video: Option<&str>) -> Result<(), String> {
    // Ensure video_url column exists in source (older DBs may lack it)
    let _ = src_conn.execute("ALTER TABLE replays ADD COLUMN video_url TEXT", []);

    src_conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS dst.replays (
            replay_id TEXT PRIMARY KEY,
            video_path TEXT,
            video_url TEXT,
            duration_ms REAL,
            frame_count INTEGER
        );
        CREATE TABLE IF NOT EXISTS dst.frame_data (
            replay_id TEXT,
            frame_number INTEGER,
            timestamp_ms REAL,
            p1_health_pct REAL,
            p2_health_pct REAL,
            timer_value INTEGER,
            p1_rounds_won INTEGER,
            p2_rounds_won INTEGER,
            p1_tension_pct REAL,
            p2_tension_pct REAL
        );
        CREATE TABLE IF NOT EXISTS dst.damage_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id TEXT,
            timestamp_ms REAL,
            frame_start INTEGER,
            frame_end INTEGER,
            target_side INTEGER,
            damage_pct REAL,
            pre_health_pct REAL,
            post_health_pct REAL
        );
        CREATE TABLE IF NOT EXISTS dst.notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS dst.drawings (
            drawing_id INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            strokes_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(replay_id, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS dst.winner_overrides (
            replay_id   TEXT NOT NULL,
            round_index INTEGER NOT NULL,
            winner      TEXT NOT NULL,
            PRIMARY KEY (replay_id, round_index)
        );
        CREATE TABLE IF NOT EXISTS dst.questionnaire_answers (
            replay_id    TEXT NOT NULL,
            question_key TEXT NOT NULL,
            answer_text  TEXT,
            updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (replay_id, question_key)
        );"
    ).map_err(|e| e.to_string())?;

    // Copy replays row (explicit columns for video_url compatibility)
    if let Some(vp) = rewrite_video {
        src_conn.execute(
            "INSERT INTO dst.replays (replay_id, video_path, video_url, duration_ms, frame_count)
             SELECT replay_id, ?, video_url, duration_ms, frame_count FROM main.replays WHERE replay_id = ?",
            rusqlite::params![vp, replay_id],
        ).map_err(|e| e.to_string())?;
    } else {
        src_conn.execute(
            "INSERT INTO dst.replays (replay_id, video_path, video_url, duration_ms, frame_count)
             SELECT replay_id, video_path, video_url, duration_ms, frame_count FROM main.replays WHERE replay_id = ?",
            [replay_id],
        ).map_err(|e| e.to_string())?;
    }

    src_conn.execute(
        "INSERT INTO dst.frame_data SELECT * FROM main.frame_data WHERE replay_id = ?",
        [replay_id],
    ).map_err(|e| e.to_string())?;

    src_conn.execute(
        "INSERT INTO dst.damage_events SELECT * FROM main.damage_events WHERE replay_id = ?",
        [replay_id],
    ).map_err(|e| e.to_string())?;

    ensure_notes_table(src_conn)?;
    src_conn.execute(
        "INSERT INTO dst.notes SELECT * FROM main.notes WHERE replay_id = ?",
        [replay_id],
    ).map_err(|e| e.to_string())?;

    ensure_drawings_table(src_conn)?;
    src_conn.execute(
        "INSERT INTO dst.drawings SELECT * FROM main.drawings WHERE replay_id = ?",
        [replay_id],
    ).map_err(|e| e.to_string())?;

    ensure_winner_overrides_table(src_conn)?;
    src_conn.execute(
        "INSERT INTO dst.winner_overrides SELECT * FROM main.winner_overrides WHERE replay_id = ?",
        [replay_id],
    ).map_err(|e| e.to_string())?;

    ensure_questionnaire_answers_table(src_conn)?;
    src_conn.execute(
        "INSERT INTO dst.questionnaire_answers SELECT * FROM main.questionnaire_answers WHERE replay_id = ?",
        [replay_id],
    ).map_err(|e| e.to_string())?;

    // Stamp the destination with our current schema version so re-openers
    // know no migrations are needed.
    src_conn.execute_batch(&format!("PRAGMA dst.user_version = {}", SCHEMA_VERSION))
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
async fn export_spagz(
    db_path: String,
    replay_id: String,
    output_path: String,
) -> Result<(), String> {
    let src_conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Create temp DB with just this replay's data — preserve original video_path
    let tmp_db = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    let tmp_db_path = tmp_db.path().to_path_buf();
    {
        let tmp_str = tmp_db_path.to_string_lossy().replace('\\', "/");
        src_conn.execute(&format!("ATTACH DATABASE '{}' AS dst", tmp_str), [])
            .map_err(|e| e.to_string())?;

        copy_replay_tables_to_dst(&src_conn, &replay_id, None)?;

        src_conn.execute("DETACH DATABASE dst", []).map_err(|e| e.to_string())?;
    }

    // Build ZIP with only replay.db (no video)
    let out_file = std::fs::File::create(&output_path)
        .map_err(|e| format!("Cannot create {}: {}", output_path, e))?;
    let mut zip = zip::ZipWriter::new(out_file);

    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("replay.db", options).map_err(|e| e.to_string())?;
    let db_bytes = std::fs::read(&tmp_db_path).map_err(|e| e.to_string())?;
    zip.write_all(&db_bytes).map_err(|e| e.to_string())?;

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn open_spagz(spagz_path: String) -> Result<SpagzSession, String> {
    let spagz = PathBuf::from(&spagz_path);
    if !spagz.exists() {
        return Err(format!("File not found: {}", spagz_path));
    }

    // Deterministic extraction dir based on file path
    let app_data = dirs::data_local_dir()
        .unwrap_or_else(|| dirs_fallback());
    let hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        spagz_path.hash(&mut h);
        h.finish()
    };
    let session_dir = app_data.join("SpaghettiLab").join("spag_sessions").join(format!("{:016x}", hash));
    std::fs::create_dir_all(&session_dir).map_err(|e| e.to_string())?;

    let db_dest = session_dir.join("replay.db");

    // Extract only replay.db
    let file = std::fs::File::open(&spagz).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    {
        let mut entry = archive.by_name("replay.db").map_err(|e| e.to_string())?;
        let mut out = std::fs::File::create(&db_dest).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
    }

    // Read replay_id, video_path hint, and video_url from extracted DB
    let conn = Connection::open(&db_dest).map_err(|e| e.to_string())?;

    // Add video_url column if missing (older .spagz files)
    let _ = conn.execute("ALTER TABLE replays ADD COLUMN video_url TEXT", []);

    // Run any pending schema migrations in-place on the extracted DB
    ensure_schema_version(&conn)?;

    let (replay_id, video_hint, video_url): (String, String, String) = conn
        .query_row(
            "SELECT replay_id, COALESCE(video_path, ''), COALESCE(video_url, '') FROM replays LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|e| e.to_string())?;

    Ok(SpagzSession {
        db_path: db_dest.to_string_lossy().to_string(),
        video_hint,
        video_url,
        spagz_path,
        replay_id,
    })
}

#[tauri::command]
async fn save_spagz(spagz_path: String, db_path: String) -> Result<(), String> {
    let spagz = PathBuf::from(&spagz_path);

    // Write to a temp file next to the original, then rename
    let tmp_path = spagz.with_extension("spagz.tmp");
    {
        let out_file = std::fs::File::create(&tmp_path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipWriter::new(out_file);

        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("replay.db", options).map_err(|e| e.to_string())?;
        let db_bytes = std::fs::read(&db_path).map_err(|e| e.to_string())?;
        zip.write_all(&db_bytes).map_err(|e| e.to_string())?;

        zip.finish().map_err(|e| e.to_string())?;
    }

    // Atomic rename
    std::fs::rename(&tmp_path, &spagz).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn create_quick_session(
    video_path: String,
    video_url: Option<String>,
) -> Result<SpagzSession, String> {
    // Derive replay_id from URL video ID or filename
    let replay_id = if let Some(ref url) = video_url {
        // Try to extract video ID from YouTube/Twitch URL
        if let Some(caps) = regex_lite_extract_id(url) {
            caps
        } else {
            url_to_slug(url)
        }
    } else if !video_path.is_empty() {
        PathBuf::from(&video_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "untitled".to_string())
    } else {
        "untitled".to_string()
    };

    // Session dir based on replay_id
    let app_data = dirs::data_local_dir().unwrap_or_else(|| dirs_fallback());
    let session_dir = app_data
        .join("SpaghettiLab")
        .join("quick_sessions")
        .join(&replay_id);
    std::fs::create_dir_all(&session_dir).map_err(|e| e.to_string())?;

    let db_path = session_dir.join("replay.db");
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS replays (
            replay_id TEXT PRIMARY KEY,
            video_path TEXT,
            video_url TEXT,
            duration_ms REAL,
            frame_count INTEGER
        );
        CREATE TABLE IF NOT EXISTS frame_data (
            replay_id TEXT,
            frame_number INTEGER,
            timestamp_ms REAL,
            p1_health_pct REAL,
            p2_health_pct REAL,
            timer_value INTEGER,
            p1_rounds_won INTEGER,
            p2_rounds_won INTEGER,
            p1_tension_pct REAL,
            p2_tension_pct REAL
        );
        CREATE TABLE IF NOT EXISTS damage_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id TEXT,
            timestamp_ms REAL,
            frame_start INTEGER,
            frame_end INTEGER,
            target_side INTEGER,
            damage_pct REAL,
            pre_health_pct REAL,
            post_health_pct REAL
        );
        CREATE TABLE IF NOT EXISTS notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS drawings (
            drawing_id INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            strokes_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(replay_id, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS winner_overrides (
            replay_id TEXT NOT NULL,
            round_index INTEGER NOT NULL,
            winner TEXT NOT NULL,
            PRIMARY KEY (replay_id, round_index)
        );"
    ).map_err(|e| e.to_string())?;

    // Insert or update replay entry
    conn.execute(
        "INSERT OR REPLACE INTO replays (replay_id, video_path, video_url, duration_ms, frame_count) VALUES (?, ?, ?, 0, 0)",
        rusqlite::params![replay_id, video_path, video_url],
    ).map_err(|e| e.to_string())?;

    Ok(SpagzSession {
        db_path: db_path.to_string_lossy().to_string(),
        video_hint: video_path,
        video_url: video_url.unwrap_or_default(),
        spagz_path: String::new(),
        replay_id,
    })
}

/// Extract video ID from YouTube/Twitch URLs
fn regex_lite_extract_id(url: &str) -> Option<String> {
    // YouTube: various URL patterns
    if url.contains("youtu.be/") || url.contains("youtube.com/") {
        // Find the video ID after watch?v=, embed/, shorts/, live/, or youtu.be/
        let markers = ["watch?v=", "embed/", "shorts/", "live/", "youtu.be/"];
        for marker in markers {
            if let Some(pos) = url.find(marker) {
                let start = pos + marker.len();
                let id: String = url[start..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                    .collect();
                if !id.is_empty() {
                    return Some(id);
                }
            }
        }
    }
    // Twitch: /videos/DIGITS
    if let Some(pos) = url.find("twitch.tv/videos/") {
        let start = pos + "twitch.tv/videos/".len();
        let id: String = url[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if !id.is_empty() {
            return Some(id);
        }
    }
    None
}

/// Fallback: convert a URL to a filesystem-safe slug
fn url_to_slug(url: &str) -> String {
    url.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
        .chars()
        .take(64)
        .collect()
}

#[tauri::command]
async fn reanalyze_all(
    app_handle: tauri::AppHandle,
    db_path: String,
) -> Result<(), String> {
    let ids: Vec<String> = {
        let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT replay_id FROM replays ORDER BY replay_id")
            .map_err(|e| e.to_string())?;
        let result: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    };

    let total = ids.len();
    for (i, id) in ids.iter().enumerate() {
        let msg = format!("Replay {}/{}: {}", i + 1, total, id);
        let _ = app_handle.emit("reanalyze-progress", &msg);
        eprintln!("[REANALYZE] {}", msg);
        // Run on blocking thread so the event loop can dispatch the emit
        let handle = app_handle.clone();
        let db = db_path.clone();
        let rid = id.clone();
        tauri::async_runtime::spawn_blocking(move || {
            tauri::async_runtime::block_on(reanalyze_replay(handle, db, rid))
        }).await.map_err(|e| e.to_string())??;
    }
    let _ = app_handle.emit("reanalyze-progress", format!("Done — {} replays reanalyzed", total));
    Ok(())
}

// ── App setup ────────────────────────────────────────────────────────────────

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            // Check if launched with a .spag file argument (file association)
            let args: Vec<String> = std::env::args().collect();
            if let Some(path) = args.get(1) {
                let event_name = if path.ends_with(".spagz") {
                    Some("open-spagz-file")
                } else if path.ends_with(".spag") {
                    Some("open-spag-file")
                } else {
                    None
                };
                if let Some(event) = event_name {
                    let path = path.clone();
                    let handle = app.handle().clone();
                    // Emit after a short delay so frontend is ready
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        let _ = handle.emit(event, &path);
                    });
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_replays,
            get_frame_data,
            get_damage_events,
            get_rounds,
            get_match_stats,
            get_highlights,
            get_default_db_path,
            analyze_video,
            reanalyze_replay,
            reanalyze_all,
            export_clip,
            resolve_video_path,
            download_vod,
            extract_preview_frame,
            scan_vod,
            cut_vod_sets,
            get_notes,
            add_note,
            update_note,
            delete_note,
            get_drawings,
            save_drawing,
            delete_drawing,
            export_spag,
            open_spag,
            save_spag,
            export_spagz,
            open_spagz,
            save_spagz,
            create_quick_session,
            set_round_winner,
            get_questionnaire_answers,
            set_questionnaire_answer,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_timer_drops_isolated_start_reading() {
        // 30 samples/s: a round at 82/81, one misread 97 on a super flash,
        // then a HUD gap; an isolated low reading is kept.
        let ts: Vec<f64> = (0..120).map(|i| i as f64 * 1000.0 / 30.0).collect();
        let mut timer: Vec<Option<i32>> = vec![None; 120];
        for i in 0..30 { timer[i] = Some(82); }
        for i in 30..50 { timer[i] = Some(81); }
        timer[50] = Some(97);
        timer[100] = Some(40);
        let out = supported_timer(&timer, &ts);
        assert_eq!(out[10], Some(82));
        assert_eq!(out[49], Some(81));
        assert_eq!(out[50], None);
        assert_eq!(out[100], Some(40));
        // A real intro (a run of 99) is kept.
        let mut intro: Vec<Option<i32>> = vec![None; 120];
        for i in 60..100 { intro[i] = Some(99); }
        assert_eq!(supported_timer(&intro, &ts)[60], Some(99));
    }

    #[test]
    fn test_sample_rate_scale() {
        let at = |rate: f64| -> Vec<f64> { (0..100).map(|i| i as f64 * 1000.0 / rate).collect() };
        assert_eq!(sample_rate_scale(&at(30.0)), 1.0);
        assert_eq!(sample_rate_scale(&at(29.97)), 1.0);
        assert!((sample_rate_scale(&at(60.0)) - 2.0).abs() < 1e-9);
        assert!((sample_rate_scale(&at(15.0)) - 0.5).abs() < 1e-9);
        assert_eq!(scale_frames(450, 1.0), 450);
        assert_eq!(scale_frames(450, 2.0), 900);
        assert_eq!(scale_frames(31, 0.5), 16);
        assert_eq!(scale_frames(1, 0.5), 1);
    }

    /// Synthetic replay: `rounds` of (p1_final, p2_final, hearts lost by P1,
    /// hearts lost by P2) with 45 s of play (timer 99 for 3 s, then counting
    /// down), separated by 4 s with nothing read.  `rate` samples per second.
    fn synthetic_replay(rounds: &[(f64, f64, i32, i32)], rate: f64) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE frame_data (replay_id TEXT, frame_number INTEGER, timestamp_ms REAL,
             p1_health_pct REAL, p2_health_pct REAL, timer_value INTEGER,
             p1_rounds_won INTEGER, p2_rounds_won INTEGER)",
        ).unwrap();
        let mut k = 0i64;
        let mut t0 = 0.0f64;
        for &(f1, f2, lost1, lost2) in rounds {
            let n_play = (45.0 * rate) as i64;
            for j in 0..n_play {
                let s = j as f64 / rate;
                let x = s / 44.0;
                let hp = |f: f64| -> f64 { HP_CEILING * (1.0 - (1.0 - f) * x.min(1.0)) };
                let timer = if s < 3.0 { 99 } else { 99 - (s - 3.0) as i32 };
                conn.execute(
                    "INSERT INTO frame_data VALUES ('r', ?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![k, (t0 + s * 1000.0), hp(f1), hp(f2), timer, lost2, lost1],
                ).unwrap();
                k += 1;
            }
            let n_gap = (4.0 * rate) as i64;
            for j in 0..n_gap {
                let s = 45.0 + j as f64 / rate;
                conn.execute(
                    "INSERT INTO frame_data VALUES ('r', ?1, ?2, NULL, NULL, NULL, NULL, NULL)",
                    rusqlite::params![k, (t0 + s * 1000.0)],
                ).unwrap();
                k += 1;
            }
            t0 += 49_000.0;
        }
        conn
    }

    #[test]
    fn test_hearts_match_starts_and_winner_check() {
        // Match 1: P1 wins 2-0.  Match 2: round 3's HP reads P1 ahead (a
        // time-out, nobody KO'd) but the hearts show P2 took it; P1 then
        // wins two rounds.  The hearts state is read early in each round.
        // The hearts are secondary: the confident HP winner is kept and the
        // disagreement only flags the round as uncertain.
        let rounds = [
            (0.6, 0.0, 0, 0),
            (0.5, 0.0, 0, 1),
            (0.3, 0.1, 0, 0),
            (0.7, 0.0, 1, 0),
            (0.4, 0.0, 1, 1),
            (0.9, 0.0, 0, 0),
        ];
        for rate in [30.0, 60.0] {
            let conn = synthetic_replay(&rounds, rate);
            let r = detect_rounds_for_replay(&conn, "r").unwrap();
            assert_eq!(r.len(), 6, "rate {}", rate);
            let winners: Vec<&str> = r.iter().map(|x| x.winner.as_str()).collect();
            assert_eq!(winners, ["P1"; 6], "rate {}", rate);
            let confident: Vec<bool> = r.iter().map(|x| x.winner_confident).collect();
            assert_eq!(confident, [true, true, false, true, true, true], "rate {}", rate);
            let starts: Vec<bool> = r.iter().map(|x| x.is_match_start).collect();
            assert_eq!(starts, [true, false, true, false, false, true], "rate {}", rate);
        }
    }

    #[test]
    fn test_hearts_confirm_or_break_tie_of_uncertain_winner() {
        // Round 1 ends in a double KO (both bars empty): on HP alone the
        // winner is an uncertain P1.  When the next round's hearts show P2
        // lost one (P1 took it) they confirm it; when they show P1 lost one
        // they break the tie for P2, and the round stays uncertain.
        let hide_next = |c: &Connection| {
            c.execute("UPDATE frame_data SET p1_rounds_won = NULL, p2_rounds_won = NULL
                       WHERE timestamp_ms >= 98000", []).unwrap();
        };
        for (lost1, lost2, winner, confident) in [(0, 1, "P1", true), (1, 0, "P2", false)] {
            let rounds = [(0.6, 0.0, 0, 0), (0.0, 0.0, 0, 0), (0.0, 0.5, lost1, lost2)];
            let conn = synthetic_replay(&rounds, 30.0);
            let r = detect_rounds_for_replay(&conn, "r").unwrap();
            assert_eq!(r.len(), 3);
            assert_eq!((r[1].winner.as_str(), r[1].winner_confident), (winner, confident),
                "next-round hearts {}/{}", lost1, lost2);
            hide_next(&conn);
            let r = detect_rounds_for_replay(&conn, "r").unwrap();
            assert_eq!((r[1].winner.as_str(), r[1].winner_confident), ("P1", false), "HP alone");
        }
    }

    #[test]
    fn test_match_start_from_result_when_hearts_unreadable() {
        // P1 wins 2-0, then the hearts of round 2 are unreadable (a bright
        // stage, an overlay): the match start comes from the previous
        // round's hearts state (P2 lost one) plus its HP winner (P1).
        let rounds = [(0.6, 0.0, 0, 0), (0.5, 0.0, 0, 1), (0.0, 0.4, 0, 0), (0.6, 0.0, 1, 0)];
        let conn = synthetic_replay(&rounds, 30.0);
        conn.execute("UPDATE frame_data SET p1_rounds_won = NULL, p2_rounds_won = NULL
                      WHERE timestamp_ms >= 98000 AND timestamp_ms < 147000", []).unwrap();
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        let starts: Vec<bool> = r.iter().map(|x| x.is_match_start).collect();
        assert_eq!(starts, [true, false, true, false]);
        // Unreadable hearts after a round that did not end the match: no start.
        let conn = synthetic_replay(&rounds, 30.0);
        conn.execute("UPDATE frame_data SET p1_rounds_won = NULL, p2_rounds_won = NULL
                      WHERE timestamp_ms >= 49000 AND timestamp_ms < 98000", []).unwrap();
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        let starts: Vec<bool> = r.iter().map(|x| x.is_match_start).collect();
        assert_eq!(starts, [true, false, true, false]);
    }

    #[test]
    fn test_match_start_needs_hud_gap_and_timer_intro() {
        // P1 wins 2-0; round 2 reads 0/0 hearts.  If the timer stays on
        // screen through the transition (no HUD gap), or round 2's timer
        // never shows its intro value, it is not a match start.
        let rounds = [(0.6, 0.0, 0, 0), (0.5, 0.0, 0, 1), (0.0, 0.4, 0, 0)];
        let conn = synthetic_replay(&rounds, 30.0);
        conn.execute("UPDATE frame_data SET timer_value = 99
                      WHERE timestamp_ms >= 94000 AND timestamp_ms < 98000", []).unwrap();
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        assert_eq!(r.len(), 3);
        assert!(!r[2].is_match_start, "no HUD gap before round 2");
        let conn = synthetic_replay(&rounds, 30.0);
        conn.execute("UPDATE frame_data SET timer_value = MIN(timer_value, 95)
                      WHERE timestamp_ms >= 98000", []).unwrap();
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        assert_eq!(r.len(), 3);
        assert!(!r[2].is_match_start, "no timer intro in round 2");
    }

    #[test]
    fn test_hearts_never_split_a_round() {
        // Mid-round, the hearts flip to 0/0 behind a 2 s HUD gap (a super
        // cinematic) with no HP or timer reset: still one round, and no
        // match start inside it.
        let rounds = [(0.6, 0.0, 0, 0), (0.5, 0.0, 0, 1), (0.0, 0.4, 1, 1)];
        let conn = synthetic_replay(&rounds, 30.0);
        conn.execute("UPDATE frame_data SET timer_value = NULL, p1_health_pct = NULL, p2_health_pct = NULL,
                      p1_rounds_won = NULL, p2_rounds_won = NULL
                      WHERE timestamp_ms >= 118000 AND timestamp_ms < 120000", []).unwrap();
        conn.execute("UPDATE frame_data SET p1_rounds_won = 0, p2_rounds_won = 0
                      WHERE timestamp_ms >= 120000 AND timestamp_ms < 143000", []).unwrap();
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        assert_eq!(r.len(), 3);
        let starts: Vec<bool> = r.iter().map(|x| x.is_match_start).collect();
        assert_eq!(starts, [true, false, false]);
    }

    #[test]
    fn test_leading_partial_round_kept_only_when_it_ends_in_a_ko() {
        // The video opens 35 s into round 0 (10 s of play left, then the 4 s
        // gap): too short for MIN_ROUND_FRAMES, but it ends in a KO.
        for (p2_final, expect) in [(0.0, 3usize), (0.3, 2usize)] {
            let conn = synthetic_replay(&[(0.6, p2_final, 0, 0), (0.5, 0.0, 0, 0), (0.0, 0.4, 0, 1)], 30.0);
            conn.execute("DELETE FROM frame_data WHERE timestamp_ms < 35000", []).unwrap();
            let r = detect_rounds_for_replay(&conn, "r").unwrap();
            assert_eq!(r.len(), expect, "p2_final {}", p2_final);
            if expect == 3 {
                assert_eq!(r[0].winner, "P1");
                assert!(r[0].round_start_ms >= 35000.0 && r[0].round_end_ms < 49000.0);
            }
        }
        // Opening on the last 4 s of a round (KO, then the post-KO screen):
        // not kept.
        let conn = synthetic_replay(&[(0.6, 0.0, 0, 0), (0.5, 0.0, 0, 0), (0.0, 0.4, 0, 1)], 30.0);
        conn.execute("DELETE FROM frame_data WHERE timestamp_ms < 40000", []).unwrap();
        conn.execute("UPDATE frame_data SET p1_health_pct = 0.5, p2_health_pct = 0.0
                      WHERE timestamp_ms >= 41000 AND timestamp_ms < 49000", []).unwrap();
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_round_end_ignores_next_rounds_intro_readings() {
        // Round 0's HUD disappears at 43.5 s; 1.3 s of bars are read on
        // the next round's intro (no timer) after a 4 s gap.  The round
        // ends after its last real reading, not on the intro.
        let conn = synthetic_replay(&[(0.0, 0.6, 0, 0), (0.5, 0.0, 0, 0), (0.0, 0.4, 0, 1)], 30.0);
        conn.execute("UPDATE frame_data SET p1_health_pct = NULL, p2_health_pct = NULL, timer_value = NULL
                      WHERE timestamp_ms >= 43500 AND timestamp_ms < 49000", []).unwrap();
        conn.execute("UPDATE frame_data SET p1_health_pct = 0.0, p2_health_pct = 0.5
                      WHERE timestamp_ms >= 47500 AND timestamp_ms < 48800", []).unwrap();
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].winner, "P2");
        assert!(r[0].round_end_ms < 46000.0, "end {}", r[0].round_end_ms);
    }

    #[test]
    fn test_old_hearts_reader_flags_no_match_start() {
        // The old reader read 0/0 nearly everywhere: no round after a 0/0
        // round is a match start.
        let rounds = [(0.6, 0.0, 0, 0), (0.5, 0.0, 0, 0), (0.7, 0.0, 0, 0)];
        let conn = synthetic_replay(&rounds, 30.0);
        let r = detect_rounds_for_replay(&conn, "r").unwrap();
        let starts: Vec<bool> = r.iter().map(|x| x.is_match_start).collect();
        assert_eq!(starts, [true, false, false]);
    }

    #[test]
    fn test_slaythatdude_rounds() {
        let db_path = std::path::Path::new("C:/Projects/replanal/output/analysis.db");
        let conn = rusqlite::Connection::open(db_path).unwrap();
        let rounds = detect_rounds_for_replay(&conn, "03_UF_SlayThatDude_vs_Salkazko").unwrap();
        for r in &rounds {
            eprintln!(
                "Round {} t={:.1}s-{:.1}s winner={} w_hp={:.3} l_hp={:.3} match_start={}",
                r.round_index,
                r.round_start_ms / 1000.0,
                r.round_end_ms / 1000.0,
                r.winner,
                r.winner_final_hp,
                r.loser_final_hp,
                r.is_match_start,
            );
        }
    }

    #[test]
    fn test_wfinal_guessinggame_vs_jason() {
        let db_path = std::path::Path::new("C:/Projects/replanal/output/analysis.db");
        let conn = rusqlite::Connection::open(db_path).unwrap();
        let rounds = detect_rounds_for_replay(&conn, "11_WFinal_GuessingGame_vs_Jason").unwrap();
        for r in &rounds {
            eprintln!(
                "Round {} t={:.1}s-{:.1}s winner={} w_hp={:.3} l_hp={:.3} match_start={}",
                r.round_index,
                r.round_start_ms / 1000.0,
                r.round_end_ms / 1000.0,
                r.winner,
                r.winner_final_hp,
                r.loser_final_hp,
                r.is_match_start,
            );
        }
        // First round should NOT start before ~120s (lobby ends around there)
        assert!(rounds[0].round_start_ms > 100_000.0,
            "First round starts too early at {:.1}s — lobby not filtered",
            rounds[0].round_start_ms / 1000.0);
    }

    /// Prints detected rounds for any DB/replay (opened read-only):
    ///   SPAG_TEST_DB=path/to/analysis.db SPAG_TEST_REPLAY=replay_id \
    ///     cargo test test_rounds_from_env -- --nocapture
    /// Skipped (passes) when SPAG_TEST_DB is unset.
    #[test]
    fn test_rounds_from_env() {
        let (Ok(db), Ok(replay)) = (std::env::var("SPAG_TEST_DB"), std::env::var("SPAG_TEST_REPLAY")) else {
            return;
        };
        let conn = rusqlite::Connection::open_with_flags(
            &db,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        let rounds = detect_rounds_for_replay(&conn, &replay).unwrap();
        for r in &rounds {
            println!(
                "ROUND {} t={:.1}s-{:.1}s winner={} w_hp={:.3} l_hp={:.3} confident={} match_start={}",
                r.round_index,
                r.round_start_ms / 1000.0,
                r.round_end_ms / 1000.0,
                r.winner,
                r.winner_final_hp,
                r.loser_final_hp,
                r.winner_confident,
                r.is_match_start,
            );
        }
    }

    #[test]
    fn test_slycoops_rounds() {
        let db_path = std::path::Path::new("C:/Projects/replanal/output/analysis.db");
        let conn = rusqlite::Connection::open(db_path).unwrap();
        let rounds = detect_rounds_for_replay(&conn, "01_WRI_Slycoops_vs_Leftover").unwrap();
        for r in &rounds {
            eprintln!(
                "Round {} t={:.1}s-{:.1}s winner={} w_hp={:.3} l_hp={:.3} match_start={}",
                r.round_index,
                r.round_start_ms / 1000.0,
                r.round_end_ms / 1000.0,
                r.winner,
                r.winner_final_hp,
                r.loser_final_hp,
                r.is_match_start,
            );
        }
    }
}
