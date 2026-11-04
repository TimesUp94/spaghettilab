# Tauri/Rust Backend Commands

All Tauri commands are defined in `app/src-tauri/src/lib.rs` and registered in the `invoke_handler` at the bottom of `run()`. The frontend calls them via `@tauri-apps/api/core` invoke.

---

## Data Types

### Rust Structs (Serialized to Frontend)

```rust
pub struct Replay {
    pub replay_id: String,
    pub video_path: String,
    pub duration_ms: f64,
    pub frame_count: i64,
}

pub struct FrameDataPoint {
    pub timestamp_ms: f64,
    pub p1_health_pct: Option<f64>,
    pub p2_health_pct: Option<f64>,
    pub p1_tension_pct: Option<f64>,
    pub p2_tension_pct: Option<f64>,
}

pub struct DamageEvent {
    pub event_id: i64,
    pub replay_id: String,
    pub timestamp_ms: f64,
    pub frame_start: i64,
    pub frame_end: i64,
    pub target_side: i64,       // 1 = P1, 2 = P2
    pub damage_pct: f64,
    pub pre_health_pct: f64,
    pub post_health_pct: f64,
}

pub struct RoundResult {
    pub replay_id: String,
    pub round_index: usize,
    pub round_start_ms: f64,
    pub round_end_ms: f64,
    pub winner: String,         // "P1" or "P2"
    pub winner_final_hp: f64,
    pub loser_final_hp: f64,
    pub winner_min_hp: f64,
    pub max_deficit: f64,
    pub deficit_timestamp_ms: f64,
    pub is_comeback: bool,
    pub is_match_start: bool,
    pub winner_confident: bool,
}

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

pub struct Highlight {
    pub kind: String,           // "comeback", "close_round", "big_damage", "perfect"
    pub label: String,
    pub timestamp_ms: f64,
    pub end_ms: f64,
    pub details: String,
    pub severity: f64,          // 0.0-1.0, for sorting
}

pub struct Note {
    pub note_id: i64,
    pub replay_id: String,
    pub timestamp_ms: f64,
    pub text: String,
    pub created_at: String,
}

pub struct AnalysisStatus {
    pub running: bool,
    pub progress_lines: Vec<String>,
    pub error: Option<String>,
    pub db_path: Option<String>,
}

pub struct SpagSession {
    pub db_path: String,
    pub video_path: String,
    pub spag_path: String,
    pub replay_id: String,
}
```

### Rust Structs (Deserialized from Frontend)

```rust
pub struct RoiRect {
    pub y1: u32, pub y2: u32,
    pub x1: u32, pub x2: u32,
}

pub struct RoiQuad {
    pub tl: [u32; 2],   // [x, y]
    pub tr: [u32; 2],
    pub br: [u32; 2],
    pub bl: [u32; 2],
}

pub struct VodRoiConfig {
    pub p1_tension: RoiRect,
    pub p2_tension: RoiRect,
    pub timer: RoiRect,
    pub p1_name: RoiQuad,
    pub p2_name: RoiQuad,
}

pub struct DetectedSetInfo {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
    pub gameplay_duration_secs: f64,
    pub game_count: u32,
    pub p1_name: String,
    pub p2_name: String,
}

pub struct CutSetRequest {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
    pub p1_name: Option<String>,
    pub p2_name: Option<String>,
}

pub struct StriveExportOptions {
    pub output_dir: String,
    pub mode: String,            // "all" | "count" | "since"
    pub count: Option<u32>,      // mode "count"
    pub since: Option<String>,   // YYYY-MM-DD, mode "since"
    pub start: Option<u32>,      // skip the N newest first
    pub dry_run: Option<bool>,   // walk the list, record nothing
    pub lang: Option<String>,    // UI language for the text checks ("auto" when absent)
}
```

---

## Database Query Commands

### get_replays

```rust
fn get_replays(db_path: String) -> Result<Vec<Replay>, String>
```

**Purpose:** Fetch all replays from the database.

**Query:** `SELECT replay_id, video_path, duration_ms, frame_count FROM replays ORDER BY replay_id`

**Notes:** Returns replays sorted alphabetically by replay_id.

---

### get_frame_data

```rust
fn get_frame_data(db_path: String, replay_id: String) -> Result<Vec<FrameDataPoint>, String>
```

**Purpose:** Fetch per-frame health and tension data for a replay.

**Query:** `SELECT timestamp_ms, p1_health_pct, p2_health_pct, p1_tension_pct, p2_tension_pct FROM frame_data WHERE replay_id = ? ORDER BY timestamp_ms`

**Notes:** Handles schema migrations gracefully. If the `p1_tension_pct` column doesn't exist (older DB), falls back to a query without tension columns and returns `None` for tension fields. HP from older data that reads a full bar at 0.875 is divided by 0.875 (`replay_hp_full`, see `docs/round-detection.md`, "HP scale").

---

### get_damage_events

```rust
fn get_damage_events(db_path: String, replay_id: String) -> Result<Vec<DamageEvent>, String>
```

**Purpose:** Fetch all damage events for a replay.

**Query:** `SELECT event_id, replay_id, timestamp_ms, frame_start, frame_end, target_side, damage_pct, pre_health_pct, post_health_pct FROM damage_events WHERE replay_id = ? ORDER BY timestamp_ms`

HP values of older (0.875-scale) data are rescaled as in `get_frame_data`.

---

### get_rounds

```rust
fn get_rounds(db_path: String, replay_id: String) -> Result<Vec<RoundResult>, String>
```

**Purpose:** Detect rounds and determine winners for a replay.

**Algorithm:**
1. Calls `detect_rounds_for_replay()` which runs the full round detection algorithm (see `docs/round-detection.md`), including the manual corrections from `winner_overrides` (applied before the match rules, so rounds that depend on a corrected one follow it)

**Notes:** Rounds are computed on-demand from raw frame data each time this command is called. Manual overrides are stored separately and applied at read-time, so they survive reanalysis.

---

### get_match_stats

```rust
fn get_match_stats(db_path: String, replay_id: String) -> Result<MatchStats, String>
```

**Purpose:** Compute aggregated statistics for a replay.

**Algorithm:**
1. Fetch replay metadata from `replays` table
2. Run round detection via `detect_rounds_for_replay()`
3. Fetch damage events
4. Call `compute_match_stats()` which aggregates across all rounds and events

**Computed fields:** Round wins, damage totals, biggest hits, duration stats, comeback count, close rounds (winner ended under 20 % of the bar), average winner HP.

All HP values (`winner_final_hp`, `loser_final_hp`, `winner_min_hp`, `max_deficit`, `damage_pct`, `p1/p2_health_pct`) are fractions of the bar: full = 1.0, KO = 0 (no `HP_CEILING` normalisation; see `docs/round-detection.md`, "HP scale").

---

### get_highlights

```rust
fn get_highlights(db_path: String, replay_id: String) -> Result<Vec<Highlight>, String>
```

**Purpose:** Generate notable moments for a replay.

**Algorithm:**
1. Run round detection
2. Fetch damage events
3. Call `generate_highlights()` which flags: comebacks, close rounds, big damage (>30%), perfects (>90% HP)
4. Sort by severity descending

---

## Notes Commands

### get_notes

```rust
fn get_notes(db_path: String, replay_id: String) -> Result<Vec<Note>, String>
```

**Purpose:** Fetch all user notes for a replay, sorted by timestamp.

**Notes:** Creates the `notes` table if it doesn't exist (migration).

---

### add_note

```rust
fn add_note(db_path: String, replay_id: String, timestamp_ms: f64, text: String) -> Result<Note, String>
```

**Purpose:** Create a new note at a specific timestamp.

**Returns:** The created `Note` with auto-generated `note_id` and `created_at`.

---

### update_note

```rust
fn update_note(db_path: String, note_id: i64, text: String, timestamp_ms: Option<f64>) -> Result<(), String>
```

**Purpose:** Update an existing note's text and optionally its timestamp.

---

### delete_note

```rust
fn delete_note(db_path: String, note_id: i64) -> Result<(), String>
```

**Purpose:** Delete a note by ID.

---

## Analysis Commands

### analyze_video

```rust
async fn analyze_video(video_path: String, _sample_every: Option<u32>) -> Result<String, String>
```

**Purpose:** Run the Python CV pipeline on a new video.

**Process:**
1. Find project root (contains `scripts/analyze_replay.py`)
2. Determine output directory and DB path via `default_output_paths()`
3. Create output directory
4. Spawn Python: `python scripts/analyze_replay.py <video> --config <yaml> --output <dir>`
5. Set `PYTHONPATH` to project root so `import replanal` works

**Returns:** Path to the analysis database.

**Notes:** Blocks until Python completes. `_sample_every` parameter is currently unused (reserved).

---

### reanalyze_replay

```rust
async fn reanalyze_replay(db_path: String, replay_id: String) -> Result<(), String>
```

**Purpose:** Re-run the Python CV pipeline on an existing replay.

**Process:**
1. Open the database, resolve the video path for the replay
2. Close the database (Python will re-open it)
3. Spawn Python with the same arguments as `analyze_video`
4. The existing data for this replay_id is overwritten (INSERT OR REPLACE)

---

### reanalyze_all

```rust
async fn reanalyze_all(db_path: String) -> Result<(), String>
```

**Purpose:** Re-run the Python CV pipeline on every replay in the database.

**Process:**
1. Open database, fetch all replay IDs
2. Call `reanalyze_replay()` sequentially for each one
3. Any failure stops the batch

---

## Winner Override Command

### set_round_winner

```rust
fn set_round_winner(db_path: String, replay_id: String, round_index: usize, winner: String) -> Result<(), String>
```

**Purpose:** Manually override the winner of a specific round.

**Storage:**
```sql
CREATE TABLE IF NOT EXISTS winner_overrides (
    replay_id   TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    winner      TEXT NOT NULL,
    PRIMARY KEY (replay_id, round_index)
);
```

Uses `INSERT OR REPLACE` so calling again for the same round overwrites the previous override.

**Notes:** Overrides are applied at read-time inside `detect_rounds_for_replay` (before the match rules), not stored in the round detection output. This means they persist across reanalysis.

---

## Export Commands

### export_clip

```rust
async fn export_clip(video_path: String, start_ms: f64, end_ms: f64, output_path: String) -> Result<(), String>
```

**Purpose:** Export a video clip using ffmpeg stream copy.

**Command:**
```
ffmpeg -y -ss <start_s> -i <video> -t <duration_s> -c copy -avoid_negative_ts make_zero <output>
```

No re-encoding -- fast stream copy.

---

### export_spag

```rust
async fn export_spag(db_path: String, replay_id: String, output_path: String) -> Result<(), String>
```

**Purpose:** Export a single replay as a self-contained `.spag` file (ZIP archive).

**Process:**
1. Resolve the video path for the replay
2. Create a temporary database with only this replay's data using `ATTACH DATABASE`:
   - Copy `replays` row (rewrite video_path to `video.mp4`)
   - Copy `frame_data` rows
   - Copy `damage_events` rows
   - Copy `notes` rows
3. Build a ZIP file:
   - `replay.db`: Deflate-compressed temporary database
   - `video.mp4`: Stored (no compression, since video is already compressed)

**Notes:** Video is written in 8MB chunks to handle large files.

---

### open_spag

```rust
async fn open_spag(spag_path: String) -> Result<SpagSession, String>
```

**Purpose:** Open a `.spag` file for viewing and editing.

**Process:**
1. Compute a deterministic extraction directory based on a hash of the file path:
   `%LOCALAPPDATA%/SpaghettiLab/spag_sessions/<16-char-hex-hash>/`
2. Extract `replay.db` and `video.mp4` from the ZIP
3. Open the extracted database, read the replay_id
4. Return a `SpagSession` with paths to the extracted DB and video

**Notes:** Always re-extracts to get fresh data. The session directory is stable per file path.

---

### save_spag

```rust
async fn save_spag(spag_path: String, db_path: String) -> Result<(), String>
```

**Purpose:** Save edits back to an open `.spag` file.

**Process:**
1. Read the original .spag to get the video
2. Write to a temporary `.spag.tmp` file:
   - Updated `replay.db` from the session directory
   - Original `video.mp4` copied from the archive
3. Atomic rename: `.spag.tmp` -> `.spag`

**Notes:** Only the database is updated. The video is copied from the original archive without modification.

---

## VOD Splitter Commands

### extract_preview_frame

```rust
async fn extract_preview_frame(video_path: String, timestamp_secs: f64) -> Result<String, String>
```

**Purpose:** Extract a single frame from a video at a given timestamp for the ROI picker preview.

**Command:**
```
ffmpeg -y -ss <timestamp> -i <video> -frames:v 1 -q:v 2 -v quiet <output.png>
```

**Output:** Saved to `<output_dir>/.vod_preview.png`. Returns the file path.

---

### scan_vod

```rust
async fn scan_vod(app_handle: AppHandle, video_path: String, roi_config: VodRoiConfig) -> Result<Vec<DetectedSetInfo>, String>
```

**Purpose:** Scan a VOD for gameplay segments and detect set boundaries.

**Process:**
1. Find project root, locate `scripts/split_vod.py`
2. Spawn Python with `--preview --json-output` and ROI arguments
3. Stream stdout line-by-line:
   - `PROGRESS:N/M` -> emit `vod-scan-progress` Tauri event
   - `JSON_RESULT:{...}` -> parse as final result
4. Read stderr in a separate thread
5. Parse JSON into `Vec<DetectedSetInfo>`

**Events emitted:** `vod-scan-progress` (progress string)

---

### cut_vod_sets

```rust
async fn cut_vod_sets(app_handle: AppHandle, video_path: String, sets: Vec<CutSetRequest>, output_dir: String) -> Result<Vec<String>, String>
```

**Purpose:** Cut selected sets from a VOD into individual files.

**Process:**
1. Create output directory
2. For each `CutSetRequest`:
   - Compute cut start (set start - 10s padding) and duration (to set end + 5s padding)
   - Generate filename: `{index:02d}_{p1}_vs_{p2}.mp4` or `set_{index:02d}.mp4`
   - Run ffmpeg stream copy
   - Emit `vod-cut-progress` event (`i/total`)

**Returns:** List of output file paths.

**Events emitted:** `vod-cut-progress` (progress string)

---

## Strive Replay Export Commands

See `docs/strive-export.md` for the exporter itself.

### export_strive_replays

```rust
async fn export_strive_replays(app_handle: AppHandle, options: StriveExportOptions) -> Result<serde_json::Value, String>
```

**Purpose:** Drive the running Guilty Gear -Strive- to its saved replays and record each one as an mp4 in `options.output_dir`.

**Process:**
1. Windows only; refuses a second export while one runs.
2. Build the CLI arguments (`strive_export_args`): `export --out DIR --json-progress` plus `--all` / `--count N` / `--since DATE`, `--start`, `--dry-run`, `--lang`.
3. Spawn `<python> -u -m replanal.strive_export ...` with `PYTHONPATH` = project root and `SPAGHETTI_FFMPEG` = `ffmpeg_bin()` (one more candidate; the module prefers any FFmpeg with `gfxcapture`).
4. Keep the child's stdin in `STRIVE_EXPORT_STDIN` for `cancel_strive_export`.
5. Forward every stdout JSON line verbatim as a `strive-export-progress` event; remember the `done` event (summary) and the last `error` message.

**Returns:** the `done` event: `{event, reason, listed, exported: [paths], skipped: [paths], manifest}`. Errors with the exporter's `error` message (or stderr's last line).

**Events emitted:** `strive-export-progress` (one JSON object per event: `ffmpeg`, `start`, `screen`, `status`, `warning`, `list`, `entry`, `recording`, `recorded`, `skipped`, `done`, `error`)

---

### cancel_strive_export

```rust
fn cancel_strive_export(now: bool) -> Result<(), String>
```

**Purpose:** Stop the running export. `now = false` writes `cancel` to its stdin (finish the replay being recorded, return to the list, stop); `now = true` writes `stop` (finalise the current mp4 at once, marked `partial` in the manifest, and leave the game in the replay). Errors when no export runs.

---

## Lab Analysis Commands

Normal users analyse on the Spaghetti Showdown servers behind a sign-in: those commands (`lab_sign_in`,
`run_lab_analysis_remote`, `lab_quota`...) and their structured errors are in `docs/desktop-login.md`, in
`lab_api.rs`, `lab_auth.rs` and `lab_remote.rs`. `run_lab_analysis` below is the local research mode: the UI offers it
only with `SPAGHETTI_LAB_MODELS` set.

| Command | What it does |
|---|---|
| `lab_config`, `lab_account`, `lab_service_status`, `lab_quota` | Read the settings, signed-in state (never tokens), service state and quota |
| `lab_sign_in`, `lab_sign_in_reopen`, `lab_sign_in_cancel`, `lab_sign_out` | The browser sign-in and sign-out |
| `lab_open_web` | Opens `support`, `terms` or `privacy` in the system browser (allow-list) |
| `run_lab_analysis_remote`, `cancel_lab_analysis_remote` | Start or cancel a hosted run (`lab-progress` events) |
| `lab_remote_cancel_phase` | Fresh job state for the cancel warning |
| `lab_remote_running`, `lab_remote_take_error`, `lab_remote_resume` | The run in progress, how the last run ended, and saved jobs at startup |

Remote commands return `{code, message, retry_after?, refunded?, quota?}` errors rather than strings. Production has neither the
login nor analysis on yet, so the app shows a readable message there; dev runs the whole flow.

`app/src-tauri/src/lab.rs`. Run the research models (`tools/ggst_logger/model/lab_analyze.py`) on a replay's
video and keep the result, `lab_result.json` (schema `spaghetti-lab-result/1`, described in
`tools/ggst_logger/model/LAB_RESULT.md`), in a per-replay cache: `<data dir>/SpaghettiLab/lab/<replay_id>/lab_result.json`
(`<data dir>` = `SPAGHETTI_LAB_DATA_DIR` or the user's local app data). It is not in the analysis DB, so there is no
schema change and `.spag` / `.spagz` files and the web app are unaffected (a shared file does not carry the lab result).

Errors that mean the lab isn't set up start with `Lab analysis is not installed` (`lab::NOT_INSTALLED`); the UI shows
them as a setup hint rather than a failure.

### run_lab_analysis

```rust
async fn run_lab_analysis(app_handle: AppHandle, db_path: String, replay_id: String, options: Option<LabOptions>) -> Result<serde_json::Value, String>

pub struct LabOptions {
    pub transcript: Option<String>,  // "auto" (default: only with CUDA) | "on" | "off"
    pub light: Option<bool>,         // --light
    pub device: Option<String>,      // "auto" | "cuda" | "cpu"
    pub video_path: Option<String>,  // the video to read; else the replay's (resolve_video_for_replay)
    pub models_dir: Option<String>,  // the app setting (Lab panel); see below
}
```

**Process:**
1. Refuses when `SPAGHETTI_LAB_NO_MODELS` is set (the e2e run) and while another lab run is going (one at a time: it can use the whole GPU).
2. Finds the script (`<project root>/tools/ggst_logger/model/lab_analyze.py`), the models folder (`options.models_dir`, `SPAGHETTI_LAB_MODELS`, `GGST_MODEL_DIR`, then `<GGST_DATA or G:\Projects\replanal-data>/ggst_dataset/model`; the first that exists), the video and `require_python()`, and checks that the interpreter imports `torch`. Any of these missing is a "not installed" error.
3. Spawns `<python> lab_analyze.py <video> --out <lab>/<id>/run --transcript MODE [--light] [--device D] [--analysis <output dir>]` (`--analysis` when the CV pipeline's `frames/<id>_frames.parquet` is there) with `GGST_MODEL_DIR` = the models folder and `GGST_FFMPEG` = `ffmpeg_bin()`.
4. Forwards each `stage: <name> ...` stdout line as a `lab-progress` event; other lines go to the console.
5. On success, checks the result (`parse_lab_result`: JSON object, `schema` `spaghetti-lab-result/1[.x]`, an `openers` list), writes it to the cache, keeps the run's other files in `<lab>/<id>/last_run/`, and returns it. A failed run leaves the previous result in place; a `ModuleNotFoundError` is reported as "not installed".

**Events emitted:** `lab-progress` `{replay_id, stage, line}`; `stage` is `hud`, `detect`, `moves`, `transcript`, `facts` or `done` (the UI shows "Reading the HUD / Recognising moves / Transcript / Summary").

**Errors:** `Cancelled` after `cancel_lab_analysis` (the UI ignores it); otherwise the script's last output lines.

---

### cancel_lab_analysis

```rust
fn cancel_lab_analysis(replay_id: Option<String>) -> Result<(), String>
```

**Purpose:** Kill the running lab analysis and its children (`taskkill /T /F` on Windows). With a `replay_id`, only if that replay's run is the one going. Errors when nothing runs.

---

### lab_analysis_running

```rust
fn lab_analysis_running() -> Option<String>
```

**Purpose:** The replay whose lab analysis is running, or null. The review screen uses it to show the progress again (and reload the result when it ends) after the user left and came back.

---

### get_lab_result

```rust
fn get_lab_result(replay_id: String) -> Result<Option<serde_json::Value>, String>
```

**Purpose:** The cached `lab_result.json` of a replay (checked like a fresh one), or null when it was never run. Errors on a corrupt or unsupported file.

---

### delete_lab_result

```rust
fn delete_lab_result(replay_id: String) -> Result<(), String>
```

**Purpose:** Remove `<lab>/<replay_id>/` (the result and the last run's files). Refused while that replay's analysis runs.

---

## Utility Commands

### get_default_db_path

```rust
fn get_default_db_path() -> Result<String, String>
```

**Purpose:** Return the default database path.

**Logic:**
- Dev environment (pyproject.toml or .git present): `<project_root>/output/analysis.db`
- Installed: `%LOCALAPPDATA%/SpaghettiLab/output/analysis.db`
- Fallback: next to the executable

---

### resolve_video_path

```rust
fn resolve_video_path(db_path: String, replay_id: String) -> Result<String, String>
```

**Purpose:** Resolve a replay's video path to an absolute filesystem path.

**Resolution order:**
1. If stored path is absolute and exists, return as-is
2. Try resolving relative to project root
3. Return the raw path (frontend handles "not found")

---

## Internal Helpers

### find_project_root()

Locates the project root directory containing `scripts/analyze_replay.py`.

**Search order:**
1. Walk up from `CARGO_MANIFEST_DIR` (dev builds)
2. Walk up from executable directory (dev/portable)
3. `<exe_dir>/resources/` (NSIS installer)
4. `<exe_dir>` itself

### default_output_paths()

Returns `(output_dir, db_path)`:
- Dev: `<project_root>/output/` and `<project_root>/output/analysis.db`
- Installed: `%LOCALAPPDATA%/SpaghettiLab/output/` and `analysis.db` within

### ensure_winner_overrides_table(conn)

Creates the `winner_overrides` table if it doesn't exist. Called before any winner override operations.

### ensure_notes_table(conn)

Creates the `notes` table if it doesn't exist. Called before any note operations.

### resolve_video_for_replay(conn, replay_id)

Shared helper to resolve video path. Returns `PathBuf` or error. Used by `reanalyze_replay` and `export_spag`.

---

## App Setup

The `run()` function configures the Tauri application:

**Plugins:**
- `tauri_plugin_dialog`: File picker dialogs
- `tauri_plugin_shell`: Shell command execution
- `tauri_plugin_updater`: Auto-update from GitHub releases
- `tauri_plugin_process`: App relaunch after update

**Setup hook:**
- Checks command-line arguments for `.spag` file association
- If launched with a `.spag` argument, emits `open-spag-file` event after 500ms delay (waits for frontend to be ready)

**All registered commands:**
```rust
get_replays, get_frame_data, get_damage_events, get_rounds,
get_match_stats, get_highlights, get_default_db_path,
analyze_video, reanalyze_replay, reanalyze_all,
export_clip, resolve_video_path, extract_preview_frame,
scan_vod, cut_vod_sets,
export_strive_replays, cancel_strive_export,
get_notes, add_note, update_note, delete_note,
export_spag, open_spag, save_spag, set_round_winner,
```
