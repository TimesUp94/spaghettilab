# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spaghetti Lab is a fighting game replay analysis tool (primarily Guilty Gear Strive, also Street Fighter 6). A Python CV pipeline extracts per-frame game state (health, tension, timer, round wins) from recorded matches, stores it in SQLite, and a Tauri desktop app (Rust backend + React frontend) visualizes and analyzes the data. A separate VOD splitter uses EasyOCR to detect player names and split tournament recordings into individual sets.

## Build & Run Commands

```bash
# Python dependencies (editable install; extras: ocr, viz, dev)
pip install -e ".[ocr,dev]"

# Python tests (pytest, testpaths = tests/)
pytest
pytest -k test_reads_health_from_ggs_frame   # single test

# Re-run health cleaning / damage events / combos from a frames parquet, no video
# decode (after changing replanal/health_clean.py, aggregator.py or combo grouping)
python scripts/reclean_replay.py output/<id>/frames/<id>_frames.parquet --video-path <mp4> --duration-ms <ms> --frame-count <n> --output <dir>

# Frontend dev (from app/)
cd app && npm install
npx tauri dev          # Dev server on port 1420 + Rust backend

# Or use helper scripts from repo root:
./dev.sh               # Git Bash — kills zombies, frees port, launches
.\dev.ps1              # PowerShell equivalent

# Production build (from app/)
npx tauri build --bundles nsis

# Rust-only build check (from app/src-tauri/)
cargo build

# TypeScript type check (from app/)
npx tsc --noEmit

# End-to-end tests of the whole app (from app/; Windows; see app/e2e/README.md). Builds the app,
# runs it isolated (SPAGHETTI_LAB_DATA_DIR = e2e/.run/data, never output/analysis.db;
# SPAGHETTI_LAB_NO_GAME = the Strive export refuses to drive the game) and drives it with Playwright
npm run e2e                                    # everything, ~10 min (real analyses)
npm run e2e:fast                               # without the @slow reanalyse tests
E2E_SKIP_BUILD=1 E2E_REUSE_DATA=1 npx playwright test 03   # iterate on one spec

# Rust tests (from app/src-tauri/)
cargo test -- --nocapture                      # --nocapture to see eprintln debug output
cargo test test_slycoops_rounds -- --nocapture # single test
# NOTE: Rust tests are round-detection debug harnesses that open the hardcoded
# C:/Projects/replanal/output/analysis.db and print rounds; they fail without that DB.

# Export saved Strive replays from the running game (Windows; game at the title screen,
# keyboard controls; the tool presses keys in the game window)
python -m replanal.strive_export --out <dir> --count 10    # or --all / --since today; --dry-run
python -m replanal.strive_export probe                     # ffmpeg/capture/encoder + current screen, no keys
python -m replanal.strive_export calibrate --lang en       # reference screens for another UI language

# Release (from repo root)
./release.sh                      # auto-version YYYY.M.DD-preview.N
./release.sh 2026.4.1-preview.1   # explicit version
./release.sh --skip-build         # reuse existing artifacts
./release.sh --skip-push          # don't git push or create GH release
.\release.ps1                     # PowerShell equivalent
```

`.github/workflows/release.yml` (manual `workflow_dispatch`) builds cross-platform bundles: Windows NSIS, macOS dmg (arm64 + x64), Linux deb/AppImage.

## Architecture

See `docs/` for detailed documentation of each subsystem:
- `docs/cv-pipeline.md` — Python CV extractors and frame processing
- `docs/round-detection.md` — Rust round boundary and winner detection algorithm
- `docs/vod-splitter.md` — VOD splitting with OCR-based name detection
- `docs/frontend.md` — React UI components and state management
- `docs/tauri-commands.md` — All Rust/Tauri backend commands
- `docs/strive-export.md` — Exporting saved Strive replays by driving the game (screen recognition, key safety, capture)

### Data Pipeline

```
Video (1080p @30fps)
  → Python CV extractors (replanal/ package)
    → SQLite DB (output/analysis.db) + Parquet files
      → Rust backend reads DB, detects rounds/matches/winners
        → React frontend renders charts and analysis
```

### Three-Layer Stack

**Python (`replanal/` + `scripts/`):** CV pipeline using OpenCV. `scripts/analyze_replay.py` is the main entry point. Extractors in `replanal/extractors/` process health bars (brightness scanning), tension (absolute fill of the fixed gauge width; green below 50 %, pale yellow above; `finalize_tension` drops readings a stream overlay caps), timer (template-matched OCR, two digits or the single warm digit of the last 9 s; `finalize_timer` keeps a single digit only when it continues the countdown; the raw timer/tension readings stay in the parquet so `reclean_replay.py` re-runs both clean-ups), round counters (heart/cross detection), combo counter (template-matched; `derive_combos` groups by HUD state: a combo continues through hidden-HUD supers/wallbreaks and while the defender's red pending segment stays up, never starts after the round ended, must deal damage, and ends when its counter goes away), and scene state. `scripts/split_vod.py` handles VOD splitting: gameplay from the pipeline's timer reader (never stream-overlay colours), sets grouped by the in-game name tags read with EasyOCR and voted per segment (the tags' look as fallback), game counts from the hearts. Names are compared by `name_key` (punctuation/case dropped, OCR look-alikes folded) and each player gets one spelling for the whole VOD (`vod_name_spellings`), so their files are named alike.

**Rust (`app/src-tauri/src/lib.rs`):** Single large file containing all Tauri commands and the round detection algorithm. Key responsibilities:
- Round boundary detection using four signals: timer jumps, HP resets, gap-based resets, and data gaps
- Winner determination: terminal state of the cleaned HP first (a side at 0 loses, else the lower final HP), then the older KO counting, HP clustering, min-HP fallback and post-round tiebreaker
- Uncertain winner detection with manual override (stored in `winner_overrides` table; applied inside `detect_rounds_for_replay` before the match rules, which never flip an overridden round or a single KO, except a near double KO (winner <= 0.03) that the hearts contradict, which is flipped but left uncertain)
- Non-gameplay filtering (no timer for 20+ seconds → NaN-out HP data, keeping a KO in its first 5 s); every round also ends at most 5 s (KO margin) after its last timer reading, so a lobby shorter than 20 s isn't counted. An HP gap over 5 s is a game break unless the round clock carries on across it (`clock_continues`: long hidden-HUD supers)
- Wallbreak detection (distinguishes mid-round stage transitions from KO animations)
- Match grouping (first-to-2 round wins), comeback detection. The match rules also resolve contradictions (`match_sequence_possible`): an impossible match (a new match after 1-1, a third win) is fixed by flipping the one round the hearts disagree with or that is uncertain; confident consistent results are never overridden
- Hearts-based winner override from round counter extraction (raw data, not smoothed). The hearts are game-native HUD (portable), but the extractor misreads lost hearts, so they stay a secondary signal; never use the Spaghetti Showdown overlay's score banner above them, which other users' videos don't have
- All SQLite queries for serving data to the frontend
- Spawning Python subprocesses for analysis and VOD splitting (`require_python()`: `SPAGHETTI_LAB_PYTHON`, the project `.venv`, `<local app data>/SpaghettiLab/venv`, then `py -3` on Windows / Homebrew and python.org `python3` on macOS, then PATH; the first that imports cv2) and ffmpeg (`ffmpeg_bin()`: `SPAGHETTI_FFMPEG`, PATH plus `/opt/homebrew/bin` and `/usr/local/bin`, the Python's `imageio_ffmpeg`, bundled copy incl. the macOS bundle's `Contents/Resources`). Only successes are cached, so installing a missing tool needs no restart
- .spag file export/import (ZIP containing SQLite snapshot)

**React (`app/src/`):** Tauri v2 frontend with React 18 + TypeScript + Tailwind + lucide icons. `App.tsx` holds the state and picks the view (library home, review, analyse, annotate, split, Strive) next to the left rail; the views are in `app/src/views/` and `app/src/components/`, shared UI in `components/ui.tsx`. The review screen fits the 16:9 video into the space beside the results panel (stacked under 1024 px) with one SVG timeline (`ReviewTimeline`) under it. `api.ts` wraps Tauri `invoke()` calls.

### Key Concepts

- **HP scale:** `p1/p2_health_pct` is the filled fraction of the bar (`health.py` measures fill/total): full bar = 1.0, KO = 0. Rust and the UI only clip it to [0,1]; there is no `HP_CEILING` any more (it divided by 0.875 a second time until 2026-10-22; the round thresholds were converted, see `docs/round-detection.md` "HP scale"). Older data (March 2026 builds and their `.spag`/`.spagz` files) really reads a full bar at 0.875: Rust recognises it per replay (`replay_hp_full`: readings piling up at 0.875 instead of 1.0) and divides HP from `frame_data`/`damage_events` by 0.875; no schema change.
- **KO vs sliver (cleaned HP):** a KO ends the round (clock stops, HUD goes for the slash). A bar at/near 0 while the clock still counts down is a live 1-px sliver (invisible at 720p). The cleaner writes exactly 0 only for the KO'd side; a live side is kept at >= 0.008 (`ALIVE_MIN`), and the winner is frozen from the KO (post-KO wipe / cinematic / HUD fade-in readings are ignored). See `docs/cv-pipeline.md`, "A KO ends the round".
- **Wallbreaks:** GGS stage transitions that briefly show both bars at 100% during a camera transition. Must be distinguished from real round resets and post-KO animations.
- **Non-gameplay detection:** If the round timer is absent for 20+ consecutive seconds, the region is classified as non-gameplay (lobby, character select) and HP data is NaN-ed out.
- **Uncertain winners:** When HP signals are ambiguous (combined signal < 0.0875 of the bar), rounds are marked `winner_confident: false`. Users can override via the UI, stored in `winner_overrides` SQLite table.
- **Quad ROIs:** VOD splitter name tag regions use 4-point quads that follow the rotated name text in GGS and leave out the game's "Player 1/2" label. Perspective-warped to rectangles before OCR. Defaults in `split_vod.py`, RoiPicker `DEFAULT_ROIS` and `docs/vod-splitter.md` must agree (a pytest checks the first two). P2's quad runs under the label to x 1795 for long names; check quad changes on both Tappa #4 (1080p) and the new VOD (720p), with and without OCR.
- **`.spag` / `.spagz` files:** `.spag` is a ZIP with an SQLite snapshot (plus video); `.spagz` is the lightweight variant (analysis DB only, no video — stores `video_url` so YouTube/Twitch sessions re-embed on reopen). Both registered as file associations.
- **Annotate mode:** Lightweight session from a YouTube/Twitch URL (`create_quick_session`, `AnnotateSetup`/`AnnotateView`) — embeds the video via IFrame API for timestamped notes, drawings and the review questionnaire, without running the CV pipeline. `VideoPlayer` abstracts playback over both `<video>` and embeds.
- **Strive replay export:** `replanal/strive_export/` drives the running game (title → Collection → Replay → Saved replays) and records each replay with FFmpeg (`gfxcapture` from FFmpeg 8+, else `ddagrab`). Keys are posted to the game window as WM_KEYDOWN/WM_KEYUP, so it works in the background and the PC stays usable (`--foreground`: the old SendInput path, game brought to the front). A key is only pressed on a recognised screen and from that screen's allow-list (`exporter.key_allowed`); ranked matchmaking and "Delete replay" must stay unreachable, and `tests/test_strive_export.py` checks this. Screen checks are language-independent (layout, colours, icons) plus per-language text checks against `refs/<lang>/` (Italian ships; `calibrate` adds others). The app runs it through `export_strive_replays` / `cancel_strive_export` (`StriveExportView`).
- **Questionnaire:** Templates in `app/src/lib/questionnaireTemplate.ts`; answers keyed by `(replay_id, question_key)` with template-id-prefixed keys (e.g. `ggst.roundstart.main_options`).

### Database Schema

SQLite with tables: `replays`, `frame_data` (per-frame health/tension/timer/rounds), `damage_events`, `notes` (user annotations), `drawings` (timestamped video drawings), `winner_overrides` (manual winner corrections), `questionnaire_answers`.

Schema is versioned via `PRAGMA user_version` (`SCHEMA_VERSION` + `ensure_schema_version()` in `lib.rs`, run on open_spag/open_spagz). When changing the schema: bump `SCHEMA_VERSION`, add a migration, and update **both** table-copy paths used for export (`export_spag` and `copy_replay_tables_to_dst`) — missing columns there has broken exports before. The schema must stay in lockstep with the web app (FGCScoreboard `app/components/spaghetti-lab`) so `.spag`/`.spagz` files round-trip between both.

## Configuration

ROI coordinates for the CV pipeline are in `config/default.yaml` (tuned for 1920x1080 GGST). Every ROI is on the game's own HUD; never read a stream overlay (score banner, logos), which is set by hand and may be stale, different or absent. Changes are checked with the overlay blacked out (`output/blackout/`). Use `scripts/calibrate_roi.py` for other layouts.

## Release Process

Releases use `release.sh` which bumps version in three files (tauri.conf.json, Cargo.toml, package.json), builds a signed NSIS installer, generates `latest.json` for the auto-updater, commits, pushes, and creates a GitHub release. The updater signing key is at `~/.tauri/spaghettilab.key`.

## Port Conflicts

Dev server uses port 1420. If it's occupied from a previous crash:
```bash
taskkill //F //IM "spaghetti-lab.exe"
```
Or use `./dev.sh` / `.\dev.ps1` which handle cleanup automatically.
