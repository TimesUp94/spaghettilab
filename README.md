# Spaghetti Lab

Replay analysis tool for Guilty Gear Strive. Extracts per-frame game state from recorded matches using computer vision, then presents the data in an interactive desktop app.

Built for the Spaghetti Showdown tournament series.

![Preview](https://img.shields.io/badge/status-experimental-orange)

## What it does

- **Analyzes replay videos**: Reads health bars, timer, tension gauges, and round counters from GGS footage using OpenCV
- **Detects rounds and matches**: Groups frames into rounds, determines winners via KO detection, and organizes rounds into first-to-2 matches
- **Interactive dashboard**: Tauri desktop app with health timelines, match breakdowns, damage logs, highlight detection, and video playback synced to analysis data
- **VOD splitter**: Select a long tournament VOD, adjust detection regions for the overlay layout, scan for gameplay segments, and cut individual sets with ffmpeg
- **Clip export**: Export any round or highlight as a standalone video file

## How it works

### Computer vision pipeline (Python)

The analysis pipeline runs per-frame extractors over the video:

1. **Scene detection** determines if the current frame is actual gameplay (vs menus, transitions, or a smaller game window) by checking tension bar and timer regions
2. **Health bar extraction** uses a brightness-based column scan. The empty portion of the bar is consistently dark regardless of character, so we threshold on brightness and measure the longest continuous filled run. A locked band activation system prevents false readings before the HUD stabilizes
3. **Timer OCR** reads the round timer digits via template matching
4. **Round counter** tracks the heart/cross indicators near health bars

Frame data is smoothed with a rolling median and stored to Parquet. An aggregator detects damage events from health deltas and writes everything to SQLite.

### Round detection (Rust)

The Tauri backend re-analyzes the stored frame data to detect rounds:

- Timer jumps (from <=70 to >=88, validated by reaching >=93) mark round starts
- HP resets (both players returning to full within a 90-frame window) confirm transitions
- KO detection counts frames where one player's HP drops below 0.15
- Game breaks (>5s gaps between gameplay) split sets into individual games
- Edge cases like wallbreak animations and long rounds are handled with specific filters

### Desktop app (Tauri + React)

The frontend displays:

- Sidebar with all analyzed replays
- Video player with round/damage event markers on the timeline
- Health timeline chart (31-frame rolling median smoothed)
- Match overview with per-match and per-game stats
- Tabbed panels for matches, rounds, highlights, and damage logs
- VOD splitter with interactive ROI picker for different overlay layouts

### VOD splitter

For splitting long tournament recordings into individual sets:

1. Pick a VOD file
2. Adjust the detection regions (tension bars, timer, banner) on a preview frame to match the overlay
3. Scan the video for gameplay segments using tension bar and timer activity
4. Banner comparison groups segments into sets (detecting when the overlay changes between sets)
5. Select which sets to keep and cut them with ffmpeg stream copy (no re-encoding)

## Project structure

```
replanal/
  replanal/           Python CV package (extractors, pipeline, aggregator, storage)
  scripts/            CLI entry points (analyze_replay.py, split_vod.py)
  config/             ROI coordinates and thresholds (default.yaml)
  app/                Tauri desktop application
    src/              React frontend
    src-tauri/        Rust backend
  data/               Video files (not in repo)
  output/             Analysis results (not in repo)
```

## Requirements

- Python 3.10+ with: `opencv-python`, `numpy`, `pyyaml`, `pyarrow`
- ffmpeg on PATH
- Node.js 18+
- Rust toolchain (for building from source)

## Running from source

```bash
# Install Python deps
pip install opencv-python numpy pyyaml pyarrow

# Install frontend deps
cd app && npm install

# Run the dev server
npx tauri dev
```

## Building

```bash
cd app && npx tauri build --bundles nsis
```

The installer will be in `app/src-tauri/target/release/bundle/nsis/`.

## Experimental notice

This project is experimental. It was built quickly with heavy use of LLM-assisted coding (Claude) due to time constraints around tournament schedules. The code works but has not been through a proper optimization or cleanup pass yet.

Known rough edges:

- Health bar detection is tuned for 1920x1080 GGS footage with the Spaghetti Showdown overlay. Other resolutions or overlays will need ROI recalibration
- The Python CV pipeline is not optimized for speed. Processing a 40-second clip takes a few minutes
- Round detection heuristics work well for standard matches but may misfire on unusual situations (e.g. double KO, timeout wins)
- The VOD splitter assumes a consistent banner/overlay that changes between sets

A hands-on optimization pass will happen when tournament scheduling allows. Contributions and bug reports welcome.

## Version scheme

`YYYY.M.DD-preview.N` - date-based versioning. The `preview` tag means this is early software under active development.

## License

Not yet specified. Contact the maintainer for usage questions.
