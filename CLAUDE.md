# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spaghetti Lab is a fighting game replay analysis tool (primarily Guilty Gear Strive, also Street Fighter 6). A Python CV pipeline extracts per-frame game state (health, tension, timer, round wins) from recorded matches, stores it in SQLite, and a Tauri desktop app (Rust backend + React frontend) visualizes and analyzes the data. A separate VOD splitter uses EasyOCR to detect player names and split tournament recordings into individual sets.

## Build & Run Commands

```bash
# Python dependencies
pip install opencv-python numpy pyyaml pyarrow easyocr

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

# Rust tests (from app/src-tauri/)
cargo test -- --nocapture   # --nocapture to see eprintln debug output

# Release (from repo root)
./release.sh                      # auto-version YYYY.M.DD-preview.N
./release.sh 2026.4.1-preview.1   # explicit version
./release.sh --skip-build         # reuse existing artifacts
./release.sh --skip-push          # don't git push or create GH release
```

## Architecture

See `docs/` for detailed documentation of each subsystem:
- `docs/cv-pipeline.md` — Python CV extractors and frame processing
- `docs/round-detection.md` — Rust round boundary and winner detection algorithm
- `docs/vod-splitter.md` — VOD splitting with OCR-based name detection
- `docs/frontend.md` — React UI components and state management
- `docs/tauri-commands.md` — All Rust/Tauri backend commands

### Data Pipeline

```
Video (1080p @30fps)
  → Python CV extractors (replanal/ package)
    → SQLite DB (output/analysis.db) + Parquet files
      → Rust backend reads DB, detects rounds/matches/winners
        → React frontend renders charts and analysis
```

### Three-Layer Stack

**Python (`replanal/` + `scripts/`):** CV pipeline using OpenCV. `scripts/analyze_replay.py` is the main entry point. Extractors in `replanal/extractors/` process health bars (brightness scanning), tension (green pixel scanning), timer (template-matched OCR), round counters (heart/cross detection), and scene state. `scripts/split_vod.py` handles VOD splitting with EasyOCR for player name detection.

**Rust (`app/src-tauri/src/lib.rs`):** Single large file containing all Tauri commands and the round detection algorithm. Key responsibilities:
- Round boundary detection using four signals: timer jumps, HP resets, gap-based resets, and data gaps
- Winner determination via KO counting, HP clustering, min-HP fallback, and post-round tiebreaker
- Uncertain winner detection with manual override (stored in `winner_overrides` table)
- Non-gameplay filtering (no timer for 20+ seconds → NaN-out HP data)
- Wallbreak detection (distinguishes mid-round stage transitions from KO animations)
- Match grouping (first-to-2 round wins), comeback detection
- Hearts-based winner override from round counter extraction (raw data, not smoothed)
- All SQLite queries for serving data to the frontend
- Spawning Python subprocesses for analysis and VOD splitting
- .spag file export/import (ZIP containing SQLite snapshot)

**React (`app/src/`):** Tauri v2 frontend with React 18 + TypeScript + Tailwind + Recharts. `App.tsx` is the main component managing all state. `api.ts` wraps Tauri `invoke()` calls. Components in `app/src/components/`.

### Key Concepts

- **HP Normalization:** Raw health values divided by `HP_CEILING = 0.875`, clipped to [0,1]. Raw 0.875 = full health.
- **Wallbreaks:** GGS stage transitions that briefly show both bars at 100% during a camera transition. Must be distinguished from real round resets and post-KO animations.
- **Non-gameplay detection:** If the round timer is absent for 20+ consecutive seconds, the region is classified as non-gameplay (lobby, character select) and HP data is NaN-ed out.
- **Uncertain winners:** When HP signals are ambiguous (combined signal < 0.10), rounds are marked `winner_confident: false`. Users can override via the UI, stored in `winner_overrides` SQLite table.
- **Quad ROIs:** VOD splitter name tag regions use 4-point quads (parallelograms) for skewed text in GGS. Perspective-warped to rectangles before OCR.
- **`.spag` files:** ZIP archives containing an SQLite snapshot. Registered as file association.

### Database Schema

SQLite with tables: `replays`, `frame_data` (per-frame health/tension/timer/rounds), `damage_events`, `notes` (user annotations), `winner_overrides` (manual winner corrections).

## Configuration

ROI coordinates for the CV pipeline are in `config/default.yaml` (tuned for 1920x1080 + Spaghetti Showdown overlay). Use `scripts/calibrate_roi.py` for other layouts.

## Release Process

Releases use `release.sh` which bumps version in three files (tauri.conf.json, Cargo.toml, package.json), builds a signed NSIS installer, generates `latest.json` for the auto-updater, commits, pushes, and creates a GitHub release. The updater signing key is at `~/.tauri/spaghettilab.key`.

## Port Conflicts

Dev server uses port 1420. If it's occupied from a previous crash:
```bash
taskkill //F //IM "spaghetti-lab.exe"
```
Or use `./dev.sh` / `.\dev.ps1` which handle cleanup automatically.
