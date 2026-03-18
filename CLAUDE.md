# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spaghetti Lab is a Guilty Gear Strive (GGS) replay analysis tool. A Python CV pipeline extracts per-frame game state (health, tension, timer, round wins) from recorded matches, stores it in SQLite, and a Tauri desktop app (Rust backend + React frontend) visualizes and analyzes the data.

## Build & Run Commands

```bash
# Python dependencies
pip install opencv-python numpy pyyaml pyarrow

# Frontend dev (from app/)
cd app && npm install
npx tauri dev          # Dev server on port 1420 + Rust backend

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

### Data Pipeline

```
Video (1080p @30fps)
  → Python CV extractors (replanal/ package)
    → SQLite DB (output/analysis.db) + Parquet files
      → Rust backend reads DB, detects rounds/matches/winners
        → React frontend renders charts and analysis
```

### Three-Layer Stack

**Python (`replanal/` + `scripts/`):** CV pipeline using OpenCV. `scripts/analyze_replay.py` is the main entry point. Extractors in `replanal/extractors/` process health bars (brightness scanning), tension (yellow/white pixel scanning), timer (template-matched OCR), round counters (heart/cross detection), and scene state.

**Rust (`app/src-tauri/src/lib.rs`):** Single ~2500-line file containing all Tauri commands and the round detection algorithm. Key responsibilities:
- Round boundary detection using three signals: timer jumps, HP resets, and data gaps
- Winner determination via KO counting, HP clustering, min-HP fallback, and asymmetric data tiebreaker
- Match grouping (first-to-2 round wins)
- Comeback detection (deficit tracking with 91-frame heavy median)
- Hearts-based winner override from round counter extraction
- All SQLite queries for serving data to the frontend
- Spawning Python subprocesses for analysis and VOD splitting

**React (`app/src/`):** Tauri v2 frontend with React 18 + TypeScript + Tailwind + Recharts. `App.tsx` is the main component (~800 lines) managing all state. `api.ts` wraps Tauri `invoke()` calls. Components in `app/src/components/`.

### Key Concepts

- **HP Normalization:** Raw health values divided by `HP_CEILING = 0.875`, clipped to [0,1]. Raw 0.875 = full health.
- **Wallbreaks:** GGS stage transitions that briefly show both bars at 100% during a camera transition. Must be distinguished from real round resets and post-KO animations.
- **Round detection heuristics:** Min round = 450 frames (~15s). Long rounds (>45s) are recursively split. Timer 99→0 with jump from ≤70→≥88 signals new round.
- **`.spag` files:** ZIP archives containing an SQLite snapshot. Registered as file association.

### Database Schema

SQLite with tables: `replays` (replay_id, video_path, duration_ms, frame_count), `frame_data` (per-frame health/tension/timer/rounds), `damage_events` (detected damage with target/amount), `notes` (user annotations with timestamps).

## Configuration

ROI coordinates for the CV pipeline are in `config/default.yaml` (tuned for 1920x1080 + Spaghetti Showdown overlay). Use `scripts/calibrate_roi.py` for other layouts.

## Release Process

Releases use `release.sh` which bumps version in three files (tauri.conf.json, Cargo.toml, package.json), builds a signed NSIS installer, generates `latest.json` for the auto-updater, commits, pushes, and creates a GitHub release. The updater signing key is at `~/.tauri/spaghettilab.key`.

## Port Conflicts

Dev server uses port 1420. If it's occupied from a previous crash, kill the process:
```bash
taskkill //F //IM "spaghetti-lab.exe"
```
