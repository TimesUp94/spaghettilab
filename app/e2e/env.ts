// Paths and settings shared by the global setup, the fixtures and the specs.
import path from "node:path";
import { fileURLToPath } from "node:url";

export const APP_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const REPO_DIR = path.resolve(APP_DIR, "..");
export const EXE = path.join(APP_DIR, "src-tauri", "target", "debug", "spaghetti-lab.exe");

/** DevTools port of the app's WebView2 (only the e2e instance opens it). */
export const CDP_PORT = Number(process.env.E2E_CDP_PORT ?? 9333);

/** Fixture clips, cached between runs (gitignored). */
export const CACHE_DIR = path.join(APP_DIR, "e2e", ".cache");
/** Everything one run writes: the app's data dir, WebView2 profile, exported files. Wiped at start. */
export const RUN_DIR = path.join(APP_DIR, "e2e", ".run");
export const DATA_DIR = path.join(RUN_DIR, "data");
export const OUTPUT_DIR = path.join(DATA_DIR, "SpaghettiLab", "output");
export const DEFAULT_DB = path.join(OUTPUT_DIR, "analysis.db");
export const FILES_DIR = path.join(RUN_DIR, "files");
export const PID_FILE = path.join(RUN_DIR, "app.pid");

/**
 * The match video every spec works on: the first two games of a set (720p, 192 s), cut from
 * E2E_SOURCE_VIDEO. The expectations below belong to this clip.
 */
export const SOURCE_VIDEO =
  process.env.E2E_SOURCE_VIDEO ?? path.join(REPO_DIR, "data", "sets_v2", "05_UF_Jason_vs_EneroNeJR.mp4");
export const CLIP_SECONDS = 192;
export const FIXTURE_VIDEO = path.join(CACHE_DIR, "e2e_set.mp4");
export const REPLAY_ID = "e2e_set";

/** What the analysis reads from the fixture clip. */
export const EXPECTED = {
  rounds: 4,
  games: 2,
  /** Every round goes to P1: two games won 2–0. */
  roundWinners: ["P1", "P1", "P1", "P1"],
  comebacks: 1,
  closeRounds: 2,
  damageEvents: 57,
  highlights: 8,
  durationLabel: "3:12",
};
