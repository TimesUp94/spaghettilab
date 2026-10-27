// Builds the app, prepares the fixture clip, and starts one isolated app instance for the whole run.
//
// Isolation: SPAGHETTI_LAB_DATA_DIR sends the analysis DB, sessions and previews to e2e/.run/data
// (never the checkout's output/analysis.db), WEBVIEW2_USER_DATA_FOLDER gives the WebView its own
// profile (localStorage), and SPAGHETTI_LAB_NO_GAME makes the Strive export refuse to drive the game.
import { execFileSync, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  APP_DIR, CACHE_DIR, CDP_PORT, CLIP_SECONDS, DATA_DIR, EXE, FILES_DIR, FIXTURE_VIDEO, PID_FILE,
  REPO_DIR, RUN_DIR, SOURCE_VIDEO,
} from "./env";

function findFfmpeg(): string {
  if (process.env.SPAGHETTI_FFMPEG) return process.env.SPAGHETTI_FFMPEG;
  const onPath = spawnSync("ffmpeg", ["-version"], { stdio: "ignore" });
  if (onPath.status === 0) return "ffmpeg";
  for (const venv of [".venv", "venv"]) {
    const dir = path.join(REPO_DIR, venv, "Lib", "site-packages", "imageio_ffmpeg", "binaries");
    const exe = fs.existsSync(dir) && fs.readdirSync(dir).find((f) => f.startsWith("ffmpeg") && f.endsWith(".exe"));
    if (exe) return path.join(dir, exe);
  }
  throw new Error("ffmpeg not found: set SPAGHETTI_FFMPEG or put ffmpeg on PATH");
}

function prepareFixture() {
  if (fs.existsSync(FIXTURE_VIDEO)) return;
  if (!fs.existsSync(SOURCE_VIDEO)) {
    throw new Error(
      `Fixture source video not found: ${SOURCE_VIDEO}\n` +
        "Set E2E_SOURCE_VIDEO to the 05_UF_Jason_vs_EneroNeJR set video (or copy a prepared clip to " +
        `${FIXTURE_VIDEO}).`,
    );
  }
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  console.log(`[e2e] cutting the fixture clip from ${SOURCE_VIDEO}`);
  const tmp = FIXTURE_VIDEO + ".part.mp4";
  execFileSync(findFfmpeg(), [
    "-v", "error", "-y", "-ss", "0", "-t", String(CLIP_SECONDS), "-i", SOURCE_VIDEO, "-an",
    "-vf", "scale=1280:720", "-r", "30", "-c:v", "libx264", "-crf", "32", "-preset", "slow",
    "-pix_fmt", "yuv420p", tmp,
  ], { stdio: "inherit" });
  fs.renameSync(tmp, FIXTURE_VIDEO);
}

function build() {
  if (process.env.E2E_SKIP_BUILD) {
    if (!fs.existsSync(EXE)) throw new Error(`E2E_SKIP_BUILD is set but ${EXE} does not exist`);
    return;
  }
  // A debug build with the frontend embedded (a `tauri dev` binary would load the dev server).
  console.log("[e2e] building the app (tauri build --debug --no-bundle)");
  const cargoBin = path.join(process.env.USERPROFILE ?? process.env.HOME ?? "", ".cargo", "bin");
  const r = spawnSync("npx", ["tauri", "build", "--debug", "--no-bundle"], {
    cwd: APP_DIR,
    stdio: "inherit",
    shell: true,
    env: { ...process.env, PATH: `${cargoBin}${path.delimiter}${process.env.PATH}` },
  });
  if (r.status !== 0) throw new Error("tauri build failed");
}

async function waitForCdp(timeoutMs: number) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    try {
      const res = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
      const list = (await res.json()) as { type: string; url: string }[];
      if (list.some((p) => p.type === "page" && p.url.startsWith("http://tauri.localhost"))) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`The app's DevTools port ${CDP_PORT} did not come up`);
}

async function portInUse(): Promise<boolean> {
  try {
    await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`);
    return true;
  } catch {
    return false;
  }
}

export default async function globalSetup() {
  if (process.platform !== "win32") throw new Error("The e2e suite drives the WebView2 app: Windows only");
  if (await portInUse()) {
    throw new Error(`Port ${CDP_PORT} is in use: is an e2e app still running? (set E2E_CDP_PORT to use another)`);
  }
  build();
  prepareFixture();

  if (process.env.E2E_REUSE_DATA) {
    // keep the previous run's data dir (its analysed DB) and start from a fresh profile
    for (const d of ["webview", "files"]) fs.rmSync(path.join(RUN_DIR, d), { recursive: true, force: true });
  } else {
    fs.rmSync(RUN_DIR, { recursive: true, force: true });
  }
  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.mkdirSync(FILES_DIR, { recursive: true });

  const log = fs.openSync(path.join(RUN_DIR, "app.log"), "w");
  const child = spawn(EXE, [], {
    cwd: RUN_DIR,
    detached: true,
    stdio: ["ignore", log, log],
    env: {
      ...process.env,
      SPAGHETTI_LAB_DATA_DIR: DATA_DIR,
      SPAGHETTI_LAB_NO_GAME: "1",
      WEBVIEW2_USER_DATA_FOLDER: path.join(RUN_DIR, "webview"),
      WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${CDP_PORT}`,
    },
  });
  child.unref();
  fs.writeFileSync(PID_FILE, String(child.pid));
  await waitForCdp(60_000);
}
