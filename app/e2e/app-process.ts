// Starts and stops the app instance under test. Used by the global setup and teardown, and by specs that
// need a different environment or a restart (`connection.restart`). Data dir and WebView2 profile stay in
// e2e/.run across restarts, like a real app's would.
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { CDP_PORT, DATA_DIR, EXE, LOCAL_EXE, KEYRING_SERVICE, PID_FILE, RUN_DIR, mockUrl } from "./env";

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

async function waitForCdpGone(timeoutMs: number) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    try {
      await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`);
    } catch {
      return;
    }
    await new Promise((r) => setTimeout(r, 200));
  }
}

export async function startApp(extraEnv: Record<string, string> = {}) {
  const env: NodeJS.ProcessEnv = { ...process.env };
  // A developer's own settings must not leak in
  delete env.SPAGHETTI_LAB_MODELS;
  delete env.GGST_MODEL_DIR;
  // The default build is what we publish; the local-models build only where a spec asks for it. A real
  // model run also needs the models allowed (16-lab-local-real).
  const { E2E_LOCAL_BUILD, E2E_REAL_MODELS, ...specEnv } = extraEnv;
  const log = fs.openSync(path.join(RUN_DIR, "app.log"), "a");
  const child = spawn(E2E_LOCAL_BUILD ? LOCAL_EXE : EXE, [], {
    cwd: RUN_DIR,
    detached: true,
    stdio: ["ignore", log, log],
    env: {
      ...env,
      SPAGHETTI_LAB_DATA_DIR: DATA_DIR,
      SPAGHETTI_LAB_NO_GAME: "1",
      // the lab specs load a fixture result: never run the research models (GPU) in a test
      ...(E2E_REAL_MODELS ? {} : { SPAGHETTI_LAB_NO_MODELS: "1" }),
      // every server is the mock; anything else is refused, and no browser opens (test mode)
      SPAGHETTI_LAB_API_URL: mockUrl(),
      SPAGHETTI_LAB_WEB_URL: mockUrl(),
      SPAGHETTI_LAB_NO_PROD: "1",
      SPAGHETTI_LAB_TEST_BROWSER: "1",
      SPAGHETTI_LAB_TEST_POLL_MS: "150",
      SPAGHETTI_LAB_KEYRING_SERVICE: KEYRING_SERVICE,
      WEBVIEW2_USER_DATA_FOLDER: path.join(RUN_DIR, "webview"),
      WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${CDP_PORT}`,
      ...specEnv,
    },
  });
  child.unref();
  fs.writeFileSync(PID_FILE, String(child.pid));
  await waitForCdp(60_000);
}

export async function stopApp() {
  if (!fs.existsSync(PID_FILE)) return;
  const pid = fs.readFileSync(PID_FILE, "utf8").trim();
  spawnSync("taskkill", ["/PID", pid, "/T", "/F"], { stdio: "ignore" });
  fs.rmSync(PID_FILE, { force: true });
  await waitForCdpGone(20_000);
}

/** The test service's entries in the Windows Credential Manager (`<account>.<service>`). */
export function clearTestCredentials() {
  for (const account of ["refresh_token", "player"]) {
    spawnSync("cmdkey", [`/delete:${account}.${KEYRING_SERVICE}`], { stdio: "ignore" });
  }
}
