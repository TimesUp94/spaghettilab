// Manual check of the app's real token and upload code against the DEV server (never production), run
// by hand: node app/e2e/live/dev-auth-check.mjs. No browser: the test account gets a login code directly,
// then the ignored Rust tests lab_live_dev_check and lab_live_dev_upload use it. They sign in, read the
// quota, refresh, upload the small e2e clip, delete it, sign out. They never start an analysis, so nothing
// is charged and no AI summary is paid for.
//
// The account file, the web JWT and every token stay out of the output: only status codes and field names
// are printed. The web JWT is used for these two calls and never stored.
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const API = "https://dev.backend.spaghettishowdown.it";
const ACCOUNT = path.join(os.homedir(), ".config", "spaghetti-lab-dev", "test-account.txt");
const TAURI_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "src-tauri");

function account() {
  const raw = fs.readFileSync(ACCOUNT, "utf-8");
  let text = raw;
  try {
    text = JSON.parse(raw).body ?? raw;
  } catch {
    /* plain text */
  }
  const field = (name) => new RegExp(`^\\s*${name}:\\s*(\\S+)`, "im").exec(text)?.[1];
  const email = field("email");
  const password = field("password");
  if (!email || !password) throw new Error("The account file doesn't have the expected fields");
  return { email, password };
}

async function post(url, body, bearer) {
  const r = await fetch(`${API}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(bearer ? { Authorization: `Bearer ${bearer}` } : {}) },
    body: JSON.stringify(body),
  });
  const json = await r.json().catch(() => ({}));
  console.log(`${url}: ${r.status} fields=${Object.keys(json).join(",")}`);
  return { status: r.status, json };
}

const { email, password } = account();
const login = await post("/playeruser_login", { playerUserEmail: email, password });
const jwt = login.json?.user?.token;
if (!jwt) process.exit(1);

async function freshCode() {
  const verifier = crypto.randomBytes(32).toString("base64url");
  const challenge = crypto.createHash("sha256").update(verifier).digest("base64url");
  const a = await post("/api/desktop-auth/authorize", { code_challenge: challenge, code_challenge_method: "S256", redirect_port: 54321 }, jwt);
  if (!a.json?.code) process.exit(1);
  return { code: a.json.code, verifier };
}

let failed = false;
for (const test of ["lab_live_dev_check", "lab_live_dev_upload"]) {
  const { code, verifier } = await freshCode();
  console.log(`--- ${test}`);
  const r = spawnSync("cargo", ["test", "--lib", test, "--", "--ignored", "--nocapture", "--test-threads=1"], {
    cwd: TAURI_DIR,
    stdio: "inherit",
    shell: true,
    env: {
      ...process.env,
      CARGO_BUILD_JOBS: "4",
      PATH: `${path.join(os.homedir(), ".cargo", "bin")}${path.delimiter}${process.env.PATH}`,
      LAB_LIVE_API: API,
      LAB_LIVE_CODE: code,
      LAB_LIVE_VERIFIER: verifier,
    },
  });
  failed ||= r.status !== 0;
}
process.exit(failed ? 1 : 0);
