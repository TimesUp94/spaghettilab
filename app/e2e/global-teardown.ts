// Stops the app instance started by the global setup (and its Python / ffmpeg children).
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { PID_FILE } from "./env";

export default async function globalTeardown() {
  if (process.env.E2E_KEEP_APP) return;
  if (!fs.existsSync(PID_FILE)) return;
  const pid = fs.readFileSync(PID_FILE, "utf8").trim();
  spawnSync("taskkill", ["/PID", pid, "/T", "/F"], { stdio: "ignore" });
  fs.rmSync(PID_FILE, { force: true });
}
