// A test-only Spaghetti Showdown server: the website's login page, desktop tokens, the upload and the opener
// API, as verified against FGCBackend (docs/desktop-login.md gives every status and body). Node http, no
// dependencies. It runs in the Playwright runner's process, on a random loopback port.
//
// Specs drive it through /__mock/*: scenario knobs, quota, sessions, and a log that never holds a token value.
// A real browser isn't involved: in test mode the app only requests the login link, the mock remembers it
// (with a code ready), and the spec calls the app's loopback callback itself.
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { APP_DIR } from "./env";

type Json = Record<string, any>;

export interface Scenario {
  /** on, off (production's 503 on every route), down (the service is down) */
  service: "on" | "off" | "down";
  /** missing: the login isn't deployed (404 page, unknown /api paths answer 401) */
  login: "ok" | "missing";
  /** the next start POSTs, in order, before a real start */
  startQueue: ("analysis_starting" | "quota_exhausted" | "one_at_a_time" | "queue_full" | "too_long" | "plain_429")[];
  upload: "ok" | "too_big" | "storage_full" | "banned" | "html_413";
  /** the answer comes this long after the last byte (the server copies the file to storage first) */
  uploadDelayMs: number;
  /** a job stops advancing when it reaches this step (a stage, or "queued") */
  holdAt: string | null;
  /** a job fails when it reaches this stage */
  failAt: string | null;
  /** the failure's give-back already happened elsewhere: analysis_refunded false */
  refundLost: boolean;
  /** job polls answer 404 */
  job404: boolean;
  result: "ok" | "bad" | "no_transcript";
  fileDelete: "ok" | "fail" | "scope";
  /** the token lookup fails: 503 unavailable on every authenticated route */
  tokenOutage: boolean;
}

const defaults = (): Scenario => ({
  service: "on",
  login: "ok",
  startQueue: [],
  upload: "ok",
  uploadDelayMs: 0,
  holdAt: null,
  failAt: null,
  refundLost: false,
  job404: false,
  result: "ok",
  fileDelete: "ok",
  tokenOutage: false,
});

interface Session {
  access: string;
  refresh: string;
  /** rotated refresh tokens: presenting one ends the session */
  old: string[];
  revoked: boolean;
}

interface Job {
  id: string;
  fileId: string;
  step: number;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  stage: string | null;
  position?: number;
  error: string | null;
  source: "free" | "credit";
  refundPending: boolean;
  created: string;
}

/** One job's life, one step per poll. */
const STEPS: { status: Job["status"]; stage: string | null; position?: number }[] = [
  { status: "queued", stage: null, position: 2 },
  { status: "queued", stage: null, position: 1 },
  { status: "running", stage: "download" },
  { status: "running", stage: "hud" },
  { status: "running", stage: "detect" },
  { status: "running", stage: "facts" },
  { status: "running", stage: "summary" },
  { status: "done", stage: "done" },
];

const PLAYER = { player_user_id: "11111111-2222-3333-4444-555555555555", nickname: "E2E Player", avatar: null };

const b64url = (b: Buffer) => b.toString("base64url");
const token = (prefix: string) => `${prefix}${b64url(crypto.randomBytes(32))}`;
const uuid = () => crypto.randomUUID();

class Mock {
  scenario = defaults();
  quota = { free_remaining: 2, free_per_week: 2, credits: 3, week_resets_at: "2026-11-08T23:00:00.000Z" };
  sessions: Session[] = [];
  codes = new Map<string, { challenge: string; player: Json }>();
  jobs = new Map<string, Job>();
  files = new Map<string, { deleted: boolean; name: string; type: string; bytes: number; contentLength: number | null }>();
  lastLogin: { seq: number; port: number; state: string; challenge: string; code: string } | null = null;
  loginSeq = 0;
  log: { method: string; path: string; token: string; status: number; note?: string }[] = [];
  /** a rotated refresh token was presented: the client broke the one-at-a-time rule */
  reuseDetected = 0;
  /** an upload deleted while its job was still alive: the job would fail */
  earlyDeletes = 0;
  peakUpload = 0;

  /** Everything but the sessions, which outlive a test the way the app's sign-in does. */
  reset() {
    this.scenario = defaults();
    this.quota = { free_remaining: 2, free_per_week: 2, credits: 3, week_resets_at: "2026-11-08T23:00:00.000Z" };
    this.jobs.clear();
    this.files.clear();
    this.log = [];
    this.lastLogin = null;
    this.reuseDetected = 0;
    this.earlyDeletes = 0;
  }

  findAccess(t: string) {
    return this.sessions.find((s) => !s.revoked && s.access === t);
  }
}

let mock = new Mock();
let server: http.Server | null = null;

function send(res: http.ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  res.writeHead(status, { "Content-Type": typeof body === "string" ? "text/plain" : "application/json", ...headers });
  res.end(text);
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

function parseJson(text: string): Json {
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

const quotaFields = () => ({ ...mock.quota });

function jobView(j: Job): Json {
  return {
    id: j.id,
    owner: PLAYER.player_user_id,
    created: j.created,
    status: j.status,
    stage: j.stage,
    started: j.status === "queued" ? null : j.created,
    finished: j.status === "done" || j.status === "failed" ? j.created : null,
    error: j.error,
    meta: {},
    ...(j.status === "queued" ? { position: j.position } : {}),
  };
}

function refund(j: Job) {
  if (j.source === "free") mock.quota.free_remaining += 1;
  else mock.quota.credits += 1;
}

/** One poll's worth of progress. */
function advance(j: Job) {
  if (j.status !== "queued" && j.status !== "running") return;
  const here = STEPS[j.step];
  if (mock.scenario.holdAt && (here.stage === mock.scenario.holdAt || (mock.scenario.holdAt === "queued" && here.status === "queued"))) return;
  if (mock.scenario.failAt && here.stage === mock.scenario.failAt) {
    j.status = "failed";
    j.stage = here.stage;
    j.error = "not a readable video";
    refund(j);
    j.refundPending = !mock.scenario.refundLost;
    return;
  }
  j.step = Math.min(j.step + 1, STEPS.length - 1);
  const next = STEPS[j.step];
  j.status = next.status;
  j.stage = next.stage;
  j.position = next.position;
}

function resultBody(): string {
  const doc = JSON.parse(fs.readFileSync(path.join(APP_DIR, "e2e", "fixtures", "lab_result.json"), "utf-8"));
  if (mock.scenario.result === "bad") return JSON.stringify({ openers: [] });
  if (mock.scenario.result === "no_transcript") {
    doc.transcript = null;
    doc.states = null;
    doc.capabilities.transcript = false;
  }
  return JSON.stringify(doc);
}

/** `isAllowedForLabScope` of FGCBackend's desktop-scope middleware. */
function inLabScope(method: string, p: string) {
  const openers = "/api/spaghetti-lab/openers";
  if (method === "GET" && (p === `${openers}/status` || p === `${openers}/quota`)) return true;
  if (method === "POST" && p === openers) return true;
  if (method === "GET" && new RegExp(`^${openers}/[^/]+(/result)?$`).test(p)) return true;
  if (method === "DELETE" && new RegExp(`^${openers}/[^/]+$`).test(p)) return true;
  if (method === "POST" && p === "/api/user-files/upload") return true;
  if (method === "DELETE" && /^\/api\/user-files\/[^/]+$/.test(p)) return true;
  return false;
}

/** The upload: only the first part's headers are kept, the rest is counted and thrown away. */
function readUpload(req: http.IncomingMessage): Promise<{ head: string; bytes: number }> {
  return new Promise((resolve, reject) => {
    let head = Buffer.alloc(0);
    let done = false;
    let bytes = 0;
    req.on("data", (c: Buffer) => {
      mock.peakUpload = Math.max(mock.peakUpload, c.length);
      bytes += c.length;
      if (!done) {
        head = Buffer.concat([head, c]);
        const i = head.indexOf("\r\n\r\n");
        if (i >= 0) {
          head = head.subarray(0, i);
          done = true;
        } else if (head.length > 64 * 1024) {
          done = true;
        }
      }
    });
    req.on("end", () => resolve({ head: head.toString("utf-8"), bytes }));
    req.on("error", reject);
  });
}

async function handleUpload(req: http.IncomingMessage, res: http.ServerResponse) {
  const { head, bytes } = await readUpload(req);
  if (mock.scenario.uploadDelayMs) await new Promise((r) => setTimeout(r, mock.scenario.uploadDelayMs));
  const type = /content-type:\s*([^\r\n]+)/i.exec(head)?.[1]?.trim() ?? "";
  const name = /filename="([^"]*)"/i.exec(head)?.[1] ?? "";
  const field = /name="([^"]*)"/i.exec(head)?.[1] ?? "";
  if (mock.scenario.upload === "too_big") return send(res, 413, { error: "file_too_large", message: "File exceeds the 500MB limit." });
  if (mock.scenario.upload === "html_413") return send(res, 413, "<html><body>413 Request Entity Too Large</body></html>");
  if (mock.scenario.upload === "storage_full") return send(res, 400, { error: "Your storage limit reached (5GB). Delete some files first." });
  if (mock.scenario.upload === "banned") return send(res, 403, { error: "banned_global", message: "This account can't upload." });
  if (field !== "file") return send(res, 400, { error: "No file uploaded" });
  if (!type.startsWith("video/")) return send(res, 400, { error: "invalid_file_type", message: "Only video files" });
  const id = uuid();
  const cl = req.headers["content-length"];
  mock.files.set(id, { deleted: false, name, type, bytes, contentLength: cl ? Number(cl) : null });
  send(res, 200, {
    success: true,
    data: { file_id: id, annotation_file_id: uuid(), original_name: name, file_size: bytes, duration_seconds: 192, compression_status: "skipped" },
  });
}

async function handleStart(req: http.IncomingMessage, res: http.ServerResponse) {
  const body = parseJson(await readBody(req));
  const fileId = body.fileId;
  if (!fileId) return send(res, 400, { error: "fileId is required" });
  const file = mock.files.get(fileId);
  if (!file || file.deleted) return send(res, 404, { error: "File not found" });
  if (!/\.(mp4|mkv|webm|mov)$/i.test(file.name)) return send(res, 415, { detail: "expected a video file (.mkv, .mov, .mp4, .webm)" });
  const forced = mock.scenario.startQueue.shift();
  switch (forced) {
    case "analysis_starting":
      return send(res, 409, { error: "analysis_starting", ...quotaFields() }, { "Retry-After": "1" });
    case "quota_exhausted":
      return send(res, 429, { error: "quota_exhausted", ...quotaFields() }, { "Retry-After": "90000" });
    case "one_at_a_time":
      return send(res, 429, { detail: "you already have an analysis queued or running" }, { "Retry-After": "60" });
    case "queue_full":
      return send(res, 503, { detail: "the analysis queue is full, try again later" }, { "Retry-After": "300" });
    case "too_long":
      return send(res, 413, { error: "The video is over 15 min: cut it to one match or set" });
    case "plain_429":
      return send(res, 429, "Too many requests from this IP, please try again later.");
  }
  const existing = [...mock.jobs.values()].find((j) => j.fileId === fileId);
  if (existing) return send(res, 200, { ...jobView(existing), reused: true, ...quotaFields() });
  let source: Job["source"];
  if (mock.quota.free_remaining > 0) {
    source = "free";
    mock.quota.free_remaining -= 1;
  } else if (mock.quota.credits > 0) {
    source = "credit";
    mock.quota.credits -= 1;
  } else {
    return send(res, 429, { error: "quota_exhausted", ...quotaFields() }, { "Retry-After": "90000" });
  }
  const job: Job = { id: uuid(), fileId, step: 0, status: "queued", stage: null, position: 2, error: null, source, refundPending: false, created: new Date().toISOString() };
  mock.jobs.set(job.id, job);
  send(res, 202, { ...jobView(job), analysis_source: source, ...quotaFields() });
}

function handleJob(method: string, jobId: string, res: http.ServerResponse) {
  if (mock.scenario.job404) mock.jobs.delete(jobId); // past the service's 24 h
  const job = mock.jobs.get(jobId);
  if (!job) return send(res, 404, { error: "Analysis not found" });
  if (method === "DELETE") {
    if (job.status === "queued") {
      refund(job);
    } else if (job.status !== "running") {
      return send(res, 409, { detail: "the job has already ended" });
    }
    job.status = "cancelled";
    return send(res, 200, { cancelled: true });
  }
  advance(job);
  const view = jobView(job);
  if (job.status === "failed") {
    view.analysis_refunded = job.refundPending;
    job.refundPending = false;
  }
  send(res, 200, view);
}

function handleFileDelete(fileId: string, res: http.ServerResponse) {
  const file = mock.files.get(fileId);
  if (mock.scenario.fileDelete === "fail") return send(res, 502, "<html>Bad gateway</html>");
  if (mock.scenario.fileDelete === "scope") return send(res, 403, { error: "scope" });
  if (!file || file.deleted) return send(res, 404, { error: "File not found" });
  if ([...mock.jobs.values()].some((j) => j.fileId === fileId && (j.status === "queued" || j.status === "running"))) mock.earlyDeletes++;
  file.deleted = true;
  send(res, 200, { success: true });
}

async function handleToken(req: http.IncomingMessage, res: http.ServerResponse) {
  const body = parseJson(await readBody(req));
  const answer = (s: Session) =>
    send(res, 200, { access_token: s.access, token_type: "Bearer", expires_in: 3600, refresh_token: s.refresh, scope: "lab", player: PLAYER }, { "Cache-Control": "no-store" });
  if (body.grant_type === "authorization_code") {
    const code = mock.codes.get(body.code);
    mock.codes.delete(body.code); // one use, and a wrong verifier burns it
    if (!code || !/^[A-Za-z0-9._~-]{43,128}$/.test(body.code_verifier ?? "")) return send(res, 400, { error: "invalid_grant" });
    if (b64url(crypto.createHash("sha256").update(body.code_verifier).digest()) !== code.challenge) return send(res, 400, { error: "invalid_grant" });
    const s: Session = { access: token("sslab_at_"), refresh: token("sslab_rt_"), old: [], revoked: false };
    mock.sessions.push(s);
    return answer(s);
  }
  if (body.grant_type === "refresh_token") {
    const reused = mock.sessions.find((s) => s.old.includes(body.refresh_token));
    if (reused) {
      mock.reuseDetected++;
      reused.revoked = true;
      return send(res, 400, { error: "invalid_grant" });
    }
    const s = mock.sessions.find((x) => !x.revoked && x.refresh === body.refresh_token);
    if (!s) return send(res, 400, { error: "invalid_grant" });
    s.old.push(s.refresh);
    s.refresh = token("sslab_rt_");
    s.access = token("sslab_at_"); // the old access token stops working at once
    return answer(s);
  }
  send(res, 400, { error: "unsupported_grant_type" });
}

async function handleControl(req: http.IncomingMessage, res: http.ServerResponse, p: string) {
  const body = parseJson(await readBody(req));
  switch (p) {
    case "/__mock/reset":
      mock.reset();
      return send(res, 200, { ok: true });
    case "/__mock/scenario":
      Object.assign(mock.scenario, body);
      return send(res, 200, { ok: true });
    case "/__mock/quota":
      Object.assign(mock.quota, body);
      return send(res, 200, { ok: true });
    case "/__mock/expire-access":
      for (const s of mock.sessions) s.access = token("sslab_at_");
      return send(res, 200, { ok: true });
    case "/__mock/revoke-all":
      for (const s of mock.sessions) s.revoked = true;
      return send(res, 200, { ok: true });
    case "/__mock/login":
      return send(res, 200, mock.lastLogin ?? {});
    case "/__mock/log":
      return send(res, 200, mock.log);
    case "/__mock/clear-log":
      mock.log = [];
      return send(res, 200, { ok: true });
    case "/__mock/state":
      return send(res, 200, {
        quota: mock.quota,
        files: [...mock.files.entries()].map(([id, f]) => ({ id, ...f })),
        jobs: [...mock.jobs.values()].map((j) => ({ id: j.id, status: j.status, stage: j.stage, source: j.source })),
        sessions: mock.sessions.filter((s) => !s.revoked).length,
        reuseDetected: mock.reuseDetected,
        earlyDeletes: mock.earlyDeletes,
      });
  }
  send(res, 404, { error: "unknown control" });
}

async function handle(req: http.IncomingMessage, res: http.ServerResponse) {
  const url = new URL(req.url ?? "/", "http://127.0.0.1");
  const p = url.pathname;
  const method = req.method ?? "GET";
  if (p.startsWith("/__mock/")) return handleControl(req, res, p);

  const auth = req.headers.authorization ?? "";
  const bearer = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  const tokenKind = bearer.startsWith("sslab_at_") ? "access" : bearer ? "other" : "none";
  const entry: Mock["log"][number] = { method, path: p, token: tokenKind, status: 0 };
  mock.log.push(entry);
  res.on("finish", () => (entry.status = res.statusCode));

  // The website
  if (p === "/desktop-login") {
    if (mock.scenario.login === "missing") return send(res, 404, "<html>Not found</html>");
    if (url.searchParams.size === 0) return send(res, 200, "<html>Desktop login</html>"); // the app's look before opening a browser
    const port = url.searchParams.get("redirect_port") ?? "";
    const state = url.searchParams.get("state") ?? "";
    const challenge = url.searchParams.get("code_challenge") ?? "";
    // desktopLogin.js: an invalid link shows a message and never redirects
    const valid =
      /^\d{4,5}$/.test(port) && Number(port) >= 1024 && Number(port) <= 65535 && state.length >= 1 && state.length <= 512 &&
      /^[A-Za-z0-9_-]{43}$/.test(challenge) && url.searchParams.get("code_challenge_method") === "S256";
    if (!valid) {
      entry.note = "invalid login link";
      return send(res, 200, "<html>This link is not valid</html>");
    }
    const code = b64url(crypto.randomBytes(24));
    mock.codes.set(code, { challenge, player: PLAYER });
    mock.lastLogin = { seq: ++mock.loginSeq, port: Number(port), state, challenge, code };
    return send(res, 200, "<html>Allow Spaghetti Lab on this PC?</html>");
  }
  if (p === "/support" || p === "/terms" || p === "/privacy") return send(res, 200, "<html>page</html>");

  // Login API
  if (p === "/api/desktop-auth/token" && method === "POST") {
    if (mock.scenario.login === "missing") {
      await readBody(req);
      return send(res, 401, { error: "Authentication required", code: "credentials_required" });
    }
    entry.token = "refresh-or-code";
    return handleToken(req, res);
  }
  if (p === "/api/desktop-auth/revoke" && method === "POST") {
    const body = parseJson(await readBody(req));
    entry.token = String(body.refresh_token ?? "").startsWith("sslab_rt_") ? "refresh" : "none";
    const s = mock.sessions.find((x) => x.refresh === body.refresh_token);
    if (s) s.revoked = true;
    return send(res, 200, { ok: true });
  }

  if (!p.startsWith("/api/")) return send(res, 404, "<html>Not found</html>");

  // desktop-scope.js runs before every route, /status included
  if (bearer) {
    if (mock.scenario.tokenOutage) {
      req.resume();
      return send(res, 503, { error: "unavailable" });
    }
    if (bearer.startsWith("sslab_at_") && !mock.findAccess(bearer)) {
      req.resume();
      return send(res, 401, { error: "invalid_token" });
    }
    if (!inLabScope(method, p)) {
      req.resume();
      return send(res, 403, { error: "scope" });
    }
  }

  const openers = "/api/spaghetti-lab/openers";
  if (p.startsWith(openers) && mock.scenario.service === "off") {
    req.resume();
    return send(res, 503, { error: "Opener analysis is not available" });
  }
  if (p === `${openers}/status` && method === "GET") {
    if (mock.scenario.service === "down") return send(res, 503, { available: false });
    return send(res, 200, {
      available: true,
      running: 0,
      queued: 0,
      max_concurrent: 1,
      max_queue: 8,
      capabilities: { openers: true, transcript: false, facts: true, ai_summary: true },
      free_per_week: 2,
    });
  }
  if (!bearer) {
    req.resume();
    return send(res, 401, { error: "Authentication required", code: "credentials_required" });
  }
  if (p === `${openers}/quota` && method === "GET") return send(res, 200, quotaFields());
  if (p === "/api/user-files/upload" && method === "POST") return handleUpload(req, res);
  if (p === openers && method === "POST") return handleStart(req, res);
  const result = new RegExp(`^${openers}/([^/]+)/result$`).exec(p);
  if (result && method === "GET") {
    const job = mock.jobs.get(result[1]);
    if (!job) return send(res, 404, { error: "Analysis not found" });
    if (job.status !== "done") return send(res, 409, { detail: "the job has no result (yet)" });
    return send(res, 200, parseJson(resultBody()));
  }
  const jobMatch = new RegExp(`^${openers}/([^/]+)$`).exec(p);
  if (jobMatch && (method === "GET" || method === "DELETE")) return handleJob(method, jobMatch[1], res);
  const fileMatch = /^\/api\/user-files\/([^/]+)$/.exec(p);
  if (fileMatch && method === "DELETE") return handleFileDelete(fileMatch[1], res);
  req.resume();
  send(res, 401, { error: "Authentication required", code: "credentials_required" });
}

export async function startMockBackend(): Promise<string> {
  mock = new Mock();
  server = http.createServer((req, res) => {
    handle(req, res).catch(() => {
      if (!res.headersSent) send(res, 500, { error: "mock failed" });
    });
  });
  await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("no port");
  // this module may be loaded twice (setup, teardown): the teardown finds the server here
  (globalThis as any).__labMockServer = server;
  return `http://127.0.0.1:${addr.port}`;
}

export async function stopMockBackend() {
  const s: http.Server | undefined = (globalThis as any).__labMockServer;
  if (!s) return;
  s.closeAllConnections();
  await new Promise((resolve) => s.close(resolve));
}
