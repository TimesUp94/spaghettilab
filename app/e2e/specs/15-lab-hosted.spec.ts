// The Lab tab for a normal user: the sign-in and the hosted analysis through the real Rust code, against the
// mock Spaghetti Showdown server (mock-lab-backend.ts). Only the system browser is simulated: the spec calls
// the app's loopback callback itself, as the website's redirect would. The upload is the small e2e clip,
// stream-copied, so no encode and no model runs. Nothing here can reach a real server (SPAGHETTI_LAB_NO_PROD).
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { test, expect, type App } from "../fixtures";
import { APP_DIR, DATA_DIR, EXE, REPLAY_ID, RUN_DIR, mockUrl } from "../env";

const LAB_DIR = path.join(DATA_DIR, "SpaghettiLab", "lab");
const CACHE = path.join(LAB_DIR, REPLAY_ID, "lab_result.json");
const fixture = () => JSON.parse(fs.readFileSync(path.join(APP_DIR, "e2e", "fixtures", "lab_result.json"), "utf-8"));

function writeCache(doc: unknown | null) {
  fs.rmSync(path.dirname(CACHE), { recursive: true, force: true });
  if (doc === null) return;
  fs.mkdirSync(path.dirname(CACHE), { recursive: true });
  fs.writeFileSync(CACHE, JSON.stringify(doc));
}

async function control(what: string, body: unknown = {}) {
  const r = await fetch(`${mockUrl()}/__mock/${what}`, { method: "POST", body: JSON.stringify(body) });
  return r.json() as Promise<any>;
}
const scenario = (s: Record<string, unknown>) => control("scenario", s);
const state = () => control("state");
const log = () => control("log") as Promise<{ method: string; path: string; token: string; status: number }[]>;
const loginLink = () => control("login") as Promise<{ seq?: number; port: number; state: string; code: string }>;

async function openLab(app: App) {
  await app.openReview();
  await app.page.getByRole("tab", { name: /^Lab/ }).click();
}

const signInPanel = (app: App) => app.page.getByTestId("lab-sign-in");
const signInButton = (app: App) => app.page.getByRole("button", { name: "Sign in with spaghettishowdown.it" });
const analyse = (app: App) => app.page.getByRole("button", { name: "Analyse this replay" });
const tryAgain = (app: App) => app.page.getByRole("button", { name: "Try again" });
const progress = (app: App) => app.page.getByTestId("lab-progress");
const remoteError = (app: App) => app.page.getByTestId("lab-remote-error");
const quotaText = (app: App) => app.page.getByTestId("lab-quota");

/** The website's answer: what the system browser would be redirected to. */
async function callback(query: string) {
  const link = await loginLink();
  return fetch(`http://127.0.0.1:${link.port}/callback?${query}`);
}

async function startSignIn(app: App): Promise<{ seq?: number; port: number; state: string; code: string }> {
  const before = (await loginLink()).seq ?? 0;
  await signInButton(app).click();
  await expect.poll(async () => (await loginLink()).seq ?? 0, { timeout: 10_000 }).toBeGreaterThan(before);
  return loginLink();
}

async function signIn(app: App) {
  const link = await startSignIn(app);
  const r = await callback(`code=${encodeURIComponent(link.code)}&state=${encodeURIComponent(link.state)}`);
  expect(r.status).toBe(200);
  await expect(app.page.getByTestId("lab-nickname")).toHaveText("E2E Player");
}

async function ensureSignedOut(app: App) {
  await app.invoke("lab_sign_out").catch(() => {});
}

/** Signed in, with the upload notice already accepted. */
async function ready(app: App, opts: { cache?: unknown } = {}) {
  writeCache(opts.cache ?? null);
  await openLab(app);
  if (!(await app.invoke<{ signed_in: boolean }>("lab_account")).signed_in) {
    if (opts.cache) await app.page.getByRole("button", { name: "Run options" }).click();
    await signIn(app);
  }
  await app.page.evaluate(() => localStorage.setItem("spaghettiLab.lab.uploadNotice", "2026-11-04b"));
  await expect(app.page.getByTestId("lab-nickname")).toBeVisible();
}

const jobCalls = (l: { method: string; path: string }[], method: string) =>
  l.filter((c) => c.method === method && /\/api\/spaghetti-lab\/openers\/[^/]+$/.test(c.path) && !/(status|quota)$/.test(c.path));
const uploadsLeft = async () => (await state()).files.filter((f: { deleted: boolean }) => !f.deleted).length;

test.beforeEach(async ({ app }) => {
  await control("reset");
  // A stale saved job of an earlier test would be collected in the background
  writeCache(null);
  await app.page.evaluate(() => localStorage.removeItem("spaghettiLab.lab.uploadNotice"));
});

test.afterEach(async () => {
  // The app must never present a rotated refresh token, nor delete an upload its job still needs
  const s = await state();
  expect(s.reuseDetected).toBe(0);
  expect(s.earlyDeletes).toBe(0);
});

test.afterAll(async ({ connection }) => {
  writeCache(null);
  await connection.page.evaluate(() => (window as any).__TAURI_INTERNALS__.invoke("lab_sign_out")).catch(() => {});
});

test("signed out: the gate says what leaves the PC, links the policies, and offers no way to run", async ({ app }) => {
  await ensureSignedOut(app);
  await openLab(app);
  const panel = signInPanel(app);
  await expect(panel).toContainText("a 720p copy of the video, without audio");
  await expect(panel).toContainText("the app deletes it afterwards");
  await expect(panel).toContainText("Discord");
  await expect(panel).not.toContainText(/e-?mail/i);
  await expect(signInButton(app)).toBeVisible();
  // no local mode and no missing-models error for a normal user
  await expect(app.page.getByRole("group", { name: "Analysis mode" })).toHaveCount(0);
  await expect(app.page.getByRole("group", { name: "Transcript" })).toHaveCount(0);
  await expect(app.page.getByRole("button", { name: "Run lab analysis" })).toHaveCount(0);
  await expect(analyse(app)).toHaveCount(0);

  await panel.getByRole("button", { name: "Terms" }).click();
  await panel.getByRole("button", { name: "Privacy" }).click();
  await expect.poll(async () => (await log()).map((c) => c.path)).toEqual(expect.arrayContaining(["/terms", "/privacy"]));
});

test("signed out: a cached result stays readable, with no call that needs an account", async ({ app }) => {
  await ensureSignedOut(app);
  const doc = fixture();
  writeCache(doc);
  await openLab(app);
  await expect(app.page.getByTestId("lab-openers-count")).toHaveText(`${doc.openers.length} openers`);
  await expect(signInPanel(app)).toHaveCount(0);
  await app.page.getByRole("group", { name: "Lab section" }).getByRole("button", { name: /^Summary/ }).click();
  await expect(app.page.getByText("no AI wrote them")).toBeVisible();
  await expect(app.page.getByText("nothing is sent anywhere")).toHaveCount(0);
  const calls = await log();
  expect(calls.filter((c) => c.token !== "none")).toEqual([]);
  // the way back to analysing goes through the sign-in
  await app.page.getByRole("button", { name: "Run options" }).click();
  await expect(signInPanel(app)).toBeVisible();
  writeCache(null);
});

test("sign-in fails readably when the login isn't deployed", async ({ app }) => {
  await ensureSignedOut(app);
  await scenario({ login: "missing" });
  await openLab(app);
  await signInButton(app).click();
  await expect(app.page.getByTestId("lab-login-error")).toContainText("isn't available yet");
  await expect(app.page.getByRole("status").filter({ hasText: "Waiting for the browser" })).toHaveCount(0);
  await expect(signInButton(app)).toBeVisible();
});

test("sign-in: a wrong state is refused, a cancel is a cancel, stray requests change nothing", async ({ app }) => {
  await ensureSignedOut(app);
  await openLab(app);

  // wrong state: refused by the listener, and the login waits on (anything on this PC can reach the port)
  let link = await startSignIn(app);
  await expect(app.page.getByText("Waiting for the browser")).toBeVisible();
  const refused = await callback(`code=${encodeURIComponent(link.code)}&state=not-the-state`);
  expect(refused.status).toBe(400);
  expect((await state()).sessions).toBe(0);
  await expect(app.page.getByText("Waiting for the browser")).toBeVisible();
  await expect(app.page.getByTestId("lab-login-error")).toHaveCount(0);

  // cancelled on the website
  await callback(`error=access_denied&state=${encodeURIComponent(link.state)}`);
  await expect(app.page.getByTestId("lab-login-error")).toContainText("cancelled in the browser");

  // cancelled in the app, and "open the page again" asks for the same page
  link = await startSignIn(app);
  await app.page.getByRole("button", { name: "Open the page again" }).click();
  await expect.poll(async () => (await loginLink()).seq).toBeGreaterThan(link.seq!);
  await app.page.getByRole("button", { name: "Cancel sign-in" }).click();
  await expect(signInButton(app)).toBeVisible();
  await expect(app.page.getByTestId("lab-login-error")).toHaveCount(0);

  // an unrelated request doesn't use up the single callback
  link = await startSignIn(app);
  const stray = await fetch(`http://127.0.0.1:${link.port}/favicon.ico`);
  expect(stray.status).toBe(404);
  await callback(`code=${encodeURIComponent(link.code)}&state=${encodeURIComponent(link.state)}`);
  await expect(app.page.getByTestId("lab-nickname")).toHaveText("E2E Player");
});

test("signed in: nickname, initial for a missing avatar, the quota line; the webview holds no token", async ({ app }) => {
  await ready(app);
  await expect(quotaText(app)).toContainText("2 free this week, 3 credits");
  await expect(app.page.getByTestId("lab-spend")).toContainText("This will use a free analysis");
  await expect(app.page.getByTestId("lab-account").locator("img")).toHaveCount(0);
  await expect(app.page.getByTestId("lab-account")).toContainText("E");

  const account = await app.invoke("lab_account");
  expect(JSON.stringify(account)).not.toContain("sslab_");
  expect(Object.keys(account as object).sort()).toEqual(["avatar", "nickname", "session_only", "signed_in", "store_unavailable"]);
  const stored = await app.page.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }));
  expect(stored).not.toContain("sslab_");
  const calls = await app.calls();
  expect(JSON.stringify(calls)).not.toContain("sslab_");

  // /status never carries a token, so a stale one can't turn "not available" into a sign-out
  const status = (await log()).filter((c) => c.path.endsWith("/openers/status"));
  expect(status.length).toBeGreaterThan(0);
  expect(status.every((c) => c.token === "none")).toBe(true);
});

test("the quota line follows the window's focus", async ({ app }) => {
  await ready(app);
  await expect(quotaText(app)).toContainText("2 free this week, 3 credits");
  await control("quota", { free_remaining: 1, credits: 1 });
  await app.page.evaluate(() => {
    // past the 30 s between refreshes
    const real = Date.now.bind(Date);
    Date.now = () => real() + 60_000;
    window.dispatchEvent(new Event("focus"));
    Date.now = real;
  });
  await expect(quotaText(app)).toContainText("1 free this week, 1 credit");
  await expect(quotaText(app)).not.toContainText("credits");
});

test("the whole run: notice once, upload, queue, stages, result, cached, upload deleted", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await app.page.evaluate(() => localStorage.removeItem("spaghettiLab.lab.uploadNotice"));
  await scenario({ holdAt: "queued" });

  // the notice comes first, and "not now" sends nothing
  await analyse(app).click();
  const notice = page.getByTestId("lab-notice");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText(/720p|720p,/);
  await notice.getByRole("button", { name: "Not now" }).click();
  expect((await state()).files).toHaveLength(0);
  await analyse(app).click();
  await notice.getByRole("button", { name: "I understand, analyse" }).click();

  // queued: position, what was charged, the quota line from the POST's answer
  await expect(progress(app)).toBeVisible();
  await expect(page.getByTestId("lab-queue-position")).toHaveText(/Position \d in the queue/, { timeout: 30_000 });
  await expect(page.getByTestId("lab-source")).toHaveText("Using a free analysis.");
  await expect(quotaText(app)).toContainText("1 free this week, 3 credits");
  await expect(progress(app).getByText("Uploading")).toBeVisible();

  // the file: always replay.mp4 (the name reaches the AI prompt), a video part, the whole clip, known length, streamed
  const s = await state();
  expect(s.files).toHaveLength(1);
  expect(s.files[0]).toMatchObject({ name: "replay.mp4", type: "video/mp4", deleted: false });
  expect(s.files[0].bytes).toBeGreaterThan(100_000);
  expect(s.files[0].contentLength).toBe(s.files[0].bytes);
  // the file waits for the job
  expect((await log()).filter((c) => c.method === "DELETE")).toEqual([]);
  // no temp copy once uploaded
  const tmp = path.join(LAB_DIR, ".tmp");
  expect(fs.existsSync(tmp) ? fs.readdirSync(tmp) : []).toEqual([]);

  // a stage
  await scenario({ holdAt: "hud" });
  await expect(progress(app).getByText("Fetching the video")).toBeVisible();
  await expect(page.getByTestId("lab-queue-position")).toHaveCount(0);

  await scenario({ holdAt: null });
  const doc = fixture();
  await expect(page.getByTestId("lab-openers-count")).toHaveText(`${doc.openers.length} openers`, { timeout: 30_000 });
  await expect(progress(app)).toHaveCount(0);
  expect(fs.existsSync(CACHE)).toBe(true);

  // cached, deleted, polled a few times
  await expect.poll(uploadsLeft).toBe(0);
  const calls = await log();
  const del = calls.findIndex((c) => c.method === "DELETE" && c.path.startsWith("/api/user-files/"));
  const res = calls.findIndex((c) => c.path.endsWith("/result"));
  expect(res).toBeGreaterThan(-1);
  expect(del).toBeGreaterThan(res);
  expect(fs.existsSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"))).toBe(false);

  // coming back shows the result and doesn't upload again
  await control("clear-log");
  await openLab(app);
  await expect(page.getByTestId("lab-openers-count")).toHaveText(`${doc.openers.length} openers`);
  expect((await log()).filter((c) => c.path === "/api/user-files/upload")).toEqual([]);

  // again: confirmed, because it spends one; the notice isn't shown twice
  await control("reset");
  await control("quota", { free_remaining: 0, credits: 2 });
  await page.getByRole("button", { name: "Run options" }).click();
  await app.page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await page.getByRole("button", { name: "Analyse again", exact: true }).last().click();
  const confirm = page.getByTestId("lab-again-confirm");
  await expect(confirm).toContainText("spends another analysis");
  await confirm.getByRole("button", { name: "Keep this result" }).click();
  expect((await state()).files).toHaveLength(0);
  await page.getByRole("button", { name: "Analyse again", exact: true }).last().click();
  await confirm.getByRole("button", { name: "Spend an analysis" }).click();
  await expect(page.getByTestId("lab-notice")).toHaveCount(0);
  await expect(progress(app)).toHaveCount(0, { timeout: 30_000 });
  const after = await state();
  expect(after.jobs).toHaveLength(1);
  expect(after.jobs[0].source).toBe("credit");
  expect(after.quota.credits).toBe(1);
});

test("nothing left: stops before the upload and offers the website, no payment in the app", async ({ app }) => {
  await ready(app);
  await control("quota", { free_remaining: 0, credits: 0 });
  await app.page.evaluate(() => {
    const real = Date.now.bind(Date);
    Date.now = () => real() + 60_000;
    window.dispatchEvent(new Event("focus"));
    Date.now = real;
  });
  const exhausted = app.page.getByTestId("lab-exhausted");
  await expect(exhausted).toContainText("No analyses left this week");
  await expect(exhausted).toContainText("Monday 9 November");
  await expect(analyse(app)).toBeDisabled();
  await control("clear-log");
  await exhausted.getByRole("button", { name: "Get credits on the website" }).click();
  await expect.poll(async () => (await log()).map((c) => c.path)).toContain("/support");
  expect((await state()).files).toHaveLength(0);
});

test("quota_exhausted from the server: the /support offer, and the upload is deleted", async ({ app }) => {
  await ready(app);
  await scenario({ startQueue: ["quota_exhausted"] });
  await analyse(app).click();
  await expect(remoteError(app)).toHaveAttribute("data-code", "quota_exhausted", { timeout: 30_000 });
  await expect(remoteError(app)).toContainText("Free analyses come back on");
  await remoteError(app).getByRole("button", { name: "Get credits on the website" }).click();
  await expect.poll(async () => (await log()).map((c) => c.path)).toContain("/support");
  await expect.poll(uploadsLeft).toBe(0);
  expect((await state()).jobs).toHaveLength(0);
});

test("analysis_starting: the same POST again after Retry-After", async ({ app }) => {
  await ready(app);
  await scenario({ startQueue: ["analysis_starting", "analysis_starting"], holdAt: "queued" });
  await analyse(app).click();
  await expect(app.page.getByTestId("lab-queue-position")).toBeVisible({ timeout: 30_000 });
  const posts = (await log()).filter((c) => c.method === "POST" && c.path === "/api/spaghetti-lab/openers");
  expect(posts.map((c) => c.status)).toEqual([409, 409, 202]);
  // one charge, one file
  expect((await state()).quota.free_remaining).toBe(1);
  await app.page.getByRole("button", { name: "Cancel", exact: true }).click();
  await app.page.getByRole("button", { name: "Cancel the analysis" }).click();
  await expect(progress(app)).toHaveCount(0);
});

test("the server's refusals have their own messages, spend nothing and delete the upload", async ({ app }) => {
  test.setTimeout(180_000);
  await ready(app);
  const cases: [Record<string, unknown>, string, string][] = [
    [{ startQueue: ["one_at_a_time"] }, "one_at_a_time", "already have an analysis queued or running"],
    [{ startQueue: ["queue_full"] }, "queue_full", "queue is full"],
    [{ startQueue: ["plain_429"] }, "rate_limited", "Too many requests"],
    [{ startQueue: ["too_long"] }, "too_long", "over 15 min"],
    [{ upload: "too_big" }, "too_big", "cut it to one set"],
    [{ upload: "html_413" }, "too_big", "cut it to one set"],
    [{ upload: "storage_full" }, "storage_full", "storage limit"],
    [{ upload: "banned" }, "banned", "can't upload"],
  ];
  for (const [s, code, text] of cases) {
    await control("reset");
    await scenario(s);
    await analyse(app).click();
    await expect(remoteError(app)).toHaveAttribute("data-code", code, { timeout: 30_000 });
    await expect(remoteError(app)).toContainText(text);
    await expect(app.page.getByTestId("lab-nickname")).toBeVisible();
    await expect.poll(uploadsLeft).toBe(0);
    expect((await state()).quota).toMatchObject({ free_remaining: 2, credits: 3 });
  }
});

test("status 503, both shapes: not available right now, and still signed in", async ({ app }) => {
  await ready(app);
  for (const service of ["off", "down"]) {
    await scenario({ service });
    await openLab(app);
    await expect(app.page.getByTestId("lab-service-down")).toContainText("not available right now");
    await expect(app.page.getByTestId("lab-nickname")).toHaveText("E2E Player");
    await expect(analyse(app)).toBeDisabled();
    await expect(signInPanel(app)).toHaveCount(0);
  }
  // back on: it reads as if nothing happened
  await scenario({ service: "on" });
  await openLab(app);
  await expect(app.page.getByTestId("lab-service-down")).toHaveCount(0);
  await expect(analyse(app)).toBeEnabled();
  await expect(quotaText(app)).toContainText("2 free this week");

  // signed out, the panel says the same and the sign-in still works
  await ensureSignedOut(app);
  await scenario({ service: "off" });
  await openLab(app);
  await expect(signInPanel(app)).toContainText("not available right now");
  await expect(signInButton(app)).toBeEnabled();
  await signIn(app);
});

test("a failed job: the server's reason, the give-back only when it said so, the upload deleted", async ({ app }) => {
  await ready(app);
  await scenario({ failAt: "hud" });
  await analyse(app).click();
  await expect(remoteError(app)).toHaveAttribute("data-code", "job_failed", { timeout: 30_000 });
  await expect(remoteError(app)).toContainText("not a readable video");
  await expect(app.page.getByTestId("lab-refunded")).toHaveText("Your analysis was given back.");
  await expect(quotaText(app)).toContainText("2 free this week, 3 credits");
  await expect.poll(uploadsLeft).toBe(0);

  // analysis_refunded false says nothing either way: no claim, just the fresh quota
  await control("reset");
  await scenario({ failAt: "hud", refundLost: true });
  await analyse(app).click();
  await expect(app.page.getByTestId("lab-progress")).toBeVisible();
  await expect(remoteError(app)).toHaveAttribute("data-code", "job_failed", { timeout: 30_000 });
  await expect(app.page.getByTestId("lab-refunded")).toHaveCount(0);
  await expect(remoteError(app)).not.toContainText("not given back");
  await expect(quotaText(app)).toContainText("2 free this week, 3 credits");
});

test("a job the server no longer has, and a result this app can't read", async ({ app }) => {
  const doc = fixture();
  await ready(app);
  await scenario({ job404: true });
  await analyse(app).click();
  await expect(remoteError(app)).toHaveAttribute("data-code", "expired", { timeout: 30_000 });
  await expect.poll(uploadsLeft).toBe(0);

  // the previous result stays until a new one is valid
  await control("reset");
  writeCache(doc);
  await scenario({ result: "bad" });
  await openLab(app);
  await app.page.getByRole("button", { name: "Run options" }).click();
  await app.page.getByRole("button", { name: "Analyse again", exact: true }).last().click();
  await app.page.getByRole("button", { name: "Spend an analysis" }).click();
  await expect(remoteError(app)).toHaveAttribute("data-code", "bad_result", { timeout: 30_000 });
  await expect(app.page.getByTestId("lab-openers-count")).toHaveText(`${doc.openers.length} openers`);
  expect(JSON.parse(fs.readFileSync(CACHE, "utf-8")).openers).toHaveLength(doc.openers.length);
});

test("a result without a transcript or AI summary doesn't assume them", async ({ app }) => {
  await ready(app);
  await scenario({ result: "no_transcript" });
  await analyse(app).click();
  await expect(app.page.getByTestId("lab-no-transcript")).toBeVisible({ timeout: 30_000 });
  await expect(app.page.getByTestId("lab-no-transcript")).not.toContainText("GPU");
});

test("an AI summary the fact check refused: dev's real answer shape reads as text", async ({ app }) => {
  await ready(app);
  await scenario({ result: "ai_withheld" });
  await analyse(app).click();
  await expect(app.page.getByTestId("lab-no-transcript")).toBeVisible({ timeout: 30_000 });
  await app.page.getByRole("group", { name: "Lab section" }).getByRole("button", { name: /^Summary/ }).click();
  const withheld = app.page.getByTestId("lab-ai-withheld");
  await expect(withheld).toBeVisible();
  await withheld.locator("summary").click();
  await expect(withheld).toContainText(/unsupported_percentage: 57%|note_without_timestamp: note 1/);
  await expect(withheld).not.toContainText("[object Object]");
});

test("cancel while queued: says it's given back, and it is", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await scenario({ holdAt: "queued" });
  await analyse(app).click();
  await expect(page.getByTestId("lab-queue-position")).toBeVisible({ timeout: 30_000 });
  await expect(quotaText(app)).toContainText("1 free this week");

  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  const confirm = page.getByTestId("lab-cancel-confirm");
  await expect(confirm).toContainText("it's given back");
  await confirm.getByRole("button", { name: "Keep it going" }).click();
  await expect(confirm).toHaveCount(0);
  expect(jobCalls(await log(), "DELETE")).toEqual([]);

  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await confirm.getByRole("button", { name: "Cancel the analysis" }).click();
  await expect(progress(app)).toHaveCount(0);
  await expect(page.getByTestId("lab-note")).toHaveText("Analysis cancelled. It was given back.");
  await expect(quotaText(app)).toContainText("2 free this week, 3 credits");
  expect(jobCalls(await log(), "DELETE")).toHaveLength(1);
  await expect.poll(uploadsLeft).toBe(0);
  expect((await state()).jobs[0].status).toBe("cancelled");
  expect(fs.existsSync(CACHE)).toBe(false);
});

test("cancel while running: warns it won't be given back, and it isn't", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await scenario({ holdAt: "hud" });
  await analyse(app).click();
  await expect(progress(app).getByText("Fetching the video")).toBeVisible({ timeout: 30_000 });
  await expect.poll(async () => (await state()).jobs[0]?.stage).toBe("hud");
  await expect(page.getByTestId("lab-queue-position")).toHaveCount(0);

  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  const confirm = page.getByTestId("lab-cancel-confirm");
  await expect(confirm).toContainText("won't be given back");
  await confirm.getByRole("button", { name: "Cancel the analysis" }).click();
  await expect(progress(app)).toHaveCount(0);
  await expect(page.getByTestId("lab-note")).toHaveText("Analysis cancelled. It had already started, so it wasn't given back.");
  await expect(quotaText(app)).toContainText("1 free this week, 3 credits");
  await expect.poll(uploadsLeft).toBe(0);
});

test("the warning comes from the server when Cancel is pressed, not from the last poll", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await scenario({ holdAt: "queued" });
  await analyse(app).click();
  await expect(page.getByTestId("lab-queue-position")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByTestId("lab-cancel-confirm")).toContainText("it's given back");
  await page.getByRole("button", { name: "Keep it going" }).click();
  // The job's turn comes: the same button now warns the other way
  await scenario({ holdAt: "hud" });
  await expect.poll(async () => (await state()).jobs[0]?.stage).toBe("hud");
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByTestId("lab-cancel-confirm")).toContainText("won't be given back");
  await page.getByRole("button", { name: "Keep it going" }).click();
  await scenario({ holdAt: null });
  await expect(page.getByTestId("lab-openers-count")).toBeVisible({ timeout: 30_000 });
});

test("a start answered by a gateway's error page: the same POST finds the job, one charge, the upload stays until the end", async ({ app }) => {
  await ready(app);
  await scenario({ startQueue: ["gateway_after_start"], holdAt: "hud" });
  await analyse(app).click();
  await expect.poll(async () => (await state()).jobs[0]?.stage, { timeout: 30_000 }).toBe("hud");
  await expect(remoteError(app)).toHaveCount(0);
  let s = await state();
  expect(s.jobs).toHaveLength(1);
  expect(s.quota.free_remaining).toBe(1);
  expect(await uploadsLeft()).toBe(1);
  await scenario({ holdAt: null });
  await expect(app.page.getByTestId("lab-openers-count")).toBeVisible({ timeout: 30_000 });
  s = await state();
  expect(s.jobs).toHaveLength(1);
  expect(s.quota.free_remaining).toBe(1);
  await expect.poll(uploadsLeft).toBe(0);
});

test("no answer to the start at all: the upload and the record stay, and trying again asks the server", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await scenario({ startQueue: ["gateway_after_start", "gateway", "gateway", "gateway"], holdAt: "queued" });
  await analyse(app).click();
  await expect(remoteError(app)).toHaveAttribute("data-code", "start_unconfirmed", { timeout: 30_000 });
  expect(await uploadsLeft()).toBe(1);
  const saved = JSON.parse(fs.readFileSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"), "utf-8"));
  expect(saved.post_sent).toBe(true);
  expect(saved.job_id ?? null).toBeNull();

  await scenario({ holdAt: null });
  await tryAgain(app).click();
  await expect(page.getByTestId("lab-openers-count")).toBeVisible({ timeout: 30_000 });
  const s = await state();
  expect(s.files).toHaveLength(1);
  expect(s.jobs).toHaveLength(1);
  expect(s.quota.free_remaining).toBe(1);
  await expect.poll(uploadsLeft).toBe(0);
});

test("a paid result behind a gateway's error page is kept and collected, for free", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await scenario({ result: "gateway" });
  await analyse(app).click();
  await expect(remoteError(app)).toHaveAttribute("data-code", "result_pending", { timeout: 30_000 });
  await expect.poll(uploadsLeft).toBe(0);
  expect(fs.existsSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"))).toBe(true);

  // with nothing left it can still be collected: it's paid for
  await control("quota", { free_remaining: 0, credits: 0 });
  await scenario({ result: "ok" });
  await tryAgain(app).click();
  await expect(page.getByTestId("lab-openers-count")).toBeVisible({ timeout: 30_000 });
  const s = await state();
  expect(s.files).toHaveLength(1);
  expect(s.quota).toMatchObject({ free_remaining: 0, credits: 0 });
  expect(fs.existsSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"))).toBe(false);
});

test("a cancel the server didn't confirm says so, keeps the job and its upload, and is tried again", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await scenario({ holdAt: "hud", jobDelete: "gateway" });
  await analyse(app).click();
  await expect.poll(async () => (await state()).jobs[0]?.stage, { timeout: 30_000 }).toBe("hud");
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.getByTestId("lab-cancel-confirm").getByRole("button", { name: "Cancel the analysis" }).click();
  await expect(remoteError(app)).toHaveAttribute("data-code", "cancel_unconfirmed", { timeout: 30_000 });
  await expect(page.getByTestId("lab-note")).toHaveCount(0);
  expect(await uploadsLeft()).toBe(1);
  expect((await state()).jobs[0].status).toBe("running");

  await scenario({ jobDelete: "ok" });
  await tryAgain(app).click();
  await expect(page.getByTestId("lab-note")).toContainText("Analysis cancelled.", { timeout: 30_000 });
  expect((await state()).jobs[0].status).toBe("cancelled");
  await expect.poll(uploadsLeft).toBe(0);
  expect(fs.existsSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"))).toBe(false);
});

test("a failure while the screen was closed is shown when it opens again, with the give-back", async ({ app }) => {
  await ready(app);
  const { page } = app;
  await scenario({ holdAt: "queued" });
  await analyse(app).click();
  await expect(page.getByTestId("lab-queue-position")).toBeVisible({ timeout: 30_000 });

  await app.openLibrary();
  await expect(progress(app)).toHaveCount(0);
  await scenario({ holdAt: null, failAt: "hud" });
  await expect.poll(async () => (await state()).jobs[0]?.status, { timeout: 30_000 }).toBe("failed");
  await expect.poll(uploadsLeft).toBe(0);

  await openLab(app);
  await expect(remoteError(app)).toHaveAttribute("data-code", "job_failed", { timeout: 15_000 });
  await expect(page.getByTestId("lab-refunded")).toBeVisible();
});

test("cancel before the charge: no question, the upload that arrived is deleted, nothing spent", async ({ app }) => {
  await ready(app);
  await scenario({ uploadDelayMs: 2500 });
  await analyse(app).click();
  await expect(app.page.getByText("Finishing the upload")).toBeVisible({ timeout: 30_000 });
  await app.page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(app.page.getByTestId("lab-cancel-confirm")).toHaveCount(0);
  await expect(app.page.getByText("Cancelling…")).toBeVisible();
  await expect(progress(app)).toHaveCount(0, { timeout: 30_000 });
  await expect(app.page.getByTestId("lab-note")).toHaveText("Analysis cancelled.");
  const s = await state();
  expect(s.files).toHaveLength(1);
  expect(s.files[0].deleted).toBe(true);
  expect(s.jobs).toHaveLength(0);
  expect(s.quota).toMatchObject({ free_remaining: 2, credits: 3 });
  const tmp = path.join(LAB_DIR, ".tmp");
  expect(fs.existsSync(tmp) ? fs.readdirSync(tmp) : []).toEqual([]);
  expect(fs.existsSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"))).toBe(false);
});

test("a 401: one refresh, one retry, still signed in; a token lookup failure isn't a sign-out", async ({ app }) => {
  await ready(app);
  await expect(quotaText(app)).toBeVisible();

  await control("clear-log");
  await control("expire-access");
  await control("quota", { free_remaining: 1 });
  await openLab(app);
  await expect(quotaText(app)).toContainText("1 free this week");
  const calls = await log();
  const refreshes = calls.filter((c) => c.path === "/api/desktop-auth/token");
  expect(refreshes).toHaveLength(1);
  const quota = calls.filter((c) => c.path.endsWith("/openers/quota"));
  expect(quota[0].status).toBe(401);
  expect(quota.at(-1)!.status).toBe(200);
  await expect(signInPanel(app)).toHaveCount(0);

  await scenario({ tokenOutage: true });
  await openLab(app);
  await expect(app.page.getByTestId("lab-nickname")).toHaveText("E2E Player");
  await expect(signInPanel(app)).toHaveCount(0);
  await scenario({ tokenOutage: false });
});

test("an expired session in the middle of the run: refresh, retry, and the job carries on", async ({ app }) => {
  await ready(app);
  await scenario({ holdAt: "hud" });
  await analyse(app).click();
  await expect.poll(async () => (await state()).jobs[0]?.stage, { timeout: 30_000 }).toBe("hud");
  await control("clear-log");
  await control("expire-access");
  await scenario({ holdAt: null });
  await expect(app.page.getByTestId("lab-openers-count")).toBeVisible({ timeout: 30_000 });
  await expect(remoteError(app)).toHaveCount(0);
  expect((await log()).filter((c) => c.path === "/api/desktop-auth/token")).toHaveLength(1);
});

test("refresh refused (disconnected on the website): signed out, nothing lost, sign in again", async ({ app }) => {
  const doc = fixture();
  await ready(app);
  await scenario({ holdAt: "hud" });
  await analyse(app).click();
  await expect.poll(async () => (await state()).jobs[0]?.stage, { timeout: 30_000 }).toBe("hud");
  await control("revoke-all");

  // the run ends with the sign-in panel; the paid job stays saved
  await expect(signInPanel(app)).toContainText("signed out", { timeout: 30_000 });
  await expect(progress(app)).toHaveCount(0);
  await expect(remoteError(app)).toHaveCount(0);
  expect(fs.existsSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"))).toBe(true);
  expect(await app.invoke<{ signed_in: boolean }>("lab_account")).toMatchObject({ signed_in: false, nickname: null });

  // signing in again collects it without spending again
  await scenario({ holdAt: null });
  await control("quota", { free_remaining: 1 });
  await signIn(app);
  await expect(app.page.getByTestId("lab-openers-count")).toHaveText(`${doc.openers.length} openers`, { timeout: 30_000 });
  await expect.poll(uploadsLeft).toBe(0);
  expect((await state()).quota.free_remaining).toBe(1);
});

test("sign out: revokes, forgets, and the result stays", async ({ app }) => {
  const doc = fixture();
  await ready(app, { cache: doc });
  await control("clear-log");
  await app.page.getByRole("button", { name: "Sign out" }).click();
  await expect(app.page.getByTestId("lab-nickname")).toHaveCount(0);
  await expect(app.page.getByTestId("lab-openers-count")).toHaveText(`${doc.openers.length} openers`);
  await expect.poll(async () => (await log()).filter((c) => c.path === "/api/desktop-auth/revoke").map((c) => c.token)).toEqual(["refresh"]);
  expect((await state()).sessions).toBe(0);
  expect((await app.invoke<{ signed_in: boolean }>("lab_account")).signed_in).toBe(false);
});

test("the app closed during a run: the sign-in survives (credential store) and the job is collected", async ({ app, connection }) => {
  test.setTimeout(150_000);
  await ready(app);
  await scenario({ holdAt: "hud" });
  await analyse(app).click();
  await expect.poll(async () => (await state()).jobs[0]?.stage, { timeout: 30_000 }).toBe("hud");
  expect((await state()).quota.free_remaining).toBe(1);

  await connection.restart();
  const page = connection.page;
  app.page = page;
  // the file with the job on disk holds no token
  const saved = fs.readFileSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"), "utf-8");
  expect(saved).not.toContain("sslab_");
  await scenario({ holdAt: null });
  await app.openReview();
  await page.getByRole("tab", { name: /^Lab/ }).click();
  await expect(page.getByTestId("lab-nickname")).toHaveText("E2E Player", { timeout: 20_000 });
  await expect(page.getByTestId("lab-openers-count")).toBeVisible({ timeout: 40_000 });
  await expect.poll(uploadsLeft).toBe(0);
  expect((await state()).quota.free_remaining).toBe(1);
  expect(fs.existsSync(path.join(LAB_DIR, REPLAY_ID, "remote_job.json"))).toBe(false);
});

test("a second launch opens its file in the running app and ends", async ({ connection }) => {
  test.setTimeout(120_000);
  // The single-instance lock is per PC: a real copy of the app running would take the file instead
  const others = spawnSync("tasklist", ["/FI", "IMAGENAME eq spaghetti-lab.exe", "/NH"], { encoding: "utf-8" }).stdout.match(/spaghetti-lab\.exe/gi)?.length ?? 0;
  test.skip(others > 1, "another copy of the app is running");
  const env = { SPAGHETTI_LAB_SINGLE_INSTANCE: "1" };
  await connection.restart(env);
  try {
    const opened: string[] = [];
    connection.page.on("console", (m) => opened.push(m.text()));
    const missing = path.join(RUN_DIR, "not-there-second-launch.spag");
    const second = spawn(EXE, [missing], { cwd: RUN_DIR, stdio: "ignore", env: { ...process.env, ...env, SPAGHETTI_LAB_DATA_DIR: DATA_DIR } });
    const exit = await new Promise<number | null>((resolve) => second.on("exit", resolve));
    expect(exit).toBe(0);
    await expect.poll(() => opened.some((t) => t.includes("Failed to open .spag")), { timeout: 15_000 }).toBe(true);
    await expect(connection.page.locator("body")).toBeVisible();
  } finally {
    await connection.restart();
  }
});

test.describe("with SPAGHETTI_LAB_MODELS", () => {
  test.beforeAll(async ({ connection }) => {
    await connection.restart({ SPAGHETTI_LAB_MODELS: path.join(DATA_DIR, "no-models-here") });
  });
  test.afterAll(async ({ connection }) => {
    await connection.restart();
  });

  test("the local mode shows, labelled, with no sign-in and no spending; the server is one click away", async ({ app }) => {
    await ensureSignedOut(app);
    writeCache(null);
    await openLab(app);
    const modes = app.page.getByRole("group", { name: "Analysis mode" });
    await expect(modes.getByRole("button", { name: "local models" })).toHaveAttribute("aria-pressed", "true");
    await expect(signInPanel(app)).toHaveCount(0);
    await expect(app.page.getByRole("group", { name: "Transcript" })).toBeVisible();
    await app.page.getByRole("button", { name: "Run lab analysis" }).click();
    await expect(app.page.getByRole("alert")).toContainText("SPAGHETTI_LAB_NO_MODELS");
    const calls = await log();
    expect(calls.filter((c) => c.path.startsWith("/api/user-files") || c.path.endsWith("/quota"))).toEqual([]);

    await modes.getByRole("button", { name: "server" }).click();
    await expect(signInPanel(app)).toBeVisible();
    await expect(app.page.getByRole("group", { name: "Transcript" })).toHaveCount(0);
  });
});
