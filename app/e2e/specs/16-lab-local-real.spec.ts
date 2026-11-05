// The local-models build's real run: lab_analyze.py with the real models and the GPU, on the fixture clip, through the
// app's Rust and UI. Not part of a normal e2e run: it needs the models and takes minutes of GPU. Only with
//   E2E_REAL_MODELS=1 npx playwright test 16 --timeout 3600000
// Writes e2e/.run/real/: the result, a screenshot of each section, and the stages with their times.
import fs from "node:fs";
import path from "node:path";
import { test, expect } from "../fixtures";
import { DATA_DIR, DEFAULT_DB, REPLAY_ID, RUN_DIR } from "../env";

const OUT = path.join(RUN_DIR, "real");
const CACHE = path.join(DATA_DIR, "SpaghettiLab", "lab", REPLAY_ID, "lab_result.json");

test.skip(!process.env.E2E_REAL_MODELS, "needs E2E_REAL_MODELS=1: the real models, on the GPU");
test.describe.configure({ timeout: 3_600_000 });

test("a real local run: openers, transcript events and states show", async ({ app, connection }) => {
  fs.mkdirSync(OUT, { recursive: true });
  fs.rmSync(path.dirname(CACHE), { recursive: true, force: true });
  await connection.restart({ E2E_LOCAL_BUILD: "1", E2E_REAL_MODELS: "1" });
  app.page = connection.page;
  await app.ensureAnalysed();
  const { page } = app;
  await app.openReview();

  const config = await app.invoke<{ local_models: boolean; models_dir: string | null }>("lab_config");
  expect(config.local_models).toBe(true);
  expect(config.models_dir, "the default models folder, with no variable set").toBeTruthy();

  await page.getByRole("tab", { name: /^Lab/ }).click();
  const modes = page.getByRole("group", { name: "Analysis mode" });
  await expect(modes.getByRole("button", { name: "local models" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("group", { name: "Transcript" }).getByRole("button", { name: "On", exact: true }).click();

  // Each stage as it arrives, with the moment it arrived
  await page.evaluate(() => {
    (window as any).__stages = [];
    (window as any).__TAURI_INTERNALS__.invoke("plugin:event|listen", {
      event: "lab-progress",
      target: { kind: "Any" },
      handler: (window as any).__TAURI_INTERNALS__.transformCallback((e: any) => (window as any).__stages.push({ at: Date.now(), ...e.payload })),
    });
  });
  const t0 = Date.now();
  await page.getByRole("button", { name: "Run lab analysis" }).click();
  await expect(page.getByTestId("lab-openers-count")).toBeVisible({ timeout: 3_500_000 });
  const stages = await page.evaluate(() => (window as any).__stages);
  const doc = JSON.parse(fs.readFileSync(CACHE, "utf-8"));
  fs.copyFileSync(CACHE, path.join(OUT, "lab_result.json"));

  // What the app shows is what the run produced
  expect(doc.capabilities.transcript, "the full transcript ran").toBe(true);
  expect(doc.openers.length).toBeGreaterThan(0);
  expect(doc.transcript.events.length).toBeGreaterThan(doc.openers.length);
  // the whole clip, not its first part: the moves reach its last round, and the states aren't one flat class there
  const lastEvent = Math.max(...doc.transcript.events.map((e: { t: number }) => e.t));
  expect(lastEvent).toBeGreaterThan(doc.duration_s * 0.8);
  const tail = doc.states.P1.slice(Math.round(doc.states.P1.length * 0.75));
  expect(new Set(tail).size).toBeGreaterThan(1);
  expect(doc.states.P1.length).toBeGreaterThan(0);
  expect(doc.states.P2.length).toBe(doc.states.P1.length);
  await expect(page.getByTestId("lab-openers-count")).toHaveText(`${doc.openers.length} openers`);
  await expect(page.locator("[data-lab-opener]")).toHaveCount(doc.openers.length);
  await page.screenshot({ path: path.join(OUT, "openers.png") });

  const hitsInRows = await page.getByTestId("lab-opener-hits").count();
  const section = (name: string) => page.getByRole("group", { name: "Lab section" }).getByRole("button", { name: new RegExp(`^${name}`) });
  await section("Moves").click();
  await expect(page.getByTestId("lab-moves-count")).toContainText(String(doc.transcript.events.length));
  await page.screenshot({ path: path.join(OUT, "moves.png") });
  await section("Search").click();
  await page.screenshot({ path: path.join(OUT, "search.png") });
  await section("Summary").click();
  await page.screenshot({ path: path.join(OUT, "summary.png") });

  const report = {
    wall_s: Math.round((Date.now() - t0) / 1000),
    timings_s: doc.timings_s,
    stages: stages.map((s: { at: number; line: string }) => ({ at_s: Math.round((s.at - t0) / 1000), line: s.line })),
    openers: doc.openers.length,
    openers_with_hits: doc.openers.filter((o: { combo_hits?: number }) => o.combo_hits).length,
    opener_rows_with_hits_on_screen: hitsInRows,
    events: doc.transcript.events.length,
    states: doc.states.P1.length,
    characters: doc.characters,
    db: DEFAULT_DB,
  };
  fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 1));
  console.log(JSON.stringify(report, null, 1));

  await connection.restart();
  app.page = connection.page;
  fs.rmSync(path.dirname(CACHE), { recursive: true, force: true });
});
