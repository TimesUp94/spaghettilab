// The "Lab" tab of the review screen. No model ever runs: the backend refuses (SPAGHETTI_LAB_NO_MODELS),
// the cached result is the fixture (e2e/fixtures/lab_result.json, a copy of
// tools/ggst_logger/model/fixtures/lab_result_sample.json) written into the run's data dir, and the
// "running" flow is a mocked run_lab_analysis that sends the same progress events as the real one.
// The expected search counts are computed here from the fixture, independently of the app's code.
import fs from "node:fs";
import path from "node:path";
import { test, expect, type App } from "../fixtures";
import { APP_DIR, DATA_DIR, REPLAY_ID } from "../env";

type Ev = { t: number; frame: number; side: string; char: string; category: string; move: string; move_p: number; outcome: string; outcome_p: number | null; opener: boolean };
const FIXTURE = path.join(APP_DIR, "e2e", "fixtures", "lab_result.json");
const fixture = () => JSON.parse(fs.readFileSync(FIXTURE, "utf-8"));
const CACHE = path.join(DATA_DIR, "SpaghettiLab", "lab", REPLAY_ID, "lab_result.json");

function writeCache(doc: unknown | null) {
  fs.rmSync(path.dirname(CACHE), { recursive: true, force: true });
  if (doc === null) return;
  fs.mkdirSync(path.dirname(CACHE), { recursive: true });
  fs.writeFileSync(CACHE, JSON.stringify(doc));
}

function withoutTranscript() {
  const d = fixture();
  d.transcript = null;
  d.states = null;
  d.capabilities.transcript = false;
  return d;
}

async function openLab(app: App) {
  await app.openReview();
  await app.page.getByRole("tab", { name: /^Lab/ }).click();
}

const section = (app: App, name: string) => app.page.getByRole("group", { name: "Lab section" }).getByRole("button", { name: new RegExp(`^${name}`) });
const count = (app: App, id: string) => app.page.getByTestId(id);

/** The video went to about `t - 1` s (a lab row seeks a second before the moment). */
async function expectSeekedTo(app: App, t: number) {
  await expect.poll(() => app.videoTime(), { timeout: 10_000 }).toBeGreaterThan(Math.max(0, t - 1) - 0.25);
  expect(await app.videoTime()).toBeLessThan(Math.max(0, t - 1) + 0.25);
}

test.afterAll(() => writeCache(null));

test("without a result: the run form, and the not-installed error", async ({ app }) => {
  writeCache(null);
  await openLab(app);
  const { page } = app;
  await expect(page.getByText("Lab analysis", { exact: true })).toBeVisible();
  await expect(page.getByRole("group", { name: "Transcript" })).toBeVisible();

  // the real backend refuses to run the models in tests
  await page.getByRole("button", { name: "Run lab analysis" }).click();
  await expect(page.getByRole("alert")).toContainText("SPAGHETTI_LAB_NO_MODELS");

  // the setup problem is shown as such
  await page.evaluate(() => {
    (window as any).__e2e.handlers["run_lab_analysis"] = () => {
      throw "Lab analysis is not installed: python has no PyTorch.";
    };
  });
  await page.getByRole("group", { name: "Transcript" }).getByRole("button", { name: "Off" }).click();
  await page.getByRole("checkbox", { name: "Light" }).check();
  await page.getByRole("button", { name: "Run lab analysis" }).click();
  await expect(page.getByRole("alert")).toContainText("The lab models aren't installed on this PC");
  const call = (await app.calls("run_lab_analysis")).at(-1)!;
  expect(call.args.replayId).toBe(REPLAY_ID);
  expect(call.args.options).toMatchObject({ transcript: "off", light: true });
  expect(call.args.options.video_path).toMatch(/e2e_set\.mp4$/);
});

test("a run shows its stages, can be cancelled, then completes", async ({ app }) => {
  writeCache(null);
  await openLab(app);
  const { page } = app;
  // a run that ends when cancel_lab_analysis is called, or when the test resolves it
  await page.evaluate(() => {
    const e2e = (window as any).__e2e;
    e2e.handlers["run_lab_analysis"] = () =>
      new Promise((resolve, reject) => {
        e2e.labRun = { resolve, reject };
      });
    e2e.handlers["cancel_lab_analysis"] = () => {
      e2e.labRun?.reject("Cancelled");
      return null;
    };
  });
  await page.getByRole("button", { name: "Run lab analysis" }).click();
  const progress = page.getByTestId("lab-progress");
  await expect(progress).toBeVisible();
  for (const [stage, line] of [["hud", "hud"], ["detect", "detect 3/9"]]) {
    await app.emit("lab-progress", { replay_id: REPLAY_ID, stage, line });
  }
  // another replay's progress is ignored
  await app.emit("lab-progress", { replay_id: "other", stage: "facts", line: "facts" });
  await expect(progress).toContainText("detect 3/9");
  await expect(progress.getByText("Reading the HUD")).toBeVisible();
  await expect(progress.getByText("Recognising moves")).toBeVisible();
  await expect(progress.getByText("Summary")).toBeVisible();

  await progress.getByRole("button", { name: "Cancel" }).click();
  await expect(progress).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(await app.calls("cancel_lab_analysis")).toHaveLength(1);

  // run again, to the end
  await page.getByRole("button", { name: "Run lab analysis" }).click();
  await expect(progress).toBeVisible();
  await app.emit("lab-progress", { replay_id: REPLAY_ID, stage: "transcript", line: "transcript" });
  await expect(progress.getByText("Transcript", { exact: true })).toBeVisible();
  const doc = fixture();
  await page.evaluate((d) => (window as any).__e2e.labRun.resolve(d), doc);
  await expect(progress).toHaveCount(0);
  await expect(count(app, "lab-openers-count")).toHaveText(`${doc.openers.length} openers`);
});

test("openers: list, timeline markers and click to seek", async ({ app }) => {
  const doc = fixture();
  writeCache(doc);
  await openLab(app);
  const { page } = app;
  await expect(page.getByRole("tab", { name: /^Lab/ })).toContainText(String(doc.openers.length));
  await expect(page.getByText(`${doc.characters.P1}`, { exact: true }).first()).toBeVisible();
  await expect(count(app, "lab-openers-count")).toHaveText(`${doc.openers.length} openers`);
  await expect(page.locator("[data-lab-opener]")).toHaveCount(doc.openers.length);
  await expect(page.locator("[data-lab-move]")).toHaveCount(doc.transcript.events.length);

  // one side only
  const p2 = doc.openers.filter((o: { attacker: string }) => o.attacker === "P2").length;
  await page.getByRole("group", { name: "Openers side" }).getByRole("button", { name: /^P2/ }).click();
  await expect(count(app, "lab-openers-count")).toHaveText(`${p2} openers`);
  await page.getByRole("group", { name: "Openers side" }).getByRole("button", { name: "Both" }).click();

  const o = [...doc.openers].sort((a, b) => a.time_s - b.time_s)[3];
  await page.getByRole("button", { name: new RegExp(`${o.attacker} ${o.move.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} opener$`) }).first().click();
  await expectSeekedTo(app, o.time_s);

  // the moves lanes can be hidden
  await page.getByRole("button", { name: "moves", exact: true }).click();
  await expect(page.locator("[data-lab-move]")).toHaveCount(0);
  await page.getByRole("button", { name: "moves", exact: true }).click();
  await expect(page.locator("[data-lab-move]")).toHaveCount(doc.transcript.events.length);
});

test("moves: filters, uncertain items, seek", async ({ app }) => {
  const doc = fixture();
  writeCache(doc);
  await openLab(app);
  const { page } = app;
  const ev: Ev[] = doc.transcript.events;
  await section(app, "Moves").click();
  await expect(count(app, "lab-moves-count")).toHaveText(`${ev.length} moves`);
  // low-confidence moves are marked
  await expect(page.locator("[data-uncertain]")).toHaveCount(ev.filter((e) => e.move_p < 0.5).length);

  await page.getByRole("group", { name: "Moves side" }).getByRole("button", { name: "P1" }).click();
  await page.getByRole("combobox", { name: "Outcome" }).selectOption("whiff");
  const whiffs = ev.filter((e) => e.side === "P1" && e.outcome === "whiff");
  await expect(count(app, "lab-moves-count")).toHaveText(`${whiffs.length} moves`);
  await page.getByRole("combobox", { name: "Outcome" }).selectOption("connected");
  await expect(count(app, "lab-moves-count")).toHaveText(`${ev.filter((e) => e.side === "P1" && (e.outcome === "hit" || e.outcome === "counter")).length} moves`);

  const e = ev.filter((x) => x.side === "P1" && (x.outcome === "hit" || x.outcome === "counter"))[2];
  await page.getByRole("button", { name: new RegExp(`^\\d+:\\d+\\.\\d P1 `) }).nth(2).click();
  await expectSeekedTo(app, e.t);
});

test("search: the presets and free filters return the fixture's moments", async ({ app }) => {
  const doc = fixture();
  writeCache(doc);
  await openLab(app);
  const { page } = app;
  const ev: Ev[] = [...doc.transcript.events].sort((a: Ev, b: Ev) => a.frame - b.frame);
  await section(app, "Search").click();
  const preset = (name: string) => page.getByRole("group", { name: "Search preset" }).getByRole("button", { name, exact: true });
  const result = count(app, "lab-search-count");
  const n = (k: number) => `${k} ${k === 1 ? "moment" : "moments"}`;

  // 1. move + outcome: P1's moves (any), then P1 whiffed <top move>
  await expect(result).toHaveText(n(ev.filter((e) => e.side === "P1").length));
  const topMove = Object.entries(
    ev.filter((e) => e.side === "P1" && e.move !== "(rare)").reduce<Record<string, number>>((m, e) => ((m[e.move] = (m[e.move] ?? 0) + 1), m), {}),
  ).sort((a, b) => b[1] - a[1])[0][0];
  await page.getByRole("combobox", { name: "Move" }).selectOption(topMove);
  await page.getByRole("combobox", { name: "Outcome" }).selectOption("whiff");
  await expect(result).toHaveText(n(ev.filter((e) => e.side === "P1" && e.move === topMove && e.outcome === "whiff").length));
  await page.getByRole("combobox", { name: "Outcome" }).selectOption("any");
  await page.getByRole("checkbox", { name: "openers only" }).check();
  await expect(result).toHaveText(n(ev.filter((e) => e.side === "P1" && e.move === topMove && e.opener).length));

  // 2. counter-hit by: when P1 got counter-hit
  await preset("Counter-hit by").click();
  const ch = ev.filter((e) => e.side !== "P1" && e.outcome === "counter");
  await expect(result).toHaveText(n(ch.length));
  await page.getByRole("group", { name: "Victim" }).getByRole("button", { name: /^P2/ }).click();
  await expect(result).toHaveText(n(ev.filter((e) => e.side !== "P2" && e.outcome === "counter").length));

  // 3. after blocked
  await preset("After blocked").click();
  for (const side of ["P1", "P2"]) {
    await page.getByRole("group", { name: "Side", exact: true }).getByRole("button", { name: new RegExp(`^${side}`) }).click();
    const own = ev.filter((e) => e.side === side);
    const k = own.filter((r, i) => i > 0 && own[i - 1].outcome === "block" && r.frame - own[i - 1].frame <= 40).length;
    await expect(result).toHaveText(n(k));
  }

  // 4. after knockdown (wake-ups from the states)
  await preset("After knockdown").click();
  await page.getByRole("group", { name: "Side", exact: true }).getByRole("button", { name: /^P1/ }).click();
  const cls: string[] = doc.states.P1;
  const wakes: number[] = [];
  for (let i = 0; i < cls.length; ) {
    if (cls[i] !== "knockdown") { i++; continue; }
    let j = i;
    while (j < cls.length && cls[j] === "knockdown") j++;
    if (j - i >= 2 && j < cls.length && cls[j] !== "hitstun") wakes.push(8 * j);
    i = j;
  }
  const own = ev.filter((e) => e.side === "P1");
  let w = 0;
  let kd = 0;
  own.forEach((r, k) => {
    const p = k ? own[k - 1] : null;
    while (w < wakes.length && wakes[w] < r.frame - 60) w++;
    if (w < wakes.length && wakes[w] - 8 <= r.frame && r.frame <= wakes[w] + 60 && (!p || p.frame < wakes[w] - 8)) { kd++; w++; }
  });
  await expect(result).toHaveText(n(kd));
  await expect(page.getByText(`${wakes.length} wake-ups`, { exact: false })).toBeVisible();

  // free filters: character + category + outcome, then click to seek
  await preset("Filters").click();
  await page.getByRole("combobox", { name: "Character" }).selectOption(doc.characters.P2);
  await page.getByRole("combobox", { name: "Category" }).selectOption("normal");
  await page.getByRole("combobox", { name: "Outcome" }).selectOption("block");
  const free = ev.filter((e) => e.char === doc.characters.P2 && e.category === "normal" && e.outcome === "block");
  await expect(result).toHaveText(n(free.length));
  await page.getByRole("button", { name: new RegExp(`^\\d+:\\d+\\.\\d P2 .* block$`) }).first().click();
  await expectSeekedTo(app, free[0].t);
});

test("summary: facts in English and Italian, copy prompt, timestamps seek", async ({ app }) => {
  const doc = fixture();
  writeCache(doc);
  await openLab(app);
  const { page } = app;
  await section(app, "Summary").click();
  await page.getByRole("group", { name: "Summary language" }).getByRole("button", { name: "English" }).click();
  const h1 = (md: string) => md.split("\n").find((l) => l.startsWith("# "))!.slice(2);
  await expect(page.getByRole("heading", { name: h1(doc.facts.md.en) })).toBeVisible();
  await expect(page.getByText(/no AI wrote them and nothing is sent anywhere/)).toBeVisible();
  await page.getByRole("group", { name: "Summary language" }).getByRole("button", { name: "Italiano" }).click();
  await expect(page.getByRole("heading", { name: h1(doc.facts.md.it) })).toBeVisible();

  await page.getByRole("button", { name: "Copy prompt" }).click();
  await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();

  const ts = /\b(\d{1,2}):(\d{2}\.\d)\b/.exec(doc.facts.md.it.split("## ")[2])!;
  await page.getByRole("button", { name: ts[0], exact: true }).first().click();
  await expectSeekedTo(app, Number(ts[1]) * 60 + Number(ts[2]));
  await page.getByRole("group", { name: "Summary language" }).getByRole("button", { name: "English" }).click();
});

/** The fixture with an AI summary (LAB_RESULT.md "ai_summary"): English grounded, Italian withheld. */
function withAiSummary() {
  const d = fixture();
  const ts = /\b\d{1,2}:\d{2}\.\d\b/.exec(d.facts.md.en)![0];
  d.capabilities.ai_summary = true;
  d.ai_summary = {
    provider: "claude",
    model: "claude-opus-5-5",
    langs: {
      en: {
        text: `## Summary
${d.characters.P1} took the set, the key opener at [${ts}].

## Coaching notes
1. **Watch** the opener at [${ts}].
2. A second note.`,
        grounded: true,
        errors: [],
        warnings: ["note 2 cites no timestamp"],
        usage: { model: "claude-opus-5-5", input_tokens: 4000, output_tokens: 300 },
      },
      it: { text: null, grounded: false, errors: ["timestamp 9:59.9 is not in the facts", "move 6H is not in the facts"], warnings: [] },
    },
  };
  return { d, ts };
}

test("summary: the AI summary above the facts, only when grounded", async ({ app }) => {
  const { d, ts } = withAiSummary();
  writeCache(d);
  await openLab(app);
  const { page } = app;
  await section(app, "Summary").click();
  const lang = (name: string) => page.getByRole("group", { name: "Summary language" }).getByRole("button", { name }).click();
  const h1 = (md: string) => md.split("\n").find((l) => l.startsWith("# "))!.slice(2);

  // English: grounded, labelled as AI, above the facts (which stay), timestamps seek
  await lang("English");
  const ai = page.getByTestId("lab-ai-summary");
  await expect(ai).toContainText("Written by AI from the match facts");
  await expect(ai.getByRole("heading", { name: "Coaching notes" })).toBeVisible();
  await expect(ai.locator("ol > li")).toHaveCount(2);
  await expect(ai.locator("strong", { hasText: "Watch" })).toBeVisible();
  await expect(page.getByRole("heading", { name: h1(d.facts.md.en) })).toBeVisible();
  const aiBox = (await ai.boundingBox())!;
  const factsBox = (await page.getByRole("heading", { name: h1(d.facts.md.en) }).boundingBox())!;
  expect(aiBox.y).toBeLessThan(factsBox.y);
  await expect(page.getByTestId("lab-ai-withheld")).toHaveCount(0);
  await ai.getByRole("button", { name: ts, exact: true }).first().click();
  const [mm, ss] = ts.split(":");
  await expectSeekedTo(app, Number(mm) * 60 + Number(ss));
  await expect(ai.getByText("Check warnings (1)")).toBeVisible();

  // Italian: withheld, no text, the errors in a collapsed detail, the facts still there
  await lang("Italiano");
  await expect(page.getByTestId("lab-ai-summary")).toHaveCount(0);
  const withheld = page.getByTestId("lab-ai-withheld");
  await expect(withheld).toContainText("non corrispondeva ai fatti");
  await expect(withheld.getByText("move 6H is not in the facts")).toBeHidden();
  await withheld.getByText("Dettagli (2)").click();
  await expect(withheld.getByText("move 6H is not in the facts")).toBeVisible();
  await expect(page.getByRole("heading", { name: h1(d.facts.md.it) })).toBeVisible();

  // a missing API key gets a hint
  d.ai_summary.langs.it.errors = ["AuthenticationError: Could not resolve authentication method. Expected the api_key to be set."];
  writeCache(d);
  await openLab(app);
  await section(app, "Summary").click();
  await lang("Italiano");
  await expect(page.getByTestId("lab-ai-withheld")).toContainText("ANTHROPIC_API_KEY");
  await lang("English");
});

test("summary: no ai_summary, no AI block", async ({ app }) => {
  writeCache(fixture());
  await openLab(app);
  const { page } = app;
  await section(app, "Summary").click();
  await expect(page.getByText(/no AI wrote them and nothing is sent anywhere/)).toBeVisible();
  await expect(page.getByTestId("lab-ai-summary")).toHaveCount(0);
  await expect(page.getByTestId("lab-ai-withheld")).toHaveCount(0);
});

test("run option: the AI summary (Claude) is off by default and passes llm", async ({ app }) => {
  writeCache(null);
  await openLab(app);
  const { page } = app;
  await page.evaluate(() => {
    const e2e = (window as any).__e2e;
    e2e.handlers["run_lab_analysis"] = () =>
      new Promise((resolve, reject) => {
        e2e.labRun = { resolve, reject };
      });
  });
  const box = page.getByRole("checkbox", { name: "AI summary (Claude)" });
  await expect(box).not.toBeChecked();
  await page.getByRole("button", { name: "Run lab analysis" }).click();
  expect((await app.calls("run_lab_analysis")).at(-1)!.args.options).toMatchObject({ llm: "none" });
  await page.evaluate(() => (window as any).__e2e.labRun.reject("Cancelled"));
  await expect(page.getByTestId("lab-progress")).toHaveCount(0);

  await box.check();
  await page.getByRole("button", { name: "Run lab analysis" }).click();
  expect((await app.calls("run_lab_analysis")).at(-1)!.args.options).toMatchObject({ llm: "claude" });
  const progress = page.getByTestId("lab-progress");
  await expect(progress.getByText("AI summary", { exact: true })).toBeVisible();
  await app.emit("lab-progress", { replay_id: REPLAY_ID, stage: "summary", line: "summary" });
  await expect(progress.getByText("AI summary", { exact: true })).toBeVisible();
  await page.evaluate((d) => (window as any).__e2e.labRun.resolve(d), withAiSummary().d);
  await expect(progress).toHaveCount(0);
  await section(app, "Summary").click();
  await page.getByRole("group", { name: "Summary language" }).getByRole("button", { name: "English" }).click();
  await expect(page.getByTestId("lab-ai-summary")).toBeVisible();
});

test("without the transcript: openers only, with the GPU hint", async ({ app }) => {
  const doc = withoutTranscript();
  writeCache(doc);
  await openLab(app);
  const { page } = app;
  await expect(page.getByTestId("lab-no-transcript")).toContainText("needs a GPU");
  await expect(page.locator("[data-lab-opener]")).toHaveCount(doc.openers.length);
  await expect(page.locator("[data-lab-move]")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "moves", exact: true })).toHaveCount(0);

  await section(app, "Moves").click();
  await expect(page.getByTestId("lab-no-transcript")).toBeVisible();

  await section(app, "Search").click();
  const preset = (name: string) => page.getByRole("group", { name: "Search preset" }).getByRole("button", { name, exact: true });
  await expect(preset("After blocked")).toBeDisabled();
  await expect(preset("After knockdown")).toBeDisabled();
  // the openers as events: outcome counter for counter-hit openers, else hit
  const p1 = doc.openers.filter((o: { attacker: string }) => o.attacker === "P1");
  await expect(count(app, "lab-search-count")).toHaveText(`${p1.length} moments`);
  await page.getByRole("combobox", { name: "Outcome" }).selectOption("counter");
  const k = p1.filter((o: { hit_kind: string }) => o.hit_kind === "counter").length;
  await expect(count(app, "lab-search-count")).toHaveText(`${k} ${k === 1 ? "moment" : "moments"}`);
});

test("delete the result", async ({ app }) => {
  writeCache(fixture());
  await openLab(app);
  const { page } = app;
  await page.getByRole("button", { name: "Delete the lab result" }).click();
  await expect(page.getByRole("button", { name: "Run lab analysis" })).toBeVisible();
  expect(fs.existsSync(CACHE)).toBe(false);
  expect(await app.invoke("get_lab_result", { replayId: REPLAY_ID })).toBeNull();
});
