// The review screen of the analysed fixture clip: header, games, results tabs, timeline and player.
import { test, expect, type App } from "../fixtures";
import { DEFAULT_DB, EXPECTED, REPLAY_ID } from "../env";

type Round = { round_index: number; winner: string; round_start_ms: number; round_end_ms: number; is_comeback: boolean };
type Damage = { timestamp_ms: number; target_side: number | null; damage_pct: number };

const data = (app: App) => ({
  rounds: () => app.invoke<Round[]>("get_rounds", { dbPath: DEFAULT_DB, replayId: REPLAY_ID }),
  damage: () => app.invoke<Damage[]>("get_damage_events", { dbPath: DEFAULT_DB, replayId: REPLAY_ID }),
  highlights: () => app.invoke<{ kind: string; timestamp_ms: number }[]>("get_highlights", { dbPath: DEFAULT_DB, replayId: REPLAY_ID }),
});


test.beforeEach(async ({ app }) => {
  await app.openReview();
});

test("the analysis reads the fixture's rounds and winners", async ({ app }) => {
  const rounds = await data(app).rounds();
  expect(rounds.map((r) => r.winner)).toEqual(EXPECTED.roundWinners);
  expect(rounds.filter((r) => r.is_comeback)).toHaveLength(EXPECTED.comebacks);
});

test("header: title, game picker, back", async ({ app }) => {
  const { page } = app;
  const header = page.locator("header");
  await expect(header.getByText("e2e set", { exact: true })).toBeVisible();
  await expect(header.getByRole("button", { name: "All matches" })).toBeVisible();
  for (let g = 1; g <= EXPECTED.games; g++) {
    await expect(header.getByRole("button", { name: `M${g}`, exact: true })).toHaveAttribute(
      "title",
      `Match ${g}: 2–0, won by P1`,
    );
  }
  await expect(header.getByRole("button", { name: `M${EXPECTED.games + 1}`, exact: true })).toHaveCount(0);

  await page.getByRole("button", { name: "Back to the library" }).click();
  await expect(page.getByRole("heading", { name: "Library" })).toBeVisible();
});

test("overview: score, stats and the games list", async ({ app }) => {
  const { page } = app;
  const statRow = (label: string) => page.getByText(label, { exact: true }).locator("..");
  await expect(statRow("Total Rounds")).toContainText(String(EXPECTED.rounds));
  await expect(statRow("Duration")).toContainText(EXPECTED.durationLabel);
  await expect(statRow("Comebacks")).toContainText(String(EXPECTED.comebacks));
  await expect(statRow("Damage Events")).toContainText(String((await data(app).damage()).length));

  await expect(page.getByRole("heading", { name: "Matches" }).or(page.getByText("Matches", { exact: true }))).toBeVisible();
  await expect(page.getByText("P1 wins", { exact: true })).toHaveCount(EXPECTED.games);
  await expect(page.getByText("COMEBACK", { exact: true })).not.toHaveCount(0);
});

test("tabs show the counts of the data", async ({ app }) => {
  const { page } = app;
  const d = data(app);
  const tab = (name: string) => page.getByRole("tab", { name: new RegExp(`^${name}`) });
  await expect(tab("Rounds")).toHaveText(`Rounds${(await d.rounds()).length}`);
  await expect(tab("Highlights")).toHaveText(`Highlights${(await d.highlights()).length}`);
  await expect(tab("Damage")).toHaveText(`Damage${(await d.damage()).length}`);
  await expect(tab("Overview")).toHaveAttribute("aria-selected", "true");
});

test("rounds tab lists every round and seeks to it", async ({ app }) => {
  const { page } = app;
  const rounds = await data(app).rounds();
  await page.getByRole("tab", { name: /^Rounds/ }).click();
  await expect(page.getByText("P1 wins", { exact: true })).toHaveCount(rounds.length);
  await expect(page.getByText(/^COMEBACK \d+%$/)).toHaveCount(EXPECTED.comebacks);

  // clicking the third round seeks there and selects its game
  await page.getByText("P1 wins", { exact: true }).nth(2).click();
  await expect.poll(() => app.videoTime()).toBeCloseTo(rounds[2].round_start_ms / 1000, 0);
  await expect(page.getByText("Match 2 only ·")).toBeVisible();
  await expect(page.getByText("P1 wins", { exact: true })).toHaveCount(2);
  await page.getByRole("button", { name: "show all matches" }).click();
  await expect(page.getByText("P1 wins", { exact: true })).toHaveCount(rounds.length);
});

test("game picker zooms to a game and filters the lists", async ({ app }) => {
  const { page } = app;
  const rounds = await data(app).rounds();
  const header = page.locator("header");
  await header.getByRole("button", { name: "M2", exact: true }).click();
  await expect(page.getByText("Match 2", { exact: true }).first()).toBeVisible(); // timeline legend
  await expect.poll(() => app.videoTime()).toBeGreaterThan(rounds[1].round_end_ms / 1000);

  await page.getByRole("tab", { name: /^Rounds/ }).click();
  await expect(page.getByText("Match 2 only ·")).toBeVisible();
  await page.getByRole("tab", { name: /^Overview/ }).click();
  await expect(page.getByRole("button", { name: "Show all" })).toBeVisible();
  await page.getByRole("button", { name: "Show all" }).click();
  await expect(page.getByText("Timeline", { exact: true })).toBeVisible();

  await header.getByRole("button", { name: "M1", exact: true }).click();
  await expect(page.getByText("Match 1", { exact: true }).first()).toBeVisible();
  await header.getByRole("button", { name: "All matches" }).click();
  await expect(page.getByText("Timeline", { exact: true })).toBeVisible();
});

test("highlights tab groups the highlights and seeks to them", async ({ app }) => {
  const { page } = app;
  const hl = await data(app).highlights();
  await page.getByRole("tab", { name: /^Highlights/ }).click();
  await expect(page.getByText(/^\(\d+\)$/).first()).toBeVisible();
  const comeback = hl.find((h) => h.kind === "comeback")!;
  await expect(page.getByText("COMEBACK", { exact: true })).toBeVisible();
  await expect(page.getByText(/^BIG HIT/).first()).toBeVisible();
  // the first row of the first group (comebacks) seeks to it
  await page.getByRole("button", { name: "Export" }).first().locator("..").click({ position: { x: 5, y: 5 } });
  await expect.poll(() => app.videoTime()).toBeCloseTo(comeback.timestamp_ms / 1000, 0);
});

test("damage tab filters, sorts and jumps", async ({ app }) => {
  const { page } = app;
  const dmg = await data(app).damage();
  await page.getByRole("tab", { name: /^Damage/ }).click();
  await expect(page.getByText(`${dmg.length} events`, { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "P1", exact: true }).first().click();
  await expect(page.getByText(`${dmg.filter((d) => d.target_side === 1).length} events`, { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "P2", exact: true }).first().click();
  await expect(page.getByText(`${dmg.filter((d) => d.target_side === 2).length} events`, { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "All", exact: true }).click();

  await page.getByRole("button", { name: "Damage", exact: true }).click();
  const biggest = [...dmg].sort((a, b) => b.damage_pct - a.damage_pct)[0];
  await page.getByRole("button", { name: "Jump" }).first().click();
  await expect.poll(() => app.videoTime()).toBeCloseTo(biggest.timestamp_ms / 1000, 0);
});

test("timeline: click and drag to seek, hover shows the time", async ({ app }) => {
  const { page } = app;
  const svg = page.locator("svg").filter({ has: page.locator("text", { hasText: /^R1$/ }) }).first();
  await expect(svg).toBeVisible();
  const box = (await svg.boundingBox())!;
  const duration = await page.locator("video").evaluate((v: HTMLVideoElement) => v.duration);

  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await expect.poll(() => app.videoTime()).toBeCloseTo(duration * 0.5, -1);

  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  await expect.poll(() => app.videoTime()).toBeCloseTo(duration * 0.6, -1);

  await page.mouse.move(box.x + box.width * 0.1, box.y + box.height / 2);
  await expect(page.getByText(/^\d+:\d\d · R1$/)).toBeVisible();
});

test("player controls and keyboard shortcuts", async ({ app }) => {
  const { page } = app;
  await app.seek(20);
  await expect(page.getByText("Round 1", { exact: true })).toBeVisible(); // round badge over the video

  await page.getByRole("button", { name: /^Forward 5 s/ }).click();
  await expect.poll(() => app.videoTime()).toBeCloseTo(25, 0);
  await page.getByRole("button", { name: /^Back 5 s/ }).click();
  await expect.poll(() => app.videoTime()).toBeCloseTo(20, 0);
  await page.getByRole("button", { name: "Next frame (.)" }).click();
  await expect.poll(() => app.videoTime()).toBeCloseTo(20 + 1 / 30, 2);
  await page.getByRole("button", { name: "Previous frame (,)" }).click();
  await expect.poll(() => app.videoTime()).toBeCloseTo(20, 2);

  await page.getByRole("button", { name: "Play / pause (Space)" }).click();
  await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.paused)).toBe(false);
  await expect.poll(() => app.videoTime()).toBeGreaterThan(20.5);
  await page.getByRole("button", { name: "Play / pause (Space)" }).click();
  await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.paused)).toBe(true);

  // keyboard (focus outside any input)
  await app.seek(40);
  await page.locator("header").getByText("e2e set").click();
  await page.keyboard.press("ArrowRight");
  await expect.poll(() => app.videoTime()).toBeCloseTo(45, 0);
  await page.keyboard.press("Shift+ArrowLeft");
  await expect.poll(() => app.videoTime()).toBeCloseTo(35, 0);
  await page.keyboard.press(".");
  await expect.poll(() => app.videoTime()).toBeCloseTo(35 + 1 / 30, 2);
  await page.keyboard.press(" ");
  await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.paused)).toBe(false);
  await page.keyboard.press(" ");
  await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.paused)).toBe(true);
  await page.keyboard.press("m");
  await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.muted || v.volume === 0)).toBe(true);
  await page.keyboard.press("m");
  await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.volume > 0 && !v.muted)).toBe(true);
  await page.keyboard.press("d");
  await expect(page.getByText("Drawing — Esc to finish")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByText("Drawing — Esc to finish")).toHaveCount(0);

  // speed
  await page.getByTitle("Playback speed").selectOption("2");
  await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.playbackRate)).toBe(2);
  await page.getByTitle("Playback speed").selectOption("1");
  await page.locator("header").getByText("e2e set").click();
});

test("reload the results keeps the screen", async ({ app }) => {
  const { page } = app;
  await app.clearCalls();
  await page.getByRole("button", { name: "Reload the results" }).click();
  await expect.poll(async () => (await app.calls("get_rounds")).length).toBeGreaterThan(0);
  await expect(page.getByRole("tab", { name: /^Overview/ })).toBeVisible();
  await expect(page.locator("header").getByRole("button", { name: "M2", exact: true })).toBeVisible();
});
