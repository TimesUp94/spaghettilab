// Running the analysis again: one video from its review screen, and the whole library.
import { test, expect, type App } from "../fixtures";
import { DEFAULT_DB, EXPECTED, REPLAY_ID } from "../env";

const winners = async (app: App) =>
  (await app.invoke<{ winner: string }[]>("get_rounds", { dbPath: DEFAULT_DB, replayId: REPLAY_ID })).map((r) => r.winner);

test("reanalyse one video from its review screen @slow", async ({ app }) => {
  test.setTimeout(8 * 60_000);
  const { page } = app;
  await app.openReview();
  await page.getByTitle("Run the analysis again on this video (after an app update)").click();
  await expect(page.getByText("Running the analysis again…")).toBeVisible();
  await expect(page.getByText("Running the analysis again…")).toHaveCount(0, { timeout: 6 * 60_000 });
  await expect(page.getByText(/^The analysis failed/)).toHaveCount(0);
  expect(await app.calls("reanalyze_replay")).toHaveLength(1);
  await expect(page.locator("header").getByRole("button", { name: "M2", exact: true })).toBeVisible();
  expect(await winners(app)).toEqual(EXPECTED.roundWinners);
});

test("reanalyse all from the library @slow", async ({ app }) => {
  test.setTimeout(8 * 60_000);
  const { page } = app;
  await app.openLibrary();
  await page.getByRole("button", { name: "Reanalyse all" }).click();
  await expect(page.getByText("Running the analysis again on every video…")).toBeVisible();
  await expect(page.getByText("Running the analysis again on every video…")).toHaveCount(0, { timeout: 6 * 60_000 });
  await expect(page.getByText(/^The analysis failed/)).toHaveCount(0);
  await expect(app.replayCard()).toBeVisible();
  expect(await winners(app)).toEqual(EXPECTED.roundWinners);
});
