// Correcting a round's winner from the games list, and undoing it.
import { test, expect, type App } from "../fixtures";
import { DEFAULT_DB, REPLAY_ID } from "../env";

const winners = async (app: App) =>
  (await app.invoke<{ winner: string }[]>("get_rounds", { dbPath: DEFAULT_DB, replayId: REPLAY_ID })).map((r) => r.winner);

test("override round 1 to P2, then back to P1", async ({ app }) => {
  const { page } = app;
  await app.openReview();
  const g1 = page.locator("header").getByRole("button", { name: "M1", exact: true });
  await expect(g1).toHaveAttribute("title", "Match 1: 2–0, won by P1");

  const setP2 = page.getByTitle("Set P2 as winner").first();
  await setP2.hover();
  await setP2.click();
  const [call] = await app.calls("set_round_winner");
  expect(call.args).toMatchObject({ roundIndex: 0, winner: "P2", replayId: REPLAY_ID });

  await expect.poll(() => winners(app)).toEqual(["P2", "P1", "P1", "P1"]);
  await expect(g1).toHaveAttribute("title", /^Match 1: 1–1/);

  // survives a reload
  await page.getByRole("button", { name: "Reload the results" }).click();
  await expect(g1).toHaveAttribute("title", /^Match 1: 1–1/);

  // back to P1 (round 1 now shows P2, so its "P1" button sets it)
  await app.clearCalls();
  const setP1 = page.getByTitle("Set P1 as winner").first();
  await setP1.hover();
  await setP1.click();
  expect((await app.calls("set_round_winner"))[0].args).toMatchObject({ roundIndex: 0, winner: "P1" });
  await expect.poll(() => winners(app)).toEqual(["P1", "P1", "P1", "P1"]);
  await expect(g1).toHaveAttribute("title", "Match 1: 2–0, won by P1");
});
