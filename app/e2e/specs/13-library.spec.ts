// Library details: search (shown above 6 videos), set cards, and the update banner.
import { test, expect } from "../fixtures";

const replay = (replay_id: string, minutes: number) => ({
  replay_id,
  video_path: `C:/videos/${replay_id}.mp4`,
  duration_ms: minutes * 60_000,
  frame_count: minutes * 1800,
});

test("search filters a large library, set names are read from the file name", async ({ app }) => {
  const { page } = app;
  await page.evaluate(
    (list) => ((window as any).__e2e.handlers["get_replays"] = () => list),
    [
      replay("01_WRI_Slycoops_vs_Leftover", 10),
      replay("02_UF_Bromay_vs_Kurisu", 5),
      replay("03_UF_SlayThatDude_vs_Salkazko", 6),
      replay("04_UF_GuessingGame_vs_Ricbionicle", 5),
      replay("05_UF_Jason_vs_EneroNeJR", 5),
      replay("06_UF_Jason_vs_Salkazko", 5),
      replay("07_UF_Dona_vs_GuessingGame", 8),
      replay("my_casual_match", 3),
    ],
  );
  await app.openLibrary();
  await expect(page.getByText("8 analysed videos")).toBeVisible();

  // a set card: players, bracket and set number, length
  const card = page.getByRole("button", { name: /Slycoops.*Leftover/ });
  await expect(card).toContainText("VS");
  await expect(card).toContainText("WRI · Set 1");
  await expect(card).toContainText("10 min");

  const search = page.getByPlaceholder("Search players or file names");
  await search.fill("jason");
  await expect(page.getByRole("button", { name: /Jason/ })).toHaveCount(2);
  await expect(page.getByRole("button", { name: /Slycoops/ })).toHaveCount(0);
  await search.fill("casual");
  await expect(page.getByRole("button", { name: /my casual match/ })).toHaveCount(1);
  await search.fill("nobody");
  await expect(page.getByText("No match for your search")).toBeVisible();
  await search.fill("");
  await expect(page.getByText("No match for your search")).toHaveCount(0);

  // back to the real library
  await page.evaluate(() => delete (window as any).__e2e.handlers["get_replays"]);
  await app.openLibrary();
  await expect(page.getByPlaceholder("Search players or file names")).toHaveCount(0);
});

test("an available update shows the banner, which can be put off", async ({ app }) => {
  const { page } = app;
  await page.evaluate(() => sessionStorage.setItem("e2e.update", "9.9.9"));
  await page.reload();
  try {
    await expect(page.getByText("v9.9.9 available")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole("button", { name: "Update & restart" })).toBeVisible();
    await page.getByRole("button", { name: "Later" }).click();
    await expect(page.getByText("v9.9.9 available")).toHaveCount(0);
  } finally {
    await page.evaluate(() => sessionStorage.removeItem("e2e.update"));
  }
});
