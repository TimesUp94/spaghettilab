// Split VOD: pick a video, the detection regions, a real scan (EasyOCR), cut the sets.
import fs from "node:fs";
import { test, expect } from "../fixtures";
import { FIXTURE_VIDEO } from "../env";

test.beforeEach(async ({ app }) => {
  await app.rail("Split VOD").click();
  await expect(app.page.getByRole("heading", { name: "Split a tournament VOD" })).toBeVisible();
});

test("Next needs a video; Download needs a link", async ({ app }) => {
  const { page } = app;
  await expect(page.getByRole("button", { name: "Next" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Download" })).toBeDisabled();
  await page.getByPlaceholder("https://www.twitch.tv/videos/...").fill("https://www.twitch.tv/videos/1");
  await expect(page.getByRole("button", { name: "Download" })).toBeEnabled();
  await page.getByPlaceholder("https://www.twitch.tv/videos/...").fill("");

  await app.queueDialogs(FIXTURE_VIDEO);
  await page.getByRole("button", { name: "Browse", exact: true }).click();
  await expect(page.getByText(FIXTURE_VIDEO)).toBeVisible();
  await expect(page.getByRole("button", { name: "Next" })).toBeEnabled();
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(app.rail("Library")).toHaveAttribute("aria-current", "page");
});

test("scan the fixture and cut its set", async ({ app }) => {
  test.setTimeout(10 * 60_000);
  const { page } = app;
  await app.queueDialogs(FIXTURE_VIDEO);
  await page.getByRole("button", { name: "Browse", exact: true }).click();
  await page.getByRole("button", { name: "Next" }).click();

  // detection regions on a preview frame
  await expect(page.getByRole("heading", { name: "Adjust Detection Regions" })).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("img").filter({ visible: true }).first()).toBeVisible();
  await expect(page.getByText("Timer", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Back" }).click();
  await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByRole("button", { name: "Start Scan" }).click();

  await expect(page.getByText("Scanning VOD for gameplay...")).toBeVisible();
  await expect(page.getByText(/^Found \d+ sets?\. Select which ones to cut:$/)).toBeVisible({ timeout: 8 * 60_000 });
  const found = Number((await page.getByText(/^Found \d+ sets?/).textContent())!.match(/\d+/)![0]);
  expect(found).toBeGreaterThanOrEqual(1);
  const sets = page.getByRole("checkbox");
  await expect(sets).toHaveCount(found);
  await expect(page.getByText(/\d+ match(es)?$/).first()).toBeVisible();

  const cut = page.getByRole("button", { name: /^Cut \d+ Sets?$/ });
  await expect(cut).toBeDisabled(); // no output folder yet
  const out = app.file("split-out");
  fs.mkdirSync(out);
  await app.queueDialogs(out);
  await page.getByRole("button", { name: "Browse", exact: true }).click();
  await expect(page.getByText(out)).toBeVisible();
  await expect(cut).toHaveText(found === 1 ? "Cut 1 Set" : `Cut ${found} Sets`);
  await cut.click();

  // done: back to the library, one video per set
  await expect(app.rail("Library")).toHaveAttribute("aria-current", "page", { timeout: 4 * 60_000 });
  const files = fs.readdirSync(out).filter((f) => f.endsWith(".mp4"));
  expect(files).toHaveLength(found);
  for (const f of files) expect(fs.statSync(`${out}/${f}`).size).toBeGreaterThan(100_000);
});
