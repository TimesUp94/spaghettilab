// Analysing a video: picking it, failures, and a full run of the real pipeline on the fixture clip
// (dropped onto the window). The later specs use its result.
import fs from "node:fs";
import { test, expect } from "../fixtures";
import { FIXTURE_VIDEO } from "../env";

test.describe("analyse page", () => {
  test("pick a video, change it, cancel", async ({ app }) => {
    const { page } = app;
    await app.rail("Analyse").click();
    await expect(page.getByRole("button", { name: "Start the analysis" })).toBeDisabled();

    await app.queueDialogs("C:/videos/first.mp4");
    await page.getByRole("button", { name: /Choose a match video/ }).click();
    await expect(page.getByText("first.mp4", { exact: true })).toBeVisible();
    const [open] = await app.calls("plugin:dialog|open");
    expect(open.args.options.filters[0].extensions).toEqual(expect.arrayContaining(["mp4", "mkv", "avi"]));

    await app.queueDialogs("C:/videos/second.mkv");
    await page.getByRole("button", { name: "Change" }).click();
    await expect(page.getByText("second.mkv", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Start the analysis" })).toBeEnabled();

    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(app.rail("Library")).toHaveAttribute("aria-current", "page");
  });

  test("a file that isn't a video fails with the error and can be retried", async ({ app }) => {
    const { page } = app;
    const broken = app.file("broken.mp4");
    fs.writeFileSync(broken, Buffer.alloc(4096, 7));

    await app.rail("Analyse").click();
    await app.queueDialogs(broken);
    await page.getByRole("button", { name: /Choose a match video/ }).click();
    await page.getByRole("button", { name: "Start the analysis" }).click();
    await expect(page.getByText("The analysis failed")).toBeVisible({ timeout: 60_000 });

    await page.getByRole("button", { name: "Try again" }).click();
    await expect(page.getByRole("button", { name: "Start the analysis" })).toBeVisible();
    await page.getByRole("main").getByRole("button", { name: "Library", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Welcome to Spaghetti Lab" })).toBeVisible();
  });
});

test("dropping a match video analyses it and adds it to the library", async ({ app }) => {
  test.setTimeout(8 * 60_000);
  const { page } = app;
  await app.rail("Library").click();
  await app.dropFiles([FIXTURE_VIDEO]);

  // starts at once, with real progress from the pipeline
  await expect(page.getByRole("heading", { name: "Analyse a match" })).toBeVisible();
  await expect(page.getByText("e2e_set.mp4")).toBeVisible();
  await expect(page.getByRole("main").getByRole("button", { name: "Library", exact: true })).toBeDisabled();
  await expect(page.getByText(/^\d{1,3}%$/)).toBeVisible({ timeout: 60_000 });
  for (const stage of [
    "Reading the video",
    "Reading health, tension, timer and rounds",
    "Finding damage, combos and rounds",
    "Saving the results",
  ]) {
    await expect(page.getByText(stage)).toBeVisible();
  }
  const seen = new Set<number>();
  // sample the percentage while it runs: it must move forward
  for (let i = 0; i < 20; i++) {
    const t = await page.getByText(/^\d{1,3}%$/).textContent().catch(() => null);
    if (t) seen.add(Number(t.replace("%", "")));
    if (seen.size >= 3) break;
    await page.waitForTimeout(2_000);
  }
  expect(seen.size).toBeGreaterThanOrEqual(2);

  // done: the library of the new database
  await expect(page.getByRole("heading", { name: "Library" })).toBeVisible({ timeout: 6 * 60_000 });
  await expect(page.getByText("1 analysed video · analysis.db")).toBeVisible();
  await expect(app.replayCard()).toBeVisible();
  await expect(app.replayCard()).toContainText("3 min");
});
