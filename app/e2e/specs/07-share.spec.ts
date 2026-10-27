// Sharing: export .spag / .spagz, open them again (with or without the video), save a session.
import fs from "node:fs";
import { test, expect } from "../fixtures";
import { FIXTURE_VIDEO } from "../env";

const SPAG = "shared.spag";
const SPAGZ = "shared.spagz";

test.describe.serial("share files", () => {
  test("export .spag and .spagz from the review screen", async ({ app }) => {
    const { page } = app;
    await app.openReview();

    const spag = app.file(SPAG);
    await app.queueDialogs(spag);
    await page.getByRole("button", { name: "Share" }).click();
    await page.getByRole("button", { name: /^Export \.spag\b/ }).click();
    await expect.poll(() => fs.existsSync(spag) && fs.statSync(spag).size, { timeout: 60_000 }).toBeGreaterThan(1_000_000);
    const [save] = await app.calls("plugin:dialog|save");
    expect(save.args.options.defaultPath).toBe("e2e_set.spag");

    const spagz = app.file(SPAGZ);
    await app.queueDialogs(spagz);
    await page.getByRole("button", { name: "Share" }).click();
    await page.getByRole("button", { name: /^Export \.spagz/ }).click();
    await expect.poll(() => fs.existsSync(spagz) && fs.statSync(spagz).size, { timeout: 30_000 }).toBeGreaterThan(1_000);
    // no video inside: much smaller than the .spag
    expect(fs.statSync(spagz).size).toBeLessThan(fs.statSync(spag).size / 4);
  });

  test("a cancelled export writes nothing", async ({ app }) => {
    const { page } = app;
    await app.openReview();
    await page.getByRole("button", { name: "Share" }).click();
    await page.getByRole("button", { name: /^Export \.spag\b/ }).click();
    await expect.poll(async () => (await app.calls("plugin:dialog|save")).length).toBe(1);
    expect(await app.calls("export_spag")).toHaveLength(0);
  });

  test("open a .spag: the review with its own video, and Save", async ({ app }) => {
    const { page } = app;
    await app.rail("Library").click();
    await app.queueDialogs(app.file(SPAG, false));
    await app.rail("Open").click();
    await expect(page.getByRole("tab", { name: /^Overview/ })).toBeVisible();
    await expect(page.locator("header").getByRole("button", { name: "M2", exact: true })).toBeVisible();
    await expect(page.locator("video")).toHaveCount(1);
    await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.readyState)).toBeGreaterThanOrEqual(1);

    // a session saves back into its file
    await page.getByRole("button", { name: "Save .spag" }).click();
    await expect.poll(async () => (await app.calls("save_spag")).length).toBe(1);
    await expect(page.getByText(/^Saving failed/)).toHaveCount(0);

    // its library is the session's
    await page.getByRole("button", { name: "Back to the library" }).click();
    await expect(page.getByText(/1 analysed video · /)).toBeVisible();
    await expect(page.getByRole("button", { name: "Reanalyse all" })).toHaveCount(0);
  });

  test("open a .spagz without the video, then locate it", async ({ app }) => {
    const { page } = app;
    await app.rail("Library").click();
    await app.queueDialogs(app.file(SPAGZ, false));
    await app.rail("Open").click();
    await expect(page.getByRole("heading", { name: "Choose Video Source" })).toBeVisible();
    for (const t of ["Browse File", "Download VOD", "Stream URL", "No Video"]) {
      await expect(page.getByRole("button", { name: t, exact: true })).toBeVisible();
    }
    await page.getByRole("button", { name: "No Video", exact: true }).click();
    await page.getByRole("button", { name: "Open Without Video" }).click();

    await expect(page.getByText("Video file not found")).toBeVisible();
    await expect(page.getByRole("tab", { name: /^Overview/ })).toBeVisible();
    await expect(page.getByRole("button", { name: "Save .spagz" })).toBeVisible();

    await app.queueDialogs(FIXTURE_VIDEO);
    await page.getByRole("button", { name: "Locate video" }).first().click();
    await expect(page.locator("video")).toHaveCount(1);
    await expect(page.getByText("Video file not found")).toHaveCount(0);
  });

  test("open a .spagz and browse for its video", async ({ app }) => {
    const { page } = app;
    await app.rail("Library").click();
    await app.queueDialogs(app.file(SPAGZ, false));
    await app.rail("Open").click();
    await page.getByRole("button", { name: "Browse File", exact: true }).click();
    await app.queueDialogs(FIXTURE_VIDEO);
    await page.getByRole("button", { name: "Browse...", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Choose Video Source" })).toHaveCount(0);
    await expect(page.locator("video")).toHaveCount(1);
    await expect(page.getByRole("tab", { name: /^Overview/ })).toBeVisible();
  });

  test("cancelling the video choice opens nothing", async ({ app }) => {
    const { page } = app;
    await app.openLibrary();
    await app.queueDialogs(app.file(SPAGZ, false));
    await app.rail("Open").click();
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Choose Video Source" })).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Library" })).toBeVisible();
  });

  test("dropping a .spag onto the window opens it", async ({ app }) => {
    const { page } = app;
    await app.openLibrary();
    await app.dropFiles([app.file(SPAG, false)]);
    await expect(page.getByRole("button", { name: "Save .spag" })).toBeVisible();
    // back to the default database
    await app.openLibrary();
    await expect(page.getByRole("button", { name: "Reanalyse all" })).toBeVisible();
  });
});
