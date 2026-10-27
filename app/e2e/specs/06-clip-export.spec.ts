// Exporting a clip of a round / game / highlight with ffmpeg.
import fs from "node:fs";
import { test, expect } from "../fixtures";

test.beforeEach(async ({ app }) => {
  await app.openReview();
});

test("export a round clip with padding", async ({ app }) => {
  const { page } = app;
  await page.getByRole("tab", { name: /^Rounds/ }).click();
  await page.getByTitle("Export round clip").first().click();
  await expect(page.getByRole("heading", { name: "Export Clip" })).toBeVisible();
  await expect(page.getByText(/^Segment duration: [\d.]+s \(with padding: [\d.]+s\)$/)).toBeVisible();

  const pads = page.getByRole("spinbutton");
  await pads.nth(0).fill("1");
  await pads.nth(1).fill("0");

  const out = app.file("round1.mp4");
  await app.queueDialogs(out);
  await page.getByRole("button", { name: "Export", exact: true }).last().click();
  await expect(page.getByText("Clip exported successfully")).toBeVisible({ timeout: 60_000 });
  expect(fs.statSync(out).size).toBeGreaterThan(10_000);
  const [call] = await app.calls("export_clip");
  expect(call.args.endMs - call.args.startMs).toBeGreaterThan(20_000);

  await page.getByRole("button", { name: "Close", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Export Clip" })).toHaveCount(0);
});

test("a cancelled save dialog exports nothing", async ({ app }) => {
  const { page } = app;
  await page.getByTitle("Export match clip").first().click();
  await expect(page.getByRole("heading", { name: "Export Clip" })).toBeVisible();
  await page.getByRole("button", { name: "Export", exact: true }).last().click();
  await expect.poll(async () => (await app.calls("plugin:dialog|save")).length).toBe(1);
  expect(await app.calls("export_clip")).toHaveLength(0);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Export Clip" })).toHaveCount(0);
});

test("highlights open the clip export too", async ({ app }) => {
  const { page } = app;
  await page.getByRole("tab", { name: /^Highlights/ }).click();
  await page.getByRole("button", { name: "Export", exact: true }).first().click();
  await expect(page.getByRole("heading", { name: "Export Clip" })).toBeVisible();
  await page.getByRole("button", { name: "x", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Export Clip" })).toHaveCount(0);
});
