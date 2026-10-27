// Layout at different window sizes: no horizontal scrolling, the video fits, the panel moves under
// the video on narrow windows.
import { test, expect, type App } from "../fixtures";

const SIZES: [number, number][] = [
  [1920, 1080],
  [1440, 900],
  [1366, 768],
  [1280, 720],
  [1100, 700],
  [1023, 768],
  [900, 800],
  [800, 700],
];

async function noHorizontalScroll(app: App) {
  const { scroll, width } = await app.page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    width: window.innerWidth,
  }));
  expect(scroll).toBeLessThanOrEqual(width);
}

test("review screen at every size", async ({ app }) => {
  const { page } = app;
  await app.openReview();
  for (const [w, h] of SIZES) {
    await test.step(`${w}×${h}`, async () => {
      await page.setViewportSize({ width: w, height: h });
      await page.waitForTimeout(150);
      await noHorizontalScroll(app);

      const video = (await page.locator("video").boundingBox())!;
      const tabs = (await page.getByRole("tablist").boundingBox())!;
      // 16:9, inside the window
      expect(Math.abs(video.width / video.height - 16 / 9)).toBeLessThan(0.03);
      expect(video.x + video.width).toBeLessThanOrEqual(w + 1);
      expect(video.x).toBeGreaterThanOrEqual(0);

      if (w >= 1024) {
        // side by side, and the video column fits the window height
        expect(tabs.x).toBeGreaterThanOrEqual(video.x + video.width - 1);
        expect(video.y + video.height).toBeLessThanOrEqual(h);
        const timeline = (await page.getByText("P1 health").boundingBox())!;
        expect(timeline.y).toBeGreaterThan(video.y + video.height - 1);
        expect(timeline.y + timeline.height).toBeLessThanOrEqual(h);
      } else {
        // stacked: results under the video
        expect(tabs.y).toBeGreaterThan(video.y + video.height - 1);
      }
    });
  }
});

test("library, welcome-style pages and tools at narrow and wide sizes", async ({ app }) => {
  const { page } = app;
  await app.openLibrary();
  for (const [w, h] of [[1440, 900], [800, 700], [640, 700]] as [number, number][]) {
    await page.setViewportSize({ width: w, height: h });
    for (const target of ["Library", "Analyse", "Annotate", "Split VOD", "Strive"]) {
      await test.step(`${target} at ${w}×${h}`, async () => {
        await app.rail(target).click();
        await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
        await noHorizontalScroll(app);
        const heading = (await page.getByRole("heading", { level: 1 }).boundingBox())!;
        expect(heading.x + heading.width).toBeLessThanOrEqual(w);
      });
    }
  }
});
