// Annotate: notes, drawings and the review checklist on a video without analysing it.
import fs from "node:fs";
import { test, expect } from "../fixtures";
import path from "node:path";
import { DATA_DIR, FIXTURE_VIDEO } from "../env";

// start without the notes sessions of earlier runs (E2E_REUSE_DATA keeps them)
test.beforeAll(() => {
  fs.rmSync(path.join(DATA_DIR, "SpaghettiLab", "quick_sessions"), { recursive: true, force: true });
});

test.describe("annotate page", () => {
  test("validates YouTube and Twitch links", async ({ app }) => {
    const { page } = app;
    await app.rail("Annotate").click();
    const input = page.getByPlaceholder("Paste a YouTube or Twitch link");
    const open = page.getByRole("main").getByRole("button", { name: "Open", exact: true });
    await expect(open).toBeDisabled();

    await input.fill("https://example.com/video");
    await expect(page.getByText("That doesn't look like a YouTube or Twitch video link.")).toBeVisible();
    await expect(open).toBeDisabled();

    await input.fill("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    await expect(page.getByText("YouTube video dQw4w9WgXcQ")).toBeVisible();
    await expect(open).toBeEnabled();

    await input.fill("https://youtu.be/dQw4w9WgXcQ?t=42");
    await expect(page.getByText("YouTube video dQw4w9WgXcQ")).toBeVisible();

    await input.fill("https://www.twitch.tv/videos/2468013579");
    await expect(page.getByText("Twitch video 2468013579")).toBeVisible();
    await expect(open).toBeEnabled();
  });

  test("a YouTube link opens an embedded session", async ({ app }) => {
    const { page } = app;
    await app.rail("Annotate").click();
    await page.getByPlaceholder("Paste a YouTube or Twitch link").fill("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    await page.getByPlaceholder("Paste a YouTube or Twitch link").press("Enter");
    await expect(page.getByText("Notes and drawings")).toBeVisible();
    const [call] = await app.calls("create_quick_session");
    expect(call.args).toMatchObject({ videoPath: "", videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ" });
    await expect(page.locator("video")).toHaveCount(0); // an embed, not a file
    await page.getByRole("button", { name: "Back to the library" }).click();
    // the library is the session's: one notes-only video, nothing to reanalyse
    await expect(page.getByText("Notes only")).toHaveCount(1);
    await expect(page.getByText(/^e2e set$/)).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Reanalyse all" })).toHaveCount(0);
  });
});

test.describe.serial("annotate a local file", () => {
  test("browse a file, take a note, draw, fill the checklist, save as .spagz", async ({ app }) => {
    const { page } = app;
    await app.rail("Annotate").click();
    await app.queueDialogs(FIXTURE_VIDEO);
    await page.getByRole("button", { name: /Drop a video file here/ }).click();
    await expect(page.getByText("Notes and drawings")).toBeVisible();
    await expect(app.rail("Annotate")).toHaveAttribute("aria-current", "page");
    await expect.poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.readyState)).toBeGreaterThanOrEqual(1);

    // note
    await app.seek(12);
    const input = page.getByPlaceholder("Add a note at current timestamp...");
    await input.fill("Annotate-only note");
    await input.press("Enter");
    await expect(page.getByText("Annotate-only note")).toBeVisible();
    expect(await app.calls("add_note")).toHaveLength(1);

    // drawing (D with the focus out of the note field)
    await input.blur();
    await page.keyboard.press("d");
    await expect(page.getByText("Drawing — Esc to finish")).toBeVisible();
    const canvas = page.locator("canvas").filter({ visible: true }).first();
    const box = (await canvas.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.6, { steps: 8 });
    await page.mouse.up();
    await page.keyboard.press("Escape");
    await expect(page.getByText("Drawing", { exact: true })).toBeVisible();

    // checklist
    await page.getByRole("button", { name: "Checklist" }).click();
    const panel = page.getByRole("dialog", { name: "Review Questionnaire" });
    await expect(panel).toBeVisible();
    const general = panel.getByPlaceholder("Write your overall impressions of the match...");
    await general.fill("Good defence, too many jumps.");
    await expect(panel.getByText("Saved")).toBeVisible();
    const [answer] = await app.calls("set_questionnaire_answer");
    expect(answer.args.answerText).toBe("Good defence, too many jumps.");
    expect(answer.args.questionKey).toMatch(/^ggst\./);
    await panel.getByRole("button", { name: "Close" }).click();
    await expect(panel).toHaveCount(0);
    await page.getByRole("button", { name: "Checklist" }).click();
    await expect(page.getByPlaceholder("Write your overall impressions of the match...")).toHaveValue("Good defence, too many jumps.");
    await page.getByRole("dialog", { name: "Review Questionnaire" }).getByRole("button", { name: "Close" }).click();

    // save as .spagz
    const out = app.file("annotated.spagz");
    await app.queueDialogs(out);
    await page.getByRole("button", { name: /Save as \.spagz/ }).click();
    await expect.poll(() => fs.existsSync(out)).toBe(true);

    // the library of the session lists it as notes only, and it reopens with its note
    await page.getByRole("button", { name: "Back to the library" }).click();
    await expect(page.getByText("Notes only")).toBeVisible();
    await page.getByText("Notes only").click();
    await expect(page.getByText("Annotate-only note")).toBeVisible();
  });

  test("the saved .spagz opens straight into the annotate view", async ({ app }) => {
    const { page } = app;
    await app.openLibrary();
    await app.queueDialogs(app.file("annotated.spagz", false));
    await app.rail("Open").click();
    await expect(page.getByText("Notes and drawings")).toBeVisible();
    await expect(page.getByText("Annotate-only note")).toBeVisible();
  });

  test("a video dropped on the annotate page opens a notes session", async ({ app }) => {
    const { page } = app;
    await app.rail("Annotate").click();
    await app.dropFiles([FIXTURE_VIDEO]);
    await expect(page.getByText("Notes and drawings")).toBeVisible();
    expect(await app.calls("analyze_video")).toHaveLength(0);
    await app.openLibrary();
  });
});
