// Notes and drawings on the review screen: add, edit, delete, persist.
import { test, expect, type App } from "../fixtures";
import { DEFAULT_DB, REPLAY_ID } from "../env";

const notes = (app: App) =>
  app.invoke<{ note_id: number; text: string; timestamp_ms: number }[]>("get_notes", { dbPath: DEFAULT_DB, replayId: REPLAY_ID });
const drawings = (app: App) =>
  app.invoke<{ drawing_id: number; timestamp_ms: number; strokes_json: string }[]>("get_drawings", { dbPath: DEFAULT_DB, replayId: REPLAY_ID });


test.beforeEach(async ({ app }) => {
  // start from no notes / drawings
  for (const n of await notes(app)) await app.invoke("delete_note", { dbPath: DEFAULT_DB, noteId: n.note_id });
  for (const d of await drawings(app)) await app.invoke("delete_drawing", { dbPath: DEFAULT_DB, drawingId: d.drawing_id });
  await app.openReview();
  await app.page.getByRole("tab", { name: /^Notes/ }).click();
});

test("add, edit and delete a note", async ({ app }) => {
  const { page } = app;
  await expect(page.getByText(/^No notes yet/)).toBeVisible();
  await app.seek(30);
  await expect(page.getByText("@ 0:30")).toBeVisible();

  const input = page.getByPlaceholder("Add a note at current timestamp...");
  await expect(page.getByRole("button", { name: "Add", exact: true })).toBeDisabled();
  await input.fill("Whiff punish missed");
  await input.press("Enter");
  await expect(page.getByText("Whiff punish missed")).toBeVisible();
  await expect(input).toHaveValue("");
  await expect(page.getByRole("tab", { name: /^Notes/ })).toHaveText("Notes1");
  let saved = await notes(app);
  expect(saved).toHaveLength(1);
  expect(saved[0].timestamp_ms).toBeGreaterThanOrEqual(29_500);
  expect(saved[0].timestamp_ms).toBeLessThanOrEqual(30_500);

  // a second note with the button, at another time: the list is in time order
  await app.seek(10);
  await expect(page.getByText("@ 0:10")).toBeVisible();
  await input.fill("Opening round start");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.getByText("Opening round start")).toBeVisible();
  const texts = await page.getByText(/^(Opening round start|Whiff punish missed)$/).allTextContents();
  expect(texts).toEqual(["Opening round start", "Whiff punish missed"]);

  // clicking a note seeks to it
  await page.getByText("Whiff punish missed").click();
  await expect.poll(() => app.videoTime()).toBeCloseTo(30, 0);

  // edit (Enter saves, Escape cancels)
  const row = page.getByText("Whiff punish missed").locator("xpath=ancestor::*[.//button[text()='edit']][1]");
  await row.hover();
  await row.getByRole("button", { name: "edit" }).click();
  const edit = page.locator("input:focus");
  await edit.fill("Should have blocked low");
  await edit.press("Enter");
  await expect(page.getByText("Should have blocked low")).toBeVisible();
  saved = await notes(app);
  expect(saved.map((n) => n.text).sort()).toEqual(["Opening round start", "Should have blocked low"]);

  const row2 = page.getByText("Opening round start").locator("xpath=ancestor::*[.//button[text()='edit']][1]");
  await row2.hover();
  await row2.getByRole("button", { name: "edit" }).click();
  await page.locator("input:focus").fill("discarded");
  await page.locator("input:focus").press("Escape");
  await expect(page.getByText("Opening round start")).toBeVisible();

  // notes survive a reload of the results
  await page.getByRole("button", { name: "Reload the results" }).click();
  await page.getByRole("tab", { name: /^Notes/ }).click();
  await expect(page.getByText("Should have blocked low")).toBeVisible();

  // delete
  const row3 = page.getByText("Should have blocked low").locator("xpath=ancestor::*[.//button[text()='delete']][1]");
  await row3.hover();
  await row3.getByRole("button", { name: "delete" }).click();
  await expect(page.getByText("Should have blocked low")).toHaveCount(0);
  expect(await notes(app)).toHaveLength(1);
});

test("draw on a frame, then delete the drawing", async ({ app }) => {
  const { page } = app;
  await app.seek(50);
  await page.getByRole("button", { name: "Draw", exact: true }).click();
  await expect(page.getByText("Drawing — Esc to finish")).toBeVisible();
  for (const b of ["Pen", "Eraser", "Clear", "Done"]) await expect(page.getByRole("button", { name: b, exact: true })).toBeVisible();

  await page.getByTitle("Blue", { exact: true }).click();
  const canvas = page.locator("canvas").filter({ visible: true }).first();
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 10 });
  await page.mouse.up();
  await page.getByRole("button", { name: "Done", exact: true }).click();
  await expect(page.getByText("Drawing — Esc to finish")).toHaveCount(0);

  await expect.poll(async () => (await drawings(app)).length).toBe(1);
  const [d] = await drawings(app);
  expect(d.timestamp_ms).toBeGreaterThanOrEqual(49_500);
  const strokes = JSON.parse(d.strokes_json);
  expect(strokes.length).toBeGreaterThan(0);
  expect(strokes[0].points.length).toBeGreaterThan(2);

  await expect(page.getByText("Drawing", { exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: /^Notes/ })).toHaveText("Notes1");

  const row = page.getByText("Drawing", { exact: true }).locator("xpath=ancestor::*[.//button[text()='delete']][1]");
  await row.hover();
  await row.getByRole("button", { name: "delete" }).click();
  await expect(page.getByText("Drawing", { exact: true })).toHaveCount(0);
  await expect.poll(async () => (await drawings(app)).length).toBe(0);
});
