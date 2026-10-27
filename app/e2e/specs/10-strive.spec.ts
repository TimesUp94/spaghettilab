// Strive replay export page. The app never drives the game here: the run has SPAGHETTI_LAB_NO_GAME
// set (the backend refuses), and the "running" flow is a mocked export that sends the same
// progress events as the real one.
import { test, expect } from "../fixtures";

test.beforeEach(async ({ app }) => {
  await app.rail("Strive").click();
  await expect(app.page.getByRole("heading", { name: "Export Strive replays" })).toBeVisible();
});

async function fillSetup(app: import("../fixtures").App, folder: string) {
  const { page } = app;
  await app.queueDialogs(folder);
  await page.getByRole("button", { name: "Browse", exact: true }).click();
  await expect(page.getByText(folder)).toBeVisible();
  const ack = page.getByRole("checkbox", { name: /The game is at the title screen/ });
  if (!(await ack.isChecked())) await ack.check();
}

test("start needs an output folder and the acknowledgement", async ({ app }) => {
  const { page } = app;
  const start = page.getByRole("button", { name: "Start Export" });
  await expect(start).toBeDisabled();

  const folder = app.file("strive-out");
  await app.queueDialogs(folder);
  await page.getByRole("button", { name: "Browse", exact: true }).click();
  await expect(page.getByText(folder)).toBeVisible();
  const [open] = await app.calls("plugin:dialog|open");
  expect(open.args.options.directory).toBe(true);
  await expect(start).toBeDisabled();

  await page.getByRole("checkbox", { name: /The game is at the title screen/ }).check();
  await expect(start).toBeEnabled();

  // dry run renames the button
  await page.getByText("Advanced").click();
  await page.getByRole("checkbox", { name: /List only/ }).check();
  await expect(page.getByRole("button", { name: "List Replays" })).toBeVisible();
  await page.getByRole("checkbox", { name: /List only/ }).uncheck();
  await expect(start).toBeVisible();
});

test("the output folder is remembered", async ({ app }) => {
  const { page } = app;
  const folder = app.file("strive-remembered");
  await app.queueDialogs(folder);
  await page.getByRole("button", { name: "Browse", exact: true }).click();
  await app.rail("Library").click();
  await app.rail("Strive").click();
  await expect(page.getByText(folder)).toBeVisible();
});

test("the backend refuses to drive the game in tests, and the error is shown", async ({ app }) => {
  const { page } = app;
  await fillSetup(app, app.file("strive-refused"));
  await page.getByRole("radio", { name: /All saved replays/ }).check();
  await page.getByRole("button", { name: "Start Export" }).click();
  await expect(page.getByText("Export stopped")).toBeVisible();
  await expect(page.getByText(/Driving the game is disabled/)).toBeVisible();
  const [call] = await app.calls("export_strive_replays");
  expect(call.args.options.mode).toBe("all");

  await page.getByRole("button", { name: "Export more" }).click();
  await expect(page.getByRole("button", { name: "Start Export" })).toBeVisible();
});

test("an export run: progress, stop after this replay, summary", async ({ app }) => {
  const { page } = app;
  const folder = app.file("strive-mock");
  // the mocked command: progress events like the real exporter, then waits for the stop
  await page.evaluate(() => {
    const w = window as any;
    const emit = (payload: object) =>
      w.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "strive-export-progress", payload: JSON.stringify(payload) });
    let stop: () => void = () => {};
    const stopped = new Promise<void>((r) => (stop = r));
    w.__e2e.handlers["cancel_strive_export"] = () => {
      stop();
      return null;
    };
    w.__e2e.handlers["export_strive_replays"] = async () => {
      await emit({ event: "ffmpeg", capture: "gfxcapture", encoder: "h264_nvenc" });
      await emit({ event: "screen", screen: "saved_list", detail: "Saved replays" });
      await emit({ event: "list", total: 12 });
      await emit({ event: "entry", index: 0, file: "replay_0001.mp4", date: "2026/10/27", time: "10:15:00", p1: "Alpha", p2: "Bravo" });
      await emit({ event: "entry", index: 1, file: "replay_0002.mp4", date: "2026/10/27", time: "10:20:00", p1: "Charlie", p2: "Delta" });
      await emit({ event: "recording", index: 0 });
      await stopped;
      await emit({ event: "recorded", index: 0, partial: false, duration_s: 104 });
      return { event: "done", reason: "stopped", listed: 2, exported: ["replay_0001.mp4"], skipped: [], manifest: "manifest.json" };
    };
  });

  await fillSetup(app, folder);
  await page.getByRole("spinbutton").first().fill("2");
  await page.getByRole("button", { name: "Start Export" }).click();

  await expect(page.getByText("Exporting - don't use the game window")).toBeVisible();
  await expect(page.getByText("Alpha vs Bravo")).toBeVisible();
  await expect(page.getByText("Charlie vs Delta")).toBeVisible();
  await expect(page.getByText("Recording replay 1...")).toBeVisible();
  await expect(page.getByText(/window capture, h264_nvenc/)).toBeVisible();
  await expect(page.getByText(/12 saved in the game/)).toBeVisible();
  await expect(page.getByRole("main").getByRole("button", { name: "Library", exact: true })).toBeDisabled();
  const [call] = await app.calls("export_strive_replays");
  expect(call.args.options).toMatchObject({ mode: "count", count: 2, output_dir: folder, dry_run: false });

  await page.getByRole("button", { name: "Stop after this replay" }).click();
  expect((await app.calls("cancel_strive_export"))[0].args).toEqual({ now: false });

  await expect(page.getByText(/^1 exported/)).toBeVisible();
  await expect(page.getByText("done", { exact: true })).toBeVisible();
  const analyze = page.getByRole("button", { name: "Analyze 1 video" });
  await expect(analyze).toBeVisible();

  // the recorded file doesn't really exist: its analysis fails and says so
  await analyze.click();
  await expect(page.getByText("No video could be analyzed.")).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Close" }).click();
  await expect(app.rail("Library")).toHaveAttribute("aria-current", "page");
});
