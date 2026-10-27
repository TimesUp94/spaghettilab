// The `app` fixture: the running Spaghetti Lab window (one instance for the whole run, see
// global-setup.ts), plus helpers to drive what Playwright can't reach directly (native dialogs,
// file drops, backend events).
import { test as base, expect, chromium, type Browser, type Locator, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { CDP_PORT, DEFAULT_DB, FILES_DIR, FIXTURE_VIDEO, REPLAY_ID } from "./env";

/**
 * Runs in the page before the app's own code (and again now, for the page already loaded).
 *
 * The Tauri IPC goes through `fetch("http://ipc.localhost/<command>")` (window.__TAURI_INTERNALS__
 * itself is frozen), so commands are mocked by wrapping fetch:
 * - native dialogs (`plugin:dialog|open` / `save`) answer from a queue the test fills; an empty
 *   queue means the user cancelled;
 * - the updater check finds no update, and opening a link in the browser is only recorded;
 * - a test can add its own handler for any command (`__e2e.handlers[cmd] = async (args) => ...`).
 * Every command is logged in `__e2e.calls`.
 */
function installMocks() {
  const w = window as any;
  if (w.__e2e) return;
  const realFetch = window.fetch.bind(window);
  const e2e = {
    dialogs: [] as unknown[],
    calls: [] as { cmd: string; args: any }[],
    handlers: {} as Record<string, (args: any) => unknown>,
    builtin: {
      "plugin:dialog|open": () => (e2e.dialogs.length ? e2e.dialogs.shift() : null),
      "plugin:dialog|save": () => (e2e.dialogs.length ? e2e.dialogs.shift() : null),
      // an update is offered only when a test asked for one (it has to be there at startup)
      "plugin:updater|check": () => {
        const version = sessionStorage.getItem("e2e.update");
        return version ? { rid: 0, currentVersion: "0.0.0", version, date: null, body: "", rawJson: {} } : null;
      },
      "plugin:shell|open": () => null,
    } as Record<string, (args: any) => unknown>,
  };
  w.__e2e = e2e;
  const reply = (ok: boolean, value: unknown) =>
    new Response(JSON.stringify(value ?? null), {
      headers: { "Tauri-Response": ok ? "ok" : "error", "Content-Type": "application/json" },
    });
  window.fetch = async function (input: RequestInfo | URL, init?: RequestInit) {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url.startsWith("http://ipc.localhost/")) {
      const cmd = decodeURIComponent(url.slice("http://ipc.localhost/".length).split("?")[0]);
      let args: any = {};
      try {
        if (typeof init?.body === "string") args = JSON.parse(init.body);
      } catch {
        /* binary payload */
      }
      e2e.calls.push({ cmd, args });
      const h = e2e.handlers[cmd] ?? e2e.builtin[cmd];
      if (h) {
        try {
          return reply(true, await h(args));
        } catch (err) {
          return reply(false, String(err));
        }
      }
    }
    return realFetch(input as RequestInfo, init);
  } as typeof window.fetch;
}

export class App {
  page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  /** Answers for the next native file dialogs, in order (null = cancelled). */
  async queueDialogs(...results: (string | string[] | null)[]) {
    await this.page.evaluate((r) => (window as any).__e2e.dialogs.push(...r), results);
  }

  /** The commands the app sent since the last `clearCalls()` (optionally one command only). */
  async calls(cmd?: string): Promise<{ cmd: string; args: any }[]> {
    const all = await this.page.evaluate(() => (window as any).__e2e.calls);
    return cmd ? all.filter((c: { cmd: string }) => c.cmd === cmd) : all;
  }

  async clearCalls() {
    await this.page.evaluate(() => ((window as any).__e2e.calls = []));
  }

  /** Call a backend command directly (bypassing the UI). */
  async invoke<T = unknown>(cmd: string, args: Record<string, unknown> = {}): Promise<T> {
    return this.page.evaluate(
      ([c, a]) => (window as any).__TAURI_INTERNALS__.invoke(c, a),
      [cmd, args] as const,
    ) as Promise<T>;
  }

  /** Emit an app event through the backend, as Rust would (`listen()` handlers receive it). */
  async emit(event: string, payload: unknown) {
    await this.invoke("plugin:event|emit", { event, payload });
  }

  /** Simulate dragging files onto the window and dropping them. */
  async dropFiles(paths: string[]) {
    const target = { kind: "Webview", label: "main" };
    const position = { x: 400, y: 300 };
    await this.invoke("plugin:event|emit_to", { target, event: "tauri://drag-enter", payload: { paths, position } });
    await this.invoke("plugin:event|emit_to", { target, event: "tauri://drag-over", payload: { position } });
    await this.invoke("plugin:event|emit_to", { target, event: "tauri://drag-drop", payload: { paths, position } });
  }

  async dragOver(paths: string[]) {
    const target = { kind: "Webview", label: "main" };
    await this.invoke("plugin:event|emit_to", { target, event: "tauri://drag-enter", payload: { paths, position: { x: 400, y: 300 } } });
  }

  async dragLeave() {
    await this.invoke("plugin:event|emit_to", { target: { kind: "Webview", label: "main" }, event: "tauri://drag-leave", payload: null });
  }

  rail(label: string): Locator {
    return this.page.getByRole("navigation").getByRole("button", { name: label, exact: true });
  }

  /** The library (or welcome page) of the default analysis DB, reloaded from disk. */
  async openLibrary() {
    await this.rail("Library").click();
    const reopen = this.page.getByRole("button", { name: /^Open (a )?database$/ });
    // "Open database" re-reads the default DB (no dialog when it exists)
    await reopen.first().click();
    await expect(this.page.getByRole("heading", { level: 1 })).toHaveText(/Library|Welcome to Spaghetti Lab/);
  }

  /** Make sure the default DB holds the analysed fixture clip (runs the analysis once if not). */
  async ensureAnalysed() {
    const has = async () => {
      try {
        const replays = await this.invoke<{ replay_id: string }[]>("get_replays", { dbPath: DEFAULT_DB });
        return replays.some((r) => r.replay_id === REPLAY_ID);
      } catch {
        return false;
      }
    };
    if (!(await has())) await this.invoke("analyze_video", { videoPath: FIXTURE_VIDEO });
    expect(await has()).toBe(true);
  }

  /** The library card of the fixture clip. */
  replayCard(): Locator {
    return this.page.getByRole("button", { name: /e2e set/ });
  }

  /** Open the fixture clip's review screen. */
  async openReview() {
    await this.ensureAnalysed();
    await this.openLibrary();
    await this.replayCard().click();
    await expect(this.page.getByRole("tab", { name: /^Overview/ })).toBeVisible();
    // the screen shows at once (with the last data when it's the same replay) under a "Loading…"
    // overlay; reload is disabled until the results are in
    await expect(this.page.getByRole("button", { name: "Reload the results" })).toBeEnabled({ timeout: 30_000 });
    await expect(this.page.locator("video")).toHaveCount(1);
    // the video is loaded (seeking works)
    await expect
      .poll(() => this.page.locator("video").evaluate((v: HTMLVideoElement) => v.readyState), { timeout: 20_000 })
      .toBeGreaterThanOrEqual(1);
  }

  /** A path under the run's files folder (`fresh`: nothing there yet). */
  file(name: string, fresh = true): string {
    fs.mkdirSync(FILES_DIR, { recursive: true });
    const p = path.join(FILES_DIR, name);
    if (fresh) fs.rmSync(p, { force: true, recursive: true });
    return p;
  }

  /** Pause the video and seek it (as the player would), waiting until the app has seen the new time. */
  async seek(seconds: number) {
    await this.page.locator("video").evaluate(
      (v: HTMLVideoElement, s) =>
        new Promise<void>((resolve) => {
          v.pause();
          v.addEventListener("seeked", () => setTimeout(resolve, 50), { once: true });
          v.currentTime = s;
        }),
      seconds,
    );
    const clock = `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")} /`;
    await expect(this.page.getByText(new RegExp(`^${clock}`))).toBeVisible();
  }

  /** The current time of the video, in seconds. */
  async videoTime(): Promise<number> {
    return this.page.locator("video").evaluate((v: HTMLVideoElement) => v.currentTime);
  }
}

export const test = base.extend<{ app: App }, { connection: { browser: Browser; page: Page } }>({
  connection: [
    async ({}, use) => {
      const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP_PORT}`);
      const page = browser.contexts()[0].pages().find((p) => p.url().startsWith("http://tauri.localhost"));
      if (!page) throw new Error("The app window was not found over CDP");
      await page.addInitScript(installMocks);
      await page.evaluate(installMocks);
      await use({ browser, page });
      // Leave the app running for the next spec file; the global teardown stops it.
      await browser.close().catch(() => {});
    },
    { scope: "worker" },
  ],
  app: async ({ connection }, use, testInfo) => {
    const app = new App(connection.page);
    await app.page.setViewportSize({ width: 1440, height: 900 });
    await app.clearCalls();
    await app.page.evaluate(() => {
      const e2e = (window as any).__e2e;
      e2e.dialogs = [];
      e2e.handlers = {};
    });
    await use(app);
    if (testInfo.status !== testInfo.expectedStatus) {
      await testInfo.attach("screenshot", { body: await app.page.screenshot(), contentType: "image/png" });
    }
  },
});

export { expect };
