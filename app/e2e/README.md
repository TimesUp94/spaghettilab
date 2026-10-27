# End-to-end tests

Playwright drives the real desktop app (Rust backend + Python pipeline + React UI) through the WebView2
DevTools port. It covers every feature, from first launch to sharing files. Windows only.

```bash
cd app
npm run e2e            # build, then run everything (~10 min, of which ~6 are real analyses)
npm run e2e:fast       # skip the @slow tests (reanalyse one / all)
npx playwright test 03 # one spec file (it analyses the fixture first if the DB doesn't have it)
npx playwright show-report e2e/.report
```

Environment variables:

| Variable | Effect |
|---|---|
| `E2E_SKIP_BUILD=1` | Reuse `src-tauri/target/debug/spaghetti-lab.exe` (it must come from `tauri build --debug`, not `tauri dev`) |
| `E2E_REUSE_DATA=1` | Keep the previous run's data dir (its analysed DB) instead of starting empty; for iterating on one spec |
| `E2E_SOURCE_VIDEO` | Where the fixture clip is cut from (default `data/sets_v2/05_UF_Jason_vs_EneroNeJR.mp4`) |
| `E2E_CDP_PORT` | DevTools port (default 9333) |
| `E2E_KEEP_APP=1` | Leave the app running after the run |

## How it works

- **One app instance** for the run (`global-setup.ts`). It builds with `tauri build --debug --no-bundle`
  (frontend embedded), then starts the exe with:
  - `SPAGHETTI_LAB_DATA_DIR=e2e/.run/data`: the analysis DB, sessions and previews go there, never
    to the checkout's `output/analysis.db`;
  - `SPAGHETTI_LAB_NO_GAME=1`: the backend refuses to run the Strive export, so a test can never press
    keys in the game;
  - its own WebView2 profile and `--remote-debugging-port`.
- **The fixture clip** is the first two games of a set (192 s, 720p), cut once into `e2e/.cache/` with
  ffmpeg. The analysis reads 4 rounds, all won by P1, as two games of 2–0 with one comeback
  (`env.ts` `EXPECTED`).
- **Specs run in file order** in one worker, because they share the window. `02-analyze` runs the
  real pipeline on the clip by dropping it onto the window. Later specs use the result, and call
  `app.ensureAnalysed()` so they also run alone.
- **Mocks** (`fixtures.ts`): the Tauri IPC is a `fetch` to `http://ipc.localhost/<command>`, which
  the page wraps:
  - native open/save dialogs answer from `app.queueDialogs(...)`, and an empty queue means cancelled;
  - the updater finds nothing, and opening a link in the browser is only recorded;
  - every command is logged, so `app.calls(cmd)` shows what was sent;
  - a test can mock a command with `__e2e.handlers[cmd]`.

  `app.dropFiles()` sends the same drag-and-drop events as Windows, and `app.emit()` sends backend
  events.

| Spec | Covers |
|---|---|
| 01-shell | Welcome page, rail, tool cards, Open / Support, drag highlight, error banner |
| 02-analyze | Pick / change / cancel, a failing file, drop → real analysis with progress → library |
| 03-review | Header and game picker, overview stats, tabs and counts, rounds / highlights / damage lists, timeline seek and drag, player controls and shortcuts, reload |
| 04-notes-drawings | Add / edit / cancel / delete notes, time order, persistence; draw, save, delete a drawing |
| 05-winner-override | Override a round's winner, persistence, undo |
| 06-clip-export | Clip export with padding (real ffmpeg), cancelled save, from highlights / games |
| 07-share | Export .spag / .spagz, open them (with the video, without, locating it, browsing for it), Save, drop a .spag |
| 08-annotate | Link validation, YouTube session, local file: note, drawing, checklist, save .spagz, reopen, drop |
| 09-split | Form validation, preview regions, real scan (EasyOCR), cut the sets |
| 10-strive | Setup validation, remembered folder, backend refusal, a mocked export run with stop and summary |
| 11-responsive | Review layout from 1920×1080 to 800×700 (16:9 video in the window, panel beside or below), every page at 640–1440 px |
| 12-reanalyse | Reanalyse one video and the whole library (@slow) |
| 13-library | Search in a large library (mocked list), set cards, the update banner |
