# Strive Replay Export

`replanal/strive_export/` exports a player's saved Guilty Gear -Strive- replays as video files, so they can be analysed like any other recording. It drives the running game itself: from the title screen to Collection > Replay > Saved replays, then for each entry it picks "View replay", records the game window with FFmpeg until the end-of-replay menu shows, and returns to the list.

It needs no game mod. The research prototype in `tools/ggst_logger/drive/` (which uses the logger mod to know when a replay starts) is separate and unchanged.

Windows only: it uses SendInput for keys and Windows capture APIs. Elsewhere it fails with a clear message.

## Usage

```bash
# Game running at the title screen (or main menu), keyboard controls
# (U confirm, W A S D move, K / L switch tabs). Don't touch the keyboard while it runs.
python -m replanal.strive_export --out D:\strive --count 10       # the 10 newest replays
python -m replanal.strive_export --out D:\strive --since today    # today's replays
python -m replanal.strive_export --out D:\strive --since 2026-10-20
python -m replanal.strive_export --out D:\strive --all            # every saved replay
python -m replanal.strive_export --out D:\strive --all --dry-run  # walk the list, record nothing
python -m replanal.strive_export --out D:\strive --count 5 --start 20   # skip the 20 newest first

python -m replanal.strive_export probe --save shot.png   # FFmpeg/capture/encoder choice + current screen (no keys)
python -m replanal.strive_export classify shot.png ...   # classify saved screenshots (no game)
python -m replanal.strive_export calibrate --lang en     # record reference screens for another UI language
```

Other `export` options: `--json-progress` (machine-readable events, see below), `--no-ocr`, `--overwrite`, `--date-order DMY|MDY|YMD`, `--fps` (default 60), `--encoder`, `--max-minutes` (default 15), `--lang`, `--refs-dir`, `--ffmpeg`.

Ctrl+C once stops after the replay being recorded (back on the list); twice stops at once, finalising the current file.

In the app: Welcome screen > **Export Strive Replays** (`StriveExportView`, Tauri command `export_strive_replays`). Afterwards **Analyze N videos** runs the normal analysis on each exported file.

## Output

One mp4 per replay in `--out`:

- `2026-10-20_22-10-28_TimesUp_vs_J4rusty.mp4` when the list row was read (match date and time, P1 and P2 names);
- `replay_003.mp4` (position in the list, 0 = newest) when it wasn't (no easyocr, or an unreadable row).

A file that already exists under a date-based name is skipped (`--overwrite` records it again), so "export today's replays" can be run repeatedly. A file the manifest marks `partial` (the export was stopped during it) is recorded again under the same name. Index-based names never overwrite: a `_1` suffix is added.

`manifest.json` lists every exported file: `index, file, date, time, p1, p2, duration_s, partial, recorded_at, capture, encoder, fps`. It is rewritten after each replay and merged with an existing one in the same folder.

Recording starts right after "View replay" is confirmed, so each file begins with a few seconds of loading screen. The pipeline's non-gameplay detection (no timer for 20 s) NaN-s that part out, so it isn't trimmed. Recording stops when the end-of-replay menu is recognised, within about 2 s. Video only: window capture has no game audio.

## How it works

`exporter.Exporter` is the state machine. It gets every game interaction through a `game` object (`grab()`, `press(key)`, `start_recording(path)`), so tests can drive it with a simulated game.

1. **Reach the list** (`reach_saved_list`): title → `U`; main menu → `D`/`A` one tab at a time until the Collection tab is selected and its Replay entry highlighted → `U`; replay room → `L`/`K` one tab at a time to the Saved tab (the third icon); end-of-replay menu → `U`. A "communicating with server" dialog, loading screens and the title's attract video are waited out. **No key is pressed on them.** If the export starts on the list itself, the cursor is first moved to the top (`W` until it stops moving or wraps).
2. **Per replay:** wait for a stable list (two looks alike, since portraits load late), read the selected row (OCR), apply `--since`, press `U` (retried while the list still shows: presses during loading are swallowed), check that the entry menu's highlight is on item 0, press `U` ("View replay"), start FFmpeg, poll the screen every 2 s until the end menu, stop FFmpeg (`q`, so the mp4 is finalised), press `U` ("Back to the replay list"), then `S` to the next entry.
3. **Stop** when the count is reached, a row is older than `--since` (read twice before stopping), `S` wraps to the top or no longer moves (end of the list), or on cancel.

### Safety

A key is pressed only right after the screen was recognised, and only a key on that screen's allow-list (`exporter.key_allowed`):

| Screen | Keys |
|---|---|
| title | `U` |
| main menu | `A`, `D`; `U` only on the Collection tab with Replay highlighted |
| replay room (other tabs) | `K`, `L` |
| saved replays list | `U`, `S`, `W` |
| replay entry menu | `U` only with item 0 ("View replay") highlighted and 5 items |
| end-of-replay menu | `U` only with item 0 ("Back to the replay list") highlighted and 3 items |
| anything else | nothing |

Ranked matchmaking (the main menu's first tab, where `U` starts matchmaking) and "Delete replay" (item 3 of the entry menu) are therefore never confirmed. The cursor is never moved inside the entry menu. If an entry menu is already open when the export starts, it stops and asks the user to close it. Keys go out only while the Strive window is the foreground window (`winio.press(verify_focus=True)`); if it loses focus, the export stops.

## Screen recognition and UI language

`screens.Classifier` works on the capture scaled to 960x540, in three layers. A window that isn't 16:9 shows the game between black bars; only the centred 16:9 picture is used (`screens.game_area`, and FFmpeg's `crop` when recording).

1. **Structure, language independent** (no reference image):
   - main menu: exactly one of the nine bottom tab icons on the dark-red highlight square gives the selected tab index;
   - replay room: the bright red header banner, and which of the five tab icons under it is highlighted;
   - saved list: the bright red selected row (sampled left and right of the portraits);
   - pop-up menus: evenly spaced grey item bars with exactly one red, counted. The entry menu has 5 items over the dimmed replay room; the end menu has 3 over the game. The replay search's 3-item menu also sits over the room, so it is not taken for the end menu.
2. **Text-free reference regions, language independent:** the GUILTY GEAR logo, the main menu's tab icons (correlation), the Collection sub-menu's highlighted first entry, the replay room's film icon, the pop-up's header picture, the server dialog's icons. They are taken from the reference screenshots.
3. **Text regions, language specific:** the title prompt, the main menu tab labels, the "Saved replays / sort by match date" heading, and each pop-up item label. They are compared (correlation ≥ 0.85) only against references of the UI language. A smaller or larger window renders the UI text a few percent narrower or wider and a pixel or two off, so each side's text is searched in the other's region at sizes 0.92–1.08 within 3 px, and the lower of the two directions counts. On a live 1196x1076 window the entry menu's labels score 0.90–0.95 against their own references and at most 0.82 against the other labels; "View replay" against "Delete replay" scores 0.65. With `--lang auto` (the default) the first screen whose text matches a reference language fixes it. Until then, and when no reference language matches (an uncalibrated UI language), only layers 1 and 2 are used.

The other language-specific parts are:

- **List dates:** the order of day, month and year (`date_order` in the language's `profile.json`, or `--date-order`). It is inferred when a part settles it: a 4-digit first part is a year, a part over 12 is a day.
- **Player names:** read with EasyOCR's Latin model, so names in other scripts come out garbled. They only affect file names.

### References

`refs/<lang>/<screen>.jpg` (960x540) for `title, main, collection, saved_list, replay_menu, replay_end, server`, plus an optional `profile.json` (`{"date_order": "DMY"}`). The search order is:

1. `--refs-dir` folders;
2. `%LOCALAPPDATA%\SpaghettiLab\strive_refs`;
3. the package's `replanal/strive_export/refs/`, which ships Italian (`it`).

Layer 2 uses the language's own references when it has them, else Italian's.

`calibrate --lang xx` records a new language without pressing any key:

1. It asks you to open each screen in the game and press Enter.
2. It captures the screen and checks it with the language-independent layers, warning on a mismatch.
3. It saves the images and asks for the date order.

## Capture and encoding (`capture.py`)

- **FFmpeg:** the candidates are tried in this order:
  1. `--ffmpeg`;
  2. `GGST_FFMPEG`;
  3. `SPAGHETTI_FFMPEG` (the app passes its `ffmpeg_bin()`);
  4. `ffmpeg` on PATH;
  5. FFmpeg builds unpacked under `$GGST_DATA/ggst_mod/dl` or `%LOCALAPPDATA%\SpaghettiLab\ffmpeg`;
  6. imageio-ffmpeg's FFmpeg (version 7).

  The first one with the `gfxcapture` filter (FFmpeg 8+, Windows Graphics Capture) wins. It records the game window itself, even while the window is covered. Without one, the first with `ddagrab` (Desktop Duplication) is used, cropped to the game's client rectangle. The window must then stay visible and on top, so the exporter warns and brings the window forward before each screenshot. The monitor index follows EnumDisplayMonitors' order, which matches DXGI on single-GPU systems.
- **Encoder:** test-encodes 10 frames at 1080p60 with `h264_nvenc`, then `h264_mf` (Media Foundation, any GPU), then `libx264`, and uses the first that works. For example, NVENC fails when the NVIDIA driver is older than FFmpeg's NVENC API, and is skipped.
- Frames are cropped to the centred 16:9 picture and scaled to 1920x1080 (the pipeline's ROIs) at `--fps` (60). A window smaller than 1080p is upscaled; the pipeline still reads it (checked on a 1196x1076 window).
- A recording is capped at `--max-minutes` + 30 s by FFmpeg's `-t`, so an orphaned FFmpeg ends by itself.
- Screenshots are single FFmpeg grabs piped as PNG. An empty capture is retried up to 5 times.

## Progress events (`--json-progress`)

One JSON object per stdout line, each with an `event` field:

| event | fields |
|---|---|
| `ffmpeg` | `path, version, capture, encoder` |
| `start` | `out, count, since, start, dry_run` |
| `screen` | `screen, detail, lang` (only when it changes) |
| `status` / `warning` / `error` | `message` |
| `list` | `total` (the "100 / 100" counter, when read) |
| `entry` | `index, file, date, time, p1, p2` |
| `recording` | `index, file` |
| `recorded` | the manifest entry |
| `skipped` | `index, file, reason` |
| `done` | `reason, listed, exported, skipped, manifest` |

With `--json-progress` the exporter reads stdin. A `cancel` line stops it after the current replay; a `stop` line stops it now.

## Tests

`tests/test_strive_export.py` runs without the game:

- **Classification** of the shipped references and of two live 1080p frames (`tests/data/strive_export/`).
- **Negatives:** gameplay, character select, dojo, the replay search screens, the filter menu.
- **Perturbations:** brightness, noise, JPEG, 720p and 1440p windows, blur.
- **Language independence:** labels repainted with other words are still recognised with `auto`, and rejected with `--lang it`.
- **Highlight on a different item:** the entry menu's highlight moved to "Delete" is never confirmed.
- **Parsing:** dates, times, names and the counter.
- **FFmpeg and encoder selection**, with a mocked FFmpeg.
- **The state machine on a simulated game:** a full export, the attract video, swallowed presses, dry run, wrap and end of the list, `--since`, `--start` with skip-existing, starting on the list, soft and hard cancel, and the manifest merge.
- **A fuzz test:** random screens, dangerous ones included, must never receive a key outside their allow-list.

The OCR test is skipped without easyocr.
