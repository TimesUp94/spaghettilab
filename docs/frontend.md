# React Frontend

The frontend is a React + TypeScript application built with Vite, running inside a Tauri webview. A left
rail (`components/shell/AppRail.tsx`) is always visible; the area next to it shows one view at a time.

---

## Application Views

The top-level `App` component manages a `view` state (no router):

| View | Component | Description |
|------|-----------|-------------|
| `library` | `views/LibraryView.tsx` | Home. The analysed videos of the open database as cards (search above 6), or a getting-started page with a drop zone when there are none |
| `review` | `views/ReviewView.tsx` | One video's results: fitted video, timeline, tabbed results panel |
| `analyze` | `AnalysisProgress` | Pick (or receive a dropped) video and analyse it, with live progress |
| `annotateSetup` | `AnnotateSetup` | Pick a YouTube / Twitch link or a local video (browse or drop) to annotate |
| `annotate` | `AnnotateView` | Notes, drawings and the checklist on a file or a YouTube / Twitch embed (no analysis) |
| `split` | `SplitVodView` | VOD splitter multi-step workflow |
| `strive` | `StriveExportView` | Export saved Strive replays from the running game, then analyse them |

The rail's entries: Library, Analyse, Annotate (the `annotateSetup` page), Split VOD, Strive, and at the bottom
Open (.spag / .spagz) and Support. Errors from any view show as a dismissable banner above the view.

**Drag and drop** (`getCurrentWebview().onDragDropEvent` in `App`): a dropped video (mp4/mkv/avi/mov/webm)
goes to `analyze` and starts at once (on the annotate page it opens a notes session instead); a `.spag` / `.spagz` opens. The library shows its drop zone while a
drag is over the window.

---

## App.tsx State

### Database & Replay State

| State | Type | Description |
|-------|------|-------------|
| `dbPath` | `string \| null` | Path to the SQLite database |
| `replays` | `Replay[]` | All replays in the database |
| `selectedReplay` | `Replay \| null` | Currently selected replay |
| `loading` | `boolean` | Global loading indicator |
| `error` | `string \| null` | Error message display |

### Analysis Data (for selected replay)

| State | Type | Description |
|-------|------|-------------|
| `frameData` | `FrameDataPoint[]` | Per-frame health + tension data |
| `damageEvents` | `DamageEvent[]` | Detected damage events |
| `rounds` | `RoundResult[]` | Detected rounds with winners |
| `stats` | `MatchStats \| null` | Aggregated statistics |
| `highlights` | `Highlight[]` | Notable moments |
| `notes` | `Note[]` | User-created notes |

### Video State

| State | Type | Description |
|-------|------|-------------|
| `videoSrc` | `string \| null` | Tauri asset URL for HTML5 video |
| `videoPath` | `string \| null` | Absolute filesystem path |
| `seekToMs` | `number \| null` | Pending seek target (consumed by VideoPlayer) |
| `currentTimeMs` | `number` | Current video playback position |

### UI State

| State | Type | Description |
|-------|------|-------------|
| `activeTab` | `ActiveTab` | Selected tab: "matches" \| "rounds" \| "highlights" \| "damage" \| "notes" |
| `selectedMatchIndex` | `number \| null` | Zoomed-in match index (null = all matches) |
| `topHeight` | `number \| string` | Height of the top section (default "55%", becomes pixel value on drag) |
| `exportTarget` | `{ startMs, endMs } \| null` | Pending clip export |
| `spagSession` | `SpagSession \| null` | Active .spag file session |
| `savingSpag` | `boolean` | .spag save in progress |
| `exportingSpag` | `boolean` | .spag export in progress |
| `reanalyzing` | `boolean` | Single replay reanalysis in progress |
| `reanalyzingAll` | `boolean` | Batch reanalysis in progress |

---

## groupRoundsIntoMatches()

Groups a flat list of `RoundResult[]` into `Match[]` objects using first-to-2 (best-of-3) round wins.

### Algorithm

1. Iterate through rounds, tracking P1 and P2 win counts
2. When `is_match_start` is `true` and there are accumulated rounds, flush the current match
3. When either player reaches 2 wins, flush the current match
4. Remaining rounds at the end are flushed as an incomplete match

### Match winner determination

Priority order:
1. First player to reach 2 wins
2. Player with more round wins
3. `"??"` if tied/ambiguous

### Match Object

```typescript
interface Match {
  match_index: number;
  rounds: RoundResult[];
  winner: string;           // "P1", "P2", or "??"
  p1_rounds_won: number;
  p2_rounds_won: number;
  start_ms: number;         // First round start
  end_ms: number;           // Last round end
}
```

---

## Match-Scoped Stats

When a match is selected (`selectedMatchIndex !== null`), `displayStats` is recomputed from the global `stats` by filtering damage events to the match's time range and recalculating:
- Round counts and wins
- Damage totals and biggest hits
- Duration statistics
- Comeback and close round counts
- Average winner final HP

---

## Data Loading Flow

### Startup

1. `getDefaultDbPath()` attempts to find the default database
2. If found, `openDatabase(path)` loads the replay list and shows the library
3. If not found, the library shows the getting-started page

Clicking a card calls `handleOpenReplay`: `loadReplayData` and the review view (a notes-only replay,
`frame_count === 0`, opens the annotate view).

### loadReplayData(db, replay)

Loads all analysis data for a replay in parallel:
```typescript
const [fd, de, rn, st, hl, nt, vp] = await Promise.all([
  getFrameData(db, replay.replay_id),
  getDamageEvents(db, replay.replay_id),
  getRounds(db, replay.replay_id),
  getMatchStats(db, replay.replay_id),
  getHighlights(db, replay.replay_id),
  getNotes(db, replay.replay_id),
  resolveVideoPath(db, replay.replay_id),
]);
```

Video path is resolved and converted to a Tauri asset URL via `convertFileSrc()`.

### .spag File Support

- **Open:** `openSpag()` extracts the ZIP, loads the embedded DB and video, sets up a session
- **Save:** `saveSpag()` re-packs the current DB into the existing .spag file
- **Export:** `exportSpag()` creates a new .spag from the current replay
- **File association:** Listens for `open-spag-file` Tauri event (app launched with .spag argument)

---

## Components

### ReviewView

**Source:** `app/src/views/ReviewView.tsx`

- **Header:** back to the library, the set's players (from `setInfo(replay_id)`), a game picker (All games,
  G1, G2…; a dot in the winner's colour), Reload, Reanalyse (local video only) and Share (export `.spag` /
  `.spagz`; inside a `.spag` / `.spagz` session it is Save).
- **Wide windows (≥ 1024 px, `useMediaQuery`):** a grid of the video column and a results panel
  (`minmax(340px, 30%)`, 480 px from 2xl). The video column is the player (`fill`: the 16:9 video is fitted
  into the space left above the timeline, so it never overflows) and the `ReviewTimeline`.
- **Narrow windows:** one scrolling column: video (full width), timeline, then the panel.
- **Panel tabs:** Overview (`MatchOverview` stats + `MatchList` games), Rounds, Highlights, Damage, Notes.
  With a game selected, Rounds / Highlights / Damage list only that game, and the timeline zooms to it.

### VideoPlayer

### VideoPlayer

**Source:** `app/src/components/VideoPlayer.tsx`

HTML5 `<video>` or a YouTube / Twitch embed, with:
- `fill`: fit the 16:9 video inside the parent's height (a `container-type: size` box; the video is
  `min(100cqw, 100cqh * 16 / 9)` wide). Without it the video takes the full width.
- `showProgress`: its own progress bar (rounds, big hits, notes, drawings); off under a `ReviewTimeline`
- Programmatic seeking via `seekToMs` prop (consumed then cleared)
- Controls: ±5 s, play, frame step, time, Draw, speed, volume, shortcuts hint (lucide icons)
- "Locate video" button when video file is not found
- `onTimeUpdate`: on `timeupdate` and, while playing, every ~66 ms (smooth playhead)

### ReviewTimeline

**Source:** `app/src/components/ReviewTimeline.tsx`

One SVG on a shared time axis (the whole replay, or the selected game): a rounds lane (one block per round
in the winner's colour, dashed when the winner is uncertain, game starts marked), both health lines,
both tension gauges as bars, big hits (≥ 30 %), notes and drawings, time ticks, the playhead and a hover
line with the time and round. Click or drag anywhere to seek. Data is bucketed to ~2 px per sample.

### MatchOverview

**Source:** `app/src/components/MatchOverview.tsx`

Stats display panel (320px wide, right of video):
- "Viewing Game N" banner when a match is selected
- Round win counts per player
- Damage totals, biggest hits
- Round duration stats
- Comeback and close round counts
- Average winner HP

### MatchList

**Source:** `app/src/components/MatchList.tsx`

Tab content showing matches in a card layout.

**For each match:**
- Match header: game number (G1, G2...), score, winner label, duration, comeback badge
- Round rows within each match card

**Uncertain winner UI:**
- Rounds with `winner_confident === false` show an amber "?" indicator
- Hover reveals P1/P2 override buttons
- Rounds with `winner_confident === true` also show override buttons on hover (for corrections)
- Clicking an override button calls `setRoundWinner()` and refreshes round data

### RoundBreakdown

**Source:** `app/src/components/RoundBreakdown.tsx`

Per-round table showing:
- Round index, time range, winner, final HPs, max deficit
- Comeback and close round indicators
- Click-to-seek and export buttons

### Highlights

**Source:** `app/src/components/Highlights.tsx`

List of notable moments sorted by severity (descending):
- Comeback rounds
- Close rounds
- Big damage events (>30%)
- Perfect rounds

Each highlight shows kind, label, timestamp, details, and a seek button.

### DamageLog

**Source:** `app/src/components/DamageLog.tsx`

Scrollable table of all damage events:
- Timestamp, target player, damage percentage, health before/after
- Click-to-seek

### NotesPanel

**Source:** `app/src/components/NotesPanel.tsx`

CRUD interface for user notes:
- Add note at current video timestamp with text
- Edit existing notes
- Delete notes
- Click timestamp to seek video
- Notes are stored per-replay in the SQLite `notes` table

### ExportModal

**Source:** `app/src/components/ExportModal.tsx`

Modal dialog for clip export:
- Shows start/end time range
- Calls `export_clip` Tauri command (ffmpeg stream copy)
- Save dialog for output path

### SplitVodView

**Source:** `app/src/components/SplitVodView.tsx`

Multi-step VOD splitter workflow. See `docs/vod-splitter.md` for full details.

Steps: select -> roi -> scanning -> results -> cutting -> done

### StriveExportView

**Source:** `app/src/components/StriveExportView.tsx`

Exports the player's saved Strive replays by driving the running game (see `docs/strive-export.md`).

- **Setup:** a warning box (game at the title screen with keyboard controls; the app sends keys to the game window, which can stay behind other windows; don't click into, type in or minimise the game meanwhile, and leave the PC alone without FFmpeg 8+), which replays (newest N / played on or after a date, with a "today" shortcut / all), output folder (remembered in `localStorage`), and an Advanced section (skip the newest N, list only = `--dry-run`, UI language). Start needs a ticked acknowledgement.
- **Running:** calls `export_strive_replays` and listens to `strive-export-progress`; shows the current game screen / status, capture and encoder, warnings, and one row per replay (players, match date, `listed` / `recording` / `done` / `partial` / `already exported`). "Stop after this replay" calls `cancel_strive_export(false)`; the button then becomes "Stop now" (`cancel_strive_export(true)`).
- **Done / error:** summary, "Analyze N videos" runs the existing `analyze_video` on each exported file in turn and opens the dashboard on the result (`onComplete(dbPath)`); "Export more" goes back to setup. After an error the files recorded so far can still be analyzed.

### RoiPicker

**Source:** `app/src/components/RoiPicker.tsx`

Interactive ROI editor. See `docs/vod-splitter.md` for full details.

Supports both rectangular ROIs (drag move/resize) and quad ROIs (SVG polygon with independent corner drag handles).

### UpdateBanner

**Source:** `app/src/components/UpdateBanner.tsx`

Auto-updater banner:
- Checks GitHub releases 3 seconds after app launch via `@tauri-apps/plugin-updater`
- If an update is available, shows a banner with version number
- "Update & restart" button: downloads, installs, and relaunches via `@tauri-apps/plugin-process`
- "Later" button dismisses the banner for the session

### AnalysisProgress

**Source:** `app/src/components/AnalysisProgress.tsx`

- File picker (or `initialPath` from a drop, which starts at once)
- Progress bar and stages from the `analyze-progress` event: `analyze_video` streams the script's stdout
  lines, and `scripts/analyze_replay.py` prints `Progress NN%` during the frame pass
- On failure the last output lines and stderr; on success the library of the result database

---

## Styling

The app uses **Tailwind CSS** with custom color tokens defined in the Tailwind config. Dark only.

### Custom Color Tokens

| Token | Usage |
|-------|-------|
| `p1` | Player 1 colour (red, `#e84040`) |
| `p2` | Player 2 colour (blue, `#4088e8`) |
| `accent-purple` | Primary accent; despite the name it is yellow (`#e8c840`): primary buttons, active rail item |
| `accent-gold` | Comebacks, drawings |
| `accent-green` | Notes |
| `surface-0` … `surface-4` | Backgrounds from deepest to borders |
| `text-primary` / `text-secondary` / `text-muted` | Text |

### Shared UI

`components/ui.tsx`: `Button` (primary / secondary / ghost / danger, sm / md / lg, lucide icon, loading),
`IconButton`, `Badge`, `Spinner`, `Modal`, `Tabs`, `EmptyState`, `PageHeader`. Icons come from
`lucide-react`. The older `.btn-primary` / `.btn-ghost` classes in `index.css` match `Button`.
