# React Frontend

The frontend is a React + TypeScript application built with Vite, running inside a Tauri webview. It provides a dashboard for viewing replay analysis, a VOD splitter workflow, and an analysis progress view.

---

## Application Views

The top-level `App` component manages a `view` state with four modes:

| View | Component/Layout | Description |
|------|-----------------|-------------|
| `welcome` | `WelcomeScreen` | Intro screen with action buttons |
| `analyze` | `AnalysisProgress` | Live progress from Python subprocess |
| `split` | `SplitVodView` | VOD splitter multi-step workflow |
| `dashboard` | Full dashboard layout | Main analysis viewer |

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
2. If found, `openDatabase(path)` loads replays and switches to dashboard
3. If not found, stays on welcome screen

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

### Sidebar

**Source:** `app/src/components/Sidebar.tsx`

Left sidebar containing:
- Replay list with selection highlighting
- Database path display
- Action buttons:
  - Reload data
  - Reanalyze current replay (re-run Python CV pipeline)
  - Reanalyze all replays
  - Analyze new video
  - Open database

### VideoPlayer

**Source:** `app/src/components/VideoPlayer.tsx`

HTML5 `<video>` element with:
- Programmatic seeking via `seekToMs` prop (consumed then cleared)
- Round boundary markers on the progress bar
- Damage event markers
- "Locate video" button when video file is not found
- Note markers display
- Time update callback for synchronizing with other components

### HealthTimeline

**Source:** `app/src/components/HealthTimeline.tsx`

Built with Recharts (`AreaChart` / `ResponsiveContainer`).

**Layout:**
- Health chart: 120px height, shows P1 and P2 health over time as overlapping area charts
- P1 tension chart: 44px height, shows P1 tension gauge
- P2 tension chart: 44px height, shows P2 tension gauge

**Data structure:**
```typescript
interface ChartPoint {
  time_s: number;
  p1: number | null;     // P1 health
  p2: number | null;     // P2 health
  t1: number | null;     // P1 tension
  t2: number | null;     // P2 tension
}
```

**Tension spend markers:**
```typescript
interface TensionSpend {
  time_s: number;
  side: "p1" | "p2";
  drop: number;  // drops >= 12% are marked
}
```

**Visual features:**
- Round boundaries as `ReferenceLine` components
- Match selection as `ReferenceArea` highlighting
- Highlight and note markers
- Click-to-seek interaction
- Damage events displayed contextually

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

Live progress display during Python analysis:
- File picker for selecting video to analyze
- Real-time output streaming from the Python subprocess
- Completion callback with the database path

### WelcomeScreen

**Source:** `app/src/components/WelcomeScreen.tsx`

Intro view with action cards:
- "Analyze a Set" -> analyze view
- "Split a VOD" -> split view
- "Open .spag file" -> file picker
- "Open Database" -> database picker

---

## Styling

The app uses **Tailwind CSS** with custom color tokens defined in the Tailwind config.

### Custom Color Tokens

| Token | Usage |
|-------|-------|
| `p1` | Player 1 color (blue-ish) |
| `p2` | Player 2 color (red-ish) |
| `accent-purple` | Primary accent (buttons, highlights, branding) |
| `accent-gold` | Secondary accent (comebacks, highlights badge) |
| `accent-green` | Tertiary accent (notes, tension, save actions) |
| `surface-0` | Deepest background |
| `surface-1` | Header/sidebar background |
| `surface-2` | Card background |
| `surface-3` | Hover/active states |
| `surface-4` | Borders, dividers |
| `text-primary` | Primary text color |
| `text-secondary` | Secondary text color |
| `text-muted` | Muted/de-emphasized text |

### Component Classes

- `btn-ghost`: Ghost-style button with border
- `tab-btn`: Tab button with active state indicator
- All interactive elements use `cursor-pointer` and transition effects
- Responsive layout with `flex`, `overflow-hidden`, and percentage-based heights

### Resizable Split

The dashboard has a draggable resize handle between the top section (video + timeline) and the bottom section (tab content):
- Default top height: "55%"
- Drag handle: 6px tall, centered dot indicator
- Mouse drag computes delta from start position, minimum height 150px
- Body cursor changes to `row-resize` during drag
