# VOD Splitter

The VOD splitter detects individual tournament sets within a long VOD recording and cuts them into separate files. It consists of a Python script (`scripts/split_vod.py`), a Rust bridge in the Tauri backend, and a React frontend workflow.

---

## Architecture: 4-Step Pipeline

```
Step 1: HUD scan (1 fps via ffmpeg pipe)
  -> per second: timer value, hearts, name-tag crops of clean HUD seconds
Step 2: Gameplay segments (timer reads, merged over short HUD gaps)
  -> list of Segment objects with their hearts states
Step 3: Names + set grouping (OCR'd in-game name tags, voted per segment)
  -> list of DetectedSet objects with player names and game counts
Step 4: ffmpeg stream copy cutting (CLI only; the app cuts in Rust)
  -> individual MP4 files
```

Every input is the game's own HUD: the round timer, the round hearts and the
in-game name tags under the health bars. A stream overlay (name/score
banner, logos, the stream's own lobby scene) is set by hand during a stream
and may be stale, different or absent, so it is never read.

### Results (Tappa #4, 3 h 10 min, 16 true sets)

Scored against `output/e2e/split/truth_rounds_vod.json` (runs in
`output/e2e2/split/`, overlay blacked out):

| | Before (colour fractions + 80x4 thumbnail) | Now |
|---|---|---|
| Auto sets / clean (one whole set per file) | 24 / 10 | 16 / 16 |
| False sets (stream "TIMESUP" lobby, intro) | 5 | 0 |
| Split / merged sets | 4 split (04, 07, 09, 13), 05b+06 merged | none |
| Game counts right | - | 16 / 16 |
| Names exact / near / wrong (32 names) | 1 / 6 / 27 (+2 empty) | 31 / 1 / 0 |
| Round spans inside their file (10 s / end pad) | 143 / 152 (5 s pad) | 152 / 152 (20 s pad) |

The same 16/16 holds on the original (overlay) VOD, with the scan shifted by
1-3 s, with 15 % of timer reads and 30 % of name crops dropped at random, and
without OCR (grouping by the tags' look). The one near name is "LeftOver" for
"Left0ver" (OCR can't tell the zero from O in the game's font).

### Results (new VOD, 2 h 20 min 720p60, Italian HUD, webcam and chat over the game)

YouTube 4bSjzC6a59Q, truth in `output/newvod/` (`stage1_report.md`,
`sets.json`), runs in `output/fix2/track4/`: 10 files for 11 true sets (the
grand final and its bracket reset are one file, see Grouping), no false or
missed sets, game counts 10/10, names **20/20 exact**; without OCR the same
10 sets. Before the wider P2 quad and the name normalisation: 16 exact,
"Elegant Dand" (cut by the P2 quad) and "Junker Boy" in 3 files (OCR lost
the underscore), so one player's files had two spellings.

---

## Step 1: HUD Scan

**Function:** `scan_hud()`

Decodes the whole VOD once at 1 fps, forced to 1920x1080:

```
ffmpeg -threads <cpus-2> -i <vod> -an -sn -vf "fps=1,scale=1920:1080" -f rawvideo -pix_fmt bgr24 -v quiet pipe:1
```

The decoder leaves two logical CPUs free. For every second it records:

- **Timer:** the pipeline's own reader (`replanal/extractors/timer.py`,
  template-matched glyphs) on the timer digits. `--timer-roi` is the timer
  badge (y 110-180, x 920-1000, like `rois.timer` in `config/default.yaml`);
  the digit box is derived from it the same way as `rois.timer_digits`
  (y 115-170).
- **Hearts:** `read_lost_hearts()` from `replanal/extractors/round_counter.py`
  for both sides (fixed 1080p geometry), or none when either side is not read.
- **Name-tag crops:** for every 2nd second where the timer *and* both hearts
  are read (the full HUD is on screen: no super, flash or transition), the grey
  bounding boxes of both name quads are kept, so names are read later without
  seeking back into the VOD (~50 MB for a 3 h VOD).

`--scan-cache <file.npz>` (debug) saves this scan, or reuses it when the file
exists, so grouping and OCR can be re-run without decoding again.

The old test (saturated pixels in the tension areas and 2 % bright pixels in
the timer box) also passed on the stream's composited "TIMESUP" lobby scene,
which made 5 false sets, bridged 05b and 06 into one segment and left one set
without names. The timer reader returns a value only for real timer glyphs on
the dark badge, so lobbies (stream or game) never read. The tension ROIs are
still accepted on the command line but no longer used.

### Progress Reporting

Machine-readable lines are printed to stdout:
```
PROGRESS:123/4567
WARNING:<message>
```

The Rust bridge forwards both as `vod-scan-progress` events.

---

## Step 2: Gameplay Segments

**Function:** `build_segments()`

1. **Valid timer seconds** (`valid_timer_seconds`): a read counts only when
   another read within 5 s agrees with a countdown (same value, or lower by at
   most the elapsed seconds + 1). The timer only counts down or freezes, so an
   isolated false read has no partner.
2. **Merging:** valid seconds closer than `merge_gap` (12 s) form one segment.
   This bridges the HUD hiding in supers, KO cinematics and round intros.
   Games are separated by character select / loading screens, so in practice
   a segment is one game, or a few rounds of one.
3. **Filtering:** segments with fewer than `min_hud` (15) valid seconds are
   dropped.
4. **Hearts:** each segment keeps its *stable* hearts states (`stable_hearts`:
   a state read twice in a row, unread seconds skipped, repeats collapsed),
   e.g. `00 10 11 00 01`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--merge-gap` | 12 s | Merge timer runs closer than this |
| `--min-duration` | 15 | Minimum valid timer seconds per segment |

---

## Step 3: Names and Set Grouping

### Name tag quads

GGS draws the player name slightly rotated (P1 rising to the right, P2
falling), centred in the tag (P1 ~x 245, P2 ~x 1670) with the platform icon
just before it, under a smaller "Player 1/2" label ("Giocatore 1", "Joueur 1",
...) that sits up and out (left of the P1 name, right of the P2 name). The
quads follow the name text and leave the label out:

| Player | Quad Corners (tl, tr, br, bl) |
|--------|-------------------------------|
| P1 | (145,168), (400,147), (400,174), (145,195) |
| P2 | (1550,153), (1795,177), (1795,200), (1550,178) |

A long name also grows towards the label, since it is centred. The P2 label
starts at x ~1755 and ends at y ~175, and a long name's end sits just below
it ("Elegant Dandy" ends at x ~1750): P2's quad runs on under the label to
x 1795, its top edge dropping below the label's baseline (a steeper slope
than the bottom edge). The P1 label ends at x ~150, so the P1 quad still
starts at 145; its right edge stays at 400, because past the tag's end
(x ~405) the background changes from segment to segment and breaks the
no-OCR look comparison. Both bottom edges leave a few pixels below the text
for descenders and underscores (the old P1 bottom edge touched the text at
x ~300).

`scripts/split_vod.py` (`P1_NAME_QUAD`, `P2_NAME_QUAD`), the RoiPicker
`DEFAULT_ROIS` (checked by `test_name_quads_match_the_app_defaults`) and this
table must agree. Earlier quads: P2 ended at x 1748, which cut long names
("Elegant Dand"); before that the quads included the label (17 of 36 names
read as "Giocatore") and the app's P1 quad slanted the wrong way.

`name_strip()` warps a quad to a rectangle (`warp_quad()`,
`cv2.getPerspectiveTransform`) at 2x for OCR.

### EasyOCR

The reader is created lazily (GPU when torch sees CUDA). It runs with
`width_ths=0.2` (`OCR_JOIN_WIDTH`; EasyOCR's default 0.5 joins boxes closer
than half their height). At the default the platform icon was often glued to
the name ("XUMAmusume", from the crossplay icon) and the underscore was lost
("Junker Boy" in 9 of 10 P1 reads). With the boxes kept apart, the icon is
dropped and the words are joined below. Per strip (`pick_name`):

1. Drop the "Player 1/2" label (`_drop_labels`): a box that looks like it (a
   fuzzy match, ratio >= 0.7, trailing 1/2/z/l ignored, against the label in
   GGST's languages) is only dropped when it carries the label's number
   ("Giocatore 1") or sits above the name line (its centre a quarter of the
   name's height above the widest other box). The quads leave the label out,
   so a label-like box on the name line is a handle ("Payer", "Spieler").
2. Drop the platform icon: a box of at most 2 characters that is not much
   wider than tall.
3. Take the **widest** remaining box (not the largest, which was often the
   label), and join boxes next to it on the same line (names with spaces).
4. Drop a 1-character token, or a 2-character token made of icon-like
   characters ("CI"), that the icon left glued to the front ("CI Dona").
5. `clean_text` keeps ASCII letters, digits and ` _-.`, and drops the space
   OCR often reads next to an underscore ("Junker_ Boy" -> "Junker_Boy").

Per segment, up to 9 clean HUD seconds spread over it are OCR'd and each side
is **voted** (`vote_names`): reads are clustered by similarity (>= 0.75), the
biggest cluster wins, represented by `best_spelling`.

### Name keys and one spelling per player

Names are compared, clustered and counted by **key** (`name_key`), never by
raw spelling. A key is lower case, letters and digits only, with the glyphs
OCR swaps in the game's font folded (0 = o, 1 = i = l). "Junker_Boy",
"Junker Boy" and "JunkerBoy" share a key, as do "U.M.A.musume" and
"UMAmusume", and "Left0ver" and "LeftOver". `name_similarity` (set grouping,
side swaps) compares keys.

`best_spelling` picks the spelling shown for a group of reads: the most
common key, then that key's most common raw spelling. OCR drops a thin mark
inside a name (`_`, `.` or `-` between letters), or reads it as a space, far
more often than it invents one. So a spelling with a mark wins once at least
a quarter of the key's reads have it (`MARK_MIN_SHARE`; "Junker_Boy" kept
its underscore in about a third of the reads before `OCR_JOIN_WIDTH`). A mark
at an edge ("TimesUp.") doesn't count.

After grouping, `vod_name_spellings` gives every player **one spelling for
the whole VOD**, so all their files are named alike. Per key it applies
`best_spelling` to every read of every set, on both sides. A rarely read key
is mapped to a more common key it misreads by a letter ("UMAmusumi" for
"UMAmusume") when all of these hold:

- similarity >= 0.85 (`NAME_VARIANT_SAME`);
- both keys have at least 6 characters;
- the common key is read at least twice as often;
- the two names never face each other in a set.

Short names never merge ("Dona" and "Dana", or "Revil" and "Revill", may be
two players). A key is only mapped to a key that keeps its own spelling: a
chain of misreads ("UMAmusumii" -> "UMAmusumi" -> "UMAmusume") ends at the most
read one. Two different long handles this alike that never meet in a set
would still merge; the count and similarity tests are the only guard.

The reader is loaded before the 1 fps scan, so a problem shows at once. If
easyocr is not installed, or fails to load (a torch DLL error on Windows) or
to start (its first run downloads models; offline that fails), the scan still
runs: it prints a `WARNING:` line saying why (with the pip command for the
interpreter in use), names stay empty and grouping falls back to the tags'
look.

### Grouping

**Function:** `group_segments()`

Each segment is compared with the current set's names (votes pooled over its
segments, side swaps aligned, `set_names`):

- **Names:** a side "matches" at similarity >= 0.6. Same set when every side
  read in both matches; a new set when a side read in both clearly differs
  (so a winner who stays on for the next set still starts a new one).
  Players may swap sides between games: the swapped pairing makes it the same
  set when all four names were read, and "unknown" (the look decides) when
  some were not (one player seen on the other side may also be them starting
  a new set).
- **Names unreadable:** the tags' look decides (`segment_signature`: each
  strip binarised with Otsu's threshold, so the encode's levels don't matter,
  median over the segment, 96x8 per side). Same players at a per-side mean
  difference at or below a threshold that follows the VOD's own noise
  (`signature_threshold`: 10x the median distance between a segment's even-
  and odd-second signatures, kept within 0.03-0.08). Tappa #4, also
  re-encoded to limited range, darker, brighter, 720p/480p upscaled and with
  added noise: noise 0.003-0.004, same set 0.001-0.013, changed player
  0.12-0.21. A side swap by look (`swapped_look_same`): each side's cropped
  tag text against the other side's, over a range of slants (P2's name is
  drawn in italics), both <= 0.15. The two sides never match as closely as
  one side with itself (Tappa #4: same player 0.10-0.26, other players >=
  0.24), so only about half of real swaps are caught this way and a missed
  one splits the set; with easyocr every swap is handled by the names.
- **No tag crops either:** a segment whose first hearts state already shows a
  lost heart continues the previous one (gap up to 270 s); otherwise a gap
  over 90 s starts a new set.

There is no hard gap rule: players often spend 12-100 s in the lobby between
games of one set, which split 4 of 16 sets before. The flip side: consecutive
sets of the same two players (a grand final and its bracket reset, a
rematch) become one set, however long the break between them; split that
file by hand. On the new VOD the grand final (4 games) and its reset (4 games)
are one 8-game file; Tappa #4's truth counts its grand final and reset as one
set too. Splitting by game wins (a new set starts once a player reaches the
set's win count) is not done. It needs each game's winner, which the hearts
reader gets wrong too often (it misses lost hearts), and the set length,
which is not on the HUD and can change within one bracket. The hearts only break ties:
the reader misses lost hearts often enough (a new set's first frames can read
as 0-1) that they never override readable names or tags.

### Game count

`DetectedSet.game_count`: every segment starts a game, plus each reset of the
hearts to 0-0 inside a segment. A segment only resumes the previous game when
its first hearts state strictly follows the previous segment's last one (0-1
then 1-1).

---

## Step 4: ffmpeg Stream Copy Cutting

**Function:** `cut_sets()` (CLI); the app uses Rust `cut_vod_sets`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `padding_before` | 10 seconds | Extra time before set start |
| `padding_after` | 20 seconds | Extra time after set end |

A set ends at its last timer-read second. The HUD fades at the match-deciding
KO (and hides during a finishing super), so with the old 5 s pad the last KO
could land within 2 s of the file end, which round detection treats as an
unfinished round, and the win screen was trimmed by up to 15 s. 20 s keeps
both. Cut start is clamped to >= 0; cut end to <= the video duration.

```
ffmpeg -ss <start> -i <vod> -t <duration> -c copy -avoid_negative_ts make_zero -y -v warning <output>
```

Stream copy starts at the keyframe before `start`, so a file may begin up to
one GOP (8 s on Tappa #4) early. That only adds lead-in; an exact start would
need a re-encode.

Output files: `{index:02d}_{p1}_vs_{p2}.mp4` (names sanitised to
alphanumerics, `_` and `-`), or `set_{index:02d}.mp4` without names.

### Tools

`find_tool()` resolves ffmpeg/ffprobe in order: `SPAGHETTI_FFMPEG` /
`SPAGHETTI_FFPROBE`, PATH, `imageio_ffmpeg` (ffmpeg only; in the `ocr`
extra), then a bundled copy (`ffmpeg/`, `resources/ffmpeg/` or `bin/` next to
`scripts/`). Without ffprobe, duration and size are parsed from `ffmpeg -i`.
Without ffmpeg the script exits with code 2 and an actionable `ERROR:` line.

---

## Rust Bridge

### Interpreter and ffmpeg

`python_bin()` / `require_python()` pick the interpreter once per session:
`SPAGHETTI_LAB_PYTHON`, the project's `.venv` / `venv` (dev checkout), a venv
at `<local app data>/SpaghettiLab/venv` (installed app), then `python3` /
`python` from PATH. The first that can `import cv2, numpy` wins; a bare
`python` on Windows is often the Store alias or an unrelated install. When none
can, `scan_vod`, `analyze_video` and `reanalyze_replay` fail with a message
naming the pip command and `SPAGHETTI_LAB_PYTHON`.

`ffmpeg_bin()` resolves ffmpeg: `SPAGHETTI_FFMPEG`, PATH, the chosen Python's
`imageio_ffmpeg`, then a bundled copy (`resources/ffmpeg/` or `ffmpeg/` next
to the exe, or in the project root). Used by `scan_vod` (passed to the script
as `SPAGHETTI_FFMPEG` so both agree), `cut_vod_sets`, `extract_preview_frame`
and `export_clip`.

### scan_vod Command

**Signature:** `async fn scan_vod(app_handle, video_path, roi_config) -> Result<Vec<DetectedSetInfo>, String>`

1. Spawns `split_vod.py` with `--preview --json-output`
2. Passes ROI configuration as CLI arguments:
   - `--p1-tension y1,y2,x1,x2`, `--p2-tension y1,y2,x1,x2` (accepted, unused)
   - `--timer-roi y1,y2,x1,x2`
   - `--p1-name-quad tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y`
   - `--p2-name-quad tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y`
3. Reads stdout line-by-line:
   - `PROGRESS:N/M` and `WARNING:...` lines are emitted as `vod-scan-progress` Tauri events
   - `JSON_RESULT:{...}` line is parsed as the final result
4. Reads stderr in a separate thread (avoids deadlock)
5. Parses JSON into `Vec<DetectedSetInfo>` (extra JSON fields such as
   `segment_count` and `warnings` are ignored)

### cut_vod_sets Command

**Signature:** `async fn cut_vod_sets(app_handle, video_path, sets, output_dir) -> Result<Vec<String>, String>`

Performs the ffmpeg cutting directly in Rust (does not call Python):
1. Creates the output directory
2. For each `CutSetRequest`, runs ffmpeg stream copy with 10 s / 20 s padding
   (same as the script)
3. Emits `vod-cut-progress` events (`i/total`)
4. Returns list of output file paths

---

## Frontend: SplitVodView

**Source:** `app/src/components/SplitVodView.tsx`

### Step State Machine

```
select -> roi -> scanning -> results -> cutting -> done
                                   \-> error
```

| Step | Description |
|------|-------------|
| `select` | File picker for VOD video |
| `roi` | RoiPicker component for adjusting ROI regions on a preview frame |
| `scanning` | Live progress display while Python scans the VOD |
| `results` | Table of detected sets with checkboxes, all selected (every set has the game's timer on screen); scan warnings (e.g. easyocr missing) shown above |
| `cutting` | Progress display during ffmpeg cutting |
| `done` | Success confirmation with output directory |
| `error` | Error display with full Python traceback and Copy Error button |

### Preview Frame Extraction

On entering the `roi` step, `extractPreviewFrame(videoPath, 30)` extracts a frame at 30 seconds (likely gameplay) via the Tauri `extract_preview_frame` command (ffmpeg single-frame extraction).

### Results Display

Detected sets are shown in a table with:
- Set number, player names, time range, gameplay duration, game count
- Checkboxes for selecting which sets to cut
- All sets are selected. A set whose names weren't read is shown and cut as "P1 vs P2" (`NN_P1_vs_P2.mp4`) instead of being dropped

---

## Frontend: RoiPicker

**Source:** `app/src/components/RoiPicker.tsx`

An interactive ROI editor overlaid on the preview frame.

### ROI Types

| Key | Type | Color | Description |
|-----|------|-------|-------------|
| `p1_tension` | RoiRect | (hidden) | Sent with defaults, unused by the scan |
| `p2_tension` | RoiRect | (hidden) | Sent with defaults, unused by the scan |
| `timer` | RoiRect | Yellow | Timer region |
| `p1_name` | RoiQuad | Blue (lighter) | P1 name tag parallelogram |
| `p2_name` | RoiQuad | Red (lighter) | P2 name tag parallelogram |

### Default ROIs (1920x1080 GGS)

```typescript
p1_tension: { y1: 1000, y2: 1040, x1: 260, x2: 750 },
p2_tension: { y1: 1000, y2: 1040, x1: 1170, x2: 1660 },
timer: { y1: 110, y2: 180, x1: 920, x2: 1000 },
p1_name: { tl: [145, 168], tr: [400, 147], br: [400, 174], bl: [145, 195] },
p2_name: { tl: [1550, 153], tr: [1795, 177], br: [1795, 200], bl: [1550, 178] },
```

### Rect ROI Interaction

- **Move:** Click and drag the rectangle body to reposition
- **Resize:** Click and drag corner handles to resize

Drag state tracks the initial mouse position and original ROI, computing deltas scaled to video coordinates.

### Quad ROI Interaction

- Rendered as SVG polygons with filled background and stroke border
- Each of the 4 corners (tl, tr, br, bl) has an independent drag handle
- Corners can be moved independently, allowing arbitrary quadrilateral shapes (not just rectangles)

### Time Slider for Preview Scrubbing

When `videoDurationSecs` is provided, a time slider allows scrubbing through the VOD to preview different timestamps. Calls `onSeekPreview(timestampSecs)` which re-extracts a preview frame at the new position.

---

## Data Types

### Python

```python
@dataclass
class Segment:
    start: float        # seconds
    end: float          # seconds
    hud_secs: int       # valid timer seconds
    hearts: list[tuple[int, int]]  # stable (p1_lost, p2_lost) states
    p1_name: str        # voted OCR names, and the per-frame reads
    p2_name: str
    p1_votes: list[str]
    p2_votes: list[str]
    signature: np.ndarray | None   # tags' look, for the no-OCR fallback
    tag_shapes: tuple | None       # cropped tag texts, for side swaps by look
    signature_noise: float | None  # even- vs odd-second signature distance

@dataclass
class DetectedSet:
    segments: list[Segment]
    index: int
    p1_name: str
    p2_name: str
    # game_count (property): games from segments + hearts resets
```

The JSON per set has `index`, `start_secs`, `end_secs`,
`gameplay_duration_secs`, `game_count` (from the hearts; was the number of
segments), `segment_count`, `p1_name`, `p2_name`; the top level also has
`warnings`.

### Rust

```rust
pub struct RoiRect {
    pub y1: u32, pub y2: u32,
    pub x1: u32, pub x2: u32,
}

pub struct RoiQuad {
    pub tl: [u32; 2],  // [x, y]
    pub tr: [u32; 2],
    pub br: [u32; 2],
    pub bl: [u32; 2],
}

pub struct VodRoiConfig {
    pub p1_tension: RoiRect,
    pub p2_tension: RoiRect,
    pub timer: RoiRect,
    pub p1_name: RoiQuad,
    pub p2_name: RoiQuad,
}

pub struct DetectedSetInfo {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
    pub gameplay_duration_secs: f64,
    pub game_count: u32,
    pub p1_name: String,
    pub p2_name: String,
}

pub struct CutSetRequest {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
    pub p1_name: Option<String>,
    pub p2_name: Option<String>,
}
```

### TypeScript

```typescript
interface RoiRect { y1: number; y2: number; x1: number; x2: number; }
interface RoiQuad { tl: [number, number]; tr: [number, number]; br: [number, number]; bl: [number, number]; }
interface VodRoiConfig { p1_tension: RoiRect; p2_tension: RoiRect; timer: RoiRect; p1_name: RoiQuad; p2_name: RoiQuad; }
interface DetectedSetInfo { index: number; start_secs: number; end_secs: number; gameplay_duration_secs: number; game_count: number; p1_name: string; p2_name: string; }
interface CutSetRequest { index: number; start_secs: number; end_secs: number; p1_name: string | null; p2_name: string | null; }
```

---

## Error Handling

- Python exceptions produce full tracebacks on stderr; a missing ffmpeg is an `ERROR:` line (exit 2), a missing or broken easyocr only a `WARNING:` (the scan continues without names)
- The Rust bridge captures stderr in a separate thread and includes the last error line in the error message
- The frontend `error` step displays the full error text with a "Copy Error" button for debugging
- Windows-specific: stdout/stderr are reconfigured to UTF-8 at script startup to avoid charmap encoding errors from EasyOCR
