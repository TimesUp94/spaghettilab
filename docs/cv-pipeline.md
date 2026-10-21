# Python CV Pipeline

The Python computer vision pipeline extracts per-frame match state (health, timer, tension, round counter) from recorded Guilty Gear Strive gameplay video. It is orchestrated by the `ReplayPipeline` class and stores results in SQLite + Parquet.

---

## Architecture

```
analyze_replay.py
  -> PipelineConfig.from_yaml()
  -> ReplayPipeline(config, extractors)
  -> pipeline.process_video(video_path) -> List[FrameData]
  -> HealthAggregator.aggregate(frames) -> List[DamageEvent]
  -> storage.write_*() -> SQLite + Parquet
```

### Core flow

1. `iter_frames()` reads the video frame-by-frame via OpenCV
2. For each frame, `ReplayPipeline` crops all needed ROIs once, builds a `FrameContext`, then runs each extractor in order
3. If `SceneDetector` marks the frame as non-gameplay (`data.is_gameplay = False`), remaining extractors are skipped
4. After all frames are processed, `HealthAggregator` detects damage events from health deltas
5. Results are written to SQLite (for the Tauri backend) and Parquet (for offline analysis)

**Source:** `replanal/pipeline.py`

---

## Data Models

All models live in `replanal/models.py`.

### Side (Enum)

```python
class Side(Enum):
    P1 = 1
    P2 = 2
```

### FrameContext

Immutable context passed to every extractor for a single frame.

| Field | Type | Description |
|-------|------|-------------|
| `video_path` | `str` | Path to source video |
| `frame_number` | `int` | Index of the frame in the source video (0, 2, 4, ... on a 60 fps video) |
| `timestamp_ms` | `float` | `(frame_number / video_fps) * 1000.0` (real video time) |
| `frame_bgr` | `np.ndarray` | Full frame in BGR format, always resized to 1920x1080 |
| `rois` | `dict[str, np.ndarray]` | Pre-cropped ROI images keyed by name |

### FrameData

All extracted data for a single frame, populated incrementally by extractors.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `frame_number` | `int` | required | Frame index |
| `timestamp_ms` | `float` | required | Timestamp in milliseconds |
| `is_gameplay` | `bool` | `True` | Set by SceneDetector |
| `p1_health` | `Optional[HealthReading]` | `None` | P1 health bar reading |
| `p2_health` | `Optional[HealthReading]` | `None` | P2 health bar reading |
| `timer_value` | `Optional[int]` | `None` | Round timer (0-99) |
| `p1_tension_pct` | `Optional[float]` | `None` | P1 tension gauge (0.0-1.0) |
| `p2_tension_pct` | `Optional[float]` | `None` | P2 tension gauge (0.0-1.0) |
| `p1_rounds_won` | `Optional[int]` | `None` | P1 round wins (from hearts) |
| `p2_rounds_won` | `Optional[int]` | `None` | P2 round wins (from hearts) |
| `combo_count` | `Optional[int]` | `None` | Combo counter hit number (ComboCounterExtractor) |
| `combo_side` | `Optional[Side]` | `None` | HUD side showing the combo counter = the attacker (also used by the health cleaner's attacker hold) |
| `text_popups` | `list[str]` | `[]` | Reserved for future use |

### HealthReading

```python
@dataclass
class HealthReading:
    side: Side
    health_pct: float        # 0.0 to 1.0; temporally cleaned once HealthAggregator has run
    bar_pixels_filled: int
    bar_pixels_total: int    # 719 (full bar)
    raw_pct: float | None    # per-frame measurement before cleaning (None = gap filled by the cleaner)
    pending_pct: float       # red "pending" combo damage next to the fill (already-lost HP)
    confidence: float        # 0..1 per-frame fit quality (occlusion, flashes)
```

`FrameData.p1_health_raw` / `p2_health_raw` keep the extractor's per-frame readings after
the aggregator replaces `p1_health` / `p2_health` with cleaned values.

### DamageEvent

Aggregated from health deltas across frames.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_ms` | `float` | When the damage event started |
| `frame_start` | `int` | First frame of health drop |
| `frame_end` | `int` | Last frame of health drop |
| `target_side` | `Side` | Who took damage |
| `damage_pct` | `float` | Amount of damage (positive) |
| `pre_health_pct` | `float` | Health before the event |
| `post_health_pct` | `float` | Health after the event |

---

## Configuration System

**Source:** `replanal/config.py`

`PipelineConfig` is loaded from a YAML file via `PipelineConfig.from_yaml(path)`.

### PipelineConfig fields

| Field | Default | Description |
|-------|---------|-------------|
| `width` | 1920 | Expected frame width |
| `height` | 1080 | Expected frame height |
| `fps` | 30 | Fallback only, when the video reports no fps (`CAP_PROP_FPS` = 0) |
| `sample_every_n_frames` | auto | `auto` = `max(1, round(video_fps / 30))` (60 fps -> 2), so ~30 samples/s; an integer forces a step |
| `rois` | `{}` | Named ROI rectangles from YAML |
| `health_bar` | `HealthBarConfig()` | Health bar extraction config |

### HealthBarConfig

| Field | Default | Description |
|-------|---------|-------------|
| `background_threshold` | 50 | Gray-level threshold for dark background |
| `min_delta_pct` | 0.005 | Minimum health change to register as damage |
| `smoothing_window` | 3 | Aggregator moving median window size |

### ROI system

ROIs are defined in the YAML config as `{x1, y1, x2, y2}` rectangles and stored as `ROIRect` instances (`replanal/roi.py`). The `ROIRect.crop(frame)` method extracts a sub-image via NumPy slicing:

```python
frame[self.y1:self.y2, self.x1:self.x2]
```

The pipeline pre-crops all ROIs once per frame before running extractors, stored in `FrameContext.rois`.

---

## Video Module

**Source:** `replanal/video.py`

### iter_frames(video_path, fps=None, sample_every=None, max_frames=None)

Generator that yields `(frame_number, timestamp_ms, frame_bgr)` tuples.

- Opens video with `cv2.VideoCapture`
- Non-1920x1080 frames are resized to 1920x1080 with `cv2.resize`
- The fps comes from the video (`CAP_PROP_FPS`); the `fps` argument (config `fps`) is only a fallback when the video reports 0
- `sample_every=None` (auto) steps `max(1, round(video_fps / 30))` frames, so the
  sample rate stays ~30/s: 60 fps YouTube replays are read every 2nd frame. Downstream
  code (health cleaning, combo grouping, the Rust round detector) counts in samples
  and assumes ~30 samples/s. Skipped frames use `cap.grab()` (no BGR copy)
- `ComboCounterExtractor`'s `stride` (default 2) also counts samples: it derives the
  spacing between sampled frame numbers (1 at 30 fps, 2 for a 60 fps video read every 2nd
  frame) and, while no counter is shown, reads every 2nd *sample*. Once a counter is found
  it reads every sample until none has been seen for 15 samples (`ACTIVE_GRACE`), and also
  reads the sample the stride skipped just before the find. The output therefore does not
  depend on which samples the stride lands on: a cut of the same video starting one frame
  later reads the same (it used to differ by 229-804 frames per file, e2e E4)
- `frame_number` is the source-video frame index; timestamp is `(frame_number / video_fps) * 1000.0`.
  30 fps videos give exactly the same frames and timestamps as before
- `max_frames` stops after that many source frames
- `VideoCapture` is released in a `finally` block

### get_video_info(video_path, fallback_fps=None, sample_every=None) -> dict

Returns metadata dictionary:
- `width`, `height`, `fps` (as reported), `frame_count`, `duration_ms`
- `effective_fps` (reported or fallback) and `sample_every` (resolved step) used by the pipeline

---

## Extractors

All extractors inherit from `BaseExtractor` (`replanal/extractors/base.py`) and implement:
- `required_rois` property: list of ROI names the extractor needs
- `extract(ctx, data)` method: reads from `FrameContext`, writes to `FrameData`

### SceneDetector

**Source:** `replanal/extractors/scene.py`

Determines whether a frame is active gameplay vs. non-gameplay (lobby, loading, transitions, stream overlays). Sets `data.is_gameplay` to `True` or `False`. The pipeline short-circuits when `is_gameplay` is `False`, skipping all subsequent extractors.

**Required ROIs:** None (reads directly from the full frame)

**Detection signals:**

1. **Health bar warm pixel check:**
   - Examines a horizontal strip at Y=112-142 (the health bar region)
   - Converts to HSV, detects "warm" pixels: `(H < 30 | H > 150) & (S > 30) & (V > 80)`
   - Checks P1 region (x=200-900) and P2 region (x=1040-1800)
   - Either player exceeding 10% warm pixel density = gameplay detected

2. **Timer region brightness check:**
   - Examines Y=110-180, X=920-1000 (timer circle area)
   - Converts to grayscale, counts pixels above brightness 100
   - More than 5% bright pixels = timer present

**Decision logic:**
- Health bars visible (>10% warm on either side) -> gameplay
- Timer visible + partial warm signal (>3% on either side) -> gameplay
- Otherwise -> non-gameplay

### HealthBarExtractor

**Source:** `replanal/extractors/health.py`

Per-frame measurement only: no smoothing and no round state. The temporal rules live in
`replanal/health_clean.py` and are applied by `HealthAggregator` (see below).

**Bar geometry (1920x1080, same with and without the tournament overlay):** each bar is
719 px wide and anchored at the screen centre. P1 spans x=149-868 (fill ends at x=867 and
shrinks towards the portrait); P2 spans x=1052-1771 (fill starts at x=1052). Interior rows
y=116-139 form the primary band; y=88-112 and y=142-170 are fallback bands for shifted HUDs.

**Column classification** (a column takes a class when > 50 % of its rows match):

| Class | Pixels | Meaning |
|-------|--------|---------|
| F (fill) | warm hue (H<=30 or >=160), S>=30, V>=170; or the yellow/green and cool modes for other overlays | current health: pastel pink/beige, orange at low HP |
| R (pending) | red hue (H<=7 or >=168), S>=180, V 90-205 | the red segment showing combo damage; already lost, not counted as health |
| W (neutral) | white (S<30, V>=170), or pastel too dark to be lit fill (V 80-170) | sprites, flashes, dimmed bar; ignored |
| N | anything else | empty (semi-transparent) bar, coloured sprites |

Fill-coloured columns that are also fill-coloured just above (y-12..y-5) or below
(y+5..y+11) the band are not a bar (full-screen, letterboxed super cinematics) and become neutral.

**Edge fit:** with index 0 at the anchored end, the health edge `e` maximises
`#F[:e] - #(R|N)[:e]`, a 1-D step fit in which neutral columns count 0. A sprite crossing
the bar lowers the confidence instead of cutting the reading, and slivers of a few pixels
are read (the old 50 px "readable" minimum treated low HP as a KO). The pending width is
the best R run right after the edge. Per side: `fill_px`, `pending_px`, `consistency`
(fraction of informative columns agreeing with the fit), `occluded` (neutral fraction) and
`ambiguity` (best-fit plateau width), with
`confidence = consistency * (1 - occluded) * (1 - ambiguity)`.

**Emission:** nothing when neither bar shows fill or pending red (HUD absent). A side is
emitted when `consistency >= 0.70`; an empty bar only when the other bar shows the HUD is up.

### TimerExtractor

**Source:** `replanal/extractors/timer.py`

Reads the two-digit round timer (0-99) by in-place grey template matching of each digit cell.
Returns `None` rather than guess: a missing reading is cheap for the cleaner, a wrong one is not.

**Required ROIs:** `timer_digits`

**Algorithm:**

1. Grey ROI, scaled to the 80x55 reference size and zero-padded by 8 px.
2. The two digits sit at fixed, staggered cells (the ones digit is drawn ~12 px lower than the
   tens). Each cell is searched within +-4 px of its nominal box (46x36) with normalised
   cross-correlation (`TM_CCOEFF_NORMED`) against ten grey glyph templates shared by both cells.
3. A cell is accepted when the best score is >= 0.65, beats the runner-up digit by >= 0.10, and
   every quadrant of the matched patch agrees with the glyph (RMS of the normalised difference
   <= 0.95 for tens, 1.2 for ones). The quadrant check rejects partly covered digits (a 5 with its
   top covered correlates with 3, a 7 with its bar hidden with 1).
4. If a cell fails at full size but looked digit-like (score >= 0.5), glyphs squashed to 90% / 80%
   height towards the baseline are tried: a new digit grows up from the baseline for a few frames
   after each tick.
5. Both cells read -> `tens * 10 + ones`; otherwise `timer_value` stays `None`.

Templates (`replanal/extractors/timer_glyphs.npz`, shipped with the package) are averaged, aligned
crops from the hand-labelled set in `tests/data/timer/` (overlay VODs + 60 fps non-overlay YouTube
replays), rebuilt and scored with `scripts/build_timer_templates.py`. On held-out videos: 100% exact
on clean crops, 0 wrong values (the old Otsu/half-split reader: 41% exact, 45% wrong, e.g. 98->90,
67->82, 65->85). Cost ~1.3 ms per read (old: ~0.16 ms). Values below 28 never occur in the labelled
VODs, so tens digits 0/1 and the single-digit display (< 10) are unverified; an unrecognised
display reads `None`.

### TensionExtractor

**Source:** `replanal/extractors/tension.py`

Reads P1 and P2 tension gauge percentage from the green-hued tension bars.

**Required ROIs:** None (reads directly from the full frame)

**Pixel regions:**
- Y band: y=1000-1040
- P1: x=260-750 (fills right-to-left from center outward)
- P2: x=1170-1660 (fills left-to-right from center outward)

**Green hue detection:**
- H: 35-85
- S: > 40
- V: > 60
- Column fill threshold: 20% of vertical pixels must be green

**Fill measurement:** Longest continuous run of filled columns = tension bar width.

**Calibration:** Max observed width per player is tracked as the 100% reference. Percentage = smoothed_width / max_width.

**Smoothing:** 15-frame rolling median window (`_MEDIAN_WINDOW = 15`).

**Reset:** Same 15-frame gap detection as HealthBarExtractor. Resets max widths and median buffers.

### RoundCounterExtractor

**Source:** `replanal/extractors/round_counter.py`

Counts rounds won by each player from the two hearts per side next to the round timer. They are game-native HUD (same pixels with or without our stream overlay; the overlay's name/score banner above them is never read). Red = alive, grey = lost; the outer heart is lost first and a lost heart is drawn smaller and ~4 px lower.

**Required ROIs:** None (reads directly from the full frame at known positions)

**Heart positions (1920x1080 coordinates):**
- P1 hearts: inner (843, 86), outer (799, 91); grey outer heart centred at (799, 94)
- P2 hearts: inner (1077, 86), outer (1121, 91); grey outer heart centred at (1120, 94)

**Detection method** (`read_lost_hearts(frame, side)`, per side):

1. The inner heart must be a visible red heart: median of a small box on it has (R-G)/R >= 0.6, and at least 3 of 4 boxes just outside it (below both bottom corners, above it, the gap between the hearts) are clearly less red. Red screen fills and the dark-red lobby stripes fail this.
2. The outer heart is compared against that still-red heart: red if similarly red and bright (R within 0.5-2.2x, the inner heart pulses), lost/grey if its median inside the grey-heart outline is neutral (or pinkish while the surroundings are red-tinted), not yellow, at most 2x the red heart's brightness, uniform, and the ring around the outline differs from it on at least 3 of 4 sides (top, left, right, below the tip).
3. Anything else returns `None` for that side (white flash, HUD hidden by a super cinematic or fade, occluding effects) instead of a count. Two lost hearts are only accepted when both are clean neutral-grey islands; in practice the HUD fades out right after the match-deciding KO, so 2 is almost never seen.

**Scoring logic:**
- `rounds_won_by_P1 = lost hearts on P2's side` (P2 lost those rounds)
- `rounds_won_by_P2 = lost hearts on P1's side`

Values are written only when both sides are readable. Accuracy on the labeled set in `tests/data/hearts` (804 frames, our VODs + 3 non-overlay YouTube replays, per-side readings): 98.7% correct / 0% wrong on the held-out videos, 97.9% / 0.12% overall (the previous S<60/V>100 classifier: 57.5% / 42.0%).

---

## HealthAggregator

**Source:** `replanal/aggregator.py`, `replanal/health_clean.py`

`aggregate(frames)` first cleans the health readings in place (`clean_frames`), then
detects damage events on the cleaned series. Call it before anything that reads
`health_pct` (storage, combo detection).

### Temporal cleaning (`health_clean.clean_series`)

GGST rules: within a round HP only goes down; it goes back up only to full, at the next
round; the red pending segment is already-lost HP.

1. **Segments (rounds):** a new segment starts after a gap of > 150 frames with no readings,
   or after a sustained (15-frame) reset following damage in the round, at least 300 frames
   after the previous start. A reset means both sides' 9-frame medians read "full", or both
   rose >= 0.30 above their in-round lows with one of them full. "Full" is
   `min(0.90, observed full-bar reading - 0.05)` per side: the game's character name on the
   bar (or a replay input display) can hide its far end (set 03: a full P2 bar reads 0.81 in some rounds). Wallbreaks
   do not refill the bars in current GGST, so they do not split rounds.
   **Round intro:** the timer shows its start value (>= 89) while the bars are still empty
   and filling. A reset moves back to a credible run of start readings (3 in 15 frames)
   within 150 frames before it, if every timer reading in between is a start value (a
   frozen in-round timer misread as "93" interleaves with its true "83"s). Otherwise the
   empty intro bars end the previous round at 0 for both sides.
   **Timer-intro reset** (fallback, `_timer_intros` / `_intro_reset`): when a bar never
   reads full (a bright stage washes out the P1 bar, which reads 0.62 with confidence 0.05;
   the replay input display covers the outer part of both bars), the round timer still
   restarts. After a gap of > 30 frames without a timer reading, a run of >= 20 readings in
   the next 90 frames, >= 90 % of them start values (>= 89), starts a new segment if the
   segment had a credible in-round reading (< 89), is >= 300 frames old and had damage, and
   the raw bars over those 90 frames (any confidence, median) rose: one side >= 0.30 above
   its low in the segment, the other not below its low - 0.05. It is checked 90 frames
   after the intro, so the regular rule gets the first chance; the reset moves back to just
   after the last confident low reading of the old round (the bars fill before the timer
   shows).
2. **Monotone fit per side:** weighted L1 isotonic regression (non-increasing, PAV with
   weighted medians) over readings with confidence >= 0.35. Short runs of bad readings in
   either direction are outvoted. A new level needs >= 3.0 total weight (about 4 good
   frames); lighter blocks merge into their closer-valued neighbour.
3. **Empty-bar readings** count only when credible: pending red is visible next to the
   emptied bar, or a first fit without them already had the side at <= 0.20. Otherwise
   flashes or dark sprites over the HUD produce false KOs. An empty run is never credible
   if the bar later **comes back** (`_comes_back`): a run of 5 confident readings of
   0.05-0.80, counted where the other bar reads its normal level (not the post-KO fade,
   where both read near 0) or where the reading is at most the side's earlier level.
   Sprites can cover a whole HUD corner for seconds (set 14 R1).
4. **Attacker hold** (`_opp_combo_gate`): the player doing a combo essentially never loses
   HP during it, so readings of the attacker's bar below its level just before (median of
   >= 3 confident readings in the 15 frames before, runs padded by 6 frames) are occlusion:
   its own sprite, super VFX or a big object over its bar (set 13 R3, 11_WFinal R12).
   - While the combo counter is on this side's HUD (3+ consecutive frames; the counter is
     drawn on the attacker's side), or while the other side's pending red keeps growing
     (from its 2nd growth step of >= 0.01), and this side shows no red itself: readings
     more than 0.02 below the level (or below half of it) are dropped.
   - While the other side merely shows red (a single hit: its red lingers ~1 s, and this
     side can be hit back in that time, 07_UF ~304 s) only readings below half the level
     are dropped.
   - A real hit on the attacker (a trade, a reversal) shows its own red, which ends the
     hold on those frames.
5. **Round end:** play ends at the earlier of the last run of 5 frames where both bars are
   read confidently and the last credible round-timer reading (the KO slash hides the
   timer). After it, readings may confirm a side's level but not lower it, except a reading
   with pending red next to it (a real hit, e.g. a trade) or a credible empty bar.
6. **KO lock:** the KO is the first frame where a side's fit reaches 0, or earlier, a
   **raw KO** (`_raw_ko`): a confident (>= 0.6) empty reading with pending red on a side
   already at <= 0.20 — or one read with only >= 0.35 when the red covers the side's whole
   remaining level and the frame before showed fill at >= 0.6 (the green was visibly there
   and the red is drawn where it was; 07_UF R5. Without that frame, a sliver under a busy
   super background reads empty mid-combo, yt_KWXz7dmt2Pk R0) — where the next 45 frames
   stay below min(0.015, half its level) (readings above its level before the hit are
   ignored: HP cannot rise, and the KO cinematic right after a lethal hit reads as
   low-confidence junk fill, 11_WFinal R12; readings <= 0.016 count as empty: over an empty
   bar the cinematic reads a steady 1-11 px, 10_UF R1), the bar does not come back, and the round timer does not count down afterwards (the clock freezes at
   a KO; a 3 % bar flickers to empty while play goes on). The KO slash often hides the
   HUD within a frame, so the fit alone never gets there. The loser is 0 from its KO. If
   the survivor was at <= 0.20 it may still reach 0 within 30 frames (double KO);
   otherwise nothing after the KO may lower it, even with red shown (KO-screen junk shows
   fake red).
   **Survivor shown after the KO** (`_near_double_ko_survivor`): when the survivor is at
   <= 0.20 just after the KO, the last run of 5 frames within 210 frames after the KO in
   which one bar shows its level (>= 0.01, between its level after and before the KO,
   confidence >= 0.35) while the other reads empty (confidence >= 0.6) names the survivor.
   That side keeps its level: its later empty readings are KO-cinematic junk, not a double
   KO (05_UF R6). If it is the side the fit KO'd first and both were at <= 0.03 (a
   near-double KO; a 1-px sliver flickers to empty), the loser is swapped (yt_Pbs R0). The
   210-frame limit keeps out the next round's intro screen, whose junk reads like a sliver
   next to an empty bar (14_Grands R1).
   **KO fallback:** if neither side reaches 0 this way, each side is refit with full weight
   (not x0.5) on the credible empty readings after the end of play, when the other side was
   above 0.20 at the end of play. The HUD is often up for only a few frames between the KO
   and the slash (set 12 R1: 6 frames after a super cinematic). The refit is adopted only if
   exactly one side then reaches 0.
7. **Gaps:** frames between fitted points (up to 120 frames) are linearly interpolated,
   so a bar hidden during a combo keeps going down and never comes back up. Frames after
   the last fitted point that still had a (rejected) reading hold the last value.

Frames with no cleaned value get `p1_health` / `p2_health = None`.

### Pipeline (damage events)

For each side (P1, P2):
1. **Extract series:** Build a float array of cleaned `health_pct` values; non-gameplay frames become `NaN`
2. **Mask transitions:** Detect health jumps UP > 8% between consecutive valid readings (round reset). NaN-out a window of 5 frames before to 15 frames after the jump
3. **Smooth:** 3-frame moving median filter (configurable via `HealthBarConfig.smoothing_window`)
4. **Detect events:** Walk the smoothed series looking for health decreases

### Transition masking

Any upward jump > 8% in health between consecutive valid readings is treated as a round transition. A window from 5 frames before to 15 frames after the jump is set to NaN to prevent false damage events from the transition animation.

### Damage event detection

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SETTLE_FRAMES` | 6 | Lookahead window for continued decline (200ms at 30fps) |
| `MAX_COMBO_GAP` | 150 | While pending red is shown, drops up to this many frames apart stay one event |
| `min_delta` | 0.005 | Minimum health drop to register as an event |

Algorithm:
1. Scan for any frame where `smoothed[i+1] < smoothed[i] - min_delta`
2. Walk forward while health continues declining, allowing pauses up to `SETTLE_FRAMES`, extended while the pending red segment is shown (2 of 5 frames), because the combo is still running
3. Record `DamageEvent` with pre/post health, frame range, and damage amount
4. Events are sorted by timestamp across both players

---

## Storage

**Source:** `replanal/storage.py`

### SQLite Schema

**replays table:**
| Column | Type | Description |
|--------|------|-------------|
| `replay_id` | TEXT PRIMARY KEY | Typically the video filename stem |
| `video_path` | TEXT | Full path to source video |
| `duration_ms` | REAL | Video duration |
| `frame_count` | INTEGER | Total frame count |

**frame_data table:**
| Column | Type |
|--------|------|
| `replay_id` | TEXT (FK -> replays) |
| `frame_number` | INTEGER |
| `timestamp_ms` | REAL |
| `p1_health_pct` | REAL (nullable) |
| `p2_health_pct` | REAL (nullable) |
| `timer_value` | INTEGER (nullable) |
| `p1_rounds_won` | INTEGER (nullable) |
| `p2_rounds_won` | INTEGER (nullable) |
| `p1_tension_pct` | REAL (nullable, migration column) |
| `p2_tension_pct` | REAL (nullable, migration column) |

Primary key: `(replay_id, frame_number)`

**damage_events table:**
| Column | Type |
|--------|------|
| `event_id` | INTEGER PRIMARY KEY AUTOINCREMENT |
| `replay_id` | TEXT (FK -> replays) |
| `timestamp_ms` | REAL |
| `frame_start` | INTEGER |
| `frame_end` | INTEGER |
| `target_side` | INTEGER (1 = P1, 2 = P2) |
| `damage_pct` | REAL |
| `pre_health_pct` | REAL |
| `post_health_pct` | REAL |

### Migrations

The `init_db()` function uses `ALTER TABLE ... ADD COLUMN` wrapped in try/except to add columns that may not exist in older databases (e.g., `p1_tension_pct`, `p2_tension_pct`).

### Parquet Output

`write_frame_parquet()` writes per-frame data as an Apache Parquet file using PyArrow directly (no pandas dependency). Output path: `<output_dir>/frames/<replay_id>_frames.parquet`.

Schema mirrors the `frame_data` SQLite table with typed Arrow columns (int32, float64). Parquet-only health columns: `p1/p2_health_raw` (per-frame measurement before cleaning), `p1/p2_pending_pct` and `p1/p2_health_conf`. The SQLite `frame_data` table is unchanged (it must stay in lockstep with the web app, and the `.spag` export copies it with `SELECT *`); its `p1/p2_health_pct` columns now hold the cleaned values.

---

## Scripts

### analyze_replay.py

Analyzes a single GGS replay video end-to-end.

```
python scripts/analyze_replay.py <video_path> [--config <yaml>] [--output <dir>]
```

Builds the full pipeline: SceneDetector -> HealthBarExtractor -> TimerExtractor -> TensionExtractor -> RoundCounterExtractor. Writes SQLite, Parquet, and a health timeline chart (if matplotlib is available).

### analyze_sets.py

Batch-analyzes all MP4 files in a directory.

```
python scripts/analyze_sets.py <sets_dir> [--config <yaml>] [--output <dir>] [--sample-every N]
```

Default `--sample-every` is 2 (15fps processing). Omits TensionExtractor for faster batch processing. Prints a summary table of all sets at the end.

### calibrate_roi.py

Interactive ROI calibration tool using OpenCV windows.

```
python scripts/calibrate_roi.py [frame_path] [--config config/default.yaml]
```

Opens a sample frame with ROI rectangles overlaid and shows each cropped ROI in separate windows for visual verification.
