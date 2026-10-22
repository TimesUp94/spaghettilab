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
- `ComboCounterExtractor`'s `stride` (default 3) also counts samples: it derives the
  spacing between sampled frame numbers (1 at 30 fps, 2 for a 60 fps video read every 2nd
  frame) and, while no counter is shown, reads every 3rd *sample*. Once a counter is found
  it reads every sample until none has been seen for 15 samples (`ACTIVE_GRACE`), and also
  reads the samples the stride skipped just before the find. The output therefore does not
  depend on which samples the stride lands on: a cut of the same video starting one frame
  later reads the same (it used to differ by 229-804 frames per file, e2e E4). A counter on
  fewer than 3 samples can be missed, but grouping drops such blips anyway.
- Decode cost at 60 fps: skipped frames still go through `cap.grab()` (H.264 needs every
  reference frame decoded); measured ~0.26 ms per skipped 720p frame, ~1-2 % of real time.
  An ffmpeg `select`/`scale` pipe was ~10x slower on Windows. The analyzer's real cost was
  the combo reader (~60 ms per read): it now matches in threads (`workers=4`: the two
  sides, and word + big matcher per side; OpenCV releases the GIL; identical readings),
  ~3.5x faster, plus the stride of 3.

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
| F (fill) | warm hue (H<=30 or >=160), S>=30, V>=170; or the yellow/green (V>100) and cool (H 90-145, S>40, V>150) modes for other overlays | current health: pastel pink/beige, orange at low HP |
| R (pending) | red hue (H<=7 or >=168), S>=180, V 90-205 | the red segment showing combo damage; already lost, not counted as health |
| W (neutral) | white (S<30, V>=170), or pastel too dark to be lit fill (V 80-170) | sprites, flashes, dimmed bar; ignored |
| N | anything else | empty (semi-transparent) bar, coloured sprites |

Fill-coloured columns whose fill colour carries on just above (y-12..y-5) or below
(y+5..y+11) the band are not a bar (full-screen, letterboxed super cinematics) and become
neutral. In the warm mode this needs the outside rows' mean colour to be the band's own
(hue within 6, saturation within 30, brightness within 40): pink or cream stage scenery
right above a pink bar is a different shade and used to hide real fill columns (new-VOD
set 01 at 140 s read P2 0.37 instead of 0.41; set 07 R0 0.348 instead of 0.398). The
yellow/green and cool modes, whose cinematic swirls vary in shade, and pending-red columns
(red never reaches past the bar; red outside it is a sprite) keep the plain test. The cool
mode needs V > 150: the empty, semi-transparent bar over a blue stage reads H ~95, S 50-90,
V ~80 and was read as 7 % fill on a KO'd bar (set 01 at 411 s).

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

Reads the round timer (0-99) by in-place grey template matching of each digit cell, and the
single warm digit of the last 9 seconds.
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
5. Both cells read -> `tens * 10 + ones`.
6. **Single digit** (`_read_single`): from 9 down GGST draws one centred digit in orange /
   yellow (flashing), ~10-25 % larger than the two-digit glyphs and pulsing. Only when the two
   cells give no value and the lit pixels (V >= 140, >= 5 % of the ROI) are clearly warm (median
   saturation >= 90; measured 46-62 on the cream two-digit timer, 148-201 on the warm digits;
   grey-only input never takes this path) is the same glyph set matched around the ROI centre (+-8 px) at scales 1.0-1.3; the best
   (scale, digit) needs score >= 0.65, a 0.06 margin over the runner-up digit and the quadrant
   check at 1.05 (an 8 under a white streak correlates with 3). Otherwise `timer_value` stays
   `None`. New-VOD set 07 R3 (9 -> 5, 720p60): 333 of 777 frames read, 0 wrong; 8400 random
   frames of all 29 test videos read exactly as before.
7. **Replay check of single digits** (`finalize_timer`, run by `analyze_replay.py` and
   `reclean_replay.py` before the tension clean-up): the single digit is only shown for the
   last 9 seconds, so it must continue the countdown. A single-digit reading is kept only when
   3 readings within +-0.5 s show it and the last such supported reading before it (`p`, `dt`
   seconds earlier) could have counted down to it (`p - dt - 1 <= value <= p`). A tens digit
   covered by a fist or a flash read "7" for 97 (Tappa #4 04_UF 144.47 s) and for 67 (yt_KWX
   47.8 s) on one frame; both are dropped, the 140 real single-digit frames of new-VOD set 07
   are kept. The per-frame readings stay in the parquet's `timer_raw`.

Templates (`replanal/extractors/timer_glyphs.npz`, shipped with the package) are averaged, aligned
crops from the hand-labelled set in `tests/data/timer/` (overlay VODs + 60 fps non-overlay YouTube
replays), rebuilt and scored with `scripts/build_timer_templates.py`. On held-out videos: 100% exact
on clean crops, 0 wrong values (the old Otsu/half-split reader: 41% exact, 45% wrong, e.g. 98->90,
67->82, 65->85). Cost ~1.3 ms per read (old: ~0.16 ms). Values 10-27 never occur in the labelled
VODs, so tens digits 0/1 are unverified; an unrecognised display reads `None`. Single-digit test
crops are `tests/data/timer/single_*.png`.

### TensionExtractor

**Source:** `replanal/extractors/tension.py`

Reads P1 and P2 tension as an absolute fraction (0-1) of the game's tension gauge.

**Required ROIs:** `p1_tension_bar`, `p2_tension_bar` (config; defaults in `DEFAULT_BARS`): the
whole 481 px bar, fill rows y 1005-1031. P1 x 266-748, P2 x 1167-1649. The fill is anchored at
the screen-centre end (P1 x2, P2 x1) and grows outward.

**What the gauge looks like** (Tappa #4 1080p30 and the new-VOD 720p60 stream):
- Below 50 % the fill is mint green (H ~50-73, S ~70-95); from 50 % the whole fill turns pale
  yellow (H ~27-40, S ~15-100, whitish next to the anchor). V ~230-255. The old green-only window
  (H 35-85, S > 40) missed every yellow fill, at 1080p too.
- A dark gear sits on the bar at 50 % (~35 px), a second one at the outer end when full; a full
  bar also shines white over its inner half. The "TENSION" label is drawn inside the fill next
  to the anchor. Positive-bonus chevrons can be drawn over the empty bar.

**Measurement** (`measure_bar`): per column, the share of fill-coloured pixels (V >= 170,
8 <= S <= 170, 18 <= H <= 90; >= 70 % of the rows, >= 35 % in the label zone next to the anchor)
and of white shine pixels (V >= 170, S <= 40). The fill is the run starting within 19 % of the
bar from the anchor (a stream webcam may cover the anchor) and continuing over gaps of at most
8.5 % of the bar (gear, letters); white columns do not open a gap but the run must end on a
coloured column (a white flash alone reads 0). Tension = run length / bar width; >= 0.965 is
full (the end gear covers the last pixels). The gear at 50 % can hide a fill ending under it:
values 0.47-0.53 read as ~0.47.

**HUD state:** the gauge is HUD. With the round timer on screen an empty bar is 0; without it a
value is kept only when both bars show a fill, else `None` (supers, cinematics, menus).

**Replay clean-up** (`finalize_tension`, run by `analyze_replay.py` and `reclean_replay.py`):
- Overlay ceiling: if one side's readings sit at a fixed value for >= 2 s in all, (almost) none
  of its readings ever pass it (<= 0.2 %), and the other side's bar is read beyond it for >= 2 s
  too (the gauge can be read there), a stream overlay covers the bar beyond it (new VOD: the
  "CREATORS" logo caps P1 at 0.836, reached for 3.7-9 s per file). The amount is absolute, not a
  share of the file (a share of 3 % missed the cap on new-VOD sets 05-07). Readings at the
  ceiling are only a lower bound and become `None`. A player's own highest tension held still is
  shorter (Tappa #4 05_UF P2: 1.7 s at 0.70, kept). A chat box that is sometimes see-through does
  not make a ceiling; then the readings are kept.
- Median of 5 consecutive readings; `None` stays `None`.
- The per-frame readings are kept in `p1/p2_tension_raw` (parquet columns), and the clean-up
  always starts from them, so `reclean_replay.py` can re-run it without decoding the video.
  Older parquets only hold the cleaned values; re-running on them leaves them as they are
  apart from another median pass.

Validation: 26/27 frame-checked values of new-VOD sets 02/03/10 within 0.1 (old extractor 3/27;
the miss is P1 at 1.0 read 0.836 = the logo ceiling, dropped by `finalize_tension`). Test strips
of both sources in `tests/data/tension/`.

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

### ComboCounterExtractor and combo grouping

**Source:** `replanal/extractors/combo.py` (reader: see the module docstring; templates
`config/templates/combo_templates.npz`). Per frame: `combo_side` (the attacker's HUD side) and
`combo_count`.

**Grouping** (`derive_combos`, run after health cleaning; also by `scripts/reclean_replay.py`):
- HUD state from the round timer (`hud_state`): a run of >= 4 samples without a timer = HUD
  hidden (super cinematic, wallbreak, round transition); a hidden stretch after which the timer
  comes back higher (99) ended the round, as does one that runs to the end of the video. The
  values compared are the nearest supported readings (3 equal values within +-15 samples)
  before and after the gap, so a lone misread next to it (a "7" for 97) does not end the round.
- Gaps between counter readings are counted in HUD-shown samples, so a combo continues through
  a super or wallbreak (max 240 samples in all). It is also bridged when the defender's red
  pending segment stayed on screen all through the gap (GGST drains it only after the combo
  ends) and the count did not fall back below 4, and when a 10+ count comes back higher by at
  most 5 hits per second of gap. Counts falling back to 2-3 still start a new combo.
- No combo continues into the next round; a counter in the gap after a round ended can only
  extend the running combo.
- A combo must deal damage (>= 0.003): one whose defender lost no HP is a counter still on
  screen (it lingers ~1 s after the combo, and through the KO slash) or a misread: dropped.
- `hp_after` = defender HP within 3 samples of the last counter frame: the combo ends when its
  counter goes away; later loss (red draining, chip from blocking the next move) is not its
  damage.
- `hp_before` = HP before hit 1: the counter appears on hit 2 and is often read late (words under
  the COUNTER banner). The cleaned HP is followed back through the combo's drops until 45 samples
  pass without one, the red was off on 8 cleanly read frames (it drained), or a steady bar total
  (fill + red) above the combo's own shows an earlier combo's red. The nearest drop always counts.
  Red is only trusted on frames whose fill reading agrees with the cleaned HP (+-0.02).

Validation (`output/fix2/track3/`): new-VOD frame-checked combos (10 sets): matched 15 -> 26,
split 13 -> 2, phantom 15 -> 1, right hit count 11 -> 20, zero-damage combos 45 -> 0; Tappa #4
sets 02/07/09/13: 20 merges (4 checked on frames, all one combo through a super or wallbreak),
10 drops (8 zero-damage leftovers).

## HealthAggregator

**Source:** `replanal/aggregator.py`, `replanal/health_clean.py`

`aggregate(frames)` first cleans the health readings in place (`clean_frames`), then
detects damage events on the cleaned series. Call it before anything that reads
`health_pct` (storage, combo detection).

### Temporal cleaning (`health_clean.clean_series`)

GGST rules: within a round HP only goes down; it goes back up only to full, at the next
round; the red pending segment is already-lost HP.

1. **Segments (rounds):** a new segment starts after a gap of > 150 frames with no readings
   (unless the round clock carries on across it: the first credible reading after the gap is an
   in-round value no higher than the last one before it; a long super cinematic hides the whole
   HUD for 5 s or more mid-round, 14_Grands R6), or after a sustained (15-frame) reset following damage in the round, at least 300 frames
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
5. **Round end:** a segment is cleaned only up to 150 frames (`KO_MARGIN`, 5 s) after its
   last credible round-timer reading. The KO can land under a super cinematic and the HUD
   come back without a readable timer for a few seconds (up to ~4.5 s seen), but later
   frames (win pose, transition, the stream's lobby, the next video) are never the round:
   before the cut their junk changed the result with how much video followed (05b R6's
   segment ran through 100 s of lobby; 01_WRI R5 held a lobby reading). Frames after the cut
   that a fit of the whole segment covers hold the last cleaned value. Play ends at the
   earlier of the last run of 5 frames where both bars are read confidently and the last
   credible timer reading (the KO slash hides the timer). After it, readings may confirm a
   side's level but not lower it, except a reading with pending red next to it (a real
   hit, e.g. a trade) or a credible empty bar.
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
   the survivor was at <= 0.20 it may still reach 0 within 30 frames (double KO: both end at
   exactly 0; the alive floor below keeps it alive only until its own KO frame), unless
   its bar is shown with fill again later in the cleaned span (11_WFinal R4: one junk
   frame read the 7 % survivor empty; 13_LFinal R9: the survivor's 1-px sliver read empty
   while the clock still ran); otherwise nothing after the KO may lower it, even with red
   shown (KO-screen junk shows fake red).
   **Frozen-clock KO** (`_frozen_clock_ko`): the HUD is up (timer read), the clock has
   stopped (the same value over >= 40 frames, never lower afterwards) and one bar reads
   empty (<= 0.005, confidence >= 0.6) for >= 5 frames in a row while the other bar shows
   fill, and it is never read with fill again while the timer is shown. A running clock
   ticks every second, so this is the KO even when no red is seen (it went with the slash;
   e2e 02_UF R1, 14_Grands R6/R18, 09b R10). The KO is placed just after the side's last
   confident reading with fill since the clock stopped. A live 1-px sliver reads 0.008 raw
   (e2e 14_Grands R15), so the empty bound stays below it, the same as the round detector's
   `KO_HP`; both come from one 1080p stream and may need a re-check on other sources.
   **Hidden KO** (`_hidden_ko`): a round ends only by a KO or a time-out. When no KO was
   found but the next round starts (its timer intro within 150 frames) and the clock last
   read 4 or more, the KO happened while the HUD was hidden: the side clearly behind (<= 0.20
   and >= 0.10 below the other) is at 0 from just after the last timer reading (at least
   on its last cleaned frame). Not when that side's combo counter was up within 30 frames of
   the last timer reading (the attacker essentially never loses HP during its own combo, so
   it is the likely winner, e.g. a super landed from behind), nor when the clock last read
   1-3 (a time-out may have ended the round under the hidden HUD; the loser keeps its HP). 12_LSemi R1 (clock at 77, HUD hidden from the super on),
   11_WFinal R6, yt_KWX R7 (perfects whose last hit is under the KO cinematic), and losers
   left at the bar's floor junk (0.010-0.05).
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
   **A KO ends the round** (`_clock_freeze`, `_hud_back`, `_end_loser`, `ALIVE_MIN`): the
   clock stops and the HUD goes for the round-end slash. The game keeps the last HP point as a
   1-px sliver, which a 720p source cannot show at all (new-VOD set 10 R0: P2 read 0.000 for
   8 s while the clock ran 70 -> 63 and P1 took two more hits), so "at or near 0" is not a KO:
   - A KO candidate (the fit reaching 0, or the raw KO) after which the clock still counts down
     is dropped; a side with only such candidates is KO'd, if at all, when play ended (the last
     timer reading before the HUD went, see below). End of play is never before the first
     frame of the clock's final value.
   - The frozen-clock KO is no longer shadowed by a later red-backed raw KO (the earlier of the
     two wins; set 09 R4 placed the KO 6 s late, after the victory cinematic, and the survivor's
     bar was dragged from 0.97 to 0.10). It ignores empty readings while the other side shows
     red and this side none: that side is attacking and its bar is covered by its own combo's
     effects (13_LFinal R8).
   - A red segment backs an empty reading only when it covers at least half the level read just
     before (the red of a lethal combo covers the bar where the fill was). Nagoriyuki's red
     sprite over his own 0.49 bar read "empty with 0.14 red" one frame before he KO'd P2
     (yt_KWX R2).
   - **HUD back** (`_hud_back`): the timer readings from the clock's last tick on (only readings
     with 3 equal values within +-15 frames, `_supported_timer`: a lone misread on a flash would
     split the gap); the last gap of >= 15 frames in them is the slash (the clock can sit on one value through a long combo
     and a super, and tick for the last time while the HUD is hidden). Play ended with the
     reading before that gap; the frames after it that read the frozen value show the result.
     A KO first seen on those frames happened while the HUD was hidden: it is placed at the
     end of play, so the winner is frozen at its level from there (new-VOD set 06 R8: the wipe
     covered part of P2's bar on the HUD's return and dragged it from 0.957 to 0.75; 07_UF R8).
     The KO'd side is 0 on all of those frames (`_ko_shown`): its fit can leave them empty
     when the empty bar is read with middling confidence after a cinematic, and the round
     detector then ended the round at its last value before the KO (new-VOD set 08 R10:
     0.021, uncertain).
   - **Both low** (`_end_loser`): when both sides were near 0 at the end, or the KO was only
     placed at the clock's stop, who lost is decided in game terms: (1) on the HUD back after the
     slash, a bar read empty (<= 0.016, the junk floor over an empty bar) next to one read
     >= 0.0025 wider (not counting frames where the "empty" side is attacking); (2) failing that,
     the side shown with red pending (being hit) in the last 15 frames of play while the other
     shows none; (3) failing that, the side whose combo counter (the attacker) is up in those
     last 15 frames of play is the winner. The counter lingers ~1 s after a combo ends, so an
     earlier one says nothing about the last hit (a one-hit reversal shows no counter). New-VOD set 04 R3: P1's 0.003 sliver had been
     taken for the KO and P2 (being combo'd by P1's super) frozen at 0.035; set 09 R3, set 07 R1.
   - **Alive floor:** the survivor's cleaned HP, and the loser's before its KO frame, never go
     below `ALIVE_MIN` (0.008, the 1-px sliver's 1080p reading), so exactly 0 always means KO'd
     (the round detector's terminal-state rule relies on it). In a double KO the survivor is
     only floored until its own KO frame.
   - The survivor's empty readings from 30 frames before the KO on are blanked when the HUD
     shows it surviving (its own super's sprites over its bar, yt_KWX R2).
   - **Hidden KO** also resolves both-low endings with the evidence above (lead < 0.10), and
     the last round of a video (no next-round intro to prove the round ended) when the HUD back
     shows the low side empty (new-VOD set 01 R8). The loser is at 0 from the end of play.
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

Schema mirrors the `frame_data` SQLite table with typed Arrow columns (int32, float64). Parquet-only health columns: `p1/p2_health_raw` (per-frame measurement before cleaning), `p1/p2_pending_pct` and `p1/p2_health_conf`; `timer_raw` and `p1/p2_tension_raw` hold the readings before `finalize_timer` / `finalize_tension` (`timer_value` and `p1/p2_tension_pct` are the cleaned ones), so `scripts/reclean_replay.py` re-runs those clean-ups from the parquet. The SQLite `frame_data` table is unchanged (it must stay in lockstep with the web app, and the `.spag` export copies it with `SELECT *`); its `p1/p2_health_pct` columns now hold the cleaned values.

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
