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
| `frame_number` | `int` | Absolute frame index |
| `timestamp_ms` | `float` | Computed as `(frame_number / fps) * 1000.0` |
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
| `combo_count` | `Optional[int]` | `None` | Reserved for future use |
| `combo_side` | `Optional[Side]` | `None` | Reserved for future use |
| `text_popups` | `list[str]` | `[]` | Reserved for future use |

### HealthReading

```python
@dataclass
class HealthReading:
    side: Side
    health_pct: float       # 0.0 to 1.0
    bar_pixels_filled: int  # Smoothed pixel width
    bar_pixels_total: int   # Max observed pixel width (100% calibration)
```

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
| `fps` | 30 | Assumed frame rate |
| `sample_every_n_frames` | 1 | Process every Nth frame (2 = 15fps) |
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

### iter_frames(video_path, fps=30, sample_every=1)

Generator that yields `(frame_number, timestamp_ms, frame_bgr)` tuples.

- Opens video with `cv2.VideoCapture`
- Non-1920x1080 frames are resized to 1920x1080 with `cv2.resize`
- `sample_every` controls frame skipping (e.g., 2 = process every other frame)
- Timestamp is computed as `(frame_number / fps) * 1000.0`
- `VideoCapture` is released in a `finally` block

### get_video_info(video_path) -> dict

Returns metadata dictionary:
- `width`, `height`, `fps`, `frame_count`, `duration_ms`

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

The most complex extractor. Reads health percentage for P1 and P2 independently using multi-band Y scanning with multiple color modes.

**Required ROIs:** None (reads directly from the full frame at known pixel positions)

**Pixel regions:**
- P1 health bar: x=200-900 (fills from portrait edge toward center)
- P2 health bar: x=1040-1800 (fills from portrait edge toward center)
- Primary Y band: y=112-142
- Alternate Y bands (fallback): y=88-112 and y=142-170
- Full Y range for single HSV conversion: y=88-170

**Color detection modes:**

Each mode detects filled columns based on HSV thresholds (>35% of vertical pixels must match):

| Mode | Hue | Saturation | Value | Used for |
|------|-----|-----------|-------|----------|
| Warm | H < 30 or H > 150 | S > 30 | V > 80 | Standard red/pink/orange bars (P1 and P2) |
| Cool | H 90-145 | S > 40 | V > 80 | Blue/purple bars (tournament overlays, P2 only) |
| Yellow/Green | H 25-70 | S > 40 | V > 100 | Yellow/green bars (tournament overlays) |

P1 tries warm + yellow/green. P2 tries warm + cool + yellow/green. The mode producing the widest fill wins.

**Anchor validation:**

Prevents false readings from UI elements that happen to have matching pixels:
- P1: Fill must end at column >= 640 (anchored to right edge, since P1 depletes left-to-right)
- P2: Fill must start at column <= 40 (anchored to left edge, since P2 depletes right-to-left)

**Activation state machine:**

The extractor requires `_ACTIVATE_FRAMES = 10` consecutive frames where the max fill width exceeds `_ACTIVATE_THRESHOLD = 200` pixels before emitting any readings. This prevents false readings during loading screens.

**Spike rejection:**

After the median buffer warms up (at least `MEDIAN_WINDOW // 2` values), new readings are clamped:
- Upward cap: `median * 1.05 + 3` pixels (prevents 1-3 frame brightness spikes from supers/transitions)
- Downward floor: `median * 0.15` (prevents instant drops to 0 from screen flashes)
- Clamped readings are NOT added to the median buffer

**Stall detection:**

If a raw reading exceeds the median by >10% + 10 pixels, `_last_frame` is NOT advanced. This causes the gap-based reset to fire naturally during transitions, since transition screens produce artificially wide warm regions.

**Rolling median smoothing:**

A 9-frame window (`_MEDIAN_WINDOW = 9`) rolling median smooths frame-to-frame noise. Only non-clamped readings are added to the buffer.

**Monotonic constraint + max drop rate:**

Health can only decrease at most `_MAX_DROP_PER_FRAME = 3%` per frame (90%/second at 30fps). When the bar is gone (zero or clamped), a slower drop rate of 0.2%/frame is used instead.

**Death detection:**

If one player's bar is unreadable for `_DEATH_FRAMES = 10` consecutive frames while the other player is still readable, the round is considered over. `_post_round` is set to `True`, freezing `_last_frame` to trigger a gap-based reset.

Prerequisite: The now-unreadable player must have been readable at some point in the current round (`_seen_p1` / `_seen_p2` flags). This prevents false death detection when a player was never visible.

**Gap-based reset (_RESET_GAP = 15 frames):**

When the current frame number exceeds `_last_frame + 15`, all state is reset:
- Activation count and flag
- Median buffers
- Last emitted percentages (reset to 1.0)
- Max width calibration (reset to 0)
- Death detection state

This handles round/set transitions naturally.

**reset_calibration():** Explicit full reset method called between replays.

### TimerExtractor

**Source:** `replanal/extractors/timer.py`

Reads the two-digit round timer (0-99) using Otsu thresholding and template matching.

**Required ROIs:** `timer_digits`

**Algorithm:**

1. Convert ROI to grayscale
2. Check background darkness (>25% of pixels below gray level 50); reject if too bright (transition/effect)
3. Apply Otsu threshold to segment digit pixels from dark background
4. Check fill ratio: must be 15%-45% (expected range for two digits)
5. Split mask into left half (tens digit) and right half (ones digit)
6. For each half:
   - Find bounding box of white pixels
   - Crop and resize to standard 30x50 template size (`TEMPLATE_W x TEMPLATE_H`)
   - Binary threshold at 127
   - Match against all pre-extracted digit templates (0-9) stored in `data/templates/timer_digits.npz`
   - Matching metric: normalized pixel agreement `sum(resized == template) / (W * H)`
   - Confidence threshold: 0.70 (reject if best score < 0.70)
7. Return `tens * 10 + ones` as `data.timer_value`

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

Counts rounds won by each player by examining heart indicators above the health bars.

**Required ROIs:** None (reads directly from the full frame at known positions)

**Heart positions (1920x1080 coordinates):**
- P1 hearts: inner (843, 83), outer (799, 88)
- P2 hearts: inner (1077, 83), outer (1121, 88)

**Detection method:**

For each heart position, samples a 20x20 pixel region (`SAMPLE_HALF = 10`):

1. Convert frame to HSV, extract saturation (S) and value (V) channels
2. Check visibility: mean V < 40 = too dark, heart not visible (skip)
3. Classify:
   - Low saturation (mean S < 60) + bright (mean V > 100) = **white heart** (round lost)
   - Otherwise = **red heart** (active / round not lost)

**Scoring logic:**
- `rounds_won_by_P1 = number of white hearts on P2's side` (P2 lost those rounds)
- `rounds_won_by_P2 = number of white hearts on P1's side`

Returns `None` (no data) if zero hearts are visible (non-gameplay frame).

---

## HealthAggregator

**Source:** `replanal/aggregator.py`

Post-processes the sequence of `FrameData` to detect discrete damage events.

### Pipeline

For each side (P1, P2):
1. **Extract series:** Build a float array of `health_pct` values; non-gameplay frames become `NaN`
2. **Mask transitions:** Detect health jumps UP > 8% between consecutive valid readings (round reset). NaN-out a window of 5 frames before to 15 frames after the jump
3. **Smooth:** 3-frame moving median filter (configurable via `HealthBarConfig.smoothing_window`)
4. **Detect events:** Walk the smoothed series looking for health decreases

### Transition masking

Any upward jump > 8% in health between consecutive valid readings is treated as a round transition. A window from 5 frames before to 15 frames after the jump is set to NaN to prevent false damage events from the transition animation.

### Damage event detection

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SETTLE_FRAMES` | 6 | Lookahead window for continued decline (200ms at 30fps) |
| `min_delta` | 0.005 | Minimum health drop to register as an event |

Algorithm:
1. Scan for any frame where `smoothed[i+1] < smoothed[i] - min_delta`
2. Walk forward while health continues declining, allowing brief pauses up to `SETTLE_FRAMES` (handles multi-hit combo animations where the bar eases between hits)
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

Schema mirrors the `frame_data` SQLite table with typed Arrow columns (int32, float64).

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
