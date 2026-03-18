# VOD Splitter

The VOD splitter detects individual tournament sets within a long VOD recording and cuts them into separate files. It consists of a Python script (`scripts/split_vod.py`), a Rust bridge in the Tauri backend, and a React frontend workflow.

---

## Architecture: 5-Step Pipeline

```
Step 1: Gameplay Scanning (1fps via ffmpeg pipe)
  -> (second, is_gameplay) pairs
Step 2: Segment Building (merge + filter)
  -> list of Segment objects
Step 3: Set Boundary Detection (name tag comparison)
  -> list of boundary indices
Step 4: Grouping + OCR Name Reading
  -> list of DetectedSet objects with player names
Step 5: ffmpeg Stream Copy Cutting
  -> individual MP4 files
```

---

## Step 1: Gameplay Scanning

**Function:** `scan_gameplay()`

Scans the entire VOD at 1 frame per second to classify each second as gameplay or non-gameplay.

### ffmpeg Pipe

Spawns ffmpeg to decode at 1fps with forced 1920x1080 scaling:

```
ffmpeg -i <vod> -vf "fps=1,scale=1920:1080" -f rawvideo -pix_fmt bgr24 -v quiet pipe:1
```

Raw frames are read from stdout in chunks of `1920 * 1080 * 3` bytes.

### Detection Signals

**Tension bar detection:**
- P1 tension ROI: y=1040-1058, x=50-440
- P2 tension ROI: y=1040-1058, x=1480-1870
- Convert ROI to HSV
- Score: percentage of pixels with `S > 40` AND `V > 30`
- Either player scoring > 15% = tension present

**Timer brightness:**
- Timer ROI: y=30-100, x=910-1010
- Convert to HSV, check V channel
- Score: percentage of pixels with `V > 180`
- Timer bright > 2% = timer present

**Gameplay decision:**
```
is_gameplay = (t1_score > 0.15 or t2_score > 0.15) and timer_bright > 0.02
```

### Progress Reporting

Machine-readable progress lines are printed to stdout:
```
PROGRESS:123/4567
```

The Rust bridge captures these and emits them as Tauri events.

---

## Step 2: Segment Building

**Function:** `build_segments()`

Converts per-second gameplay flags into merged, filtered segments.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `merge_gap` | 12 seconds | Merge gameplay segments with < N second gaps |
| `min_duration` | 25 seconds | Discard segments shorter than this |

### Algorithm

1. Walk through `(second, is_gameplay)` pairs, building raw segments from contiguous `True` runs
2. Merge segments where the gap between them is < `merge_gap` seconds
3. Filter out segments shorter than `min_duration` seconds

---

## Step 3: Set Boundary Detection

**Function:** `detect_set_boundaries()`

Compares player name tag fingerprints between consecutive gameplay segments to identify where one set ends and another begins.

### Name Tag Fingerprinting

**Quad ROIs for GGS name tags:**

GGS name tags are parallelogram-shaped (skewed), not rectangular. They are defined as 4-point quads:

| Player | Quad Corners (tl, tr, br, bl) |
|--------|-------------------------------|
| P1 | (45,155), (340,150), (370,178), (75,178) |
| P2 | (1555,150), (1870,155), (1850,178), (1580,178) |

**Perspective warp:**

`warp_quad()` uses `cv2.getPerspectiveTransform()` to warp the parallelogram into a 300x40 rectangle, correcting for the skew. Source points are the 4 quad corners; destination points are the rectangle corners.

**Binary signature:**

`get_name_signature()`:
1. Warp both P1 and P2 name regions to rectangles
2. Convert to grayscale
3. Binary threshold at 160 (separates bright text from dark background)
4. Resize to 80x4 pixels (compact signature)
5. Concatenate P1 + P2 signatures into a single float32 vector (640 values)
6. Reject if mean intensity < 5.0 (too dark, no text visible)

**Distance metric:**

`name_distance()`: Mean absolute difference between two signatures.

### Multi-Frame Sampling

For each segment, 3 frames are sampled at 20%, 50%, and 80% through the segment duration. The signature with the highest mean intensity (most visible text) is selected.

### Boundary Detection Logic

A new set boundary is detected between consecutive segments when:
- **Gap threshold:** The gap between segments exceeds 45 seconds (`SET_GAP_THRESHOLD`), OR
- **Name distance:** The name tag signature distance exceeds 12 (`NAME_DIST_THRESHOLD`)

The first segment always starts a new set (boundary index 0).

---

## Step 4: Grouping + OCR Name Reading

### Grouping

**Function:** `group_sets()`

Groups consecutive segments between boundary indices into `DetectedSet` objects, each with an index number.

### EasyOCR Name Reading

**Lazy-loaded GPU-accelerated EasyOCR reader:**

```python
_ocr_reader = None
def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _ocr_reader
```

**Reading process (`ocr_name_region`):**
1. Run EasyOCR on the warped name region with `detail=1` (returns bounding box, text, confidence)
2. Pick the result with the **largest bounding box area** (most prominent text in the ROI)
3. Sanitize to ASCII: keep only alphanumeric characters, spaces, underscores, and hyphens

**Multi-frame sampling for names:**
- For each set, sample 3 frames from the first segment at 20%, 40%, 60%
- Pick the reading with the longest combined P1+P2 name length (best OCR result)

---

## Step 5: ffmpeg Stream Copy Cutting

**Function:** `cut_sets()`

### Padding

| Parameter | Default | Description |
|-----------|---------|-------------|
| `padding_before` | 10 seconds | Extra time before set start |
| `padding_after` | 5 seconds | Extra time after set end |

Cut start is clamped to >= 0; cut end is clamped to <= total video duration.

### ffmpeg Command

```
ffmpeg -ss <start> -i <vod> -t <duration> -c copy -avoid_negative_ts make_zero -y -v warning <output>
```

Stream copy (no re-encode) for fast cutting.

### Output Filenames

- With player names: `{index:02d}_{safe_p1}_vs_{safe_p2}.mp4`
- Without player names: `set_{index:02d}.mp4`

Player names are sanitized for filenames (alphanumeric + underscore + hyphen only).

---

## Rust Bridge

### scan_vod Command

**Signature:** `async fn scan_vod(app_handle, video_path, roi_config) -> Result<Vec<DetectedSetInfo>, String>`

1. Spawns Python with the `split_vod.py` script and `--preview --json-output` flags
2. Passes ROI configuration as CLI arguments:
   - `--p1-tension y1,y2,x1,x2`
   - `--p2-tension y1,y2,x1,x2`
   - `--timer-roi y1,y2,x1,x2`
   - `--p1-name-quad tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y`
   - `--p2-name-quad tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y`
3. Reads stdout line-by-line:
   - `PROGRESS:N/M` lines are emitted as `vod-scan-progress` Tauri events
   - `JSON_RESULT:{...}` line is parsed as the final result
4. Reads stderr in a separate thread (avoids deadlock)
5. Parses JSON into `Vec<DetectedSetInfo>`

### cut_vod_sets Command

**Signature:** `async fn cut_vod_sets(app_handle, video_path, sets, output_dir) -> Result<Vec<String>, String>`

Performs the ffmpeg cutting directly in Rust (does not call Python):
1. Creates the output directory
2. For each `CutSetRequest`, runs ffmpeg stream copy with 10s/5s padding
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
| `results` | Table of detected sets with checkboxes; auto-deselects sets with no player names |
| `cutting` | Progress display during ffmpeg cutting |
| `done` | Success confirmation with output directory |
| `error` | Error display with full Python traceback and Copy Error button |

### Preview Frame Extraction

On entering the `roi` step, `extractPreviewFrame(videoPath, 30)` extracts a frame at 30 seconds (likely gameplay) via the Tauri `extract_preview_frame` command (ffmpeg single-frame extraction).

### Results Display

Detected sets are shown in a table with:
- Set number, player names, time range, gameplay duration, game count
- Checkboxes for selecting which sets to cut
- Sets with empty player names are auto-deselected (likely false detections)

---

## Frontend: RoiPicker

**Source:** `app/src/components/RoiPicker.tsx`

An interactive ROI editor overlaid on the preview frame.

### ROI Types

| Key | Type | Color | Description |
|-----|------|-------|-------------|
| `p1_tension` | RoiRect | Blue | P1 tension bar region |
| `p2_tension` | RoiRect | Red | P2 tension bar region |
| `timer` | RoiRect | Yellow | Timer region |
| `p1_name` | RoiQuad | Blue (lighter) | P1 name tag parallelogram |
| `p2_name` | RoiQuad | Red (lighter) | P2 name tag parallelogram |

### Default ROIs (1920x1080 GGS)

```typescript
p1_tension: { y1: 1040, y2: 1058, x1: 50, x2: 440 },
p2_tension: { y1: 1040, y2: 1058, x1: 1480, x2: 1870 },
timer: { y1: 30, y2: 100, x1: 910, x2: 1010 },
p1_name: { tl: [75, 150], tr: [340, 145], br: [370, 178], bl: [45, 178] },
p2_name: { tl: [1555, 145], tr: [1850, 150], br: [1880, 178], bl: [1580, 178] },
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
    start: float  # seconds
    end: float    # seconds

@dataclass
class DetectedSet:
    segments: list[Segment]
    index: int
    p1_name: str
    p2_name: str
```

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

- Python exceptions produce full tracebacks on stderr
- The Rust bridge captures stderr in a separate thread and includes the last error line in the error message
- The frontend `error` step displays the full error text with a "Copy Error" button for debugging
- Windows-specific: stdout/stderr are reconfigured to UTF-8 at script startup to avoid charmap encoding errors from EasyOCR
