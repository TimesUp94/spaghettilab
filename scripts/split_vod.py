"""Split a GGS tournament VOD into individual sets.

Detects gameplay segments via tension bar + timer presence, then
identifies set boundaries by comparing player name tags under the
health bars. Cuts are done with ffmpeg stream copy (no re-encode).

Usage:
    python scripts/split_vod.py <vod_path> [--output <dir>] [--preview]

Examples:
    python scripts/split_vod.py "my_tournament.mp4"
    python scripts/split_vod.py "my_tournament.mp4" --preview   # detect only, don't cut
    python scripts/split_vod.py "my_tournament.mp4" --output data/my_sets
    python scripts/split_vod.py "my_tournament.mp4" --name-threshold 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Force UTF-8 output on Windows to avoid charmap encoding errors from EasyOCR
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config defaults — tuned for 1080p GGS with the "SHOW DOWN" style overlay.
# Override via CLI flags for different overlays.
# ---------------------------------------------------------------------------

# ROI coordinates (y1, y2, x1, x2) for 1920x1080
P1_TENSION = (1040, 1058, 50, 440)
P2_TENSION = (1040, 1058, 1480, 1870)
TIMER_ROI = (30, 100, 910, 1010)
# Player name tags: 4-point quads (tl_x,tl_y, tr_x,tr_y, br_x,br_y, bl_x,bl_y)
# For rectangular name tags (e.g. SF6), all corners form a rectangle.
# For skewed tags (e.g. GGS), corners trace the parallelogram.
P1_NAME_QUAD = (45, 155, 340, 150, 370, 178, 75, 178)  # GGS left name tag
P2_NAME_QUAD = (1555, 150, 1870, 155, 1850, 178, 1580, 178)  # GGS right name tag

GAMEPLAY_MERGE_GAP = 12    # merge gameplay segments with < N second gaps
MIN_SEGMENT_DURATION = 25  # discard segments shorter than this
SET_GAP_THRESHOLD = 45     # gaps > N seconds are always set boundaries
NAME_DIST_THRESHOLD = 12   # name tag distance above this = different players
PADDING_BEFORE = 10
PADDING_AFTER = 5


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DetectedSet:
    segments: list[Segment] = field(default_factory=list)
    index: int = 0
    p1_name: str = ""
    p2_name: str = ""

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def gameplay_duration(self) -> float:
        return sum(s.duration for s in self.segments)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def get_video_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frame_ffmpeg(vod_path: str, timestamp: float, w: int, h: int) -> np.ndarray | None:
    """Extract a single frame at a given timestamp using ffmpeg, scaled to w x h."""
    cmd = [
        "ffmpeg", "-ss", str(timestamp), "-i", vod_path,
        "-vf", f"scale={w}:{h}",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-v", "quiet", "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True).stdout
    if len(raw) < w * h * 3:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))


def parse_roi(s: str) -> tuple[int, int, int, int]:
    """Parse 'y1,y2,x1,x2' string into tuple."""
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must have 4 values (y1,y2,x1,x2), got: {s}")
    return tuple(parts)  # type: ignore


def parse_quad(s: str) -> tuple[int, ...]:
    """Parse 'tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y' string into 8-tuple."""
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 8:
        raise ValueError(f"Quad must have 8 values (tl_x,tl_y,...,bl_x,bl_y), got: {s}")
    return tuple(parts)


def warp_quad(frame: np.ndarray, quad: tuple[int, ...], out_w: int = 300, out_h: int = 40) -> np.ndarray:
    """Perspective-warp a quad region of the frame into a rectangle.

    quad = (tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y)
    """
    src_pts = np.array([
        [quad[0], quad[1]],  # top-left
        [quad[2], quad[3]],  # top-right
        [quad[4], quad[5]],  # bottom-right
        [quad[6], quad[7]],  # bottom-left
    ], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [out_w, 0],
        [out_w, out_h],
        [0, out_h],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, M, (out_w, out_h))


# ---------------------------------------------------------------------------
# Step 1: Scan for gameplay frames
# ---------------------------------------------------------------------------

def scan_gameplay(
    vod_path: str,
    w: int = 1920,
    h: int = 1080,
    p1_tension: tuple = P1_TENSION,
    p2_tension: tuple = P2_TENSION,
    timer_roi: tuple = TIMER_ROI,
    total_duration: float = 0,
) -> list[tuple[int, bool]]:
    """Scan the VOD at 1fps, return (second, is_gameplay) pairs."""
    cmd = [
        "ffmpeg", "-i", vod_path,
        "-vf", f"fps=1,scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-v", "quiet", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    frame_size = w * h * 3
    results = []
    sec = 0
    t0 = time.time()
    total_secs = int(total_duration) if total_duration > 0 else 0

    print("Scanning for gameplay segments (1 fps)...", flush=True)
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))

        # Verify frame shape
        if frame.shape != (h, w, 3):
            print(f"WARNING: frame shape {frame.shape} != expected ({h},{w},3) at sec {sec}", flush=True, file=sys.stderr)
            results.append((sec, False))
            sec += 1
            continue

        # Tension bars: colored (high saturation) + visible (some brightness)
        t1 = frame[p1_tension[0]:p1_tension[1], p1_tension[2]:p1_tension[3]]
        t2 = frame[p2_tension[0]:p2_tension[1], p2_tension[2]:p2_tension[3]]
        if t1.size == 0 or t2.size == 0:
            results.append((sec, False))
            sec += 1
            continue
        hsv1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(t2, cv2.COLOR_BGR2HSV)
        t1_score = float(np.mean((hsv1[:, :, 1] > 40) & (hsv1[:, :, 2] > 30)))
        t2_score = float(np.mean((hsv2[:, :, 1] > 40) & (hsv2[:, :, 2] > 30)))

        # Timer region: bright white digits present during gameplay
        tr = frame[timer_roi[0]:timer_roi[1], timer_roi[2]:timer_roi[3]]
        if tr.size == 0:
            results.append((sec, False))
            sec += 1
            continue
        timer_bright = float(np.mean(cv2.cvtColor(tr, cv2.COLOR_BGR2HSV)[:, :, 2] > 180))

        is_gameplay = bool((t1_score > 0.15 or t2_score > 0.15) and timer_bright > 0.02)
        results.append((sec, is_gameplay))
        sec += 1

        # Machine-readable progress: first frame then every 10 seconds
        if sec == 1 or sec % 10 == 0:
            total_str = f"/{total_secs}" if total_secs else ""
            print(f"PROGRESS:{sec}{total_str}", flush=True)

    proc.wait()
    print(f"PROGRESS:{sec}/{sec}", flush=True)
    print(f"  Done: {len(results)} seconds scanned in {time.time() - t0:.0f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Step 2: Build & merge gameplay segments
# ---------------------------------------------------------------------------

def build_segments(
    scan: list[tuple[int, bool]],
    merge_gap: int = GAMEPLAY_MERGE_GAP,
    min_duration: int = MIN_SEGMENT_DURATION,
) -> list[Segment]:
    """Convert per-second gameplay flags into merged, filtered segments."""
    raw: list[Segment] = []
    in_gp = False
    start = 0
    for sec, is_gp in scan:
        if is_gp and not in_gp:
            start = sec
            in_gp = True
        elif not is_gp and in_gp:
            raw.append(Segment(start, sec))
            in_gp = False
    if in_gp:
        raw.append(Segment(start, scan[-1][0]))

    if not raw:
        return []

    # Merge segments with small gaps
    merged = [Segment(raw[0].start, raw[0].end)]
    for seg in raw[1:]:
        if seg.start - merged[-1].end < merge_gap:
            merged[-1] = Segment(merged[-1].start, seg.end)
        else:
            merged.append(seg)

    # Filter short segments
    return [s for s in merged if s.duration >= min_duration]


# ---------------------------------------------------------------------------
# OCR via EasyOCR — lazy-loaded, GPU-accelerated
# ---------------------------------------------------------------------------

_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _ocr_reader


def ocr_name_region(region: np.ndarray) -> str:
    """OCR a name tag region. Returns the recognized text or empty string.

    Picks the result with the largest bounding box area, which is typically
    the player name (the most prominent text in the ROI).
    """
    if region.size == 0:
        return ""
    reader = _get_ocr_reader()
    # detail=1 returns (bbox, text, confidence) tuples
    results = reader.readtext(region, detail=1)
    if not results:
        return ""
    # Pick the result with the largest bounding box (most prominent text)
    best = max(results, key=lambda r: (
        (max(p[0] for p in r[0]) - min(p[0] for p in r[0])) *
        (max(p[1] for p in r[0]) - min(p[1] for p in r[0]))
    ))
    # Sanitize to ASCII-safe characters for filenames and JSON
    text = best[1].strip()
    return "".join(c for c in text if c.isascii() and (c.isalnum() or c in " _-"))


def read_player_names(
    frame: np.ndarray,
    p1_name_quad: tuple = P1_NAME_QUAD,
    p2_name_quad: tuple = P2_NAME_QUAD,
) -> tuple[str, str]:
    """Read P1 and P2 player names from a frame. Returns (p1_name, p2_name)."""
    p1 = warp_quad(frame, p1_name_quad)
    p2 = warp_quad(frame, p2_name_quad)
    return ocr_name_region(p1), ocr_name_region(p2)


# ---------------------------------------------------------------------------
# Step 3: Detect set boundaries via player name comparison
# ---------------------------------------------------------------------------

def get_name_signature(
    frame: np.ndarray,
    p1_name_quad: tuple = P1_NAME_QUAD,
    p2_name_quad: tuple = P2_NAME_QUAD,
) -> np.ndarray | None:
    """Extract a compact signature from both player name tags."""
    p1 = warp_quad(frame, p1_name_quad)
    p2 = warp_quad(frame, p2_name_quad)

    if p1.size == 0 or p2.size == 0:
        return None

    g1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
    _, b1 = cv2.threshold(g1, 160, 255, cv2.THRESH_BINARY)
    _, b2 = cv2.threshold(g2, 160, 255, cv2.THRESH_BINARY)

    s1 = cv2.resize(b1, (80, 4), interpolation=cv2.INTER_AREA)
    s2 = cv2.resize(b2, (80, 4), interpolation=cv2.INTER_AREA)

    sig = np.concatenate([s1.flatten(), s2.flatten()]).astype(np.float32)
    if np.mean(sig) < 5.0:
        return None
    return sig


def name_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    return float(np.mean(np.abs(sig1 - sig2)))


def detect_set_boundaries(
    segments: list[Segment],
    vod_path: str,
    w: int,
    h: int,
    gap_threshold: float = SET_GAP_THRESHOLD,
    dist_threshold: float = NAME_DIST_THRESHOLD,
    p1_name_quad: tuple = P1_NAME_QUAD,
    p2_name_quad: tuple = P2_NAME_QUAD,
) -> list[int]:
    """Return indices into *segments* where a new set begins.

    Compares player name tag fingerprints between consecutive gameplay
    segments. A boundary is detected when:
      - The gap between segments exceeds the gap threshold, OR
      - The name tag signature distance exceeds the distance threshold
    """
    if not segments:
        return []

    print("Detecting set boundaries via player name comparison...", flush=True)

    # For each segment, sample a frame from mid-gameplay and extract name sigs.
    # Sample multiple frames and pick the best (most text visible) to handle
    # transient occlusion during supers/burst.
    sigs: list[np.ndarray | None] = []
    for i, seg in enumerate(segments):
        # Sample 3 frames across the segment to find one with clear names
        candidates = []
        for frac in (0.2, 0.5, 0.8):
            t = seg.start + seg.duration * frac
            frame = extract_frame_ffmpeg(vod_path, t, w, h)
            if frame is not None:
                sig = get_name_signature(frame, p1_name_quad, p2_name_quad)
                if sig is not None:
                    # Score by amount of visible text (higher = more text)
                    candidates.append((sig, float(np.mean(sig))))

        if candidates:
            # Pick the signature with the most visible text
            best = max(candidates, key=lambda x: x[1])
            sigs.append(best[0])
        else:
            sigs.append(None)

    boundaries = [0]
    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end

        if gap > gap_threshold:
            if i not in boundaries:
                boundaries.append(i)
                print(f"  Set break before seg {i} (gap={gap:.0f}s > {gap_threshold}s)", flush=True)
            continue

        if sigs[i] is not None and sigs[i - 1] is not None:
            dist = name_distance(sigs[i], sigs[i - 1])
            if dist > dist_threshold:
                if i not in boundaries:
                    boundaries.append(i)
                print(f"  Set break before seg {i} (name dist={dist:.1f} > {dist_threshold})", flush=True)

    boundaries.sort()
    return boundaries


# ---------------------------------------------------------------------------
# Step 4: Group segments into sets
# ---------------------------------------------------------------------------

def group_sets(segments: list[Segment], boundaries: list[int]) -> list[DetectedSet]:
    sets: list[DetectedSet] = []
    for k in range(len(boundaries)):
        si = boundaries[k]
        ei = boundaries[k + 1] if k + 1 < len(boundaries) else len(segments)
        ds = DetectedSet(segments=segments[si:ei], index=k + 1)
        sets.append(ds)
    return sets


# ---------------------------------------------------------------------------
# Step 5: Cut video
# ---------------------------------------------------------------------------

def cut_sets(
    sets: list[DetectedSet],
    vod_path: str,
    output_dir: Path,
    pad_before: int = PADDING_BEFORE,
    pad_after: int = PADDING_AFTER,
    total_duration: float = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in sets:
        cut_start = max(0, ds.start - pad_before)
        cut_end = ds.end + pad_after
        if total_duration > 0:
            cut_end = min(cut_end, total_duration)
        duration = cut_end - cut_start

        if ds.p1_name and ds.p2_name:
            # Sanitize names for filenames
            safe_p1 = "".join(c for c in ds.p1_name if c.isalnum() or c in "_-")
            safe_p2 = "".join(c for c in ds.p2_name if c.isalnum() or c in "_-")
            label = f"{ds.index:02d}_{safe_p1}_vs_{safe_p2}"
        else:
            label = f"set_{ds.index:02d}"
        out_path = output_dir / f"{label}.mp4"

        print(f"  {label}: {fmt(cut_start)} -> {fmt(cut_end)} ({duration / 60:.1f} min)", flush=True)

        cmd = [
            "ffmpeg",
            "-ss", str(cut_start),
            "-i", vod_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y", "-v", "warning",
            str(out_path),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"    FAILED (exit {result.returncode})", flush=True)
        else:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"    -> {out_path.name} ({size_mb:.0f} MB)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split a GGS tournament VOD into individual sets."
    )
    parser.add_argument("vod", help="Path to the VOD MP4 file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <vod_dir>/sets/)")
    parser.add_argument("--preview", action="store_true",
                        help="Detect sets but don't cut the video")
    parser.add_argument("--json-output", action="store_true",
                        help="Output detected sets as JSON (for GUI integration)")
    parser.add_argument("--merge-gap", type=int, default=GAMEPLAY_MERGE_GAP)
    parser.add_argument("--min-duration", type=int, default=MIN_SEGMENT_DURATION)
    parser.add_argument("--set-gap", type=float, default=SET_GAP_THRESHOLD)
    parser.add_argument("--name-threshold", type=float, default=NAME_DIST_THRESHOLD)
    parser.add_argument("--banner-threshold", type=float, default=NAME_DIST_THRESHOLD,
                        help="(deprecated, use --name-threshold)")
    parser.add_argument("--padding-before", type=int, default=PADDING_BEFORE)
    parser.add_argument("--padding-after", type=int, default=PADDING_AFTER)
    # ROI overrides
    parser.add_argument("--p1-tension", type=str, default=None,
                        help="P1 tension bar ROI as y1,y2,x1,x2")
    parser.add_argument("--p2-tension", type=str, default=None,
                        help="P2 tension bar ROI as y1,y2,x1,x2")
    parser.add_argument("--timer-roi", type=str, default=None,
                        help="Timer ROI as y1,y2,x1,x2")
    parser.add_argument("--banner-roi", type=str, default=None,
                        help="(deprecated, use --p1-name / --p2-name)")
    parser.add_argument("--p1-name", type=str, default=None,
                        help="(deprecated, use --p1-name-quad)")
    parser.add_argument("--p2-name", type=str, default=None,
                        help="(deprecated, use --p2-name-quad)")
    parser.add_argument("--p1-name-quad", type=str, default=None,
                        help="P1 name tag quad as tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y")
    parser.add_argument("--p2-name-quad", type=str, default=None,
                        help="P2 name tag quad as tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y")
    args = parser.parse_args()

    # Apply ROI overrides
    p1_tension = parse_roi(args.p1_tension) if args.p1_tension else P1_TENSION
    p2_tension = parse_roi(args.p2_tension) if args.p2_tension else P2_TENSION
    timer_roi = parse_roi(args.timer_roi) if args.timer_roi else TIMER_ROI
    p1_name_quad = parse_quad(args.p1_name_quad) if args.p1_name_quad else P1_NAME_QUAD
    p2_name_quad = parse_quad(args.p2_name_quad) if args.p2_name_quad else P2_NAME_QUAD
    # Use --name-threshold if specified, fall back to --banner-threshold for compat
    name_threshold = args.name_threshold if args.name_threshold != NAME_DIST_THRESHOLD else args.banner_threshold

    vod_path = str(Path(args.vod).resolve())
    if not Path(vod_path).exists():
        print(f"VOD not found: {vod_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path(vod_path).parent / "sets"

    # Get video info
    total_duration = get_video_duration(vod_path)
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "stream=width,height",
        "-of", "default=noprint_wrappers=1", vod_path,
    ]
    probe = subprocess.run(cmd, capture_output=True, text=True)
    w = h = 0
    for line in probe.stdout.strip().splitlines():
        if line.startswith("width="):
            w = int(line.split("=")[1])
        elif line.startswith("height="):
            h = int(line.split("=")[1])
    if w == 0 or h == 0:
        print("Could not determine video resolution.")
        sys.exit(1)

    print(f"VOD: {Path(vod_path).name}", flush=True)
    print(f"  Native resolution: {w}x{h}, Duration: {fmt(total_duration)} ({total_duration / 60:.0f} min)", flush=True)

    # Force 1920x1080 for processing — ROIs are defined at this resolution
    if w != 1920 or h != 1080:
        print(f"  Scaling to 1920x1080 for processing", flush=True)
    w, h = 1920, 1080
    print(flush=True)

    # Step 1: Scan
    scan = scan_gameplay(vod_path, w, h, p1_tension, p2_tension, timer_roi, total_duration)
    print(flush=True)

    # Step 2: Build segments
    segments = build_segments(scan, args.merge_gap, args.min_duration)
    print(f"Found {len(segments)} gameplay segments", flush=True)
    for i, s in enumerate(segments):
        gap = s.start - segments[i - 1].end if i > 0 else 0
        print(f"  {i:2d}: {fmt(s.start)} - {fmt(s.end)}  ({s.duration / 60:.1f} min)  gap={gap:.0f}s", flush=True)
    print(flush=True)

    # Step 3: Detect set boundaries via player name tags
    boundaries = detect_set_boundaries(
        segments, vod_path, w, h,
        gap_threshold=args.set_gap,
        dist_threshold=name_threshold,
        p1_name_quad=p1_name_quad,
        p2_name_quad=p2_name_quad,
    )
    print(flush=True)

    # Step 4: Group into sets
    sets = group_sets(segments, boundaries)

    # Step 4b: OCR player names for each set
    print("Reading player names...", flush=True)
    for ds in sets:
        # Sample multiple frames from the first segment to get a clear read
        best_p1, best_p2 = "", ""
        best_len = 0
        seg = ds.segments[0]
        for frac in (0.2, 0.4, 0.6):
            t = seg.start + seg.duration * frac
            frame = extract_frame_ffmpeg(vod_path, t, w, h)
            if frame is not None:
                p1, p2 = read_player_names(frame, p1_name_quad, p2_name_quad)
                total_len = len(p1) + len(p2)
                if total_len > best_len:
                    best_p1, best_p2, best_len = p1, p2, total_len
        ds.p1_name = best_p1
        ds.p2_name = best_p2
        label = f"{ds.p1_name} vs {ds.p2_name}" if ds.p1_name and ds.p2_name else f"Set {ds.index}"
        print(f"  Set {ds.index}: {label}", flush=True)
    print(flush=True)

    if args.json_output:
        # Machine-readable output for GUI
        result = {
            "sets": [
                {
                    "index": ds.index,
                    "start_secs": ds.start,
                    "end_secs": ds.end,
                    "gameplay_duration_secs": ds.gameplay_duration,
                    "game_count": len(ds.segments),
                    "p1_name": ds.p1_name,
                    "p2_name": ds.p2_name,
                }
                for ds in sets
            ]
        }
        print(f"JSON_RESULT:{json.dumps(result)}", flush=True)
    else:
        print(f"Detected {len(sets)} sets:", flush=True)
        for ds in sets:
            n = len(ds.segments)
            print(f"  Set {ds.index:2d}: {fmt(ds.start)} - {fmt(ds.end)}  "
                  f"({ds.gameplay_duration / 60:.1f} min gameplay, {n} game(s))", flush=True)
        print(flush=True)

    # Step 5: Cut
    if args.preview or args.json_output:
        if not args.json_output:
            print("Preview mode — skipping video cuts.")
    else:
        print("Cutting video...", flush=True)
        cut_sets(sets, vod_path, output_dir,
                 args.padding_before, args.padding_after, total_duration)
        print(f"\nDone! {len(sets)} sets saved to {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
