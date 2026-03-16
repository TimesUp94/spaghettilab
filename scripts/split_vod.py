"""Split a GGS tournament VOD into individual sets.

Detects gameplay segments via tension bar + timer presence, then
identifies set boundaries by comparing the tournament overlay banner
between segments. Cuts are done with ffmpeg stream copy (no re-encode).

Usage:
    python scripts/split_vod.py <vod_path> [--output <dir>] [--preview]

Examples:
    python scripts/split_vod.py "my_tournament.mp4"
    python scripts/split_vod.py "my_tournament.mp4" --preview   # detect only, don't cut
    python scripts/split_vod.py "my_tournament.mp4" --output data/my_sets
    python scripts/split_vod.py "my_tournament.mp4" --banner-threshold 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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
BANNER_ROI = (2, 20, 100, 1820)  # tournament overlay at the top

GAMEPLAY_MERGE_GAP = 12    # merge gameplay segments with < N second gaps
MIN_SEGMENT_DURATION = 25  # discard segments shorter than this
SET_GAP_THRESHOLD = 45     # gaps > N seconds are always set boundaries
BANNER_DIST_THRESHOLD = 15 # banner distance above this = different players
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
    """Extract a single frame at a given timestamp using ffmpeg."""
    cmd = [
        "ffmpeg", "-ss", str(timestamp), "-i", vod_path,
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
        "-vf", "fps=1",
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

        # Tension bars: colored (high saturation) + visible (some brightness)
        t1 = frame[p1_tension[0]:p1_tension[1], p1_tension[2]:p1_tension[3]]
        t2 = frame[p2_tension[0]:p2_tension[1], p2_tension[2]:p2_tension[3]]
        hsv1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(t2, cv2.COLOR_BGR2HSV)
        t1_score = float(np.mean((hsv1[:, :, 1] > 40) & (hsv1[:, :, 2] > 30)))
        t2_score = float(np.mean((hsv2[:, :, 1] > 40) & (hsv2[:, :, 2] > 30)))

        # Timer region: bright white digits present during gameplay
        tr = frame[timer_roi[0]:timer_roi[1], timer_roi[2]:timer_roi[3]]
        timer_bright = float(np.mean(cv2.cvtColor(tr, cv2.COLOR_BGR2HSV)[:, :, 2] > 180))

        is_gameplay = bool((t1_score > 0.15 or t2_score > 0.15) and timer_bright > 0.02)
        results.append((sec, is_gameplay))
        sec += 1

        # Machine-readable progress every 10 seconds
        if sec % 10 == 0:
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
# Step 3: Detect set boundaries via banner comparison
# ---------------------------------------------------------------------------

def get_banner_signature(frame: np.ndarray, banner_roi: tuple = BANNER_ROI) -> np.ndarray:
    """Extract a compact grayscale signature of the tournament banner."""
    banner = frame[banner_roi[0]:banner_roi[1], banner_roi[2]:banner_roi[3]]
    gray = cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (172, 2), interpolation=cv2.INTER_AREA)
    return small.flatten().astype(np.float32)


def banner_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    return float(np.mean(np.abs(sig1 - sig2)))


def detect_set_boundaries(
    segments: list[Segment],
    vod_path: str,
    w: int,
    h: int,
    gap_threshold: float = SET_GAP_THRESHOLD,
    dist_threshold: float = BANNER_DIST_THRESHOLD,
    banner_roi: tuple = BANNER_ROI,
) -> list[int]:
    """Return indices into *segments* where a new set begins."""
    if not segments:
        return []

    print("Detecting set boundaries via banner comparison...", flush=True)

    sigs: list[np.ndarray | None] = []
    for i, seg in enumerate(segments):
        if i == 0:
            t = seg.start + min(8, seg.duration / 2)
        else:
            gap = seg.start - segments[i - 1].end
            if gap > 25:
                t = segments[i - 1].end + min(5, gap / 2)
            else:
                t = seg.start + min(8, seg.duration / 2)

        frame = extract_frame_ffmpeg(vod_path, t, w, h)
        sigs.append(get_banner_signature(frame, banner_roi) if frame is not None else None)

    boundaries = [0]
    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end

        if gap > gap_threshold:
            if i not in boundaries:
                boundaries.append(i)
            continue

        if sigs[i] is not None and sigs[i - 1] is not None:
            dist = banner_distance(sigs[i], sigs[i - 1])
            if dist > dist_threshold:
                if i not in boundaries:
                    boundaries.append(i)
                label = "gap" if gap > 25 else "gameplay"
                print(f"  Set break before seg {i} (dist={dist:.1f}, sampled from {label})", flush=True)

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
    parser.add_argument("--banner-threshold", type=float, default=BANNER_DIST_THRESHOLD)
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
                        help="Banner ROI as y1,y2,x1,x2")
    args = parser.parse_args()

    # Apply ROI overrides
    p1_tension = parse_roi(args.p1_tension) if args.p1_tension else P1_TENSION
    p2_tension = parse_roi(args.p2_tension) if args.p2_tension else P2_TENSION
    timer_roi = parse_roi(args.timer_roi) if args.timer_roi else TIMER_ROI
    banner_roi = parse_roi(args.banner_roi) if args.banner_roi else BANNER_ROI

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
    print(f"  Resolution: {w}x{h}, Duration: {fmt(total_duration)} ({total_duration / 60:.0f} min)", flush=True)
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

    # Step 3: Detect set boundaries
    boundaries = detect_set_boundaries(
        segments, vod_path, w, h,
        gap_threshold=args.set_gap,
        dist_threshold=args.banner_threshold,
        banner_roi=banner_roi,
    )
    print(flush=True)

    # Step 4: Group into sets
    sets = group_sets(segments, boundaries)

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
    main()
