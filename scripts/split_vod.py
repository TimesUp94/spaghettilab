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


# ---------------------------------------------------------------------------
# Step 1: Scan for gameplay frames
# ---------------------------------------------------------------------------

def scan_gameplay(vod_path: str, w: int = 1920, h: int = 1080) -> list[tuple[int, bool]]:
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

    print("Scanning for gameplay segments (1 fps)...")
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))

        # Tension bars: colored (high saturation) + visible (some brightness)
        t1 = frame[P1_TENSION[0]:P1_TENSION[1], P1_TENSION[2]:P1_TENSION[3]]
        t2 = frame[P2_TENSION[0]:P2_TENSION[1], P2_TENSION[2]:P2_TENSION[3]]
        hsv1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(t2, cv2.COLOR_BGR2HSV)
        t1_score = float(np.mean((hsv1[:, :, 1] > 40) & (hsv1[:, :, 2] > 30)))
        t2_score = float(np.mean((hsv2[:, :, 1] > 40) & (hsv2[:, :, 2] > 30)))

        # Timer region: bright white digits present during gameplay
        tr = frame[TIMER_ROI[0]:TIMER_ROI[1], TIMER_ROI[2]:TIMER_ROI[3]]
        timer_bright = float(np.mean(cv2.cvtColor(tr, cv2.COLOR_BGR2HSV)[:, :, 2] > 180))

        is_gameplay = bool((t1_score > 0.15 or t2_score > 0.15) and timer_bright > 0.02)
        results.append((sec, is_gameplay))
        sec += 1

        if sec % 600 == 0:
            elapsed = time.time() - t0
            print(f"  {sec // 60}min scanned ({elapsed:.0f}s elapsed)")

    proc.wait()
    print(f"  Done: {len(results)} seconds scanned in {time.time() - t0:.0f}s")
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

def get_banner_signature(frame: np.ndarray) -> np.ndarray:
    """Extract a compact grayscale signature of the tournament banner."""
    banner = frame[BANNER_ROI[0]:BANNER_ROI[1], BANNER_ROI[2]:BANNER_ROI[3]]
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
) -> list[int]:
    """Return indices into *segments* where a new set begins.

    Uses two signals:
    1. Large time gaps between segments (> gap_threshold) → always a set break.
    2. Banner signature change → players changed.

    For long gaps (> 30s), the banner is sampled from the gap itself (lobby
    screen, clean banner).  For short gaps, it's sampled from 8-10s into the
    next gameplay segment.
    """
    if not segments:
        return []

    print("Detecting set boundaries via banner comparison...")

    # Collect one banner signature per segment.
    # Prefer sampling from the gap before the segment (lobby = clean banner).
    # Fall back to 8s into gameplay for short gaps.
    sigs: list[np.ndarray | None] = []
    for i, seg in enumerate(segments):
        if i == 0:
            # First segment: sample from gameplay
            t = seg.start + min(8, seg.duration / 2)
        else:
            gap = seg.start - segments[i - 1].end
            if gap > 25:
                # Long gap: sample from the lobby/loading screen
                t = segments[i - 1].end + min(5, gap / 2)
            else:
                # Short gap: sample from gameplay (after SLASH! animation)
                t = seg.start + min(8, seg.duration / 2)

        frame = extract_frame_ffmpeg(vod_path, t, w, h)
        sigs.append(get_banner_signature(frame) if frame is not None else None)

    # Compare consecutive signatures + gap durations
    boundaries = [0]
    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end

        # Large gap → always a set boundary
        if gap > gap_threshold:
            if i not in boundaries:
                boundaries.append(i)
            continue

        # Compare banners
        if sigs[i] is not None and sigs[i - 1] is not None:
            dist = banner_distance(sigs[i], sigs[i - 1])
            if dist > dist_threshold:
                if i not in boundaries:
                    boundaries.append(i)
                label = "gap" if gap > 25 else "gameplay"
                print(f"  Set break before seg {i} (dist={dist:.1f}, sampled from {label})")

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

        print(f"  {label}: {fmt(cut_start)} -> {fmt(cut_end)} ({duration / 60:.1f} min)")

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
            print(f"    FAILED (exit {result.returncode})")
        else:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"    -> {out_path.name} ({size_mb:.0f} MB)")


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
    parser.add_argument("--merge-gap", type=int, default=GAMEPLAY_MERGE_GAP,
                        help=f"Merge gameplay segments with gaps < N seconds (default: {GAMEPLAY_MERGE_GAP})")
    parser.add_argument("--min-duration", type=int, default=MIN_SEGMENT_DURATION,
                        help=f"Discard gameplay segments shorter than N seconds (default: {MIN_SEGMENT_DURATION})")
    parser.add_argument("--set-gap", type=float, default=SET_GAP_THRESHOLD,
                        help=f"Time gaps > N seconds are always set boundaries (default: {SET_GAP_THRESHOLD})")
    parser.add_argument("--banner-threshold", type=float, default=BANNER_DIST_THRESHOLD,
                        help=f"Banner distance threshold for set detection (default: {BANNER_DIST_THRESHOLD})")
    parser.add_argument("--padding-before", type=int, default=PADDING_BEFORE,
                        help=f"Seconds of padding before each set (default: {PADDING_BEFORE})")
    parser.add_argument("--padding-after", type=int, default=PADDING_AFTER,
                        help=f"Seconds of padding after each set (default: {PADDING_AFTER})")
    args = parser.parse_args()

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

    print(f"VOD: {Path(vod_path).name}")
    print(f"  Resolution: {w}x{h}, Duration: {fmt(total_duration)} ({total_duration / 60:.0f} min)")
    print()

    # Step 1: Scan
    scan = scan_gameplay(vod_path, w, h)
    print()

    # Step 2: Build segments
    segments = build_segments(scan, args.merge_gap, args.min_duration)
    print(f"Found {len(segments)} gameplay segments")
    for i, s in enumerate(segments):
        gap = s.start - segments[i - 1].end if i > 0 else 0
        print(f"  {i:2d}: {fmt(s.start)} - {fmt(s.end)}  ({s.duration / 60:.1f} min)  gap={gap:.0f}s")
    print()

    # Step 3: Detect set boundaries
    boundaries = detect_set_boundaries(
        segments, vod_path, w, h,
        gap_threshold=args.set_gap,
        dist_threshold=args.banner_threshold,
    )
    print()

    # Step 4: Group into sets
    sets = group_sets(segments, boundaries)
    print(f"Detected {len(sets)} sets:")
    for ds in sets:
        n = len(ds.segments)
        print(f"  Set {ds.index:2d}: {fmt(ds.start)} - {fmt(ds.end)}  "
              f"({ds.gameplay_duration / 60:.1f} min gameplay, {n} game(s))")
    print()

    # Step 5: Cut
    if args.preview:
        print("Preview mode — skipping video cuts.")
    else:
        print("Cutting video...")
        cut_sets(sets, vod_path, output_dir,
                 args.padding_before, args.padding_after, total_duration)
        print(f"\nDone! {len(sets)} sets saved to {output_dir}")


if __name__ == "__main__":
    main()
