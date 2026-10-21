"""Split a GGS tournament VOD into individual sets.

Detects gameplay from the game's own HUD (the round timer read by the
pipeline's timer reader, the round hearts), groups the gameplay into sets by
the in-game player name tags read with OCR (and the hearts: a game still in
progress is never a set break), then cuts each set with ffmpeg stream copy
(no re-encode).

Usage:
    python scripts/split_vod.py <vod_path> [--output <dir>] [--preview]

Examples:
    python scripts/split_vod.py "my_tournament.mp4"
    python scripts/split_vod.py "my_tournament.mp4" --preview   # detect only, don't cut
    python scripts/split_vod.py "my_tournament.mp4" --output data/my_sets
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Force UTF-8 output on Windows to avoid charmap encoding errors from EasyOCR
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# `import replanal` when run as a plain script (the app also sets PYTHONPATH)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from replanal.extractors.round_counter import read_lost_hearts
from replanal.extractors.timer import TimerExtractor

# ---------------------------------------------------------------------------
# Config defaults for 1080p GGS.  Every ROI is on the game's own HUD (same
# pixels as config/default.yaml): a stream overlay (score banner, logos) is set
# by hand during a stream and may be stale, different or absent, so it is
# never read.  Override via CLI flags for other games / layouts.
# ---------------------------------------------------------------------------

# ROI coordinates (y1, y2, x1, x2) for 1920x1080
P1_TENSION = (1000, 1040, 260, 750)    # in-game tension bars (not used for detection)
P2_TENSION = (1000, 1040, 1170, 1660)
TIMER_ROI = (110, 180, 920, 1000)      # in-game round timer badge
# Player name tags: 4-point quads (tl_x,tl_y, tr_x,tr_y, br_x,br_y, bl_x,bl_y)
# around the name text only.  GGS draws the name slightly rotated (P1 rising to
# the right, P2 falling), under a smaller "Player 1/2" label ("Giocatore 1",
# ...) that sits up and out (left of the P1 name, right of the P2 name) and is
# left outside the quads.  Names are centred in the tag (P1 ~x 258, P2 ~1690)
# with the platform icon just before them.
P1_NAME_QUAD = (145, 168, 400, 147, 400, 172, 145, 193)
P2_NAME_QUAD = (1560, 150, 1748, 165, 1748, 190, 1560, 175)

TIMER_SUPPORT_WINDOW = 5   # a timer read needs a consistent read within N s
GAMEPLAY_MERGE_GAP = 12    # merge HUD runs with < N second gaps (supers, round transitions)
MIN_SEGMENT_HUD = 15       # discard segments with fewer timer-read seconds
UNKNOWN_NAMES_MAX_GAP = 90  # without readable names, a longer gap starts a new set
NAME_SAME = 0.6            # name similarity at/above this = same player
NAME_SAMPLES = 9           # HUD frames OCR'd per segment (names are voted)
PADDING_BEFORE = 10
PADDING_AFTER = 20         # the HUD fades at the match-deciding KO; keep the win screen


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    hud_secs: int = 0
    # Stable hearts states (p1_lost, p2_lost) in time order
    hearts: list[tuple[int, int]] = field(default_factory=list)
    p1_name: str = ""
    p2_name: str = ""
    p1_votes: list[str] = field(default_factory=list)
    p2_votes: list[str] = field(default_factory=list)
    signature: np.ndarray | None = None
    # Each side's tag text, cropped and binarised (side-swap comparisons)
    tag_shapes: tuple[np.ndarray, np.ndarray] | None = None
    # Signature noise: distance between the signatures of its even and odd
    # seconds (None for segments too short to split)
    signature_noise: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def games_started(self) -> int:
        """Hearts resets to 0-0 after a lost heart inside the segment."""
        n = 0
        for a, b in zip(self.hearts, self.hearts[1:]):
            if sum(a) > 0 and sum(b) == 0:
                n += 1
        return n

    @property
    def starts_mid_game(self) -> bool:
        return bool(self.hearts) and sum(self.hearts[0]) > 0


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

    @property
    def game_count(self) -> int:
        """Games played, from the HUD gaps and the hearts.

        Games are separated by character select / loading screens, so every
        segment starts a game, plus each reset of the hearts to 0-0 inside a
        segment.  A segment only resumes the previous game when its first
        hearts state strictly follows the previous segment's last one (0-1
        then 1-1): an equal or lower state is a new game whose 0-0 start
        wasn't read.
        """
        n = 0
        for i, seg in enumerate(self.segments):
            n += seg.games_started
            prev = self.segments[i - 1].hearts if i > 0 else []
            resumes = (
                bool(prev) and bool(seg.hearts)
                and all(b >= a for a, b in zip(prev[-1], seg.hearts[0]))
                and sum(seg.hearts[0]) > sum(prev[-1])
            )
            if not resumes:
                n += 1
        return n


# ---------------------------------------------------------------------------
# External tools
# ---------------------------------------------------------------------------

class ToolMissing(RuntimeError):
    pass


def _bundled_dirs() -> list[Path]:
    """Folders an installed app may ship ffmpeg in (next to scripts/)."""
    return [_ROOT / "ffmpeg", _ROOT / "resources" / "ffmpeg", _ROOT / "bin"]


def find_tool(name: str) -> str | None:
    """Resolve ffmpeg/ffprobe: env override, PATH, imageio-ffmpeg, bundled copy."""
    env = os.environ.get(f"SPAGHETTI_{name.upper()}")
    if env and Path(env).is_file():
        return env
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and Path(exe).is_file():
                return exe
        except Exception:
            pass
    exe_name = name + (".exe" if sys.platform == "win32" else "")
    for d in _bundled_dirs():
        if (d / exe_name).is_file():
            return str(d / exe_name)
    # ffprobe usually sits next to whatever ffmpeg we found
    if name == "ffprobe":
        ff = find_tool("ffmpeg")
        if ff and (Path(ff).parent / exe_name).is_file():
            return str(Path(ff).parent / exe_name)
    return None


_FFMPEG: str | None = None


def ffmpeg_bin() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = find_tool("ffmpeg")
        if _FFMPEG is None:
            raise ToolMissing(
                "ffmpeg not found. Install ffmpeg and put it on PATH, or run: "
                f'"{sys.executable}" -m pip install imageio-ffmpeg'
            )
    return _FFMPEG


def probe_video(path: str) -> tuple[float, int, int]:
    """(duration_s, width, height) via ffprobe, else parsed from `ffmpeg -i`."""
    ffprobe = find_tool("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "format=duration:stream=width,height",
             "-of", "default=noprint_wrappers=1", path],
            capture_output=True, text=True,
        ).stdout
        vals = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
        try:
            return float(vals["duration"]), int(vals["width"]), int(vals["height"])
        except (KeyError, ValueError):
            pass
    err = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", path],
                         capture_output=True, text=True, errors="replace").stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return dur, w, h


def ffmpeg_threads() -> int:
    """Decoder threads: leave two logical CPUs free for the app / OS."""
    return max(1, (os.cpu_count() or 4) - 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


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


def quad_bbox(quad: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of a quad."""
    xs, ys = quad[0::2], quad[1::2]
    return min(xs), min(ys), max(xs), max(ys)


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


def name_strip(crop: np.ndarray, quad: tuple[int, ...], scale: int = 2) -> np.ndarray:
    """Warp the name quad out of its bounding-box crop, upscaled for OCR."""
    x0, y0, _, _ = quad_bbox(quad)
    local = tuple(v - (x0 if i % 2 == 0 else y0) for i, v in enumerate(quad))
    w = int(round(np.hypot(quad[2] - quad[0], quad[3] - quad[1]))) * scale
    h = int(round(np.hypot(quad[6] - quad[0], quad[7] - quad[1]))) * scale
    return warp_quad(crop, local, w, h)


def timer_digits_roi(timer_roi: tuple) -> tuple[int, int, int, int]:
    """The timer reader's digit box inside the timer badge ROI.

    config/default.yaml: timer (y 110-180) -> timer_digits (y 115-170), same x.
    """
    y1, y2, x1, x2 = timer_roi
    hgt = y2 - y1
    return y1 + round(hgt * 5 / 70), y2 - round(hgt * 10 / 70), x1, x2


# ---------------------------------------------------------------------------
# Step 1: Scan the game's HUD (1 fps)
# ---------------------------------------------------------------------------

@dataclass
class Scan:
    """Per-second HUD readings plus name-tag crops of clean HUD seconds."""
    timer: list[int | None] = field(default_factory=list)
    hearts: list[tuple[int, int] | None] = field(default_factory=list)
    # second -> (p1 crop, p2 crop) grey bounding-box crops of the name quads
    strips: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        secs = sorted(self.strips)
        np.savez_compressed(
            path,
            timer=np.array([-1 if v is None else v for v in self.timer], np.int16),
            hearts=np.array([(-1, -1) if v is None else v for v in self.hearts], np.int8).reshape(-1, 2),
            strip_secs=np.array(secs, np.int32),
            p1=np.array([self.strips[s][0] for s in secs]) if secs else np.zeros((0, 1, 1), np.uint8),
            p2=np.array([self.strips[s][1] for s in secs]) if secs else np.zeros((0, 1, 1), np.uint8),
        )

    @classmethod
    def load(cls, path: Path) -> "Scan":
        d = np.load(path)
        sc = cls()
        sc.timer = [None if v < 0 else int(v) for v in d["timer"]]
        sc.hearts = [None if a < 0 else (int(a), int(b)) for a, b in d["hearts"]]
        p1, p2 = d["p1"], d["p2"]  # each access of an npz key decompresses it again
        sc.strips = {int(s): (p1[i], p2[i]) for i, s in enumerate(d["strip_secs"])}
        return sc


def scan_hud(
    vod_path: str,
    w: int = 1920,
    h: int = 1080,
    timer_roi: tuple = TIMER_ROI,
    p1_name_quad: tuple = P1_NAME_QUAD,
    p2_name_quad: tuple = P2_NAME_QUAD,
    total_duration: float = 0,
    strip_every: int = 2,
) -> Scan:
    """Decode the VOD at 1 fps and read the timer and hearts of every second.

    Name-tag crops are kept for every *strip_every*-th second whose timer and
    both hearts are read (the full HUD is on screen, no super or flash), so
    the names can be OCR'd later without seeking back into the VOD.
    """
    cmd = [
        ffmpeg_bin(), "-threads", str(ffmpeg_threads()), "-i", vod_path,
        "-an", "-sn", "-vf", f"fps=1,scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-v", "quiet", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    frame_size = w * h * 3
    timer_reader = TimerExtractor()
    ty1, ty2, tx1, tx2 = timer_digits_roi(timer_roi)
    b1 = quad_bbox(p1_name_quad)
    b2 = quad_bbox(p2_name_quad)
    scan = Scan()
    sec = 0
    t0 = time.time()
    total_secs = int(total_duration) if total_duration > 0 else 0
    hearts_ok = w == 1920 and h == 1080

    print("Scanning the game HUD (1 fps)...", flush=True)
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))

        value = timer_reader.read(frame[ty1:ty2, tx1:tx2])
        hearts = None
        if hearts_ok:
            p1_lost = read_lost_hearts(frame, "p1")
            p2_lost = read_lost_hearts(frame, "p2")
            if p1_lost is not None and p2_lost is not None:
                hearts = (p1_lost, p2_lost)
        scan.timer.append(value)
        scan.hearts.append(hearts)
        if value is not None and hearts is not None and sec % strip_every == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scan.strips[sec] = (g[b1[1]:b1[3], b1[0]:b1[2]].copy(), g[b2[1]:b2[3], b2[0]:b2[2]].copy())
        sec += 1

        # Machine-readable progress: first frame then every 10 seconds
        if sec == 1 or sec % 10 == 0:
            total_str = f"/{total_secs}" if total_secs else ""
            print(f"PROGRESS:{sec}{total_str}", flush=True)

    proc.wait()
    print(f"PROGRESS:{sec}/{sec}", flush=True)
    print(f"  Done: {sec} seconds scanned in {time.time() - t0:.0f}s", flush=True)
    return scan


def valid_timer_seconds(timer: list[int | None], window: int = TIMER_SUPPORT_WINDOW) -> list[bool]:
    """Seconds whose timer read is backed by a consistent nearby read.

    The round timer only counts down (about 1 per second, slower in hitstop
    and supers) or stays frozen, so a read at second *s* is kept when another
    read within *window* seconds agrees with that: an isolated false read on
    a lobby or a stream scene has no such partner.
    """
    n = len(timer)
    ok = [False] * n
    for s, v in enumerate(timer):
        if v is None:
            continue
        for d in range(1, window + 1):
            if s + d < n and timer[s + d] is not None and 0 <= v - timer[s + d] <= d + 1:
                ok[s] = True
                break
            if s - d >= 0 and timer[s - d] is not None and 0 <= timer[s - d] - v <= d + 1:
                ok[s] = True
                break
    return ok


def stable_hearts(hearts: list[tuple[int, int] | None], min_run: int = 2) -> list[tuple[int, tuple[int, int]]]:
    """(second, state) of each hearts state held for *min_run* consecutive reads.

    Unread seconds (None) are skipped, so a state interrupted by a super still
    counts.  Consecutive repeats of a state are collapsed.
    """
    out: list[tuple[int, tuple[int, int]]] = []
    run_state, run_len, run_start = None, 0, 0
    for s, v in enumerate(hearts):
        if v is None:
            continue
        if v == run_state:
            run_len += 1
        else:
            run_state, run_len, run_start = v, 1, s
        if run_len == min_run and (not out or out[-1][1] != v):
            out.append((run_start, v))
    return out


# ---------------------------------------------------------------------------
# Step 2: Build gameplay segments
# ---------------------------------------------------------------------------

def build_segments(
    scan: Scan,
    merge_gap: int = GAMEPLAY_MERGE_GAP,
    min_hud: int = MIN_SEGMENT_HUD,
) -> list[Segment]:
    """Group timer-read seconds into gameplay segments.

    Runs closer than *merge_gap* seconds are merged (the HUD hides during
    supers, KO cinematics and round intros); segments with fewer than
    *min_hud* timer-read seconds are dropped.
    """
    valid = valid_timer_seconds(scan.timer)
    secs = [s for s, ok in enumerate(valid) if ok]
    if not secs:
        return []
    groups: list[list[int]] = [[secs[0]]]
    for s in secs[1:]:
        if s - groups[-1][-1] <= merge_gap:
            groups[-1].append(s)
        else:
            groups.append([s])

    hearts_states = stable_hearts(scan.hearts)
    segments = []
    for g in groups:
        if len(g) < min_hud:
            continue
        seg = Segment(float(g[0]), float(g[-1] + 1), hud_secs=len(g))
        seg.hearts = [st for s, st in hearts_states if g[0] <= s <= g[-1]]
        segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# OCR via EasyOCR -- lazy-loaded, GPU-accelerated
# ---------------------------------------------------------------------------

_ocr_reader = None
_ocr_error: str | None = None


def _get_ocr_reader():
    """The EasyOCR reader, or None (with a warning once) when unavailable.

    Any failure degrades to grouping by the name tags' look: easyocr missing,
    a torch that can't load its DLLs (common on Windows), or the first run's
    model download failing without internet.
    """
    global _ocr_reader, _ocr_error
    if _ocr_reader is None and _ocr_error is None:
        try:
            import easyocr
        except ImportError:
            _ocr_error = (
                "easyocr is not installed, so player names can't be read "
                "(sets are still found and grouped by the name tags' look). "
                f'To read names run: "{sys.executable}" -m pip install easyocr'
            )
        except Exception as e:  # e.g. OSError: [WinError 126] ... c10.dll
            _ocr_error = (
                f"easyocr could not be loaded ({type(e).__name__}: {e}), so player names "
                "can't be read (sets are still found and grouped by the name tags' look). "
                f'Try reinstalling it: "{sys.executable}" -m pip install --force-reinstall easyocr torch'
            )
        else:
            try:
                try:
                    import torch
                    gpu = bool(torch.cuda.is_available())
                except Exception:
                    gpu = False
                _ocr_reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            except Exception as e:  # model download (offline first run), torch errors
                _ocr_error = (
                    f"the easyocr reader could not start ({type(e).__name__}: {e}), so player "
                    "names can't be read (sets are still found and grouped by the name tags' "
                    "look). On the first run easyocr downloads its models: check the internet "
                    "connection and try again."
                )
        if _ocr_error is not None:
            print(f"WARNING: {_ocr_error}", flush=True)
    return _ocr_reader


# The game's "Player 1/2" label above each name tag, in the languages GGST
# ships.  Compared with a fuzzy match because OCR mangles it ("Glocatore",
# "Giocatore z", "Giacatore").
LABEL_WORDS = (
    "giocatore", "player", "joueur", "spieler", "jugador", "jogador",
    "igrok", "gracz", "oyuncu",
)
_LABEL_RE = re.compile(r"(プレイヤー|플레이어|玩家|игрок)")


def clean_text(text: str) -> str:
    """ASCII-safe name text for filenames and JSON."""
    return "".join(c for c in text.strip() if c.isascii() and (c.isalnum() or c in " _-.")).strip()


def norm_name(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def is_label(text: str) -> bool:
    """True for text that looks like the game's "Player 1/2" label (any
    language, OCR-mangled).  Real handles can look like it too ("Payer",
    "Spieler"), so pick_name only drops such a box where the label sits."""
    if _LABEL_RE.search(text):
        return True
    t = norm_name(text).rstrip("12zl")
    if len(t) < 4:
        return False
    return any(difflib.SequenceMatcher(None, t, w).ratio() >= 0.7 for w in LABEL_WORDS)


_LABEL_NUMBER_RE = re.compile(r"\s[12zZlI]$")


def _drop_labels(boxes: list[list]) -> list[list]:
    """Remove the "Player 1/2" label from [x0, x1, y0, y1, text] boxes.

    The name quads leave the label out, so a label-like box is only dropped
    when it carries the label's number ("Giocatore 1") or sits above the
    name line (its centre at least a quarter of the name's height above the
    centre of the widest other box); otherwise it is a name.
    """
    labelish = [is_label(b[4]) for b in boxes]
    others = [b for b, lab in zip(boxes, labelish) if not lab]
    main = max(others, key=lambda b: b[1] - b[0]) if others else None
    out = []
    for b, lab in zip(boxes, labelish):
        if lab:
            if _LABEL_NUMBER_RE.search(b[4]) or _LABEL_RE.search(b[4]):
                continue
            if main is not None:
                mh = main[3] - main[2]
                if (main[2] + main[3]) / 2 - (b[2] + b[3]) / 2 >= 0.25 * mh:
                    continue
        out.append(b)
    return out


# What OCR makes of the platform icon when it glues it to the name ("CI Dona")
_ICON_CHARS = set("CcIilLDOoQ0[]|")


def pick_name(results: list) -> str:
    """The player name among EasyOCR (bbox, text, conf) results of one strip.

    Drops the "Player 1/2" label (see _drop_labels) and the platform icon (a
    short, roughly square box before the name), keeps the widest remaining box
    and joins the boxes
    next to it on the same line (names with spaces), then drops a stray
    one- or two-character token the icon left at the start ("C Thalllude",
    "CI Dona").
    """
    boxes = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        t = clean_text(text)
        if not t:
            continue
        bw, bh = x1 - x0, max(y1 - y0, 1.0)
        if len(norm_name(t)) <= 2 and bw <= 1.6 * bh:
            continue  # platform icon or a stray glyph
        boxes.append([x0, x1, y0, y1, t])
    boxes = _drop_labels(boxes)
    if not boxes:
        return ""
    boxes.sort(key=lambda b: b[0])
    main = max(range(len(boxes)), key=lambda i: boxes[i][1] - boxes[i][0])
    parts = [boxes[main]]
    lo, hi = main, main
    while lo > 0 or hi < len(boxes) - 1:
        grown = False
        h = parts[0][3] - parts[0][2]
        if lo > 0 and parts[0][0] - boxes[lo - 1][1] < 0.8 * h:
            lo -= 1
            parts.insert(0, boxes[lo])
            grown = True
        if hi < len(boxes) - 1 and boxes[hi + 1][0] - parts[-1][1] < 0.8 * h:
            hi += 1
            parts.append(boxes[hi])
            grown = True
        if not grown:
            break
    tokens = " ".join(p[4] for p in parts).split()
    if len(tokens) > 1 and (len(tokens[0]) == 1 or (len(tokens[0]) == 2 and set(tokens[0]) <= _ICON_CHARS)):
        tokens = tokens[1:]
    return " ".join(tokens)


def ocr_name_strip(strip: np.ndarray) -> str:
    """OCR a warped name strip. Returns the player name or ''."""
    reader = _get_ocr_reader()
    if reader is None or strip.size == 0:
        return ""
    return pick_name(reader.readtext(strip, detail=1))


def ocr_name_region(region: np.ndarray) -> str:
    """OCR a name tag region (compatibility wrapper around ocr_name_strip)."""
    return ocr_name_strip(region)


def vote_names(reads: list[str]) -> tuple[str, int]:
    """Most supported name among per-frame reads, and its support.

    Reads are clustered by similarity (OCR drops or swaps a letter now and
    then); the biggest cluster wins and is represented by its most frequent
    exact spelling (ties: the longer one).
    """
    reads = [r for r in reads if norm_name(r)]
    if not reads:
        return "", 0
    clusters: list[list[str]] = []
    for r in reads:
        for c in clusters:
            if name_similarity(r, c[0]) >= 0.75:
                c.append(r)
                break
        else:
            clusters.append([r])
    best = max(clusters, key=len)
    counts = Counter(best)
    rep = max(counts, key=lambda s: (counts[s], len(norm_name(s))))
    return rep, len(best)


def name_similarity(a: str, b: str) -> float:
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def read_segment_names(
    seg: Segment,
    scan: Scan,
    p1_name_quad: tuple,
    p2_name_quad: tuple,
    samples: int = NAME_SAMPLES,
) -> None:
    """OCR the name tags of up to *samples* clean HUD seconds of a segment."""
    secs = [s for s in sorted(scan.strips) if seg.start <= s < seg.end]
    if not secs:
        return
    if len(secs) > samples:
        idx = np.linspace(0, len(secs) - 1, samples).round().astype(int)
        secs = [secs[i] for i in idx]
    for s in secs:
        c1, c2 = scan.strips[s]
        seg.p1_votes.append(ocr_name_strip(name_strip(c1, p1_name_quad)))
        seg.p2_votes.append(ocr_name_strip(name_strip(c2, p2_name_quad)))
    seg.p1_name = vote_names(seg.p1_votes)[0]
    seg.p2_name = vote_names(seg.p2_votes)[0]


SIG_W, SIG_H = 96, 8


def binarize_strip(strip: np.ndarray) -> np.ndarray:
    """Name text (1) against the tag (0), with Otsu's threshold per strip.

    A fixed brightness cut depends on the encode (limited vs full range,
    gamma, an upscaled 720p source); Otsu's split doesn't.
    """
    _, b = cv2.threshold(strip, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b.astype(np.float32)


def _median_tags(secs: list[int], scan: Scan, quads: tuple) -> list[np.ndarray]:
    """Per side, the median binarised name strip over the given seconds."""
    return [np.median(np.array([binarize_strip(name_strip(scan.strips[s][k], q, scale=1)) for s in secs]), axis=0)
            for k, q in enumerate(quads)]


def _signature_of(tags: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([cv2.resize(t, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).ravel() for t in tags])


def _runs(idx: np.ndarray, gap: int) -> list[list[int]]:
    """Runs of sorted indices, joined across gaps of at most *gap*."""
    runs = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if i - runs[-1][1] <= gap:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    return runs


def tag_text_box(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    """(r0, r1, c0, c1) of the name text in a binarised strip.

    The text line is the band of rows with the most ink (a sliver of the
    label or the tag's border sits apart from it); within it the text is the
    run of ink columns with the most ink, joined across gaps narrower than
    the line is tall (spaces, the icon).
    """
    rs = ink.sum(axis=1)
    if rs.max() == 0:
        return None
    rows = np.flatnonzero(rs >= 0.05 * rs.max())
    r0, r1 = max(_runs(rows, 1), key=lambda r: rs[r[0]:r[1] + 1].sum())
    cs = ink[r0:r1 + 1].sum(axis=0)
    cols = np.flatnonzero(cs)
    c0, c1 = max(_runs(cols, max(2, r1 - r0)), key=lambda r: cs[r[0]:r[1] + 1].sum())
    return r0, r1 + 1, c0, c1 + 1


def _tag_shape(tag: np.ndarray) -> np.ndarray:
    ink = tag > 0.5
    box = tag_text_box(ink)
    if box is None:
        return np.zeros((1, 1), np.float32)
    r0, r1, c0, c1 = box
    return ink[r0:r1, c0:c1].astype(np.float32)


def segment_signature(seg: Segment, scan: Scan, p1_name_quad: tuple, p2_name_quad: tuple) -> np.ndarray | None:
    """Median binarised look of both name strips over the segment.

    Used to compare segments when OCR is unavailable or reads nothing.  Also
    fills the segment's tag shapes (for side swaps) and its signature noise.
    """
    secs = [s for s in sorted(scan.strips) if seg.start <= s < seg.end]
    if not secs:
        return None
    quads = (p1_name_quad, p2_name_quad)
    tags = _median_tags(secs, scan, quads)
    seg.tag_shapes = (_tag_shape(tags[0]), _tag_shape(tags[1]))
    if len(secs) >= 4:
        seg.signature_noise = signature_distance(_signature_of(_median_tags(secs[0::2], scan, quads)),
                                                 _signature_of(_median_tags(secs[1::2], scan, quads)))
    return _signature_of(tags)


# ---------------------------------------------------------------------------
# Step 3: Group segments into sets
# ---------------------------------------------------------------------------

def names_relation(a: tuple[str, str], b: tuple[str, str]) -> str:
    """'same', 'different' or 'unknown' for two (p1, p2) name pairs.

    A pair is the same when every side read in both matches; it is different
    when a side read in both clearly doesn't.  Players may swap sides between
    games, so the swapped pairing counts too: with all four names read it
    makes the pair the same; with some unread it only makes it 'unknown' (one
    player seen on the other side may also be them starting a new set), and
    the tags' look decides.
    """
    def rel(x: tuple[str, str], y: tuple[str, str]) -> str:
        verdicts = []
        for p, q in zip(x, y):
            if norm_name(p) and norm_name(q):
                verdicts.append(name_similarity(p, q) >= NAME_SAME)
        if not verdicts:
            return "unknown"
        return "same" if all(verdicts) else "different"

    straight = rel(a, b)
    if straight == "same":
        return "same"
    swapped = rel(a, (b[1], b[0]))
    if swapped == "same":
        return "same" if all(norm_name(v) for v in a + b) else "unknown"
    return straight


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest per-side mean difference of two segment signatures."""
    n = len(a) // 2
    return float(max(np.mean(np.abs(a[:n] - b[:n])), np.mean(np.abs(a[n:] - b[n:]))))


# Fallback look distance at/below this = same players.  The threshold
# follows the VOD's own noise (the distance between a segment's even- and
# odd-second signatures): SIGNATURE_NOISE_FACTOR times its median, within
# [SIGNATURE_SAME, SIGNATURE_SAME_MAX].  Tappa #4 (blacked overlay), also
# re-encoded to limited range, darker, brighter, 720p / 480p upscaled and
# with added noise: noise median 0.003-0.004, segments of one set
# 0.001-0.013, a changed player 0.12-0.21.
SIGNATURE_SAME = 0.03
SIGNATURE_SAME_MAX = 0.08
SIGNATURE_NOISE_FACTOR = 10


def signature_threshold(segments: list[Segment]) -> float:
    """The same-players look distance for this VOD (see SIGNATURE_SAME)."""
    noise = [s.signature_noise for s in segments if s.signature_noise is not None]
    if not noise:
        return SIGNATURE_SAME
    return float(min(SIGNATURE_SAME_MAX, max(SIGNATURE_SAME, SIGNATURE_NOISE_FACTOR * np.median(noise))))


# A side swap by look: each side's tag text against the other side's.  The
# game draws P2's name in italics and the tags differ in size, so the text is
# cropped, a range of slants is tried and the widths must agree.  The two
# sides never match as closely as one side with itself: Tappa #4 has the same
# player on opposite sides at 0.10-0.26 and different players at >= 0.24, so
# only a close match counts (about half of real swaps) and a swap the look
# misses still splits the set; reading names (easyocr) handles all of them.
SWAP_SAME = 0.15
_SWAP_SLANTS = np.linspace(-0.4, 0.4, 17)


def _shear(img: np.ndarray, k: float) -> np.ndarray:
    h, w = img.shape
    pad = int(abs(k) * h) + 1
    m = np.float32([[1, k, -k * h / 2 + pad], [0, 1, 0]])
    out = cv2.warpAffine(img, m, (w + 2 * pad, h), flags=cv2.INTER_NEAREST)
    cols = np.flatnonzero(out.sum(axis=0))
    return out[:, cols[0]:cols[-1] + 1] if len(cols) else out


def tag_shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Look distance of two cropped tag texts, over a range of slants of *b*."""
    if min(a.shape) < 2 or min(b.shape) < 2:
        return 1.0
    sa = cv2.resize(a, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)
    best = 1.0
    for k in _SWAP_SLANTS:
        bb = _shear(b, float(k))
        if not 0.8 < bb.shape[1] / a.shape[1] < 1.25:
            continue
        sb = cv2.resize(bb, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)
        best = min(best, float(np.mean(np.abs(sa - sb))))
    return best


def swapped_look_same(a: Segment, b: Segment) -> bool:
    """True when b's tags look like a's with the players on swapped sides."""
    if a.tag_shapes is None or b.tag_shapes is None:
        return False
    return (tag_shape_distance(a.tag_shapes[0], b.tag_shapes[1]) <= SWAP_SAME
            and tag_shape_distance(a.tag_shapes[1], b.tag_shapes[0]) <= SWAP_SAME)


def group_segments(segments: list[Segment]) -> list[list[Segment]]:
    """Split the segments into sets.

    A new set starts at a segment whose players differ from the current
    set's OCR'd names (either side assignment).  When the names can't be
    compared, the name tags' look decides (same look, or a close match with
    the sides swapped); without tag crops either, a segment that opens mid-game
    (the hearts already show a lost heart) continues the previous one, else
    the gap decides.  The hearts only break ties: the reader misses lost
    hearts often enough (a new set's first frames can read as 1-0) that they
    never override readable names or tags.  The gap between segments is not
    a break by itself: back-to-back sets of the same two players (a grand
    final and its bracket reset) end up in one set.
    """
    threshold = signature_threshold(segments)
    groups: list[list[Segment]] = []
    for seg in segments:
        if not groups:
            groups.append([seg])
            continue
        cur = groups[-1]
        prev = cur[-1]
        gap = seg.start - prev.end
        cur_names = set_names(cur)
        rel = names_relation(cur_names, (seg.p1_name, seg.p2_name))
        if rel == "unknown":
            if seg.signature is not None and prev.signature is not None:
                same = (signature_distance(seg.signature, prev.signature) <= threshold
                        or swapped_look_same(prev, seg))
            elif seg.starts_mid_game:
                same = gap <= 3 * UNKNOWN_NAMES_MAX_GAP
            else:
                same = gap <= UNKNOWN_NAMES_MAX_GAP
            rel = "same" if same else "different"
        if rel == "same":
            cur.append(seg)
        else:
            groups.append([seg])
    return groups


def set_names(segments: list[Segment]) -> tuple[str, str]:
    """Names of a set: votes pooled over its segments (side swaps aligned)."""
    p1_votes: list[str] = []
    p2_votes: list[str] = []
    ref: tuple[str, str] | None = None
    for seg in segments:
        a, b = seg.p1_votes, seg.p2_votes
        if ref is not None and (seg.p1_name or seg.p2_name) and any(ref):
            straight = name_similarity(ref[0], seg.p1_name) + name_similarity(ref[1], seg.p2_name)
            swapped = name_similarity(ref[0], seg.p2_name) + name_similarity(ref[1], seg.p1_name)
            if swapped > straight:
                a, b = b, a
        p1_votes += a
        p2_votes += b
        if ref is None or not all(ref):
            ref = (vote_names(p1_votes)[0], vote_names(p2_votes)[0])
    return vote_names(p1_votes)[0], vote_names(p2_votes)[0]


def group_sets(groups: list[list[Segment]]) -> list[DetectedSet]:
    sets = []
    for k, g in enumerate(groups):
        ds = DetectedSet(segments=g, index=k + 1)
        ds.p1_name, ds.p2_name = set_names(g)
        sets.append(ds)
    return sets


# ---------------------------------------------------------------------------
# Step 4: Cut video
# ---------------------------------------------------------------------------

def set_label(ds: DetectedSet) -> str:
    if ds.p1_name and ds.p2_name:
        safe_p1 = "".join(c for c in ds.p1_name if c.isalnum() or c in "_-")
        safe_p2 = "".join(c for c in ds.p2_name if c.isalnum() or c in "_-")
        return f"{ds.index:02d}_{safe_p1}_vs_{safe_p2}"
    return f"set_{ds.index:02d}"


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
        label = set_label(ds)
        out_path = output_dir / f"{label}.mp4"

        print(f"  {label}: {fmt(cut_start)} -> {fmt(cut_end)} ({duration / 60:.1f} min)", flush=True)

        cmd = [
            ffmpeg_bin(),
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
    parser.add_argument("--min-duration", type=int, default=MIN_SEGMENT_HUD,
                        help="Minimum timer-read seconds of a gameplay segment")
    parser.add_argument("--padding-before", type=int, default=PADDING_BEFORE)
    parser.add_argument("--padding-after", type=int, default=PADDING_AFTER)
    parser.add_argument("--no-ocr", action="store_true",
                        help="Don't read names (group by the name tags' look only)")
    parser.add_argument("--scan-cache", type=str, default=None,
                        help="Debug: save the 1 fps HUD scan to this .npz, or reuse it if present")
    # ROI overrides
    parser.add_argument("--p1-tension", type=str, default=None,
                        help="(unused: gameplay is detected from the timer)")
    parser.add_argument("--p2-tension", type=str, default=None,
                        help="(unused: gameplay is detected from the timer)")
    parser.add_argument("--timer-roi", type=str, default=None,
                        help="Timer badge ROI as y1,y2,x1,x2")
    parser.add_argument("--p1-name-quad", type=str, default=None,
                        help="P1 name tag quad as tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y")
    parser.add_argument("--p2-name-quad", type=str, default=None,
                        help="P2 name tag quad as tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y")
    # Deprecated flags still accepted so older callers keep working
    for flag in ("--set-gap", "--name-threshold", "--banner-threshold"):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
    for flag in ("--banner-roi", "--p1-name", "--p2-name"):
        parser.add_argument(flag, type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    timer_roi = parse_roi(args.timer_roi) if args.timer_roi else TIMER_ROI
    p1_name_quad = parse_quad(args.p1_name_quad) if args.p1_name_quad else P1_NAME_QUAD
    p2_name_quad = parse_quad(args.p2_name_quad) if args.p2_name_quad else P2_NAME_QUAD

    vod_path = str(Path(args.vod).resolve())
    if not Path(vod_path).exists():
        print(f"VOD not found: {vod_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path(vod_path).parent / "sets"

    total_duration, w, h = probe_video(vod_path)
    if w == 0 or h == 0:
        print("Could not determine video resolution.", file=sys.stderr)
        sys.exit(1)

    print(f"VOD: {Path(vod_path).name}", flush=True)
    print(f"  Native resolution: {w}x{h}, Duration: {fmt(total_duration)} ({total_duration / 60:.0f} min)", flush=True)

    # Force 1920x1080 for processing -- ROIs are defined at this resolution
    if w != 1920 or h != 1080:
        print(f"  Scaling to 1920x1080 for processing", flush=True)
    w, h = 1920, 1080
    print(flush=True)

    # Load the OCR reader before the long scan, so a missing or broken
    # easyocr is reported right away (the scan still runs; names stay empty)
    if not args.no_ocr:
        _get_ocr_reader()

    # Step 1: Scan
    cache = Path(args.scan_cache) if args.scan_cache else None
    if cache is not None and cache.exists():
        print(f"Reusing HUD scan {cache}", flush=True)
        scan = Scan.load(cache)
    else:
        scan = scan_hud(vod_path, w, h, timer_roi, p1_name_quad, p2_name_quad, total_duration)
        if cache is not None:
            scan.save(cache)
    print(flush=True)

    # Step 2: Build segments
    segments = build_segments(scan, args.merge_gap, args.min_duration)
    print(f"Found {len(segments)} gameplay segments", flush=True)

    # Step 3: Read names and group into sets
    print("Reading player names...", flush=True)
    for i, seg in enumerate(segments):
        if not args.no_ocr:
            read_segment_names(seg, scan, p1_name_quad, p2_name_quad)
        seg.signature = segment_signature(seg, scan, p1_name_quad, p2_name_quad)
        gap = seg.start - segments[i - 1].end if i > 0 else 0
        hearts = " ".join(f"{a}{b}" for a, b in seg.hearts)
        print(f"  {i:2d}: {fmt(seg.start)} - {fmt(seg.end)}  ({seg.duration / 60:.1f} min)  gap={gap:.0f}s  "
              f"hearts=[{hearts}]  names={seg.p1_name!r} vs {seg.p2_name!r}", flush=True)
    print(flush=True)

    sets = group_sets(group_segments(segments))
    for ds in sets:
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
                    "game_count": ds.game_count,
                    "segment_count": len(ds.segments),
                    "p1_name": ds.p1_name,
                    "p2_name": ds.p2_name,
                }
                for ds in sets
            ],
            "warnings": [_ocr_error] if _ocr_error else [],
        }
        print(f"JSON_RESULT:{json.dumps(result)}", flush=True)
    else:
        print(f"Detected {len(sets)} sets:", flush=True)
        for ds in sets:
            print(f"  Set {ds.index:2d}: {fmt(ds.start)} - {fmt(ds.end)}  "
                  f"({ds.gameplay_duration / 60:.1f} min gameplay, {ds.game_count} game(s))", flush=True)
        print(flush=True)

    # Step 4: Cut
    if args.preview or args.json_output:
        if not args.json_output:
            print("Preview mode -- skipping video cuts.")
    else:
        print("Cutting video...", flush=True)
        cut_sets(sets, vod_path, output_dir,
                 args.padding_before, args.padding_after, total_duration)
        print(f"\nDone! {len(sets)} sets saved to {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ToolMissing as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
