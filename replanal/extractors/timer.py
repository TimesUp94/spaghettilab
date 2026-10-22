"""Timer digit reader.

Reads the round timer from the GGS HUD: two digits, or the single warm
digit of the last 9 seconds.

The digits are cream-coloured glyphs on the dark circular "ROUND TIME" badge.
The two digits sit at fixed, *staggered* positions (the ones digit is drawn
~12 px lower than the tens digit), so instead of splitting the ROI in half
and stretching each half's bounding box (the old reader, which turned 9/8/6
into 0 and 6/5 into 8), every digit cell is matched in place:

* the ROI is converted to grey, scaled to the 1080p reference size (80x55)
  and zero-padded so small HUD shakes stay inside the search window;
* each cell is searched within +-``SEARCH`` px of its nominal position with
  normalised cross-correlation (``cv2.TM_CCOEFF_NORMED``) against ten grey
  glyph templates (one shared set: the font is identical for both cells);
* a cell is accepted only when the best score is high enough *and* clearly
  beats the runner-up digit; otherwise the whole reading is rejected
  (``timer_value`` stays None).  A missing value is cheap downstream, a
  wrong one is not.

Templates are averages of hand-labelled crops from overlay and non-overlay
VODs, built by ``scripts/build_timer_templates.py`` and stored next to this
module (``timer_glyphs.npz``) so they ship with the package.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData

# Reference ROI size (timer_digits ROI at 1920x1080: x 920-1000, y 115-170).
ROI_W = 80
ROI_H = 55
# Zero padding around the ROI so a cell can be searched beyond the ROI edge.
PAD = 8
# Glyph template size.
GLYPH_H = 46
GLYPH_W = 36
# Search radius (px) around each cell's nominal position.
SEARCH = 4

# Acceptance rule (tuned on the labelled set, see build script / tests).
MIN_SCORE = 0.65
MIN_MARGIN = 0.10
# Minimum share of dark (<60) pixels in the ROI (the badge behind the digits).
MIN_DARK = 0.0
# Vertical scales tried (after 1.0) when a cell is not read: when a digit
# changes, the new glyph grows up from the baseline for a few frames.
SQUASH_SCALES = (0.9, 0.8)
# ...only when the full-size match already looks digit-like.
RETRY_SCORE = 0.5
# Per-quadrant agreement with the chosen glyph (RMS of the normalised
# difference).  Clean tens digits stay below ~0.9; the ones cell is looser
# because it overlaps the badge rim / "TIME" lettering and the digit-change
# animation, where None is the acceptable answer anyway.
MAX_QUADRANT = {"tens": 0.95, "ones": 1.2}

_DEFAULT_TEMPLATES = Path(__file__).resolve().parent / "timer_glyphs.npz"

# Single digits (the last 9 seconds): GGST draws one centred digit, in orange
# / yellow instead of cream, pulsing slightly larger now and then.  Read only
# when the two cells give no value and the lit glyph pixels are clearly warm
# (median HSV saturation of pixels with V >= SINGLE_LIT_V at least
# SINGLE_MIN_SAT, on at least SINGLE_MIN_LIT of the ROI), so the cream
# two-digit timer and grey-only input never take this path.  The glyph is
# searched within +-SINGLE_SEARCH px of the ROI centre at each scale of
# SINGLE_SCALES (glyph size relative to the templates; the single digit is
# drawn ~10-25 % larger than the two-digit glyphs).
SINGLE_LIT_V = 140
SINGLE_MIN_SAT = 90
SINGLE_MIN_LIT = 0.05
# ... at most SINGLE_MAX_LIT of the ROI lit (the digits measure 0.12-0.24;
# fire and flash cinematics 0.9+), and at most SINGLE_MAX_STRAY of the lit
# pixels outside the centred glyph box SINGLE_BOX (y1, y2, x1, x2 at the
# 80x55 reference size; digits <= 0.08, two sprite streaks 0.15).
SINGLE_MAX_LIT = 0.35
SINGLE_MAX_STRAY = 0.10
SINGLE_BOX = (2, 53, 18, 62)
SINGLE_SEARCH = 8
SINGLE_SCALES = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)
SINGLE_MARGIN = 0.06
SINGLE_QUADRANT = 1.05

# Nominal top-left corner (y, x) of each glyph box in the padded ROI.
# Overridden by the "cells" array stored with the templates.
DEFAULT_CELLS = {"tens": (7, 15), "ones": (19, 47)}


def prepare_roi(roi: np.ndarray) -> np.ndarray:
    """BGR (or grey) timer ROI -> padded float32 grey image at reference size."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    if gray.shape != (ROI_H, ROI_W):
        gray = cv2.resize(gray, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(
        gray.astype(np.float32), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0
    )


def cell_window(padded: np.ndarray, y0: int, x0: int) -> np.ndarray:
    """Search window around a glyph box whose nominal corner is (y0, x0)."""
    return padded[
        y0 - SEARCH : y0 + GLYPH_H + SEARCH, x0 - SEARCH : x0 + GLYPH_W + SEARCH
    ]


def score_cell(window: np.ndarray, glyphs: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Best NCC score of each glyph template within *window*, and where."""
    scores = np.full(len(glyphs), -1.0, dtype=np.float32)
    locs = [(0, 0)] * len(glyphs)
    if float(window.std()) < 1.0:  # flat region: nothing to read
        return scores, locs
    for d, g in enumerate(glyphs):
        res = cv2.matchTemplate(window, g, cv2.TM_CCOEFF_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(res)
        scores[d] = best
        locs[d] = (by, bx)
    return scores, locs


def _normalise(p: np.ndarray) -> np.ndarray:
    return (p - p.mean()) / (p.std() + 1e-6)


def worst_quadrant(patch: np.ndarray, glyph: np.ndarray) -> float:
    """Largest per-quadrant RMS difference between normalised patch and glyph.

    A digit that is partly covered (a slash effect, a character, an orange
    blob) can still correlate well overall with the wrong glyph -- a 5 with
    its top covered looks like a 3, a 7 with its bar hidden like a 1 -- but
    the covered quadrant then disagrees strongly with the template.
    """
    diff = (_normalise(patch) - _normalise(glyph)) ** 2
    h, w = GLYPH_H // 2, GLYPH_W // 2
    return float(max(np.sqrt(diff[a:a + h, b:b + w].mean()) for a in (0, h) for b in (0, w)))


def decide(scores: np.ndarray) -> int | None:
    """Digit with the best score, or None when weak or ambiguous."""
    order = np.argsort(scores)[::-1]
    best, second = float(scores[order[0]]), float(scores[order[1]])
    if best < MIN_SCORE or best - second < MIN_MARGIN:
        return None
    return int(order[0])


def squash(glyphs: np.ndarray, scale: float) -> np.ndarray:
    """Glyphs shrunk vertically towards their baseline (digit-change animation)."""
    h = int(round(GLYPH_H * scale))
    out = np.empty_like(glyphs)
    for d, g in enumerate(glyphs):
        out[d] = np.percentile(g, 10)  # badge background level
        out[d, GLYPH_H - h:] = cv2.resize(g, (GLYPH_W, h), interpolation=cv2.INTER_AREA)
    return out


def read_cell(padded: np.ndarray, cell: tuple[int, int], glyph_sets: list[np.ndarray],
              max_quadrant: float) -> int | None:
    """Digit in one cell of a prepared ROI, or None when unsure.

    ``glyph_sets`` is tried in order (full size first, then squashed
    variants); the first confident, unoccluded match wins.
    """
    y0, x0 = cell
    window = cell_window(padded, y0, x0)
    for i, glyphs in enumerate(glyph_sets):
        scores, locs = score_cell(window, glyphs)
        d = decide(scores)
        if d is None:
            if i == 0 and float(scores.max()) < RETRY_SCORE:
                return None  # nothing digit-like here: skip the squashed sets
            continue
        by, bx = locs[d]
        patch = window[by:by + GLYPH_H, bx:bx + GLYPH_W]
        if worst_quadrant(patch, glyphs[d]) <= max_quadrant:
            return d
    return None


def is_warm_digit(roi_bgr: np.ndarray) -> bool:
    """Does the ROI look like the single warm digit (see SINGLE_*)?

    Lit pixels must be warm, cover a digit's share of the ROI (not a whole
    fire or flash cinematic) and stay inside the centred glyph box (two
    sprite streaks side by side are not one digit).
    """
    if roi_bgr.shape[:2] != (ROI_H, ROI_W):
        roi_bgr = cv2.resize(roi_bgr, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lit = hsv[..., 2] >= SINGLE_LIT_V
    share = float(lit.mean())
    if share < SINGLE_MIN_LIT or share > SINGLE_MAX_LIT:
        return False
    y1, y2, x1, x2 = SINGLE_BOX
    inside = int(lit[y1:y2, x1:x2].sum())
    if 1.0 - inside / lit.sum() > SINGLE_MAX_STRAY:
        return False
    return float(np.median(hsv[..., 1][lit])) >= SINGLE_MIN_SAT


def load_templates(path: Path | None = None) -> tuple[np.ndarray, dict[str, tuple[int, int]]] | None:
    path = Path(path or _DEFAULT_TEMPLATES)
    if not path.exists():
        return None
    data = np.load(str(path))
    glyphs = data["glyphs"].astype(np.float32)
    cells = dict(DEFAULT_CELLS)
    if "cells" in data.files:
        c = data["cells"]
        cells = {"tens": (int(c[0][0]), int(c[0][1])), "ones": (int(c[1][0]), int(c[1][1]))}
    return glyphs, cells


class TimerExtractor(BaseExtractor):
    """Extracts the round timer value (0-99) from the GGS HUD."""

    def __init__(self, templates_path: Path | None = None):
        loaded = load_templates(templates_path)
        self._glyphs: np.ndarray | None = None
        self._glyph_sets: list[np.ndarray] = []
        self._cells = dict(DEFAULT_CELLS)
        if loaded is not None:
            self._glyphs, self._cells = loaded
            self._glyph_sets = [self._glyphs] + [squash(self._glyphs, s) for s in SQUASH_SCALES]

    @property
    def required_rois(self) -> list[str]:
        return ["timer_digits"]

    def read(self, roi: np.ndarray) -> int | None:
        """Timer value shown in a timer_digits ROI, or None."""
        if roi is None or roi.size == 0 or self._glyphs is None:
            return None
        padded = prepare_roi(roi)
        inner = padded[PAD:-PAD, PAD:-PAD]
        # The badge behind the digits is dark; a mostly bright ROI is a
        # flash, a super background or a transition screen.
        if float(np.mean(inner < 60)) < MIN_DARK:
            return None
        digits = []
        for name in ("tens", "ones"):
            d = read_cell(padded, self._cells[name], self._glyph_sets, MAX_QUADRANT[name])
            if d is None:
                return self._read_single(roi, padded)
            digits.append(d)
        return digits[0] * 10 + digits[1]

    def _read_single(self, roi: np.ndarray, padded: np.ndarray) -> int | None:
        """The one centred warm digit shown for the last 9 seconds, or None.

        Every scale of SINGLE_SCALES is tried; the best-scoring (scale, digit)
        wins if it beats the runner-up digit at that scale by SINGLE_MARGIN
        and passes the per-quadrant check.
        """
        if roi.ndim != 3 or not is_warm_digit(roi):
            return None
        best = None
        for scale in SINGLE_SCALES:
            img = padded if scale == 1.0 else cv2.resize(
                padded, (round(padded.shape[1] / scale), round(padded.shape[0] / scale)),
                interpolation=cv2.INTER_AREA)
            h, w = img.shape
            y0 = (h - GLYPH_H) // 2 - SINGLE_SEARCH
            x0 = (w - GLYPH_W) // 2 - SINGLE_SEARCH
            window = img[max(0, y0):y0 + GLYPH_H + 2 * SINGLE_SEARCH,
                         max(0, x0):x0 + GLYPH_W + 2 * SINGLE_SEARCH]
            if window.shape[0] < GLYPH_H or window.shape[1] < GLYPH_W:
                continue
            scores, locs = score_cell(window, self._glyphs)
            order = np.argsort(scores)[::-1]
            d, top, second = int(order[0]), float(scores[order[0]]), float(scores[order[1]])
            if best is None or top > best[0]:
                best = (top, top - second, d, window, locs[d])
        if best is None:
            return None
        top, margin, d, window, (by, bx) = best
        if top < MIN_SCORE or margin < SINGLE_MARGIN or d == 1:
            # A lone "1" is a plain bar: vertical orange beams and streaks
            # match it.  It is shown for the round's last second only.
            return None
        patch = window[by:by + GLYPH_H, bx:bx + GLYPH_W]
        if worst_quadrant(patch, self._glyphs[d]) > SINGLE_QUADRANT:
            return None
        return d

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        value = self.read(ctx.rois.get("timer_digits"))
        if value is not None:
            data.timer_value = value


# finalize_timer: a reading is supported when at least SUPPORT_RUN readings
# (itself included) within +-SUPPORT_MS show the same value (the clock shows
# each value for a second).
SUPPORT_RUN = 3
SUPPORT_MS = 500.0


def finalize_timer(frames: list[FrameData]) -> int:
    """Replay-level check of the single-digit readings, in place.

    The single warm digit is only shown for the last 9 seconds of a round,
    so it must continue the countdown: a single-digit reading is kept only
    when it is supported (see SUPPORT_RUN) and the last supported reading
    before it (``p``, ``dt`` seconds earlier) could have counted down to it:
    ``p - dt - 1 <= value <= p``.  A tens digit covered by a hit spark or a
    fist reads "7" for 97 or 67 on a frame or two, which would otherwise
    look like a timer jump (round end / new round).  Two-digit readings are
    left alone.  The raw values stay in ``timer_raw``.  Returns the number
    of readings dropped.
    """
    for f in frames:
        if f.timer_raw is None:
            f.timer_raw = f.timer_value
    seen = [i for i, f in enumerate(frames) if f.timer_raw is not None]
    vals = [frames[i].timer_raw for i in seen]
    ts = [frames[i].timestamp_ms for i in seen]
    supported = []
    lo = hi = 0
    for k in range(len(seen)):
        while ts[lo] < ts[k] - SUPPORT_MS:
            lo += 1
        while hi < len(seen) and ts[hi] <= ts[k] + SUPPORT_MS:
            hi += 1
        supported.append(sum(1 for j in range(lo, hi) if vals[j] == vals[k]) >= SUPPORT_RUN)
    dropped = 0
    prev = None  # (value, timestamp_ms) of the last supported, kept reading
    for k, i in enumerate(seen):
        v = vals[k]
        keep = True
        if v <= 9:
            keep = supported[k]
            if keep and prev is not None:
                p, tp = prev
                keep = p - (ts[k] - tp) / 1000.0 - 1 <= v <= p
        frames[i].timer_value = v if keep else None
        dropped += not keep
        if keep and supported[k]:
            prev = (v, ts[k])
    return dropped
