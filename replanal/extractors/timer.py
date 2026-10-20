"""Timer digit reader.

Reads the two-digit round timer from the GGS HUD.

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
                return None
            digits.append(d)
        return digits[0] * 10 + digits[1]

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        value = self.read(ctx.rois.get("timer_digits"))
        if value is not None:
            data.timer_value = value
