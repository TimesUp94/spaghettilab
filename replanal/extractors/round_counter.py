"""Round counter (heart) reader.

GGST draws two hearts per player next to the round timer (game-native HUD,
present in every GGST video, with or without our stream overlay):

- **Red heart** = still alive.
- **Grey heart** = lost; the opponent won that round. The outer heart (closer
  to the screen edge) is lost first; a lost heart is drawn smaller and a few
  pixels lower than the red one.

So ``rounds_won_by_player = lost hearts on the opponent's side``.

Per side the reader:

1. Requires the inner heart to be visibly red: red hue on a small box at its
   centre *and* a heart-shaped red island (its surroundings - below the
   corners, above, between the hearts - are not red). A red/orange screen
   fill or the dark-red lobby stripes therefore do not count.
2. Classifies the outer heart against that still-red reference: red (similar
   redness and brightness) or grey (neutral colour, or pinkish when the whole
   area is red-tinted; brightness close to the red heart's; uniform; and a
   heart-shaped island, i.e. the ring around the grey-heart outline differs
   from it on at least 3 of 4 sides).
3. Returns ``None`` for the side when neither matches - white flashes, HUD
   hidden by a super cinematic or a fade, occluding effects - instead of
   guessing a count.

Two lost hearts (inner grey too) are only accepted when both hearts are clean
neutral grey islands. In practice the HUD fades out right after the
match-deciding KO, so this state is almost never on screen.

Labeled evaluation set: ``tests/data/hearts`` (see ``labels.csv`` there).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData

# ── Geometry (full 1920x1080 frame) ──────────────────────────────────────────
# Each heart: core = (cx, cy, half) sample box on the heart itself; out = boxes
# (x0, y0, x1, y1) just outside the red heart: bottom-left and bottom-right
# corners of its bounding box, a band above it, and the gap between the hearts.
_P1_GAP = (814, 82, 823, 90)
_P2_GAP = (1097, 82, 1106, 90)
HEARTS = {
    "p1": {
        "inner": dict(core=(843, 86, 4), out=[(826, 95, 833, 102), (853, 95, 860, 102), (832, 60, 855, 65), _P1_GAP]),
        "outer": dict(core=(799, 91, 3), out=[(784, 97, 790, 103), (809, 97, 815, 103), (790, 64, 808, 69), _P1_GAP]),
    },
    "p2": {
        "inner": dict(core=(1077, 86, 4), out=[(1060, 95, 1067, 102), (1087, 95, 1094, 102), (1065, 60, 1088, 65), _P2_GAP]),
        "outer": dict(core=(1121, 91, 3), out=[(1105, 97, 1111, 103), (1130, 97, 1136, 103), (1112, 64, 1130, 69), _P2_GAP]),
    },
}

# Lost (grey) outer heart outline: per-row (dx0, dx1) around GREY_CX, rows
# starting at y=GREY_Y0 (measured from labeled frames, identical on both sides).
GREY_SPANS = [(-8, 8), (-9, 9)] + [(-10, 10)] * 7 + [
    (-9, 9), (-8, 8), (-8, 8), (-7, 7), (-6, 6), (-5, 5), (-4, 4), (-3, 3), (-2, 2), (-1, 1)
]
GREY_Y0 = 86
GREY_CX = {"p1": 799, "p2": 1120}

# ── Thresholds ───────────────────────────────────────────────────────────────
RED_MIN_RG = 0.6          # (R-G)/R of a red heart (magenta tints still pass)
RED_MIN_R = 25            # dimmest red heart accepted (fades)
RED_SHAPE_MIN_PASS = 3    # outside boxes (of 4) that must not look like the heart
RED_REL_RG = 0.7          # outer red: redness vs the inner heart
RED_REL_R = (0.5, 2.2)    # outer red: R vs the inner heart (inner heart pulses)
GREY_V = (20, 225)        # grey heart brightness range (absolute)
GREY_MAX_V_REL = 2.0      # grey brightness <= this x inner red R
GREY_MAX_RG = 0.3         # neutral
GREY_MAX_RG_TINTED = 0.45 # pinkish grey accepted when the surroundings are red
GREY_TINT_RG = 0.5
GREY_MIN_B_MINUS_G = -25  # rejects yellow/orange occluders
GREY_MAX_STD = 16         # uniform fill (unless the ring contrast is strong)
GREY_RING_MIN_PASS = 3    # ring sides (of 4) differing from the heart


def _rg(c: np.ndarray) -> float:
    return float((c[2] - c[1]) / max(float(c[2]), 1.0))


def _contrast(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def _median(px: np.ndarray) -> np.ndarray:
    return np.median(px.reshape(-1, 3).astype(np.float32), axis=0)


def _build_grey_masks():
    h, w, cx, pad = len(GREY_SPANS) + 12, 36, 18, 6
    m = np.zeros((h, w), np.uint8)
    for i, (a, b) in enumerate(GREY_SPANS):
        m[pad + i, cx + a:cx + b + 1] = 1
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(m, k, iterations=2).astype(bool)
    ring = (cv2.dilate(m, k, iterations=4) - cv2.dilate(m, k, iterations=1)).astype(bool)
    yy, xx = np.mgrid[0:h, 0:w]
    lower = yy >= pad + 3
    segs = [
        ring & ~lower,                       # top
        ring & lower & (xx < cx - 4),        # left
        ring & lower & (xx > cx + 4),        # right
        ring & lower & (np.abs(xx - cx) <= 4),  # below the tip
    ]
    return inner, segs, h, w, cx, pad


_GREY_INNER, _GREY_SEGS, _GREY_H, _GREY_W, _GREY_CX0, _GREY_PAD = _build_grey_masks()


def _heart_stats(frame: np.ndarray, spec: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    x, y, h = spec["core"]
    core = _median(frame[y - h:y + h, x - h:x + h])
    outs = [_median(frame[y0:y1, x0:x1]) for x0, y0, x1, y1 in spec["out"]]
    return core, outs


def _is_red(core: np.ndarray, outs: list[np.ndarray], ref: Optional[tuple[float, float]] = None) -> bool:
    rg = _rg(core)
    if rg < RED_MIN_RG:
        return False
    n_pass = sum(rg - _rg(p) >= 0.3 or core[2] - p[2] >= 0.4 * core[2] for p in outs)
    if n_pass < RED_SHAPE_MIN_PASS:
        return False
    if ref is None:
        return core[2] >= RED_MIN_R
    ref_r, ref_rg = ref
    return rg >= RED_REL_RG * ref_rg and RED_REL_R[0] * ref_r <= core[2] <= RED_REL_R[1] * ref_r


def _is_grey_outer(frame: np.ndarray, side: str, ref_r: float) -> bool:
    x0 = GREY_CX[side] - _GREY_CX0
    y0 = GREY_Y0 - _GREY_PAD
    win = frame[y0:y0 + _GREY_H, x0:x0 + _GREY_W].astype(np.float32)
    px = win[_GREY_INNER]
    core = np.median(px, axis=0)
    segs = [np.median(win[m], axis=0) for m in _GREY_SEGS]
    v = float(core.max())
    if not (GREY_V[0] <= v <= GREY_V[1] and v <= GREY_MAX_V_REL * ref_r):
        return False
    rg = _rg(core)
    around = np.median(np.array(segs), axis=0)
    if not (rg < GREY_MAX_RG or (rg < GREY_MAX_RG_TINTED and _rg(around) > GREY_TINT_RG)):
        return False
    if core[0] - core[1] < GREY_MIN_B_MINUS_G:
        return False
    strong = all(_contrast(core, p) >= 0.5 * v for p in segs)
    if not strong and float(px.std(axis=0).max()) > GREY_MAX_STD:
        return False
    thr = max(12.0, 0.12 * v)
    return sum(_contrast(core, p) >= thr for p in segs) >= GREY_RING_MIN_PASS


def _is_clean_grey(core: np.ndarray, outs: list[np.ndarray]) -> bool:
    v = float(core.max())
    return (90 <= v <= 200 and abs(core[0] - core[1]) <= 15 and abs(core[2] - core[1]) <= 20
            and min(_contrast(core, p) for p in outs) >= 30)


def read_lost_hearts(frame: np.ndarray, side: str) -> Optional[int]:
    """Number of lost (grey) hearts of ``side`` ("p1"/"p2"), or None if not visible.

    ``frame`` is a full 1920x1080 BGR frame.
    """
    spec = HEARTS[side]
    ci, oi = _heart_stats(frame, spec["inner"])
    co, oo = _heart_stats(frame, spec["outer"])
    if _is_red(ci, oi):
        if _is_red(co, oo, (float(ci[2]), _rg(ci))):
            return 0
        if _is_grey_outer(frame, side, float(ci[2])):
            return 1
        return None
    if _is_clean_grey(ci, oi) and _is_clean_grey(co, oo) and abs(float(ci.max()) - float(co.max())) < 25:
        return 2
    return None


class RoundCounterExtractor(BaseExtractor):
    """Counts rounds won by each player from heart indicators."""

    @property
    def required_rois(self) -> list[str]:
        # We read directly from the full frame at known positions
        return []

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        frame = ctx.frame_bgr
        h, w = frame.shape[:2]
        if h < 1080 or w < 1920:
            return

        p1_lost = read_lost_hearts(frame, "p1")
        p2_lost = read_lost_hearts(frame, "p2")

        if p1_lost is not None and p2_lost is not None:
            # Rounds won by a player = rounds lost by their opponent
            data.p1_rounds_won = p2_lost
            data.p2_rounds_won = p1_lost
