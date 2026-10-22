"""Tension gauge extractor.

Reads P1 and P2 tension (0-1) from the GGST HUD gauge at the bottom of the
screen. What the gauge looks like (checked on Tappa #4 1080p30 and on a
720p60 stream, see output/fix2/track3/):

* Each bar is a fixed 481 px wide strip at y ~1005-1030 (1920x1080). The
  fill is anchored at the end next to the screen centre and grows outward:
  P1 from x 747 to the left, P2 from x 1167 to the right.
* Below 50% the fill is mint green (H ~50-73, S ~70-95); from 50% on the
  whole fill turns pale yellow (H ~27-40, S ~15-100, whitish next to the
  anchor). V is ~230-255 in both cases.
* A dark gear icon sits on the bar at the 50% mark (~35 px), a second one
  appears at the outer end when the bar is full. The "TENSION" label is drawn
  inside the fill next to the anchor (~75 px of blue letters).
* Unfilled bar: dark translucent strip. Positive-bonus chevrons (green
  arrows) can be drawn over it.

How it is read: per column of the bar ROI, the fraction of fill-coloured
pixels (bright, fill hue, not saturated). The fill is the run of filled
columns starting at the anchor (the first one may be up to START_SLACK px
away: the label, or a stream webcam over the anchor) that continues with
gaps of at most MAX_GAP px (gear, label letters). Tension = run length / the
bar's fixed width. Absolute, so no calibration between rounds.

The gauge is part of the HUD: with no fill and no timer on screen (supers,
cinematics, menus) the value is unknown (None), not 0.

``finalize_tension`` runs over the whole replay afterwards: it removes
readings that a static stream overlay caps (the fill end never passes a fixed
x on the bar, so the value there is only a lower bound) and median-filters
the series.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData

# Bar ROIs (full-res, also in config/default.yaml as p1/p2_tension_bar).
# (x1, y1, x2, y2); the fill anchor is x2 for P1 and x1 for P2.
DEFAULT_BARS = {
    "p1_tension_bar": (266, 1005, 748, 1031),
    "p2_tension_bar": (1167, 1005, 1649, 1031),
}
# Rows trimmed off the top/bottom of the ROI (anti-aliased bar edges).
EDGE_ROWS = 2

# Fill colour (OpenCV HSV, H 0-180)
FILL_V_MIN = 170
FILL_S_MIN = 8  # pure white/grey (flashes) is not fill on its own
FILL_S_MAX = 170
FILL_H_LO, FILL_H_HI = 18, 90
# A column is filled when this fraction of its rows is fill-coloured. Next
# to the anchor the TENSION label covers up to ~half of a column; further out
# the fill is solid (thin bright strokes there are overlay text, e.g. chat).
COL_FILL_MIN_LABEL = 0.35
LABEL_ZONE = 0.2  # fraction of the bar width from the anchor
COL_FILL_MIN = 0.7
# White shine over a full bar (any hue, low saturation; the label letters
# cover up to half a column there).
WHITE_S_MAX = 40
COL_WHITE_MIN = 0.5
# Gaps inside the fill (gear icon ~35 px, label letters), as a fraction of
# the bar width.
MAX_GAP = 0.085
# The first filled column may be this far from the anchor.
START_SLACK = 0.19
# Inside the run at least this fraction of the columns must be filled.
MIN_DENSITY = 0.7
# The full bar shows a gear over its outer end: a fill reaching this far is 1.
FULL_SNAP = 0.965

# finalize_tension: overlay ceiling detection.  A ceiling is a fill end c
# (+-CEIL_TOL) that one side's readings sit at for at least CEIL_MIN_S
# seconds in all, with (almost) none of its readings above it (at most
# CEIL_MAX_ABOVE of them: flashes), while the other side's bar is read above
# c for at least as long (the gauge itself can be read there).  The amount is
# absolute, not a share of the file: a cap is a cap however rarely a player
# reaches it.  A player's own highest tension held still is shorter (Tappa #4
# 05_UF P2: 1.7 s at 0.70; the new VOD's logo cap: 3.7-9 s per file).
CEIL_TOL = 0.006  # +-3 px
CEIL_MIN_S = 2.0
CEIL_MAX_ABOVE = 0.002
SMOOTH_WINDOW = 5


@dataclass
class TensionReading:
    pct: float  # 0..1
    first: int  # first filled column, px from the anchor (-1 = none)
    density: float


def measure_bar(bar_bgr: np.ndarray, anchor_right: bool) -> TensionReading:
    """Tension fill of one bar crop (ROI as in DEFAULT_BARS)."""
    w = bar_bgr.shape[1]
    core = bar_bgr[EDGE_ROWS:-EDGE_ROWS] if bar_bgr.shape[0] > 2 * EDGE_ROWS + 2 else bar_bgr
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fill = (v >= FILL_V_MIN) & (s >= FILL_S_MIN) & (s <= FILL_S_MAX) & (h >= FILL_H_LO) & (h <= FILL_H_HI)
    ratio = fill.mean(axis=0)
    white = ((v >= FILL_V_MIN) & (s <= WHITE_S_MAX)).mean(axis=0) >= COL_WHITE_MIN
    if anchor_right:
        ratio = ratio[::-1]
        white = white[::-1]
    cols = ratio >= COL_FILL_MIN
    label = int(LABEL_ZONE * w)
    cols[:label] = ratio[:label] >= COL_FILL_MIN_LABEL
    # the fill starts at the anchor; next to it a full bar shines white
    start = np.flatnonzero(cols | white)
    if len(start) == 0 or start[0] > START_SLACK * w:
        return TensionReading(0.0, -1, 0.0)
    first = int(start[0])
    max_gap = MAX_GAP * w
    last = first
    reached = False  # a coloured fill column was reached
    for d in np.flatnonzero(cols):
        if d < first:
            continue
        # white columns (the shine across a full bar) do not open a gap, but
        # never end the fill either (a white flash alone is not tension)
        gap = d - last - int(np.count_nonzero(white[last + 1:d]))
        if gap > max_gap:
            break
        last = int(d)
        reached = True
    if not reached:
        return TensionReading(0.0, first, 0.0)
    cols = cols | white
    n_filled = int(np.count_nonzero(cols[first:last + 1]))
    density = n_filled / (last - first + 1)
    if density < MIN_DENSITY:
        return TensionReading(0.0, first, density)
    pct = (last + 1) / w
    if pct >= FULL_SNAP:
        pct = 1.0
    return TensionReading(float(min(1.0, pct)), first, density)


class TensionExtractor(BaseExtractor):
    """Extracts tension gauge percentage for P1 and P2 (absolute, 0-1)."""

    ROIS = ("p1_tension_bar", "p2_tension_bar")

    @property
    def required_rois(self) -> list[str]:
        return list(self.ROIS)

    def read(self, frame_bgr: np.ndarray, rois: dict | None = None) -> tuple[TensionReading, TensionReading]:
        out = []
        for name in self.ROIS:
            crop = (rois or {}).get(name)
            if crop is None or not getattr(crop, "size", 0):
                x1, y1, x2, y2 = DEFAULT_BARS[name]
                crop = frame_bgr[y1:y2, x1:x2]
            out.append(measure_bar(crop, anchor_right=name.startswith("p1")))
        return out[0], out[1]

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        r1, r2 = self.read(ctx.frame_bgr, ctx.rois)
        # The gauge belongs to the HUD: an empty bar only means 0 while the
        # HUD (the round timer) is on screen; a fill with no timer is only
        # believed when the other bar shows a fill too (timer misread).
        timer_on = data.timer_value is not None
        both = r1.pct > 0 and r2.pct > 0
        for attr, r in (("p1_tension_pct", r1), ("p2_tension_pct", r2)):
            if timer_on or both:
                setattr(data, attr, r.pct)

    def reset_calibration(self) -> None:
        """Kept for API compatibility; the extractor is stateless."""


def _ceiling(vals: np.ndarray, other: np.ndarray, min_count: int) -> float | None:
    """A fixed fill end that one side's readings (``vals``) pile up at and
    never pass while the other side's bar (``other``) is read beyond it: a
    stream overlay (logo, chat, webcam) covers the bar there.  ``min_count``
    readings make CEIL_MIN_S seconds."""
    nz = vals[(vals > 0) & (vals < FULL_SNAP)]
    if len(nz) < min_count:
        return None
    # From the top down: the highest value held long enough, with (almost)
    # nothing above it.  Lower values only have more readings above them.
    for c in np.unique(np.round(nz, 3))[::-1]:
        if np.count_nonzero(vals > c + CEIL_TOL) > CEIL_MAX_ABOVE * len(vals):
            return None
        if (np.count_nonzero(np.abs(nz - c) <= CEIL_TOL) >= min_count
                and np.count_nonzero(other > c + CEIL_TOL) >= min_count):
            return float(c)
    return None


def finalize_tension(frames: list[FrameData]) -> dict[str, float | None]:
    """Replay-level clean-up of the tension readings, in place.

    1. Overlay ceiling: if one side's readings pile up at a fixed value that
       is never exceeded while the other bar is read beyond it (a stream
       overlay hides the bar there, see _ceiling), readings at that value
       are only a lower bound and become None.
    2. Median of SMOOTH_WINDOW consecutive readings (screen flashes, hit
       sparks); None stays None.

    Works from the per-frame readings kept in ``p1/p2_tension_raw`` (set
    from the current values on the first run), so it can be re-run (e.g. by
    scripts/reclean_replay.py) with the same result.
    Returns the ceiling found per side (None = bar fully visible).
    """
    raw = {}
    for attr in ("p1_tension_pct", "p2_tension_pct"):
        key = attr.replace("_pct", "_raw")
        for f in frames:
            if getattr(f, key) is None:
                setattr(f, key, getattr(f, attr))
        raw[attr] = np.array([getattr(f, key) if getattr(f, key) is not None else np.nan for f in frames],
                             dtype=float)
    ts = np.array([f.timestamp_ms for f in frames], dtype=float)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1000.0 / 30
    min_count = max(1, int(round(CEIL_MIN_S * 1000.0 / max(dt, 1e-6))))
    ceilings: dict[str, float | None] = {}
    for attr, other_attr in (("p1_tension_pct", "p2_tension_pct"), ("p2_tension_pct", "p1_tension_pct")):
        vals = raw[attr].copy()
        other = raw[other_attr]
        valid = ~np.isnan(vals)
        c = _ceiling(vals[valid], other[~np.isnan(other)], min_count) if valid.any() else None
        ceilings[attr] = c
        if c is not None:
            vals[valid & (vals >= c - CEIL_TOL) & (vals < FULL_SNAP)] = np.nan
        out = vals.copy()
        half = SMOOTH_WINDOW // 2
        for i in range(len(vals)):
            if np.isnan(vals[i]):
                continue
            win = vals[max(0, i - half): i + half + 1]
            win = win[~np.isnan(win)]
            out[i] = float(np.median(win))
        for f, v in zip(frames, out):
            setattr(f, attr, None if np.isnan(v) else float(v))
    return ceilings
