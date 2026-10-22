"""Screen-space camera model for 1920x1080 GGST recordings, from the logged positions only.

- zoom: K px per game unit/1000 (~1.47; detector calibration: median 1.42, varies ~1.35-1.6);
- the camera centre follows the players' midpoint with an exponential lag (alpha 0.1 / frame);
- the camera is clamped by the stage walls: the screen edge shows ~EDGE units beyond the wall.
  Walls move after a wallbreak (stage sections, ~3040 units wide), so they are estimated per
  section (split where the positions jump) from the positions players are pinned at.
"""
from __future__ import annotations

import numpy as np

K = 1.47
ALPHA = 0.1
EDGE = 82.0
WIDTH = 3040.0
JUMP = 600.0


def sections(x: np.ndarray) -> list[tuple[int, int]]:
    out, s = [], 0
    for i in range(1, len(x)):
        if np.abs(x[i] - x[i - 1]).max() > JUMP:
            out.append((s, i))
            s = i
    out.append((s, len(x)))
    return out


def walls(xs: np.ndarray, pin=20, tol=12.0):
    """(left, right) wall of one stage section or None where unknown."""
    v = xs[np.abs(xs) < 15000]
    if len(v) < 60:
        return None, None
    lo, hi = v.min(), v.max()
    L = lo if (v <= lo + tol).sum() >= pin else None
    R = hi if (v >= hi - tol).sum() >= pin else None
    if L is None and R is not None:
        L = R - WIDTH if lo >= R - WIDTH - tol else None
    if R is None and L is not None:
        R = L + WIDTH if hi <= L + WIDTH + tol else None
    return L, R


def camera(log: dict) -> np.ndarray:
    """Camera centre (game units/1000) per log row."""
    x = np.stack([log["p1_x"], log["p2_x"]], 1) / 1000.0
    mid = x.mean(1)
    cam = np.empty_like(mid)
    for s, e in sections(x):
        L, R = walls(x[s:e].ravel())
        c = mid[s]
        lo = -np.inf if L is None else L - EDGE + 960 / K
        hi = np.inf if R is None else R + EDGE - 960 / K
        for i in range(s, e):
            c = c + ALPHA * (mid[i] - c)
            cam[i] = min(max(c, lo), hi) if lo <= hi else (lo + hi) / 2
    return cam


def to_px(log: dict, cam: np.ndarray, row: int, side: int) -> tuple[float, float]:
    """(x, y) pixel estimate of a player's body centre."""
    px = 960 + K * (log[f"p{side}_x"][row] / 1000 - cam[row])
    py = 960 - K * 0.6 * log[f"p{side}_y"][row] / 1000 - 250
    return px, py
