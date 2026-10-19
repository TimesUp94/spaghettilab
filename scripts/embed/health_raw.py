"""Dump RAW (unsmoothed, unclamped) per-frame health-bar fill widths for a video.

Re-uses HealthBarExtractor's colour-mode measurements on the primary Y band only, with
no median filter / spike rejection / drop-rate limiting, so baseline failures of the
smoothed extractor can be compared with what the pixels actually say.

Usage:
  .venv/Scripts/python.exe scripts/embed/health_raw.py VIDEO OUT.npz
  .venv/Scripts/python.exe scripts/embed/health_raw.py VIDEO OUT.npy --pending   (red pending-damage runs)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from replanal.extractors.health import HealthBarExtractor  # noqa: E402
from replanal.video import iter_frames  # noqa: E402


def main():
    video, out = sys.argv[1], Path(sys.argv[2])
    ext = HealthBarExtractor()
    y1, y2 = ext._PRIMARY_BAND
    cols = ["p1_warm", "p1_yg", "p2_warm", "p2_cool", "p2_yg"]
    rows_w, rows_s = [], []
    t0 = time.time()
    for fn, _ts, frame in iter_frames(video):
        hsv = cv2.cvtColor(frame[y1:y2], cv2.COLOR_BGR2HSV)
        m = [
            ext._measure_warm_fill(hsv, ext.P1_X1, ext.P1_X2),
            ext._measure_yellow_green_fill(hsv, ext.P1_X1, ext.P1_X2),
            ext._measure_warm_fill(hsv, ext.P2_X1, ext.P2_X2),
            ext._measure_cool_fill(hsv, ext.P2_X1, ext.P2_X2),
            ext._measure_yellow_green_fill(hsv, ext.P2_X1, ext.P2_X2),
        ]
        rows_w.append([w for w, _ in m])
        rows_s.append([s for _, s in m])
    W, S = np.array(rows_w, np.int32), np.array(rows_s, np.int32)
    # anchored best width (same anchor rules as the extractor, primary band only)
    p1 = np.zeros(len(W), np.int32)
    p2 = np.zeros(len(W), np.int32)
    for i in range(len(W)):
        for j in (0, 1):
            w, s = W[i, j], S[i, j]
            if w >= ext._MIN_RUN and s + w < ext._P1_ANCHOR_END_MIN:
                continue
            p1[i] = max(p1[i], w)
        for j in (2, 3, 4):
            w, s = W[i, j], S[i, j]
            if w >= ext._MIN_RUN and s > ext._P2_ANCHOR_START_MAX:
                continue
            p2[i] = max(p2[i], w)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, widths=W, starts=S, cols=np.array(cols), p1=p1, p2=p2,
                        p1_max=ext.P1_MAX_W, p2_max=ext.P2_MAX_W)
    print(f"{len(W)} frames in {time.time() - t0:.1f}s -> {out}")




# --------------------------------------------------------------------------- pending-damage
def pending_red_cols(hsv_band: np.ndarray, x1: int, x2: int) -> tuple[int, int]:
    """(longest run, start) of columns dominated by GGST's dark-red pending-damage segment.

    Measured on set 17: pending red is H<=6 or H>=170, S>=200, V 100-200; normal fill is
    pink (S 70-140, V>200) or low-HP orange (H~18, S~250, V~225), so they separate cleanly.
    """
    h = hsv_band[:, x1:x2, 0]
    s = hsv_band[:, x1:x2, 1]
    v = hsv_band[:, x1:x2, 2]
    red = ((h <= 6) | (h >= 170)) & (s >= 200) & (v >= 100) & (v <= 200)
    return HealthBarExtractor._longest_run(red.mean(axis=0) > 0.5)


def dump_pending(video: str, out: Path) -> None:
    ext = HealthBarExtractor()
    y1, y2 = ext._PRIMARY_BAND
    rows = []
    for _fn, _ts, frame in iter_frames(video):
        hsv = cv2.cvtColor(frame[y1:y2], cv2.COLOR_BGR2HSV)
        a = pending_red_cols(hsv, ext.P1_X1, ext.P1_X2)
        b = pending_red_cols(hsv, ext.P2_X1, ext.P2_X2)
        rows.append([a[0], a[1], b[0], b[1]])
    np.save(out, np.array(rows, np.int32))
    print(f"pending-red columns -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "--pending":
        dump_pending(sys.argv[1], Path(sys.argv[2]))
    else:
        main()
