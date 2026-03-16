"""Dump calibration candidate scores at specific frames."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replanal.extractors.health import HealthBarExtractor

VIDEO = str(ROOT / "data" / "sets" / "04_UF_GuessingGame_vs_Ricbionicle.mp4")


def dump_candidates(frame: np.ndarray, label: str):
    """Score every candidate Y band and print top results."""
    ext = HealthBarExtractor()
    w = ext._CAL_WINDOW
    Y1, Y2 = ext.Y_SEARCH_MIN, ext.Y_SEARCH_MAX
    P1_X1, P1_X2 = ext._P1_CAL_X1, ext._P1_CAL_X2
    p1_cal_width = P1_X2 - P1_X1
    p2_cal_width = 500  # 1200-1700

    search = frame[Y1:Y2]
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)

    # P1 warm row density
    p1_hsv = hsv[:, P1_X1:P1_X2]
    h_ch, s_ch, v_ch = p1_hsv[:, :, 0], p1_hsv[:, :, 1], p1_hsv[:, :, 2]
    warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)
    row_density = np.mean(warm, axis=1).astype(float)

    results = []
    for y_off in range(0, len(row_density) - w):
        band_density = float(np.mean(row_density[y_off:y_off + w]))
        if band_density < 0.10:
            continue

        abs_y1 = Y1 + y_off

        # P1 longest warm run
        p1_strip = hsv[y_off:y_off + w, P1_X1:P1_X2]
        p1_run = ext._measure_warm_fill(p1_strip, 0, p1_cal_width)

        # P2 longest runs
        p2_strip = hsv[y_off:y_off + w, 1200:1700]
        p2_warm_run = ext._measure_warm_fill(p2_strip, 0, p2_cal_width)
        p2_cool_run = ext._measure_cool_fill(p2_strip, 0, p2_cal_width)
        p2_run = max(p2_warm_run, p2_cool_run)

        p1_norm = p1_run / p1_cal_width
        p2_norm = p2_run / p2_cal_width
        score = p1_norm * 0.3 + p2_norm * 0.7

        results.append((abs_y1, score, p1_run, p2_warm_run, p2_cool_run, band_density))

    results.sort(key=lambda x: -x[1])
    print(f"\n=== {label} ===")
    print(f"{'Y':>4}  {'Score':>6}  {'P1run':>5}  {'P2warm':>6}  {'P2cool':>6}  {'Density':>7}")
    for y, sc, p1r, p2wr, p2cr, dens in results[:15]:
        print(f"{y:>4}  {sc:>6.3f}  {p1r:>5}  {p2wr:>6}  {p2cr:>6}  {dens:>7.3f}")


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for target_s in [0, 14, 20, 45, 60, 90, 120]:
        target_frame = int(target_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if ok:
            dump_candidates(frame, f"t={target_s}s (frame {target_frame})")

    cap.release()


if __name__ == "__main__":
    main()
