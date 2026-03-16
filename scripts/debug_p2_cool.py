"""Diagnose P2 cool detection flickering at specific frames."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO = str(ROOT / "data" / "sets" / "04_UF_GuessingGame_vs_Ricbionicle.mp4")


def analyze_p2_cool(frame: np.ndarray, frame_idx: int, fps: float):
    """Show detailed P2 cool detection at a single frame."""
    ts = frame_idx / fps

    # Band LOW (73-97) — the active band for sets/ overlay
    y1, y2 = 73, 97
    x1, x2 = 1220, 1720  # P2 range

    strip = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    is_blue = (h_ch >= 90) & (h_ch <= 145) & (s_ch > 40)
    filled = is_blue & (v_ch > 80)

    col_ratio = np.mean(filled, axis=0)

    # Different thresholds
    filled_025 = col_ratio > 0.25
    filled_035 = col_ratio > 0.35

    # Longest runs
    def longest_run(arr):
        best = cur = 0
        for v in arr:
            if v:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    run_025 = longest_run(filled_025)
    run_035 = longest_run(filled_035)

    # Also check warm detection
    warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)
    warm_ratio = np.mean(warm, axis=0)
    warm_filled = warm_ratio > 0.35
    warm_run = longest_run(warm_filled)

    # Column stats — show how many cool pixels per column in the region
    cool_per_col = np.sum(filled, axis=0)  # count per column
    band_height = y2 - y1

    # Find the bar region (first and last columns with any cool pixels)
    cool_cols = np.where(cool_per_col > 0)[0]
    if len(cool_cols) > 0:
        bar_start = cool_cols[0]
        bar_end = cool_cols[-1]
        # Stats within the bar region
        bar_ratios = col_ratio[bar_start:bar_end+1]
        bar_min = float(np.min(bar_ratios)) if len(bar_ratios) > 0 else 0
        bar_mean = float(np.mean(bar_ratios)) if len(bar_ratios) > 0 else 0
        bar_max = float(np.max(bar_ratios)) if len(bar_ratios) > 0 else 0
        # Count how many columns in bar region pass each threshold
        pass_025 = int(np.sum(bar_ratios > 0.25))
        pass_035 = int(np.sum(bar_ratios > 0.35))
        bar_width = bar_end - bar_start + 1
    else:
        bar_start = bar_end = bar_width = 0
        bar_min = bar_mean = bar_max = 0.0
        pass_025 = pass_035 = 0

    print(f"  t={ts:6.1f}s f={frame_idx:4d}: "
          f"cool_run@0.25={run_025:3d}  cool_run@0.35={run_035:3d}  "
          f"warm_run={warm_run:3d}  "
          f"bar_cols={bar_width:3d} [{bar_start}-{bar_end}]  "
          f"ratio min/mean/max={bar_min:.2f}/{bar_mean:.2f}/{bar_max:.2f}  "
          f"pass@0.25={pass_025}  pass@0.35={pass_035}")

    # Also check: what fraction of the band is the actual blue bar?
    # Look at row-wise blue density to see which rows have blue
    row_blue = np.mean(is_blue, axis=1)
    row_filled = np.mean(filled, axis=1)
    for row_idx in range(band_height):
        abs_y = y1 + row_idx
        if row_blue[row_idx] > 0.05 or row_filled[row_idx] > 0.05:
            print(f"    row y={abs_y}: blue_density={row_blue[row_idx]:.3f}  "
                  f"filled_density={row_filled[row_idx]:.3f}  "
                  f"mean_H={np.mean(h_ch[row_idx]):.0f}  "
                  f"mean_S={np.mean(s_ch[row_idx]):.0f}  "
                  f"mean_V={np.mean(v_ch[row_idx]):.0f}")


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check frames around t=75s where P2 drops to 0, and t=70s where it reads 1.0
    # Also check t=46s (start of game 2) and t=50s
    targets_s = [46, 48, 50, 55, 60, 65, 70, 72, 73, 74, 75, 76, 78, 80, 85, 90]

    for target_s in targets_s:
        target_frame = int(target_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if ok:
            analyze_p2_cool(frame, target_frame, fps)

    # Also check 3 consecutive frames at t=74-75s to see frame-to-frame variation
    print("\n=== Consecutive frames around t=74s ===")
    base_frame = int(74 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame)
    for i in range(6):
        ok, frame = cap.read()
        if ok:
            analyze_p2_cool(frame, base_frame + i, fps)

    cap.release()


if __name__ == "__main__":
    main()
