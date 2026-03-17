"""Debug: visualize what the health extractor sees at specific frames.

Saves debug images showing HSV masks and column fill ratios
to help diagnose why bar width drops to ~163px at t=66s.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

VIDEO = str(ROOT / "data" / "sets" / "01_WRI_Slycoops_vs_Leftover.mp4")
DEBUG_DIR = ROOT / "output" / "debug_health"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Same constants as HealthBarExtractor
P1_X1, P1_X2 = 200, 900
P2_X1, P2_X2 = 1040, 1800
FULL_Y1 = 88
FULL_Y2 = 170
PRIMARY_BAND = (112, 142)
ALT_BANDS = [(88, 112), (142, 170)]

# Frames to inspect: good reading at t≈65s, bad at t≈66s
# Every frame from 3030-3110 to see P2 anchor behavior
TARGET_FRAMES = list(range(3030, 3110))


def longest_run(is_filled: np.ndarray) -> tuple[int, int, int]:
    """Returns (length, start, end) of longest True run."""
    best = best_start = best_end = 0
    current = start = 0
    for i, val in enumerate(is_filled):
        if val:
            if current == 0:
                start = i
            current += 1
            if current > best:
                best = current
                best_start = start
                best_end = i + 1
        else:
            current = 0
    return best, best_start, best_end


def analyze_frame(frame_bgr: np.ndarray, frame_num: int, ts: float):
    full_strip = frame_bgr[FULL_Y1:FULL_Y2]
    full_hsv = cv2.cvtColor(full_strip, cv2.COLOR_BGR2HSV)

    print(f"\n=== Frame {frame_num} (t={ts:.2f}s) ===")

    for band_name, (y1, y2) in [("PRIMARY", PRIMARY_BAND)] + [(f"ALT_{i}", b) for i, b in enumerate(ALT_BANDS)]:
        hsv = full_hsv[y1 - FULL_Y1 : y2 - FULL_Y1]

        for side, x1, x2 in [("P1", P1_X1, P1_X2), ("P2", P2_X1, P2_X2)]:
            h_ch = hsv[:, x1:x2, 0]
            s_ch = hsv[:, x1:x2, 1]
            v_ch = hsv[:, x1:x2, 2]

            # Warm mask
            warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)
            warm_ratio = np.mean(warm, axis=0)
            warm_filled = warm_ratio > 0.35
            warm_run, warm_s, warm_e = longest_run(warm_filled)

            # Cool mask
            cool = (h_ch >= 90) & (h_ch <= 145) & (s_ch > 40) & (v_ch > 80)
            cool_ratio = np.mean(cool, axis=0)
            cool_filled = cool_ratio > 0.35
            cool_run, cool_s, cool_e = longest_run(cool_filled)

            # Yellow-green mask
            yg = (h_ch >= 25) & (h_ch <= 70) & (s_ch > 40) & (v_ch > 100)
            yg_ratio = np.mean(yg, axis=0)
            yg_filled = yg_ratio > 0.35
            yg_run, yg_s, yg_e = longest_run(yg_filled)

            best = max(warm_run, cool_run, yg_run)
            label = "warm" if warm_run == best else ("cool" if cool_run == best else "yg")
            print(f"  {band_name} {side}: warm={warm_run}[{warm_s}:{warm_e}] "
                  f"cool={cool_run}[{cool_s}:{cool_e}] "
                  f"yg={yg_run}[{yg_s}:{yg_e}] -> best={best} ({label})")

    # Save annotated frame with ROI rectangles
    vis = frame_bgr.copy()
    cv2.rectangle(vis, (P1_X1, PRIMARY_BAND[0]), (P1_X2, PRIMARY_BAND[1]), (0, 255, 0), 2)
    cv2.rectangle(vis, (P2_X1, PRIMARY_BAND[0]), (P2_X2, PRIMARY_BAND[1]), (0, 255, 0), 2)
    for y1, y2 in ALT_BANDS:
        cv2.rectangle(vis, (P1_X1, y1), (P1_X2, y2), (255, 255, 0), 1)
        cv2.rectangle(vis, (P2_X1, y1), (P2_X2, y2), (255, 255, 0), 1)
    cv2.putText(vis, f"f={frame_num} t={ts:.2f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Save the ROI close-ups
    p1_roi = frame_bgr[FULL_Y1:FULL_Y2, P1_X1:P1_X2]
    p2_roi = frame_bgr[FULL_Y1:FULL_Y2, P2_X1:P2_X2]

    # Save HSV masks for primary band
    hsv_primary = full_hsv[PRIMARY_BAND[0] - FULL_Y1 : PRIMARY_BAND[1] - FULL_Y1]
    for side, x1, x2 in [("P1", P1_X1, P1_X2), ("P2", P2_X1, P2_X2)]:
        h_ch = hsv_primary[:, x1:x2, 0]
        s_ch = hsv_primary[:, x1:x2, 1]
        v_ch = hsv_primary[:, x1:x2, 2]

        warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)
        yg = (h_ch >= 25) & (h_ch <= 70) & (s_ch > 40) & (v_ch > 100)

        # Upsample masks for visibility
        warm_img = (warm.astype(np.uint8) * 255)
        yg_img = (yg.astype(np.uint8) * 255)
        warm_big = cv2.resize(warm_img, (warm_img.shape[1], 100), interpolation=cv2.INTER_NEAREST)
        yg_big = cv2.resize(yg_img, (yg_img.shape[1], 100), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(DEBUG_DIR / f"f{frame_num}_{side}_warm_mask.png"), warm_big)
        cv2.imwrite(str(DEBUG_DIR / f"f{frame_num}_{side}_yg_mask.png"), yg_big)

    # Full frame and cropped ROIs
    crop_top = frame_bgr[0:200, :]
    cv2.imwrite(str(DEBUG_DIR / f"f{frame_num}_top.png"), crop_top)
    cv2.imwrite(str(DEBUG_DIR / f"f{frame_num}_frame.png"), vis)
    p1_big = cv2.resize(p1_roi, (p1_roi.shape[1], 200), interpolation=cv2.INTER_NEAREST)
    p2_big = cv2.resize(p2_roi, (p2_roi.shape[1], 200), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(DEBUG_DIR / f"f{frame_num}_P1_strip.png"), p1_big)
    cv2.imwrite(str(DEBUG_DIR / f"f{frame_num}_P2_strip.png"), p2_big)


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {VIDEO}")
    print(f"FPS: {fps}")

    target_set = set(TARGET_FRAMES)
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in target_set:
            ts = frame_num / fps
            analyze_frame(frame, frame_num, ts)
            target_set.discard(frame_num)
            if not target_set:
                break
        frame_num += 1

    cap.release()
    print(f"\nDebug images saved to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
