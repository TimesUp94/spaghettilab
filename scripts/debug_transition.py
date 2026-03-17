"""Debug: trace health extractor state during round 2→3 transition.

Prints frame-by-frame info for frames 2900-3200 (~96.7s-106.7s)
to understand why the gap-based reset isn't firing properly.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from collections import deque

VIDEO = str(ROOT / "data" / "sets" / "01_WRI_Slycoops_vs_Leftover.mp4")

# Same constants as HealthBarExtractor
P1_X1, P1_X2 = 200, 900
P2_X1, P2_X2 = 1040, 1800
PRIMARY_BAND = (112, 142)
FULL_Y1, FULL_Y2 = 88, 170
MIN_RUN = 50
P1_ANCHOR_END_MIN = 640
P2_ANCHOR_START_MAX = 40

def longest_run(is_filled):
    best = best_start = 0
    current = start = 0
    for i, val in enumerate(is_filled):
        if val:
            if current == 0: start = i
            current += 1
            if current > best:
                best = current
                best_start = start
        else:
            current = 0
    return best, best_start

def measure_warm(hsv, x1, x2):
    h = hsv[:, x1:x2, 0]; s = hsv[:, x1:x2, 1]; v = hsv[:, x1:x2, 2]
    warm = ((h < 30) | (h > 150)) & (s > 30) & (v > 80)
    return longest_run(np.mean(warm, axis=0) > 0.35)

def measure_cool(hsv, x1, x2):
    h = hsv[:, x1:x2, 0]; s = hsv[:, x1:x2, 1]; v = hsv[:, x1:x2, 2]
    filled = (h >= 90) & (h <= 145) & (s > 40) & (v > 80)
    return longest_run(np.mean(filled, axis=0) > 0.35)

def measure_yg(hsv, x1, x2):
    h = hsv[:, x1:x2, 0]; s = hsv[:, x1:x2, 1]; v = hsv[:, x1:x2, 2]
    filled = (h >= 25) & (h <= 70) & (s > 40) & (v > 100)
    return longest_run(np.mean(filled, axis=0) > 0.35)

def anchor_filter_p1(candidates):
    best_w, best_s = 0, 0
    for w, s in candidates:
        if w >= MIN_RUN and (s + w) < P1_ANCHOR_END_MIN: continue
        if w > best_w: best_w, best_s = w, s
    return best_w

def anchor_filter_p2(candidates):
    best_w, best_s = 0, 0
    for w, s in candidates:
        if w >= MIN_RUN and s > P2_ANCHOR_START_MAX: continue
        if w > best_w: best_w, best_s = w, s
    return best_w

# Frames 2880 to 3200 (~96s to 107s at 30fps)
TARGET_START, TARGET_END = 2880, 3200

def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_num < TARGET_START:
            frame_num += 1
            continue
        if frame_num > TARGET_END:
            break

        ts = frame_num / fps
        full_strip = frame[FULL_Y1:FULL_Y2]
        full_hsv = cv2.cvtColor(full_strip, cv2.COLOR_BGR2HSV)
        hsv = full_hsv[PRIMARY_BAND[0] - FULL_Y1 : PRIMARY_BAND[1] - FULL_Y1]

        p1_w = anchor_filter_p1([measure_warm(hsv, P1_X1, P1_X2), measure_yg(hsv, P1_X1, P1_X2)])
        p2_w = anchor_filter_p2([measure_warm(hsv, P2_X1, P2_X2), measure_cool(hsv, P2_X1, P2_X2), measure_yg(hsv, P2_X1, P2_X2)])

        p1_read = "Y" if p1_w >= MIN_RUN else "N"
        p2_read = "Y" if p2_w >= MIN_RUN else "N"

        # Also check scene detector warm ratio
        bar_strip = frame[112:142]
        bar_hsv = cv2.cvtColor(bar_strip, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = bar_hsv[:,:,0], bar_hsv[:,:,1], bar_hsv[:,:,2]
        warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)
        p1_warm = float(np.mean(warm[:, P1_X1:P1_X2]))
        p2_warm = float(np.mean(warm[:, P2_X1:P2_X2]))
        scene_ok = p1_warm > 0.10 or p2_warm > 0.10

        if frame_num % 5 == 0 or not scene_ok or p1_w >= MIN_RUN:
            print(f"f={frame_num:5d} t={ts:6.2f}s  P1_w={p1_w:4d}({p1_read}) P2_w={p2_w:4d}({p2_read})  "
                  f"scene={'Y' if scene_ok else 'N'}  p1_warm={p1_warm:.3f} p2_warm={p2_warm:.3f}")

        frame_num += 1

    cap.release()

if __name__ == "__main__":
    main()
