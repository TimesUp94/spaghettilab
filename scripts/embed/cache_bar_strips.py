"""Cache the HUD health-bar strip (rows 100-150, x 140-1800) of every frame to a .npy memmap.

Lets health-extractor experiments run in seconds instead of re-decoding the video.
Usage: .venv/Scripts/python.exe scripts/embed/cache_bar_strips.py VIDEO OUT.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from replanal.video import get_video_info, iter_frames  # noqa: E402

Y1, Y2, X1, X2 = 100, 150, 140, 1800


def main():
    video, out = sys.argv[1], sys.argv[2]
    n = get_video_info(video)["frame_count"]
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(n, Y2 - Y1, X2 - X1, 3))
    k = 0
    for fn, _ts, frame in iter_frames(video):
        if fn >= n:
            break
        mm[fn] = frame[Y1:Y2, X1:X2]
        k = fn + 1
    mm.flush()
    print(f"cached {k}/{n} frames -> {out}")


if __name__ == "__main__":
    main()
