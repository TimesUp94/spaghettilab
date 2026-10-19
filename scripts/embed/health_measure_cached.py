"""Run the per-frame HealthBarExtractor measurement over a strip cache (cache_bar_strips.py).

Writes an .npz with per-frame fill/pending/consistency/occluded/ambiguity for both sides, so the
temporal cleaner can be iterated on without re-decoding the video.  Primary band only (the cache
has rows 100-150).
Usage: .venv/Scripts/python.exe scripts/embed/health_measure_cached.py STRIPS.npy OUT.npz
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
from replanal.models import Side  # noqa: E402

CY1, CX1 = 100, 140


def main():
    S = np.load(sys.argv[1], mmap_mode="r")
    ext = HealthBarExtractor()
    y1, y2 = ext._PRIMARY_BAND
    a1, a2 = y1 + ext._ABOVE[0], y1 + ext._ABOVE[1]
    b1, b2 = y2 + ext._BELOW[0], y2 + ext._BELOW[1]
    b2 = min(b2, CY1 + S.shape[1])
    n = len(S)
    keys = ["fill", "pending", "cons", "occ", "amb"]
    out = {f"p{s}_{k}": np.zeros(n, np.float32) for s in (1, 2) for k in keys}
    t0 = time.time()
    pad = np.zeros((S.shape[1], 1920, 3), np.uint8)
    for i in range(n):
        pad[:, CX1:CX1 + S.shape[2]] = S[i]
        hsv = cv2.cvtColor(pad, cv2.COLOR_BGR2HSV)
        for s, side in ((1, Side.P1), (2, Side.P2)):
            m = ext.measure_bar(hsv[y1 - CY1:y2 - CY1], side, hsv_above=hsv[a1 - CY1:a2 - CY1],
                                hsv_below=hsv[b1 - CY1:b2 - CY1])
            out[f"p{s}_fill"][i] = m.fill_px
            out[f"p{s}_pending"][i] = m.pending_px
            out[f"p{s}_cons"][i] = m.consistency
            out[f"p{s}_occ"][i] = m.occluded
            out[f"p{s}_amb"][i] = m.ambiguity
    np.savez_compressed(sys.argv[2], **out)
    print(f"{n} frames in {time.time() - t0:.1f}s -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
