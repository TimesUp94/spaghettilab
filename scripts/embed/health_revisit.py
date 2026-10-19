"""Revisit the baseline failure moments: bar strips with before (exp17) and after (exp17_health)
stored HP drawn as edge markers.  Red line = before, green line = after.
Usage: python health_revisit.py BEFORE.db AFTER.db STRIPS.npy OUT_DIR"""
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

RID = "17_DIHLScarecrow_vs_QSalkazko"
MOMENTS = {"m1": [3555, 3561, 3570], "m2": [5483, 5494, 5505], "m3": [885, 896, 907], "m4": [10488, 10500, 10520],
           "m5": [2720, 2760, 2800], "m6": [5994, 6040, 6070], "m7": [11220, 11400, 11520], "m8": [4770, 4800, 4830],
           "m9": [7923, 7935, 7960]}
P1X, P2X, W = (149, 868), (1052, 1771), 719


def load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    d = {}
    for f, a, b in c.execute("select frame_number,p1_health_pct,p2_health_pct from frame_data where replay_id=?", (RID,)):
        d[f] = (a, b)
    return d


def main():
    before, after = load(sys.argv[1]), load(sys.argv[2])
    S = np.load(sys.argv[3], mmap_mode="r")
    out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
    for name, frames in MOMENTS.items():
        rows = []
        for f in frames:
            im = np.ascontiguousarray(S[f]).copy()
            txt = f"f{f} t={f/30:.1f}s"
            for label, d, col in (("before", before, (0, 0, 255)), ("after", after, (0, 255, 0))):
                a, b = d.get(f, (None, None))
                txt += f" | {label} P1={'-' if a is None else f'{a:.2f}'} P2={'-' if b is None else f'{b:.2f}'}"
                for v, (x1, x2), p1 in ((a, P1X, True), (b, P2X, False)):
                    if v is None:
                        continue
                    x = int(x2 - v * W) if p1 else int(x1 + v * W)
                    y0, y1 = (0, 25) if label == "before" else (25, 50)
                    cv2.line(im, (x - 140, y0), (x - 140, y1), col, 3)
            lab = np.zeros((20, im.shape[1], 3), np.uint8)
            cv2.putText(lab, txt, (4, 15), 0, 0.5, (255, 255, 255), 1)
            rows += [lab, im]
        cv2.imwrite(str(out / f"{name}_revisit.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
