"""Contact sheets for checking combo-counter output by eye.

  combos  <combos.json> <out.png> [idx ...]  start / middle / end frame per combo
  frames  <out.png> frame [frame ...]          full frames with ROIs + extractor reading

Usage:
    .venv/Scripts/python.exe scripts/embed/combo_review.py combos output/exp17_combo/X_combos.json out.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from replanal.config import PipelineConfig  # noqa: E402
from replanal.extractors.combo import BIG_ROI, WORD_ROI, ComboCounterExtractor  # noqa: E402
from replanal.models import FrameContext, FrameData, Side  # noqa: E402

VIDEO = ROOT / "ggstsets" / "17_DIHLScarecrow_vs_QSalkazko.mp4"


def grab(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, img = cap.read()
    return img if ok else np.zeros((1080, 1920, 3), np.uint8)


def annotate(img, cfg, ext, n):
    rois = {k: cfg.rois[k].crop(img) for k in ext.required_rois}
    d = FrameData(n, 0.0)
    ext.extract(FrameContext("", n, 0.0, img, rois), d)
    vis = img.copy()
    for side in (Side.P1, Side.P2):
        for name, col in ((WORD_ROI[side], (0, 255, 255)), (BIG_ROI.get(side), (255, 0, 255))):
            if name is None:
                continue
            r = cfg.rois[name]
            cv2.rectangle(vis, (r.x1, r.y1), (r.x2, r.y2), col, 3)
        rd = ext.last.get(side)
        if rd is not None:
            r = cfg.rois[WORD_ROI[side] if rd.mode == "word" else BIG_ROI[side]]
            ds = 4 if rd.mode == "word" else 8
            x, y = r.x1 + rd.x * ds, r.y1 + rd.y * ds
            cv2.circle(vis, (x, y), 14, (0, 0, 255), -1)
    txt = f"f{n} {d.combo_side.name if d.combo_side else '-'} {d.combo_count if d.combo_count else ''}"
    cv2.putText(vis, txt, (20, 1060), 0, 2.0, (0, 0, 0), 10)
    cv2.putText(vis, txt, (20, 1060), 0, 2.0, (0, 255, 0), 4)
    return vis


def main():
    mode = sys.argv[1]
    cap = cv2.VideoCapture(str(VIDEO))
    if mode == "combos":
        combos = json.loads(Path(sys.argv[2]).read_text())
        out = sys.argv[3]
        idx = [int(i) for i in sys.argv[4:]] or list(range(len(combos)))
        rows = []
        for i in idx:
            c = combos[i]
            a, b = c["frame_start"], c["frame_end"]
            tiles = []
            for n in (a, (a + b) // 2, b, b + 20):
                img = grab(cap, n)
                x0 = 0 if c["attacker"] == "P1" else 960
                t = cv2.resize(img[0:900, x0 : x0 + 960], (320, 300), interpolation=cv2.INTER_AREA)
                cv2.putText(t, f"{n}", (4, 290), 0, 0.6, (0, 255, 255), 2)
                tiles.append(t)
            # health strip of the full frame at start and after end
            hud = [cv2.resize(grab(cap, n)[100:150, 0:1920], (480, 13)) for n in (max(0, a - 15), b + 20)]
            label = np.zeros((300, 330, 3), np.uint8)
            dmg = "?" if c["damage"] is None else f"{c['damage']:.2f}"
            for k, line in enumerate([f"#{i} {c['attacker']}", f"{c['start_s']:.1f}-{c['end_s']:.1f}s", f"hits {c['max_hits']}", f"dmg {dmg}"]):
                cv2.putText(label, line, (8, 40 + 45 * k), 0, 1.0, (255, 255, 255), 2)
            label[220:233, 0:330] = hud[0][:, :330]
            label[250:263, 0:330] = hud[1][:, :330]
            rows.append(np.hstack([label] + tiles))
        cv2.imwrite(out, np.vstack(rows))
    else:
        cfg = PipelineConfig.from_yaml(ROOT / "config" / "default.yaml")
        ext = ComboCounterExtractor(stride=1)
        out = sys.argv[2]
        tiles = [cv2.resize(annotate(grab(cap, int(n)), cfg, ext, int(n)), (640, 360), interpolation=cv2.INTER_AREA) for n in sys.argv[3:]]
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(out, np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]))
    print(out)


if __name__ == "__main__":
    main()
