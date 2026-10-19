"""Run the combo-counter matchers over a whole video in parallel (no thresholds).

Writes, per frame and side, the best and runner-up word matches, the
big-counter reading and the best two tens / ones glyph scores,
so thresholds can be tuned and readings checked without re-running the full
pipeline.  Output: output/embed_test/combo_counter/scan/<stem>_combo_raw.json

Usage:
    .venv/Scripts/python.exe scripts/embed/combo_scan.py [video] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from replanal.config import PipelineConfig  # noqa: E402
from replanal.extractors.combo import BIG_ROI, WORD_DS, WORD_ROI, ComboCounterExtractor, _gray_small  # noqa: E402

VIDEO = ROOT / "ggstsets" / "17_DIHLScarecrow_vs_QSalkazko.mp4"


def _top2(per):
    ranked = sorted(per.items(), key=lambda kv: -kv[1][0])[:2]
    return [[int(c), round(v[0], 4), int(v[1]), int(v[2])] for c, v in ranked]


def work(args):
    video, a, b = args
    cv2.setNumThreads(1)
    cfg = PipelineConfig.from_yaml(ROOT / "config" / "default.yaml")
    ext = ComboCounterExtractor()
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    out = []
    for fn in range(a, b):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[:2] != (1080, 1920):
            img = cv2.resize(img, (1920, 1080))
        rec = {"f": fn}
        for side in WORD_ROI:
            k = side.value
            (pw,) = ext.word.best(_gray_small(cfg.rois[WORD_ROI[side]].crop(img), WORD_DS))
            rec[f"w{k}"] = _top2(pw)
            r, dbg = ext.big.read(cfg.rois[BIG_ROI[side]].crop(img), side)
            rec[f"b{k}"] = r.count if r else None
            for part in ("tens", "ones"):
                ranked = sorted(dbg.get(part, {}).items(), key=lambda kv: -kv[1])[:2]
                rec[f"{part[0]}{k}"] = [[int(d), round(v, 4)] for d, v in ranked]
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=str(VIDEO))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    k = args.workers * 2
    chunks = [(args.video, i * n // k, (i + 1) * n // k) for i in range(k)]
    t0 = time.time()
    with Pool(args.workers) as p:
        res = p.map(work, chunks)
    recs = [r for c in res for r in c]
    out = Path(args.out) if args.out else ROOT / "output" / "embed_test" / "combo_counter" / "scan" / f"{Path(args.video).stem}_combo_raw.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs))
    print(f"{len(recs)} frames in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
