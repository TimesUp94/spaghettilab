"""Build / evaluate the round-timer glyph templates.

Usage:
    python scripts/build_timer_templates.py LABELS.json CROPS_DIR [--out NPZ]
        [--holdout] [--eval-only NPZ]

LABELS.json (tests/data/timer/labels.json) is a list of {"id", "video", "frame", "label"} entries, label being
"NN" (clear), "NNt" (digit change / partly covered: NN or NN+1 acceptable),
"none" (no timer visible) or "unreadable" (timer present but not legible;
ignored in the scores).  CROPS_DIR/<id>.png holds the 130x85 frame crop at
x 895-1025, y 100-185 of the 1080p frame (the timer_digits ROI is the
[15:70, 25:105] part of it); --extract regenerates them from the videos.
CROPS_DIR may also be tests/data/timer/rois.npz (grey ROIs of every entry),
which is enough to rebuild and evaluate without the VODs.

--holdout trains on the training videos only and reports the held-out
videos, which is the honest accuracy estimate; without it the templates are
built from every clean sample and written to --out.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replanal.extractors import timer as T  # noqa: E402

CROP_Y, CROP_X = 100, 895
ROI_SLICE = (slice(15, 70), slice(25, 105))

# Held-out videos (never used to build the templates in --holdout mode):
# 4 overlay sets, 4 ggstsets VODs and one 60 fps non-overlay YouTube replay.
HOLDOUT = {
    "data/sets/03_UF_SlayThatDude_vs_Salkazko.mp4",
    "data/sets/06_UF_Jason_vs_Salkazko.mp4",
    "data/sets/09_UF_Hysy_vs_HLScarecrow.mp4",
    "data/sets/12_LSemi_Revill_vs_Salkazko.mp4",
    "ggstsets/04_QSlaythalDude_vs_Salkazko.mp4",
    "ggstsets/10_0Dona_vs_3UMAmusume.mp4",
    "ggstsets/16_4Revll_vs_4IDona.mp4",
    "ggstsets/22_AUMAmusume_vs_3DaveFennoy.mp4",
    "output/next_steps/non_overlay/videos/yt_PbsXLfP7C1M.mp4",
}


def parse(label: str):
    """-> (kind, value): kind in clean / trans / none / skip."""
    if label == "none":
        return "none", None
    if label == "unreadable":
        return "skip", None
    if label.endswith("t"):
        return "trans", int(label[:-1])
    return "clean", int(label)


def load(labels_path: Path, crops: Path):
    """[(entry, roi)] for every labelled entry with a crop.

    *crops* is either a directory of 130x85 frame crops (<id>.png) or the
    compact test-data file (tests/data/timer/rois.npz: grey timer_digits ROIs
    keyed by id).
    """
    entries = json.loads(labels_path.read_text())
    out = []
    if crops.suffix == ".npz":
        data = np.load(str(crops))
        rois = dict(zip(data["ids"].tolist(), data["rois"]))
        return [(e, rois[e["id"]]) for e in entries if e["id"] in rois]
    for e in entries:
        img = cv2.imread(str(crops / f"{e['id']}.png"))
        if img is None:
            continue
        out.append((e, img[ROI_SLICE]))
    return out


def extract_crops(labels_path: Path, crops: Path) -> None:
    crops.mkdir(parents=True, exist_ok=True)
    entries = json.loads(labels_path.read_text())
    by_video = collections.defaultdict(list)
    for e in entries:
        by_video[e["video"]].append(e)
    for video, es in by_video.items():
        cap = cv2.VideoCapture(str(ROOT / video))
        for e in es:
            cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame"])
            ok, f = cap.read()
            if ok:
                cv2.imwrite(str(crops / f"{e['id']}.png"), f[CROP_Y:CROP_Y + 85, CROP_X:CROP_X + 130])


def _norm(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    return (p - p.mean()) / (p.std() + 1e-6)


def train(samples, iters: int = 4):
    cells = dict(T.DEFAULT_CELLS)
    clean = [(T.prepare_roi(roi), v) for e, roi in samples
             for kind, v in [parse(e["label"])] if kind == "clean"]
    # Initial templates: patches at the nominal positions.
    glyphs = np.zeros((10, T.GLYPH_H, T.GLYPH_W), np.float32)
    for it in range(iters + 1):
        acc = collections.defaultdict(list)
        offs = {"tens": [], "ones": []}
        for padded, v in clean:
            for name, d in (("tens", v // 10), ("ones", v % 10)):
                y0, x0 = cells[name]
                if it == 0:
                    dy = dx = 0
                else:
                    if not glyphs[d].any():
                        continue
                    res = cv2.matchTemplate(T.cell_window(padded, y0, x0), glyphs[d], cv2.TM_CCOEFF_NORMED)
                    _, _, _, (bx, by) = cv2.minMaxLoc(res)
                    dy, dx = by - T.SEARCH, bx - T.SEARCH
                offs[name].append((dy, dx))
                acc[d].append(_norm(padded[y0 + dy:y0 + dy + T.GLYPH_H, x0 + dx:x0 + dx + T.GLYPH_W]))
        for d in range(10):
            if acc[d]:
                glyphs[d] = np.mean(acc[d], axis=0)
        if 0 < it < iters:  # re-centre the nominal boxes on the mean offset
            for name in cells:
                o = np.array(offs[name])
                cells[name] = (cells[name][0] + int(round(np.median(o[:, 0]))),
                               cells[name][1] + int(round(np.median(o[:, 1]))))
    counts = {d: len(acc[d]) for d in range(10)}
    return glyphs, cells, counts


def evaluate(ext, samples, show_wrong=True):
    res = collections.Counter()
    conf = {"tens": collections.Counter(), "ones": collections.Counter()}
    wrong = []
    dt = 0.0
    for e, roi in samples:
        kind, v = parse(e["label"])
        t0 = time.perf_counter()
        r = ext.read(roi)
        dt += time.perf_counter() - t0
        if kind == "skip":
            res["unreadable_" + ("none" if r is None else "read")] += 1
            continue
        if kind == "none":
            res["none_ok" if r is None else "none_FALSE_READ"] += 1
            if r is not None:
                wrong.append((e["id"], e["label"], r))
            continue
        ok = {v, v + 1} if kind == "trans" else {v}
        res[f"{kind}_" + ("none" if r is None else "ok" if r in ok else "WRONG")] += 1
        if r is not None and r not in ok:
            wrong.append((e["id"], e["label"], r))
        if kind == "clean":
            rt = "-" if r is None else r // 10
            ro = "-" if r is None else r % 10
            conf["tens"][(v // 10, rt)] += 1
            conf["ones"][(v % 10, ro)] += 1
    n_clean = sum(res[f"clean_{k}"] for k in ("ok", "none", "WRONG"))
    n_all = sum(n for k, n in res.items() if not k.startswith("unreadable"))
    n_wrong = res["clean_WRONG"] + res["trans_WRONG"] + res["none_FALSE_READ"]
    print("  counts:", dict(sorted(res.items())))
    if n_clean:
        print("  clean readable: exact %.1f%%  none %.1f%%  wrong %.1f%%  (n=%d)" % (
            100 * res["clean_ok"] / n_clean, 100 * res["clean_none"] / n_clean,
            100 * res["clean_WRONG"] / n_clean, n_clean))
    print("  wrong non-None over all scored crops: %.2f%% (%d/%d)" % (100 * n_wrong / max(n_all, 1), n_wrong, n_all))
    print("  %.3f ms / read" % (1000 * dt / max(len(samples), 1)))
    for name in ("tens", "ones"):
        errs = sorted(((g, r, n) for (g, r), n in conf[name].items() if g != r), key=str)
        print(f"  {name} confusions (truth->read:n):", " ".join(f"{g}->{r}:{n}" for g, r, n in errs) or "none")
    if show_wrong and wrong:
        print("  wrong reads (id, label, read):", wrong)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("crops", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "replanal" / "extractors" / "timer_glyphs.npz")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--extract", action="store_true", help="(re)extract crops from the videos first")
    ap.add_argument("--eval-only", type=Path, default=None, help="evaluate an existing template file")
    args = ap.parse_args()

    if args.extract:
        extract_crops(args.labels, args.crops)
    samples = load(args.labels, args.crops)
    test = [s for s in samples if s[0]["video"] in HOLDOUT]
    trainset = [s for s in samples if s[0]["video"] not in HOLDOUT]

    if args.eval_only:
        ext = T.TimerExtractor(args.eval_only)
        print("all samples:"); evaluate(ext, samples)
        print("held-out videos:"); evaluate(ext, test, show_wrong=False)
        return

    # Entries with "v" ids are a later, blind-labelled validation batch
    # (random reads of the final reader across all videos); they are scored
    # but never used to build templates.
    fit = [s for s in (trainset if args.holdout else samples) if not s[0]["id"].startswith("v")]
    glyphs, cells, counts = train(fit)
    print("cells:", cells, "samples per glyph:", counts)
    path = args.out if not args.holdout else args.out.with_name("timer_glyphs_holdout.npz")
    np.savez_compressed(path, glyphs=glyphs.astype(np.float16),
                        cells=np.array([cells["tens"], cells["ones"]], np.int16))
    ext = T.TimerExtractor(path)
    if args.holdout:
        print("train videos:"); evaluate(ext, trainset, show_wrong=False)
        print("held-out videos:"); evaluate(ext, test)
    else:
        print("all samples:"); evaluate(ext, samples)
    print("wrote", path)


if __name__ == "__main__":
    main()
