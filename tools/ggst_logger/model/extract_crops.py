"""Attacker-centred crops (+ small full frames) over video frames -12..0 of every opener.

Camera model: camera.py (lagged midpoint clamped by the stage walls; median error ~50 px on
character boxes from a person detector, 85% within 200 px).

Crop: S x S (S=640 px of 1080p) centred between the attacker and 25% towards the defender,
vertically on the attacker's body, resized to 224; mirrored so the attacker always faces right.
Generous on purpose: zoom and camera errors (~+-150 px) stay inside the window.

Output (cached): <data>/ggst_dataset/crops/<stem>/<log_row>_<P>.jpg (13 tiles of 224x224, frames
-12..0 left to right) and full/<stem>/<log_row>_<P>.jpg (13 tiles of 256x144, unmirrored),
plus crops/index.json.

    .venv/Scripts/python.exe tools/ggst_logger/model/extract_crops.py --jobs 4
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ggst_logger" / "dataset"))
DATA = Path(os.environ.get("GGST_DATA", r"G:\Projects\replanal-data")) / "ggst_dataset"
OFFS = list(range(-12, 1))
S = 640
T = 224
FULL = (256, 144)


def crop_box(log, cam, row, att, dfn):
    xa, ya = camera.to_px(log, cam, row, att)
    xd, _ = camera.to_px(log, cam, row, dfn)
    cx = xa + 0.25 * (xd - xa)
    return int(round(cx - S / 2)), int(round(ya + 20 - S / 2))


def cut(img, x0, y0):
    import cv2
    h, w = img.shape[:2]
    out = np.zeros((S, S, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + S), min(h, y0 + S)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (T, T), interpolation=cv2.INTER_AREA)


def process(args):
    stem, rows, csv, video, rate, c = args
    import cv2
    import build
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # idle
    except Exception:
        pass
    cv2.setNumThreads(2)
    odir, fdir = DATA / "crops" / stem, DATA / "full" / stem
    todo = [r for r in rows if not (odir / f"{r['log_row']}_{r['attacker']}.jpg").exists()]
    if not todo:
        return stem, 0
    odir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    log = build.load_log(csv)
    cam = camera.camera(log)
    want = {}
    for k, r in enumerate(todo):
        h = r["log_row"]
        v0 = rate * log["t"][h] + c
        for o in OFFS:
            row = max(0, h + o)
            vf = int(round(v0 + o * rate / 60.0))
            want.setdefault(vf, []).append((k, o, row))
    tiles = {k: {} for k in range(len(todo))}
    full = {k: {} for k in range(len(todo))}
    cap = cv2.VideoCapture(video)
    last = max(want)
    f = 0
    while f <= last:
        if f in want:
            ok, img = cap.read()
            if not ok:
                break
            small = cv2.resize(img, FULL, interpolation=cv2.INTER_AREA)
            for k, o, row in want[f]:
                r = todo[k]
                att = int(r["attacker"][1])
                x0, y0 = crop_box(log, cam, row, att, 3 - att)
                t = cut(img, x0, y0)
                if log[f"p{att}_face"][row] != 0:     # attacker faces left -> mirror
                    t = t[:, ::-1]
                tiles[k][o] = t
                full[k][o] = small
        elif not cap.grab():
            break
        f += 1
    cap.release()
    n = 0
    for k, r in enumerate(todo):
        if len(tiles[k]) != len(OFFS):
            continue
        name = f"{r['log_row']}_{r['attacker']}.jpg"
        cv2.imwrite(str(odir / name), np.hstack([tiles[k][o] for o in OFFS]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(fdir / name), np.hstack([full[k][o] for o in OFFS]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        n += 1
    return stem, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by = {}
    for r in rows:
        by.setdefault(r["replay"], []).append(r)
    tasks = []
    for stem, rr in by.items():
        if a.only and stem not in a.only:
            continue
        al = json.loads((DATA / "runs" / stem / "openers.json").read_text())["summary"]["alignment"]
        tasks.append((stem, [{"log_row": r["log_row"], "attacker": r["attacker"]} for r in rr],
                      rr[0]["csv"], rr[0]["video"], al["rate"], al["c"]))
    with Pool(a.jobs) as p:
        for stem, n in p.imap_unordered(process, tasks):
            print(stem, n, flush=True)
    idx = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": (DATA / "crops" / r["replay"] / f"{r['log_row']}_{r['attacker']}.jpg").exists()
           for r in rows}
    (DATA / "crops" / "index.json").write_text(json.dumps({"offsets": OFFS, "tile": T, "full": FULL, "items": idx}))
    print(sum(idx.values()), "/", len(idx), "openers with crops")


if __name__ == "__main__":
    main()
