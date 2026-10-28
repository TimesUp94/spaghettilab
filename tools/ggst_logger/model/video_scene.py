"""Whole-frame context for the move net: the full frame at low resolution around every detection
of video_crops.py (same frames, same file names), so projectiles, summons and hits away from the
attacker crop are in view. Mirrored like the crop (the attacker, by the crop's positions, faces
right). Frames are read by seeking to the keyframe before each window (cv2), so only ~1/5 of the
video is decoded; a 30 fps video gives frame n // 2 for 60 fps frame n.

    .venv/Scripts/python.exe tools/ggst_logger/model/video_scene.py --crops vcrops_test --jobs 5
    .venv/Scripts/python.exe tools/ggst_logger/model/video_scene.py --crops degraded/vcrops --video-dir degraded

-> <data>/ggst_dataset/<crops dir with vcrops -> vscene>/<stem>/<frame>_P<side>.jpg (21 tiles of 384x216)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, low_priority  # noqa: E402

SW, SH = 384, 216
SEEK = 90                         # read on (no seek) when the next frame is closer than this


def frames_at(video, wanted):
    """{60 fps frame index: SWxSH BGR frame} for the sorted indices."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    step = 2 if abs(cap.get(cv2.CAP_PROP_FPS) - 30) < 1 else 1
    out, cache, pos = {}, {}, None
    for f in wanted:
        s = f // step
        if s not in cache:
            if pos is None or s < pos or s - pos > SEEK:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s)
                pos = s
            while pos < s:
                cap.grab()
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                break
            cache = {s: cv2.resize(img, (SW, SH), interpolation=cv2.INTER_AREA)}
        out[f] = cache[s]
    cap.release()
    return out


def process(args):
    stem, video, cdir, odir = args
    import cv2
    low_priority()
    cv2.setNumThreads(2)
    odir = Path(odir)
    if (odir / "done").exists():
        return stem, -1
    odir.mkdir(parents=True, exist_ok=True)
    idx = json.loads((Path(cdir) / "index.json").read_text())
    win = idx["win"]
    wanted = sorted({max(0, d["frame"] + o) for d in idx["dets"] for o in win})
    fr = frames_at(video, wanted)
    n = 0
    for d in idx["dets"]:
        fs = [max(0, d["frame"] + o) for o in win]
        if any(x not in fr for x in fs):
            continue
        p = np.array(d["pos"])
        a = d["side"] - 1
        flip = np.mean(p[:, 1 - a, 0] < p[:, a, 0]) > 0.5
        strip = np.hstack([fr[x] for x in fs])
        if flip:
            strip = np.hstack([fr[x][:, ::-1] for x in fs])
        cv2.imwrite(str(odir / d["file"]), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n += 1
    (odir / "done").write_text(str(n))
    return stem, n


def scene_dir(crops_root: Path) -> Path:
    return crops_root.parent / crops_root.name.replace("vcrops", "vscene")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="vcrops_test", help="crops dir under the data root")
    ap.add_argument("--video-dir", help="videos as <dir>/<stem>.mp4 (default: the dataset's recordings)")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    low_priority()
    import pyarrow.parquet as pq
    croot = DATA / a.crops
    oroot = scene_dir(croot)
    video = {r["replay"]: r["video"] for r in pq.read_table(DATA / "openers.parquet", columns=["replay", "video"]).to_pylist()}
    if a.video_dir:
        video = {s: str(DATA / a.video_dir / f"{s}.mp4") for s in video}
    stems = sorted(p.parent.name for p in croot.glob("*/index.json"))
    tasks = [(s, video[s], str(croot / s), str(oroot / s)) for s in stems if not a.only or s in a.only]
    t0 = time.time()
    with Pool(a.jobs) as p:
        for k, (stem, n) in enumerate(p.imap_unordered(process, tasks)):
            print(f"{k + 1}/{len(tasks)} {stem} {n} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
