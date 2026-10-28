"""Tiny whole frames (64x36 RGB) at every step of the CV pipeline's frames parquet, for the opener
detector's image stream (video_openers.py --tiny): super flash, cinematics, screen darkening and
hitsparks are visible even where the HUD is hidden. Step k is the video frame frame_number[k] of
the parquet, so it works for any frame rate the pipeline ran on.

    .venv/Scripts/python.exe tools/ggst_logger/model/video_tiny.py --jobs 6
    .venv/Scripts/python.exe tools/ggst_logger/model/video_tiny.py --degraded --jobs 6

-> <data>/ggst_dataset/tiny/<stem>.npy and degraded/tiny/<stem>.npy ([T, 36, 64, 3] uint8)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, low_priority  # noqa: E402

TW, TH = 64, 36


def tiny_frames(video, frames, threads=3):
    """[len(frames), TH, TW, 3] uint8 of the video frames at the given (sorted) indices; a frame the
    decoder does not reach repeats the last one."""
    from video_crops import FFMPEG
    frames = np.asarray(frames, int)
    d = np.diff(frames)
    step = int(d[0]) if len(d) and (d == d[0]).all() and frames[0] % d[0] == 0 else 1
    sel = f"select='not(mod(n\\,{step}))'," if step > 1 else ""
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-threads", str(threads), "-i", str(video), "-vf",
           f"{sel}scale={TW}:{TH}:flags=area", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    for tries in range(3):                    # ffmpeg fails now and then when memory is short
        r = subprocess.run(cmd, stdout=subprocess.PIPE)
        if r.returncode == 0:
            break
    r.check_returncode()
    raw = r.stdout
    dec = np.frombuffer(raw, np.uint8).reshape(-1, TH, TW, 3)
    idx = np.minimum(frames // step, len(dec) - 1)
    return dec[idx]


def process(args):
    stem, video, parquet, out = args
    low_priority()
    out = Path(out)
    if out.exists():
        return stem, -1
    import pyarrow.parquet as pq
    frames = np.asarray(pq.read_table(parquet, columns=["frame_number"]).column(0))
    x = tiny_frames(video, frames)
    tmp = out.with_suffix(".part.npy")
    np.save(tmp, x)
    tmp.rename(out)
    return stem, len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded", action="store_true", help="the 720p30 test copies under degraded/")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    low_priority()
    import pyarrow.parquet as pq
    root = DATA / "degraded" if a.degraded else DATA
    video = {r["replay"]: r["video"] for r in pq.read_table(DATA / "openers.parquet", columns=["replay", "video"]).to_pylist()}
    (root / "tiny").mkdir(parents=True, exist_ok=True)
    tasks = []
    for p in sorted((root / "runs").glob("*/frames/*_frames.parquet")):
        s = p.parent.parent.name
        v = root / f"{s}.mp4" if a.degraded else video.get(s)
        if v is not None:
            tasks.append((s, str(v), str(p), str(root / "tiny" / f"{s}.npy")))
    t0 = time.time()
    with Pool(a.jobs) as p:
        for k, (s, n) in enumerate(p.imap_unordered(process, tasks)):
            print(f"{k + 1}/{len(tasks)} {s} {n} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
