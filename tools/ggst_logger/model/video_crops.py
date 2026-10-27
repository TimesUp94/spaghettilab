"""Attacker crops placed from the video alone, around every detected opener (video_openers.py
crossfit -> detections.json): the player network (video_players.py) finds both players on each
frame, the crop is centred like extract_crops.crop_box (between the attacker and 25 % towards
the defender, on the attacker's body) and mirrored so the attacker faces the defender's side.

Frames -16..+4 around the detected HP drop are kept (WIN), so the move models can be trained
with the detector's timing jitter. Per replay also: both players' positions per tile and the
character probabilities summed over every frame (the replay's characters by vote).

    .venv/Scripts/python.exe tools/ggst_logger/model/video_crops.py --net test --jobs 3

-> <data>/ggst_dataset/vcrops_<net>/<stem>/<frame>_P<side>.jpg  (21 tiles of 224x224)
   <data>/ggst_dataset/vcrops_<net>/<stem>/index.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, low_priority  # noqa: E402

WIN = list(range(-16, 5))
S, T = 640, 224
FULL = (256, 144)
FFMPEG = os.environ.get("GGST_FFMPEG", str(DATA.parent / "ggst_mod" / "dl" / "ffmpeg-9.0.2-essentials_build" / "bin" / "ffmpeg.exe"))
FW, FH = 1920, 1080


def video_format(video):
    """(fps, width, height) of a video."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return out


def read_frames(video, wanted):
    """Yield (frame index, BGR image) for the sorted frame indices in wanted. Frames are always
    1920x1080 at 60 fps (the recordings the models learnt on): any other video is converted on
    decode, and the indices count 60 fps frames."""
    fps, w, h = video_format(video)
    canon = ""
    if abs(fps - 60) > 0.5:
        canon += "fps=60,"
    if (w, h) != (FW, FH):
        canon += f"scale={FW}:{FH}:flags=bicubic,"
    ranges, a = [], None
    for f in wanted:
        if a is None or f > b + 1:
            if a is not None:
                ranges.append((a, b))
            a = f
        b = f
    ranges.append((a, b))
    sel = "+".join(f"between(n\\,{x}\\,{y})" for x, y in ranges)
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-threads", "4", "-i", str(video), "-vf", f"{canon}select='{sel}'",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FW * FH * 3 * 4)
    n = FW * FH * 3
    for f in wanted:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield f, np.frombuffer(buf, np.uint8).reshape(FH, FW, 3)
    p.stdout.close()
    p.wait()


def cut(img, x0, y0):
    import cv2
    out = np.zeros((S, S, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(FW, x0 + S), min(FH, y0 + S)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (T, T), interpolation=cv2.INTER_AREA)


_net = None


def players(smalls, netfile):
    """Positions [N, 2, 2] (px of 1920x1080) and character log-probs [N, 2, C] for 256x144 BGR frames."""
    global _net
    import cv2
    import torch
    import video_players as vp
    if _net is None:
        ck = torch.load(netfile, map_location="cuda")
        _net = vp.make_net(len(ck["chars"])).cuda().eval()
        _net.load_state_dict(ck["state"])
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = np.stack([(cv2.resize(s[:, :, ::-1], (vp.IN_W, vp.IN_H)).astype(np.float32) / 255 - mean) / std
                  for s in smalls]).transpose(0, 3, 1, 2)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            out = _net(torch.from_numpy(np.ascontiguousarray(x)).cuda())
        gl = torch.log_softmax(out["gchar"].float(), -1)
        chars = gl.sum(0).argmax(-1)[None].expand(len(smalls), 2)     # the window's characters, by vote
        pos, gl = vp.decode(out, chars, window=True)
    return pos.cpu().numpy() * [FW, FH], gl.cpu().numpy()


def smooth(x, k=2):
    """Median over +-k tiles (positions jump less than the network's per-frame noise)."""
    return np.stack([np.median(x[max(0, i - k):i + k + 1], 0) for i in range(len(x))])


def process(args):
    stem, video, dets, odir, netfile = args
    import cv2
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    cv2.setNumThreads(2)
    odir = Path(odir)
    if (odir / "index.json").exists():
        return stem, -1
    odir.mkdir(parents=True, exist_ok=True)
    dets = sorted(dets, key=lambda d: d["frame"])
    wanted = sorted({max(0, d["frame"] + o) for d in dets for o in WIN})
    index = {"win": WIN, "chars_logp": None, "frames_seen": 0, "dets": []}
    full, done, chsum = {}, set(), None
    nxt = 0                                           # first detection not cut yet
    # stream: a detection is cut as soon as its last frame is in; frames no window needs are dropped
    for f, img in read_frames(video, wanted):
        full[f] = img.copy()
        while nxt < len(dets) and max(0, dets[nxt]["frame"] + WIN[-1]) <= f:
            d = dets[nxt]
            nxt += 1
            fs = [max(0, d["frame"] + o) for o in WIN]
            if any(x not in full for x in fs):
                continue
            pos, chlp = players([cv2.resize(full[x], FULL, interpolation=cv2.INTER_AREA) for x in fs], netfile)
            for j, x in enumerate(fs):                 # character vote: every frame once
                if x not in done:
                    done.add(x)
                    chsum = chlp[j] if chsum is None else chsum + chlp[j]
            p = smooth(pos)                            # [21, 2, 2]
            a = d["side"] - 1
            tiles = []
            for j, x in enumerate(fs):
                (xa, ya), (xd, _) = p[j, a], p[j, 1 - a]
                cx = xa + 0.25 * (xd - xa)
                t = cut(full[x], int(round(cx - S / 2)), int(round(ya + 20 - S / 2)))
                tiles.append(t[:, ::-1] if xd < xa else t)
            name = f"{d['frame']}_P{d['side']}.jpg"
            cv2.imwrite(str(odir / name), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
            index["dets"].append(dict(d, file=name, pos=np.round(p, 1).tolist()))
        lo = max(0, dets[nxt]["frame"] + WIN[0]) if nxt < len(dets) else f + 1
        for x in [x for x in full if x < lo]:
            del full[x]
    index["chars_logp"] = chsum.tolist() if chsum is not None else None
    index["frames_seen"] = len(done)
    (odir / "index.json").write_text(json.dumps(index))
    return stem, len(index["dets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="test", choices=["test", "final"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    low_priority()
    import pyarrow.parquet as pq
    dets = json.loads((OUT / "detections.json").read_text())
    video = {r["replay"]: r["video"] for r in pq.read_table(DATA / "openers.parquet", columns=["replay", "video"]).to_pylist()}
    root = DATA / f"vcrops_{a.net}"
    netfile = str(OUT / f"video_players_{a.net}.pt")
    tasks = [(s, video[s], dets[s], str(root / s), netfile) for s in sorted(dets) if dets[s] and (not a.only or s in a.only)]
    t0 = time.time()
    with Pool(a.jobs) as p:
        for k, (stem, n) in enumerate(p.imap_unordered(process, tasks)):
            print(f"{k + 1}/{len(tasks)} {stem} {n} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
