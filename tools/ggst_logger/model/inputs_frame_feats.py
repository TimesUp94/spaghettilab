"""Per-frame (60 fps) features of both player squares for the inputs model (ROADMAP point 3, stage 2).

The transcript's V-JEPA 2 tokens are 7.5 Hz (one per 8 video frames); a button press is visible as the
first frame of a new animation, so frame-exact timing needs frame-rate features. Every video frame of
the selected replays is decoded at 960 x 540, each player's square is cut like transcript_feats.py (a
640 px square in 1080p pixels, centred a quarter of the way towards the opponent and 20 px below the
player's point, mirrored so the player faces right; positions from the transcript features' player-net
track at 15 fps, linearly interpolated per frame) at 224 x 224 and encoded by frozen DINOv2-small
(ViT-S/14, the CLS output, 384-d, fp16).

Replays: the val and test splits plus N_TRAIN training replays (a fixed hash order), to stay inside the
disk / time budget; video_inputs.py --frame-feats trains on those training replays only.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/inputs_frame_feats.py smoke         # 2 replays, throughput
    .venv/Scripts/python.exe tools/ggst_logger/model/inputs_frame_feats.py extract       # resumable

-> <data>/ggst_dataset/inputs/frame_feats/<stem>.npz: f [F, 2, 384] fp16 (P1, P2), n_frames;
   inputs/frame_feats_times.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, low_priority  # noqa: E402
import transcript_feats as tf  # noqa: E402

INP = DATA / "inputs"
FF = INP / "frame_feats"
N_TRAIN = 250
TILE = 224
DEC_W, DEC_H = 960, 540
HF = "facebook/dinov2-small"
DIM = 384
CHUNK = 64


def selected():
    sp = json.loads((tf.TR / "transcript_splits.json").read_text())
    tr = sorted((k for k, v in sp.items() if v == "train"), key=lambda s: hashlib.md5(("ff:" + s).encode()).hexdigest())
    return dict(train=sorted(tr[:N_TRAIN]), val=sorted(k for k, v in sp.items() if v == "val"),
                test=sorted(k for k, v in sp.items() if v == "test"))


def track(stem, F):
    """Per video frame: square centres [F, 2, 2] (decode px) and flips [F, 2] from the 15 fps track."""
    z = np.load(tf.FEATS / f"{stem}.npz")
    fr, pos = z["frame"].astype(np.float64), z["pos"].astype(np.float64)
    f = np.arange(F, dtype=np.float64)
    P = np.stack([np.stack([np.interp(f, fr, pos[:, s, c]) for c in (0, 1)], -1) for s in (0, 1)], 1)   # [F, 2, 2]
    cen = np.zeros_like(P)
    flip = np.zeros((F, 2), bool)
    for s in (0, 1):
        xs, ys, xo = P[:, s, 0], P[:, s, 1], P[:, 1 - s, 0]
        cen[:, s, 0] = (xs + 0.25 * (xo - xs)) / 2
        cen[:, s, 1] = (ys + 20) / 2
        flip[:, s] = xo < xs
    return cen, flip


class Frames(torch.utils.data.IterableDataset):
    """Both player squares of every frame, [n, 2, TILE, TILE, 3] uint8, in chunks; last=True closes a replay."""

    def __init__(self, tasks, threads):
        self.tasks, self.threads = tasks, threads

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        mine = self.tasks if wi is None else self.tasks[wi.id::wi.num_workers]
        for stem, video, F in mine:
            yield from self.replay(stem, video, F)

    def replay(self, stem, video, F):
        import cv2
        t0 = time.time()
        cen, flip = track(stem, F)
        fps, _, _ = tf.video_format(video)
        canon = "fps=60," if abs(fps - 60) > 0.5 else ""
        cmd = [tf.FFMPEG, "-hide_banner", "-v", "error", "-threads", str(self.threads), "-i", str(video), "-vf",
               f"{canon}scale={DEC_W}:{DEC_H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=DEC_W * DEC_H * 3 * 4)
        n = DEC_W * DEC_H * 3
        half = tf.S / 4
        sc = TILE / (2 * half)
        buf, k = [], 0
        while k < F:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            img = np.frombuffer(b, np.uint8).reshape(DEC_H, DEC_W, 3)
            tiles = np.zeros((2, TILE, TILE, 3), np.uint8)
            for s in (0, 1):
                cx, cy = cen[k, s]
                if flip[k, s]:   # mirrored: x' = -(x - cx) * sc + TILE/2
                    M = np.array([[-sc, 0, cx * sc + TILE / 2], [0, sc, -(cy - half) * sc]], np.float64)
                else:
                    M = np.array([[sc, 0, -(cx - half) * sc], [0, sc, -(cy - half) * sc]], np.float64)
                tiles[s] = cv2.warpAffine(img, M, (TILE, TILE), flags=cv2.INTER_LINEAR, borderValue=0)
            buf.append(tiles)
            k += 1
            if len(buf) == CHUNK:
                yield dict(stem=stem, tiles=np.stack(buf), last=False, F=F, dec_s=time.time() - t0)
                buf = []
        p.kill()
        p.wait()
        tiles = np.stack(buf) if buf else np.zeros((0, 2, TILE, TILE, 3), np.uint8)
        yield dict(stem=stem, tiles=tiles, last=True, F=F, got=k, dec_s=time.time() - t0)


def tasks(stems):
    vid = {s: v for s, v in tf.replays()}
    out = []
    for s in stems:
        z = np.load(tf.FEATS / f"{s}.npz")
        out.append((s, vid[s], int(z["n_frames"])))
    return out


def run(stems, workers=3, threads=4, bs=256):
    from transformers import AutoModel
    low_priority()
    FF.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks(stems) if not (FF / f"{t[0]}.npz").exists()]
    print(f"{len(todo)} replays to do", flush=True)
    if not todo:
        return
    m = AutoModel.from_pretrained(HF, dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    dl = torch.utils.data.DataLoader(Frames(todo, threads), batch_size=None, num_workers=min(workers, 3), prefetch_factor=4)
    acc = {}
    t0 = time.time()
    done = 0
    for item in dl:
        st = item["stem"]
        x = torch.as_tensor(item["tiles"]).cuda(non_blocking=True)        # [n, 2, H, W, 3]
        n = len(x)
        out = []
        if n:
            x = x.view(n * 2, TILE, TILE, 3).permute(0, 3, 1, 2).float().div_(255)
            x = ((x - mean) / std).half()
            with torch.no_grad():
                for j in range(0, len(x), bs):
                    out.append(m(pixel_values=x[j:j + bs]).pooler_output.float().cpu())
            acc.setdefault(st, []).append(torch.cat(out).view(n, 2, DIM).half().numpy())
        if item["last"]:
            f = np.concatenate(acc.pop(st, [np.zeros((0, 2, DIM), np.float16)]))
            if len(f) < item["F"]:   # short decode: pad with the last frame
                f = np.concatenate([f, np.repeat(f[-1:] if len(f) else np.zeros((1, 2, DIM), np.float16), item["F"] - len(f), 0)])
            tmp = FF / f"{st}.tmp.npz"
            np.savez(tmp, f=f, n_frames=np.int32(item["F"]), decoded=np.int32(item["got"]))
            os.replace(tmp, FF / f"{st}.npz")
            done += 1
            el = time.time() - t0
            with open(INP / "frame_feats_times.jsonl", "a") as fh:
                fh.write(json.dumps(dict(stem=st, frames=item["F"], decoded=item["got"], dec_s=item["dec_s"], t=el)) + "\n")
            print(f"{done}/{len(todo)} {st} {item['F']} frames ({item['got']} decoded) {el:.0f}s "
                  f"-> eta {el / done * (len(todo) - done) / 3600:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("smoke", "extract"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sel = selected()
    if a.cmd == "smoke":
        run(sel["test"][:2], a.workers, a.threads)
    else:
        run(sel["test"] + sel["val"] + sel["train"], a.workers, a.threads)


if __name__ == "__main__":
    main()
