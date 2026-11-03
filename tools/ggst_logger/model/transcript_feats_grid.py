"""Spatial 2x2 grid tokens of the two player squares for the move transcript (ROADMAP point 2,
improvement round 1).

Why: the first transcript model (video_transcript.py on transcript_feats.py's tokens) finds moves
that touch the opponent (hits 0.90, blocks 0.88) but only 0.54 of whiffs and 0.62 of interrupted
moves; on validation most missed whiffs have a mid event score (0.3 to the threshold), i.e. the
model half sees them. transcript_feats.py mean-pools each V-JEPA 2 tubelet over all 16 x 16 patches
of the 640 px player square, which keeps little of where in the square something moved (an arm
or leg extending towards the opponent). This keeps a coarse layout: the 16 x 16 patch grid is
pooled into 2 x 2 quadrants (8 x 8 patches each; the squares are mirrored so 'right' is always
towards the opponent and 'top' is the upper body), then projected with a fixed PCA (fitted once on
training replays) to PC dims to fit on disk.

Same squares and timeline as transcript_feats.py: the player positions / flips are read back from
the existing feats/<stem>.npz (no player net run), frames decoded the same way (every 4th frame,
960 x 540), 16-frame clips, tubelets of 8 video frames. Only the two player views go through
V-JEPA 2 (the scene view is unchanged), so this costs about 2/3 of the first extraction.

Per replay -> <data>/ggst_dataset/transcript/feats_grid/<stem>.npz (atomic; resumable):
  p1g, p2g [Ntub, 4, PC] fp16   (quadrants: top-back, top-front, bottom-back, bottom-front)
PCA: transcript/feats_grid/pca.npz (mean [1024], comp [PC, 1024], explained variance).

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_feats_grid.py fitpca --n 16
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_feats_grid.py extract [--order val,train,test] [--max N]
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_feats_grid.py check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, low_priority  # noqa: E402
import transcript_feats as tf  # noqa: E402

GRID = tf.TR / "feats_grid"
PC = 192
G = 2            # 2 x 2 quadrants


class GridEncoder(tf.Encoder):
    def __init__(self, players_net, pca=None):
        super().__init__(players_net)
        self.pca = None
        if pca is not None:
            self.pca = (torch.as_tensor(pca["mean"], device=self.dev, dtype=torch.float32),
                        torch.as_tensor(pca["comp"], device=self.dev, dtype=torch.float32))

    def vjepa_grid(self, views):
        """views: list of [n, 3, TILE, TILE] -> list of [ceil(n/2), G*G, DIM] float32 (quadrant means)."""
        n = len(views[0])
        pad = (-n) % tf.CLIP
        clips = []
        for v in views:
            v = (v - self.mean) / self.std
            if pad:
                v = torch.cat([v, v[-1:].expand(pad, -1, -1, -1)])
            clips.append(v.view(-1, tf.CLIP, 3, tf.TILE, tf.TILE))
        x = torch.cat(clips).to(self.dt)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
            h = self.m(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        P = tf.TILE // 16
        h = h.view(len(x), tf.CLIP // 2, G, P // G, G, P // G, tf.DIM).float().mean((3, 5))      # [B, T, G, G, D]
        h = h.reshape(len(views), -1, G * G, tf.DIM)
        return [z[:(n + 1) // 2] for z in h]

    def chunk_grid(self, frames, pos):
        """frames [n, H, W, 3] uint8, pos [n, 2, 2] (the stored smoothed positions) -> [2] x [ntub, 4, D]."""
        with torch.no_grad():
            x = frames.to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            (p1, p2), _ = self.squares(x, torch.as_tensor(pos, device=self.dev, dtype=torch.float32))
            del x
            return self.vjepa_grid([p1, p2])


def split_of():
    sp = json.loads((tf.TR / "transcript_splits.json").read_text())
    return sp


def ordered_tasks(order, seed=0):
    tasks = tf.replays()
    sp = split_of()
    rng = np.random.default_rng(seed)
    out = []
    for s in order:
        t = [x for x in tasks if sp.get(x[0]) == s]
        rng.shuffle(t)
        out += t
    return out


def run(tasks, enc, write=True, limit=0, collect=None, workers=3):
    GRID.mkdir(parents=True, exist_ok=True)
    dl = torch.utils.data.DataLoader(tf.Chunks(tasks, 6, limit), batch_size=None, num_workers=workers,
                                     prefetch_factor=2 if workers else None, pin_memory=True)
    state, done, t0 = {}, 0, time.time()
    tlog = open(GRID / "times.jsonl", "a") if write else None
    for item in dl:
        s = item["stem"]
        if s not in state:
            z = np.load(tf.FEATS / f"{s}.npz")
            state[s] = dict(pos=z["pos"], ntub=len(z["tub_frame"]), out=[[], []], t0=time.time(), n=0)
        st = state[s]
        fr = item["frames"]
        if len(fr):
            pos = st["pos"][st["n"]:st["n"] + len(fr)]
            if len(pos) < len(fr):     # never expected: decoded more frames than the first run
                pos = np.concatenate([pos, np.repeat(pos[-1:], len(fr) - len(pos), 0)])
            st["n"] += len(fr)
            f = enc.chunk_grid(fr, pos)
            for k in (0, 1):
                z = f[k]
                if collect is not None:
                    collect.append(z.reshape(-1, tf.DIM)[::3].cpu())
                elif enc.pca is not None:
                    z = (z - enc.pca[0]) @ enc.pca[1].T
                st["out"][k].append(z.half().cpu().numpy())
        if not item["last"]:
            continue
        del state[s]
        if item["err"] and not limit:
            print(f"  SKIP {s}: {item['err']}", flush=True)
            continue
        if not write:
            done += 1
            continue
        a, b = (np.concatenate(o) for o in st["out"])
        if len(a) != st["ntub"]:
            print(f"  WARN {s}: {len(a)} tubelets vs {st['ntub']} in feats; cut / padded", flush=True)
            a, b = (np.concatenate([x[:st["ntub"]], np.repeat(x[-1:], max(0, st["ntub"] - len(x)), 0)]) for x in (a, b))
        tmp = GRID / f"{s}.tmp.npz"
        np.savez(tmp, p1g=a, p2g=b)
        os.replace(tmp, GRID / f"{s}.npz")
        done += 1
        rec = dict(stem=s, ntub=len(a), wall_s=round(time.time() - st["t0"], 1))
        tlog.write(json.dumps(rec) + "\n")
        tlog.flush()
        if done % 10 == 1 or done == len(tasks):
            el = time.time() - t0
            print(f"  {done}/{len(tasks)} {s} | {el:.0f}s, eta {(len(tasks) - done) * el / done / 3600:.2f} h", flush=True)
    if tlog:
        tlog.close()
    return done


def fitpca(n, limit):
    tasks = ordered_tasks(["train"], seed=123)[:n]
    enc = GridEncoder(str(OUT / "video_players_test.pt"))
    col = []
    run(tasks, enc, write=False, limit=limit, collect=col)
    X = torch.cat(col).float().cuda()
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / len(Xc)
    ev, vec = torch.linalg.eigh(C)
    ev, vec = ev.flip(0), vec.flip(1)
    frac = (ev.cumsum(0) / ev.sum()).cpu().numpy()
    GRID.mkdir(parents=True, exist_ok=True)
    np.savez(GRID / "pca.npz", mean=mu.cpu().numpy(), comp=vec[:, :PC].T.cpu().numpy(), explained=frac)
    print(f"PCA on {len(X)} quadrant tokens: explained at 64 {frac[63]:.3f} 128 {frac[127]:.3f} 192 {frac[191]:.3f} "
          f"256 {frac[255]:.3f} 384 {frac[383]:.3f}", flush=True)


def check():
    tasks = tf.replays()
    miss, bad, size = [], [], 0
    for s, _ in tasks:
        f = GRID / f"{s}.npz"
        if not f.exists():
            miss.append(s)
            continue
        size += f.stat().st_size
        z, y = np.load(f), np.load(tf.FEATS / f"{s}.npz")
        if not (len(z["p1g"]) == len(y["tub_frame"]) == len(z["p2g"]) and np.isfinite(z["p1g"].astype(np.float32)).all()):
            bad.append(s)
    out = dict(replays=len(tasks), present=len(tasks) - len(miss), missing=len(miss), bad=bad, gb=round(size / 2 ** 30, 2))
    print(json.dumps(out, indent=1))
    (GRID / "check.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fitpca", "extract", "check"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--limit", type=int, default=1024)
    ap.add_argument("--order", default="val,train,test")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    low_priority()
    if a.mode == "check":
        return check()
    if a.mode == "fitpca":
        return fitpca(a.n, a.limit)
    pca = dict(np.load(GRID / "pca.npz"))
    tasks = [t for t in ordered_tasks(a.order.split(",")) if not (GRID / f"{t[0]}.npz").exists()]
    if a.max:
        tasks = tasks[:a.max]
    print(f"{len(tasks)} to extract", flush=True)
    enc = GridEncoder(str(OUT / "video_players_test.pt"), pca)
    run(tasks, enc, workers=a.workers)


if __name__ == "__main__":
    main()
