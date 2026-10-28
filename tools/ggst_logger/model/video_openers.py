"""Combo openers from video alone: a temporal network over the CV pipeline's per-frame HUD readings
(raw HP, red pending segment, read confidence, combo counter, tension, timer) that marks the
frame a combo starts and who is hitting.

Truth: the logged openers' first HP drop (first_damage_video_frame; bursts deal none and are left
out). The pipeline samples every 2nd video frame (30 Hz), so a step is 2 video frames.

    .venv/Scripts/python.exe tools/ggst_logger/model/video_openers.py train   # folds 1-4, test on fold 0
    .venv/Scripts/python.exe tools/ggst_logger/model/video_openers.py final   # every replay -> weights

--tiny adds an image stream: a small CNN on the whole frame at 64x36 at every step (video_tiny.py),
for what the HUD misses (super flash, cinematics with the HUD hidden, hitsparks).

-> <data>/ggst_dataset/model/video_openers_{test,final}.pt, results_video_openers.json
   (train --frac 0.25: a share of the training replays -> video_openers_exp_frac<pct>.pt, results_exp_openers_frac<pct>.json;
    train --name x -> video_openers_exp_x.pt, results_exp_openers_x.json and the test fold's detections_exp_x.json)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, TEST_FOLD, low_priority, replay_fold_map, save_json, subsample  # noqa: E402

RUNS = DATA / "runs"
STEP = 2            # video frames per pipeline sample
MATCH = 3           # steps (+-100 ms) for a detection to count
NF = 11             # features per side
TINY_DIM = 32       # image features per step (--tiny)
THR = 0.1           # detection threshold of the chain (0.3 before the image stream: its false alarms are rarer)


def side_features(t: dict, s: int, o: int) -> list[np.ndarray]:
    """Per-step features of side s (o = the other side), NaN -> 0 with a 'read' flag."""
    def col(name):
        return np.asarray(t[name], dtype=np.float64)
    hp = col(f"p{s}_health_raw")
    pend = col(f"p{s}_pending_pct")
    conf = col(f"p{s}_health_conf")
    ten = col(f"p{s}_tension_raw")
    ok = np.isfinite(hp)
    hpf = np.where(ok, hp, 0)
    dhp = np.diff(np.where(ok, hp, np.nan), prepend=np.nan)
    dpend = np.diff(pend, prepend=np.nan)
    dten = np.diff(ten, prepend=np.nan)
    cs = np.asarray([v if v is not None else 0 for v in t["combo_side"]])
    cc = np.asarray([v if v is not None else 0 for v in t["combo_count"]], dtype=np.float64)
    # combo counter on the attacker's side: side s hits when the counter is on s's side
    mine = (cs == s) * np.minimum(cc, 30) / 10
    return [hpf, ok.astype(float), np.nan_to_num(pend), np.nan_to_num(conf), np.nan_to_num(dhp) * 10,
            np.nan_to_num(dpend) * 10, np.nan_to_num(ten), np.nan_to_num(dten) * 10, mine, (cs == s).astype(float),
            np.zeros_like(hpf)]


def replay_inputs(stem: str):
    return parquet_inputs(RUNS / stem / "frames" / f"{stem}_frames.parquet")


def replay_tiny(stem: str, root: Path = DATA):
    """The replay's tiny frames [T, 36, 64, 3] uint8 (memory-mapped), from video_tiny.py."""
    return np.load(root / "tiny" / f"{stem}.npy", mmap_mode="r")


def parquet_inputs(path):
    """[T, 2*NF + 1] float32 features and the video frame of each step, from a pipeline frames parquet."""
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pydict()
    timer = np.asarray([v if v is not None else -1 for v in t["timer_raw"]])
    f = side_features(t, 1, 2) + side_features(t, 2, 1) + [(timer >= 0).astype(float)]
    X = np.nan_to_num(np.stack(f, 1)).astype(np.float32)
    return X, np.asarray(t["frame_number"])


def swap_sides(X):
    return np.concatenate([X[:, NF:2 * NF], X[:, :NF], X[:, 2 * NF:]], 1)


def truth(rows_by_replay, stem, frames):
    """(step, attacker side 1|2, row) of every logged opener with an HP drop."""
    out = []
    for r in rows_by_replay.get(stem, []):
        v = r["first_damage_video_frame"]
        if v is None or r["is_burst"]:
            continue
        i = int(np.searchsorted(frames, v))
        i = min(max(i, 0), len(frames) - 1)
        if i > 0 and abs(frames[i - 1] - v) < abs(frames[i] - v):
            i -= 1
        out.append((i, int(r["attacker"][1]), r))
    return out


def make_net(tiny=False):
    import torch
    import torch.nn as nn

    class Tiny(nn.Module):
        """Per-step image features from the 64x36 frame and its difference to the previous step."""
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                     nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(),
                                     nn.Linear(512, TINY_DIM))

        def forward(self, im, chunk=8192):     # [B, T, H, W, 3] uint8 -> [B, T, TINY_DIM]
            B, T = im.shape[:2]
            x = im.permute(0, 1, 4, 2, 3).float() / 255 - 0.5
            d = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
            x = torch.cat([x, d], 2).flatten(0, 1)
            with torch.autocast("cuda", dtype=torch.float16):
                z = torch.cat([self.cnn(x[k:k + chunk]) for k in range(0, len(x), chunk)])
            return z.float().view(B, T, -1)

    class Net(nn.Module):
        def __init__(self, cin=2 * NF + 1, h=96):
            super().__init__()
            self.tiny = Tiny() if tiny else None
            layers, c = [], cin + (TINY_DIM if tiny else 0)
            for d in (1, 2, 4, 8, 1):
                layers += [nn.Conv1d(c, h, 5, padding=2 * d, dilation=d), nn.GELU()]
                c = h
            self.conv = nn.Sequential(*layers)
            self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * h, 3)

        def forward(self, x, im=None):           # [B, T, C] (+ tiny frames [B, T, H, W, 3]) -> [B, T, 3]
            if self.tiny is not None:
                x = torch.cat([x, self.tiny(im)], -1)
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            z, _ = self.gru(z)
            return self.out(z)
    return Net()


def train(data, epochs=30, win=600, bs=24, lr=2e-3, seed=0, tiny=False):
    """data: list of (X [T,C], y [T] int64 0/1/2, w [T] float32 (0 = ignore)[, tiny frames [T,H,W,3]])."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net(tiny).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    per_ep = sum(len(d[0]) for d in data) // (win * bs) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * per_ep, pct_start=0.1)
    cw = torch.tensor([1.0, 20.0, 20.0]).cuda()
    lens = np.array([len(d[0]) for d in data], float)

    def batch():
        xs, ys, ws, ims = [], [], [], []
        for k in rng.choice(len(data), bs, p=lens / lens.sum()):
            X, y, w = data[k][:3]
            a = rng.integers(0, max(1, len(X) - win))
            x, yy, ww = X[a:a + win], y[a:a + win], w[a:a + win]
            im = np.asarray(data[k][3][a:a + win]) if tiny else None
            if rng.random() < 0.5:                       # P1 <-> P2 (the frame mirrored)
                x = swap_sides(x)
                yy = np.where(yy == 1, 2, np.where(yy == 2, 1, 0))
                im = im[:, :, ::-1] if tiny else None
            pad = win - len(x)
            xs.append(np.pad(x, ((0, pad), (0, 0))))
            ys.append(np.pad(yy, (0, pad)))
            ws.append(np.pad(ww, (0, pad)))
            if tiny:
                ims.append(np.pad(im, ((0, pad), (0, 0), (0, 0), (0, 0))))
        return [np.stack(v) for v in (xs, ys, ws)] + [np.stack(ims) if tiny else None]

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)                         # the next batch is read while this one trains
    nxt = pool.submit(batch)
    for ep in range(epochs):
        net.train()
        tot = 0
        for _ in range(per_ep):
            xs, ys, ws, ims = nxt.result()
            nxt = pool.submit(batch)
            x, y, w = (torch.from_numpy(v).cuda() for v in (xs, ys, ws))
            logit = net(x, torch.from_numpy(ims).cuda() if tiny else None)
            loss = (torch.nn.functional.cross_entropy(logit.reshape(-1, 3), y.reshape(-1), weight=cw, reduction="none")
                    * w.reshape(-1)).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        print(f"  epoch {ep + 1}/{epochs} loss {tot / per_ep:.4f}", flush=True)
    pool.shutdown(cancel_futures=True)
    return net


def predict(net, X, im=None):
    import torch
    net.eval()
    with torch.no_grad():
        im = torch.from_numpy(np.array(im)[None]).cuda() if net.tiny is not None else None
        p = torch.softmax(net(torch.from_numpy(X[None]).cuda(), im), -1)[0].cpu().numpy()
    return p


def load(path):
    """Net on the GPU from saved weights (a state dict; an image stream when it has 'tiny.' keys)."""
    import torch
    sd = torch.load(path, map_location="cpu")
    net = make_net(any(k.startswith("tiny.") for k in sd)).cuda()
    net.load_state_dict(sd)
    return net


def peaks(p, thr=0.5, nms=4):
    """(step, side, score): local maxima of each side's opener probability above thr."""
    out = []
    for s in (1, 2):
        q = p[:, s]
        idx = np.nonzero(q >= thr)[0]
        for i in idx:
            lo, hi = max(0, i - nms), i + nms + 1
            if q[i] == q[lo:hi].max() and (i == lo or q[i] > q[lo:i].max(initial=-1)):
                out.append((int(i), s, float(q[i])))
    return sorted(out)


def labels_for(X, tru):
    y = np.zeros(len(X), np.int64)
    w = np.ones(len(X), np.float32)
    for i, s, _ in tru:
        for d in (-1, 1):          # a step off is not wrong: neither push it up nor down
            if 0 <= i + d < len(X) and y[i + d] == 0:
                w[i + d] = 0
        y[i] = s
        w[i] = 1
    return y, w


def evaluate(dets, tru, frames):
    """Greedy match by score; returns per-truth matched det (or None) and the false positives."""
    used = set()
    hit = {}
    for i, s, sc in sorted(dets, key=lambda d: -d[2]):
        best = None
        for j, (ti, ts, _) in enumerate(tru):
            if j in hit or ts != s or abs(ti - i) > MATCH:
                continue
            if best is None or abs(ti - i) < abs(tru[best][0] - i):
                best = j
        if best is not None:
            hit[best] = (i, s, sc)
            used.add((i, s))
    fp = [d for d in dets if (d[0], d[1]) not in used]
    return hit, fp


def load_all(rows, tiny=False):
    """{stem: (X, y, w, frames, truth, tiny frames | None)}."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["replay"]].append(r)
    data = {}
    for k, stem in enumerate(sorted(by)):
        X, frames = replay_inputs(stem)
        tru = truth(by, stem, frames)
        y, w = labels_for(X, tru)
        im = replay_tiny(stem) if tiny else None
        assert im is None or len(im) == len(X), stem
        data[stem] = (X, y, w, frames, tru, im)
        if k % 100 == 0:
            print(f"  loaded {k}/{len(by)}", flush=True)
    return data


def train_on(data, stems, epochs, seed=0, tiny=False):
    return train([data[s][:3] + ((data[s][5],) if tiny else ()) for s in stems], epochs, seed=seed, tiny=tiny)


def report_detection(data, test, probs, thr):
    n = tp = nfp = 0
    dt = []
    by_cat = collections.defaultdict(lambda: [0, 0])
    for stem in test:
        frames, tru = data[stem][3:5]
        dets = peaks(probs[stem], thr)
        hit, fp = evaluate(dets, tru, frames)
        n += len(tru)
        tp += len(hit)
        nfp += len(fp)
        for j, (ti, ts, r) in enumerate(tru):
            by_cat[r["category"]][0] += 1
            if j in hit:
                by_cat[r["category"]][1] += 1
                dt.append(int(frames[hit[j][0]]) - int(r["first_damage_video_frame"]))
    rec, prec = tp / max(1, n), tp / max(1, tp + nfp)
    return dict(threshold=thr, truth=n, detected=tp, false_pos=nfp, recall=rec, precision=prec,
                f1=2 * rec * prec / max(1e-9, rec + prec),
                recall_by_category={k: (v[0], round(v[1] / v[0], 3)) for k, v in sorted(by_cat.items())},
                dt_video_frames=dict(collections.Counter(dt).most_common(9)))


def matched_dets(p, frames, tru, thr=THR):
    """Detections as detections.json entries, each matched to its logged opener (row) or null."""
    dets = peaks(p, thr)
    hit, _ = evaluate(dets, tru, frames)
    row_of = {(d[0], d[1]): tru[j][2] for j, d in hit.items()}
    out = [dict(frame=int(frames[i]), side=sd, score=round(sc, 4),
                row=(f"{row_of[(i, sd)]['log_row']}_{row_of[(i, sd)]['attacker']}" if (i, sd) in row_of else None))
           for i, sd, sc in dets]
    return out, len(hit)


def crossfit(data, fold, epochs, thr=THR, tiny=False):
    """Detections of every replay by the network that did not train on its fold, matched to the
    logged openers -> detections.json {stem: [{frame, side, score, row (log_row_attacker) | null}]}.
    The move models train on these, so they see the detector's timing and its false alarms."""
    out, n_tru, n_hit, n_det = {}, 0, 0, 0
    for f in sorted(set(fold.values())):
        net = train_on(data, [s for s in data if fold[s] != f], epochs, seed=f, tiny=tiny)
        for s in (s for s in data if fold[s] == f):
            X, _, _, frames, tru, im = data[s]
            out[s], h = matched_dets(predict(net, X, im), frames, tru, thr)
            n_tru += len(tru)
            n_hit += h
            n_det += len(out[s])
        print(f"fold {f}: {n_hit}/{n_tru} openers found so far, {n_det} detections", flush=True)
    (OUT / "detections.json").write_text(json.dumps(out))
    print(f"recall {n_hit / n_tru:.3f}, precision {n_hit / n_det:.3f} at threshold {thr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "final", "crossfit"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--frac", type=float, default=1.0, help="share of the training replays (by replay, seed 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="the image stream (video_tiny.py)")
    ap.add_argument("--name", help="experiment name -> video_openers_exp_<name>.pt, results_exp_openers_<name>.json")
    a = ap.parse_args()
    low_priority()
    import pyarrow.parquet as pq
    import torch
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    data = load_all(rows, a.tiny)
    stems = sorted(data)
    if a.mode == "crossfit":
        return crossfit(data, fold, a.epochs, tiny=a.tiny)
    test = [s for s in stems if fold[s] == TEST_FOLD]
    trainset = stems if a.mode == "final" else [s for s in stems if fold[s] != TEST_FOLD]
    trainset = subsample(trainset, a.frac)
    exp = a.frac < 1 or a.seed != 0 or a.name
    name = f"exp_{a.name}" if a.name else f"exp_frac{round(a.frac * 100)}" + (f"_seed{a.seed}" if a.seed else "")
    t0 = time.time()
    net = train_on(data, trainset, a.epochs, seed=a.seed, tiny=a.tiny)
    print(f"trained on {len(trainset)} replays in {time.time() - t0:.0f}s")
    torch.save(net.state_dict(), OUT / f"video_openers_{name if exp else 'final' if a.mode == 'final' else 'test'}.pt")
    if a.mode == "train":
        probs = {s: predict(net, data[s][0], data[s][5]) for s in test}
        res = {f"{thr:.2f}": report_detection(data, test, probs, thr) for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)}
        save_json(res, OUT / (f"results_{name.replace('exp_', 'exp_openers_')}.json" if exp else "results_video_openers.json"))
        if a.name:
            dets = {s: matched_dets(probs[s], data[s][3], data[s][4])[0] for s in test}
            (OUT / f"detections_{name}.json").write_text(json.dumps(dets))
        for k, v in res.items():
            print(f"thr {k}: recall {v['recall']:.3f} precision {v['precision']:.3f} f1 {v['f1']:.3f}")
        best = res[f"{THR:.2f}"]
        print(f"by category (thr {THR}):", best["recall_by_category"])
        print("dt:", best["dt_video_frames"])


if __name__ == "__main__":
    main()
