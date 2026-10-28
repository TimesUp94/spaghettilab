"""Combo openers from video alone: a temporal network over the CV pipeline's per-frame HUD readings
(raw HP, red pending segment, read confidence, combo counter, tension, timer) that marks the
frame a combo starts and who is hitting.

Truth: the logged openers' first HP drop (first_damage_video_frame; bursts deal none and are left
out). The pipeline samples every 2nd video frame (30 Hz), so a step is 2 video frames.

    .venv/Scripts/python.exe tools/ggst_logger/model/video_openers.py train   # folds 1-4, test on fold 0
    .venv/Scripts/python.exe tools/ggst_logger/model/video_openers.py final   # every replay -> weights

-> <data>/ggst_dataset/model/video_openers_{test,final}.pt, results_video_openers.json
   (train --frac 0.25: a share of the training replays -> video_openers_exp_frac<pct>.pt, results_exp_openers_frac<pct>.json)
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


def make_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, cin=2 * NF + 1, h=96):
            super().__init__()
            layers, c = [], cin
            for d in (1, 2, 4, 8, 1):
                layers += [nn.Conv1d(c, h, 5, padding=2 * d, dilation=d), nn.GELU()]
                c = h
            self.conv = nn.Sequential(*layers)
            self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * h, 3)

        def forward(self, x):           # [B, T, C] -> [B, T, 3]
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            z, _ = self.gru(z)
            return self.out(z)
    return Net()


def train(data, epochs=30, win=600, bs=24, lr=2e-3, seed=0):
    """data: list of (X [T,C], y [T] int64 0/1/2, w [T] float32 (0 = ignore))."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    per_ep = sum(len(X) for X, _, _ in data) // (win * bs) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * per_ep, pct_start=0.1)
    cw = torch.tensor([1.0, 20.0, 20.0]).cuda()
    lens = np.array([len(X) for X, _, _ in data], float)
    for ep in range(epochs):
        net.train()
        tot = 0
        for _ in range(per_ep):
            xs, ys, ws = [], [], []
            for k in rng.choice(len(data), bs, p=lens / lens.sum()):
                X, y, w = data[k]
                a = rng.integers(0, max(1, len(X) - win))
                x, yy, ww = X[a:a + win], y[a:a + win], w[a:a + win]
                if rng.random() < 0.5:                       # P1 <-> P2
                    x = swap_sides(x)
                    yy = np.where(yy == 1, 2, np.where(yy == 2, 1, 0))
                pad = win - len(x)
                xs.append(np.pad(x, ((0, pad), (0, 0))))
                ys.append(np.pad(yy, (0, pad)))
                ws.append(np.pad(ww, (0, pad)))
            x = torch.from_numpy(np.stack(xs)).cuda()
            y = torch.from_numpy(np.stack(ys)).cuda()
            w = torch.from_numpy(np.stack(ws)).cuda()
            logit = net(x)
            loss = (torch.nn.functional.cross_entropy(logit.reshape(-1, 3), y.reshape(-1), weight=cw, reduction="none")
                    * w.reshape(-1)).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / per_ep:.4f}", flush=True)
    return net


def predict(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(torch.from_numpy(X[None]).cuda()), -1)[0].cpu().numpy()
    return p


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


def load_all(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["replay"]].append(r)
    data = {}
    for k, stem in enumerate(sorted(by)):
        X, frames = replay_inputs(stem)
        tru = truth(by, stem, frames)
        y, w = labels_for(X, tru)
        data[stem] = (X, y, w, frames, tru)
        if k % 100 == 0:
            print(f"  loaded {k}/{len(by)}", flush=True)
    return data


def report_detection(data, test, net, thr):
    n = tp = nfp = 0
    dt = []
    by_cat = collections.defaultdict(lambda: [0, 0])
    for stem in test:
        X, y, w, frames, tru = data[stem]
        dets = peaks(predict(net, X), thr)
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


def crossfit(data, fold, epochs, thr=0.3):
    """Detections of every replay by the network that did not train on its fold, matched to the
    logged openers -> detections.json {stem: [{frame, side, score, row (log_row_attacker) | null}]}.
    The move models train on these, so they see the detector's timing and its false alarms."""
    out, n_tru, n_hit, n_det = {}, 0, 0, 0
    for f in sorted(set(fold.values())):
        net = train([data[s][:3] for s in data if fold[s] != f], epochs, seed=f)
        for s in (s for s in data if fold[s] == f):
            X, _, _, frames, tru = data[s]
            dets = peaks(predict(net, X), thr)
            hit, _ = evaluate(dets, tru, frames)
            row_of = {(d[0], d[1]): tru[j][2] for j, d in hit.items()}
            out[s] = [dict(frame=int(frames[i]), side=sd, score=round(sc, 4),
                           row=(f"{row_of[(i, sd)]['log_row']}_{row_of[(i, sd)]['attacker']}" if (i, sd) in row_of else None))
                      for i, sd, sc in dets]
            n_tru += len(tru)
            n_hit += len(hit)
            n_det += len(dets)
        print(f"fold {f}: {n_hit}/{n_tru} openers found so far, {n_det} detections", flush=True)
    (OUT / "detections.json").write_text(json.dumps(out))
    print(f"recall {n_hit / n_tru:.3f}, precision {n_hit / n_det:.3f} at threshold {thr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "final", "crossfit"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--frac", type=float, default=1.0, help="share of the training replays (by replay, seed 0)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    low_priority()
    import pyarrow.parquet as pq
    import torch
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    data = load_all(rows)
    stems = sorted(data)
    if a.mode == "crossfit":
        return crossfit(data, fold, a.epochs)
    test = [s for s in stems if fold[s] == TEST_FOLD]
    trainset = stems if a.mode == "final" else [s for s in stems if fold[s] != TEST_FOLD]
    trainset = subsample(trainset, a.frac)
    exp = a.frac < 1 or a.seed != 0
    name = f"exp_frac{round(a.frac * 100)}" + (f"_seed{a.seed}" if a.seed else "")
    t0 = time.time()
    net = train([data[s][:3] for s in trainset], a.epochs, seed=a.seed)
    print(f"trained on {len(trainset)} replays in {time.time() - t0:.0f}s")
    torch.save(net.state_dict(), OUT / f"video_openers_{name if exp else 'final' if a.mode == 'final' else 'test'}.pt")
    if a.mode == "train":
        res = {f"{thr:.2f}": report_detection(data, test, net, thr) for thr in (0.2, 0.3, 0.4, 0.5, 0.6)}
        save_json(res, OUT / (f"results_{name.replace('exp_', 'exp_openers_')}.json" if exp else "results_video_openers.json"))
        for k, v in res.items():
            print(f"thr {k}: recall {v['recall']:.3f} precision {v['precision']:.3f} f1 {v['f1']:.3f}")
        best = max(res.values(), key=lambda v: v["f1"])
        print("by category:", best["recall_by_category"])
        print("dt:", best["dt_video_frames"])


if __name__ == "__main__":
    main()
