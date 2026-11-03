"""Frame-exact timing around the transcript's events from 60 fps features (ROADMAP point 3, stage 2b).

The transcript places a move start to +-2..4 frames (7.5 Hz tokens + an 8-way offset). This refiner
looks only at a window of per-frame DINOv2-small features of both player squares
(inputs_frame_feats.py) around an anchor frame and predicts where in the window the move starts and
where its button was pressed (the press of its command, inputs_eval.event_cmds). It answers "do
higher-rate features around the transcript's events improve timing?" directly, without the dense
per-frame model (video_inputs.py --frame-feats, which did not use them).

Input per event: frames anchor-16 .. anchor+15 (32), the side's square and the other's (mirrored
order mine / other), the features and their frame-to-frame differences, plus the transcript's
category as an embedding. Conv1d stack (kernel 5, 4 layers, 256 ch) -> two 32-way softmaxes.
Training (the 250 training replays with frame features): anchor = true start + uniform jitter in
[-6, 6], or (--jit-emp) an offset drawn from the transcript's own start errors on the validation
matches (mostly 0 / +-1). Test: anchor = the
start of the transcript prediction (exp_tr_ens5grid) matched to the event at the move level within +-4
frames; the refined start / press against the true ones, compared with the transcript's own start
(and the start as the press: press and start share the frame for 68% of logged moves).

    .venv/Scripts/python.exe tools/ggst_logger/model/inputs_refine.py --name refine

-> model/results_exp_in_<name>.json, model/inputs_refine_exp_in_<name>.pt, inputs/exp_in_<name>.log
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, low_priority, replay_fold_map, save_json, test_subsets  # noqa: E402
import inputs_eval as ie  # noqa: E402
import inputs_frame_feats as iff  # noqa: E402

W0, W = 16, 32
JIT = 6
CATS = ["normal", "special", "super", "throw", "other"]
DEV = "cuda"


def log(msg, f):
    s = time.strftime("%H:%M:%S ") + msg
    print(s, flush=True)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def windows(stem, items):
    """items: [(side 0/1, anchor)] -> [n, W, 2 * 384] fp16 (mine | other)."""
    f = np.load(ie.INP / "frame_feats" / f"{stem}.npz")["f"]
    Fn = len(f)
    out = np.zeros((len(items), W, 2 * 384), np.float16)
    for j, (s, a) in enumerate(items):
        idx = np.clip(np.arange(a - W0, a - W0 + W), 0, Fn - 1)
        out[j] = np.concatenate([f[idx, s], f[idx, 1 - s]], -1)
    return out


class Net(nn.Module):
    def __init__(self, c=256, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(4 * 384), nn.Dropout(drop), nn.Linear(4 * 384, c))
        self.cat = nn.Embedding(len(CATS), c)
        self.pos = nn.Parameter(torch.zeros(W, c))
        self.convs = nn.ModuleList([nn.Conv1d(c, c, 5, padding=2) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(4)])
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(c, 2)

    def forward(self, x, cat):
        x = x.float()
        d = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
        h = self.inp(torch.cat([x, d], -1)) + self.cat(cat)[:, None] + self.pos
        for cv, n in zip(self.convs, self.norms):
            h = h + self.drop(F.gelu(cv(n(h).transpose(1, 2)).transpose(1, 2)))
        o = self.out(h)                    # [n, W, 2]: start, press
        return o[..., 0], o[..., 1]


def train_set(stems, rng, reps, jit=None):
    rows = collections.defaultdict(list)
    for r in ie.event_cmds():
        rows[r["replay"]].append(r)
    ev = ie.events_by_replay(set(stems))
    X, C, S, P = [], [], [], []
    for st in stems:
        cm = {(r["side"], r["start_frame"]): r for r in rows.get(st, [])}
        items, cats, ts, ps = [], [], [], []
        for e in ev.get(st, []):
            if not e["in_video"]:
                continue
            s = 0 if e["side"] == "P1" else 1
            r = cm.get((e["side"], e["start_frame"]))
            for _ in range(reps):
                a = e["start_frame"] + (int(rng.choice(jit)) if jit is not None else int(rng.integers(-JIT, JIT + 1)))
                items.append((s, a))
                cats.append(CATS.index(e["category"]))
                ts.append(e["start_frame"] - a + W0)
                pf = None if r is None or r["press_frame"] is None else r["press_frame"] - a + W0
                ps.append(pf if pf is not None and 0 <= pf < W else -100)
        if items:
            X.append(windows(st, items))
            C += cats
            S += ts
            P += ps
    return np.concatenate(X), np.array(C), np.array(S), np.array(P)


def test_items(stems, preds_path):
    """Matched events: (stem, side, transcript start, true start, true press, category)."""
    matched, ev = ie.transcript_matches(stems, preds_path)
    rows = collections.defaultdict(dict)
    for r in ie.event_cmds():
        rows[r["replay"]][(r["side"], r["start_frame"])] = r
    out = collections.defaultdict(list)
    for (st, i), mf in matched.items():
        e = ev[st][i]
        r = rows[st].get((e["side"], e["start_frame"]))
        out[st].append((0 if e["side"] == "P1" else 1, int(mf), e["start_frame"],
                        None if r is None else r["press_frame"], CATS.index(e["category"])))
    return out


@torch.no_grad()
def predict(net, items):
    res = []
    net.eval()
    for st, its in items.items():
        x = torch.as_tensor(windows(st, [(s, a) for s, a, *_ in its]), device=DEV)
        c = torch.as_tensor([it[4] for it in its], device=DEV)
        ls, lp = net(x, c)
        ss, pp = ls.argmax(1).cpu().numpy(), lp.argmax(1).cpu().numpy()
        for j, (s, a, t, p, cat) in enumerate(its):
            res.append(dict(replay=st, cat=CATS[cat], anchor=a, true_start=t, true_press=p,
                            start=int(a - W0 + ss[j]), press=int(a - W0 + pp[j])))
    return res


def summ(rr):
    out = {}
    for name, pred, tru in (("transcript_start_vs_start", "anchor", "true_start"), ("refined_start_vs_start", "start", "true_start"),
                            ("transcript_start_vs_press", "anchor", "true_press"), ("refined_press_vs_press", "press", "true_press")):
        d = np.array([abs(r[pred] - r[tru]) for r in rr if r[tru] is not None])
        out[name] = dict(n=int(len(d)), le0=float((d == 0).mean()), le1=float((d <= 1).mean()), le2=float((d <= 2).mean()),
                         le4=float((d <= 4).mean()), mean=float(d.mean())) if len(d) else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="refine")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jit-emp", action="store_true", help="anchor jitter from the transcript's start errors on val")
    a = ap.parse_args()
    low_priority()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    logf = ie.INP / f"exp_in_{a.name}.log"
    sel = iff.selected()
    t0 = time.time()
    val_items = test_items(sel["val"], ie.INP / "exp_in_ens5grid_val_preds.parquet")
    jit = None
    if a.jit_emp:   # anchor offsets drawn from the transcript's own start errors on the val matches
        jit = np.array([a_ - t for its in val_items.values() for _, a_, t, _, _ in its])
        log(f"empirical jitter from {len(jit)} val matches: " + str(dict(zip(*np.unique(jit, return_counts=True)))), logf)
    X, C, S, P = train_set(sel["train"], rng, a.reps, jit)
    log(f"train windows {len(X)} ({X.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s", logf)
    Xg = torch.as_tensor(X, device=DEV)
    Cg, Sg, Pg = (torch.as_tensor(v, device=DEV) for v in (C, S, P))
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    bs = 512
    steps = a.epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    hist = []
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(X), device=DEV)
        tot = 0.0
        for k in range(len(X) // bs):
            b = perm[k * bs:(k + 1) * bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls, lp = net(Xg[b], Cg[b])
            loss = F.cross_entropy(ls.float(), Sg[b]) + F.cross_entropy(lp.float(), Pg[b], ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        v = summ(predict(net, val_items))
        hist.append(dict(epoch=ep + 1, loss=tot / max(1, len(X) // bs), val=v))
        log(f"ep {ep + 1} loss {tot / max(1, len(X) // bs):.4f} val start<=1 {v['refined_start_vs_start']['le1']:.3f} "
            f"(transcript {v['transcript_start_vs_start']['le1']:.3f}) press<=1 {v['refined_press_vs_press']['le1']:.3f} "
            f"(transcript start {v['transcript_start_vs_press']['le1']:.3f})", logf)
    torch.save(dict(state=net.state_dict(), args=vars(a)), OUT / f"inputs_refine_exp_in_{a.name}.pt")
    sp = ie.splits()
    test = predict(net, test_items(sp["test"], ie.TR / "exp_tr_ens5grid_test_preds.parquet"))
    sub = test_subsets(replay_fold_map())
    res = {}
    for nm in ("old", "new", "all"):
        reps = set(sub[nm])
        rr = [r for r in test if r["replay"] in reps]
        res[nm] = dict(all=summ(rr), **{c: summ([r for r in rr if r["cat"] == c]) for c in CATS})
        d = res[nm]["all"]
        log(f"[{nm}] start <=0/1/2: transcript {d['transcript_start_vs_start']['le0']:.3f}/{d['transcript_start_vs_start']['le1']:.3f}/"
            f"{d['transcript_start_vs_start']['le2']:.3f} refined {d['refined_start_vs_start']['le0']:.3f}/{d['refined_start_vs_start']['le1']:.3f}/"
            f"{d['refined_start_vs_start']['le2']:.3f} | press <=0/1/2: transcript start {d['transcript_start_vs_press']['le0']:.3f}/"
            f"{d['transcript_start_vs_press']['le1']:.3f}/{d['transcript_start_vs_press']['le2']:.3f} refined "
            f"{d['refined_press_vs_press']['le0']:.3f}/{d['refined_press_vs_press']['le1']:.3f}/{d['refined_press_vs_press']['le2']:.3f}", logf)
    save_json(dict(name=a.name, args=vars(a), doc=__doc__, train_seconds=time.time() - t0, hist=hist, test=res),
              OUT / f"results_exp_in_{a.name}.json")


if __name__ == "__main__":
    main()
