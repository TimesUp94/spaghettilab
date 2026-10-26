"""Classifiers on cached frozen features (features.py), 5-fold cross-validation grouped by
replay (no replay in both train and test), out-of-fold predictions pooled for the report.

Tasks: category | normal (button x stance) | hit_kind (counter / punish / none) | move:Asuka | move:Elphelt
Models: majority, damage+distance (gradient boosting), log features (damage, distance, airborne
flags, hits, hitstop: game-memory values, an upper reference, not visual), and the visual head
(temporal conv over the 13 frame features of one or both views).

    .venv/Scripts/python.exe tools/ggst_logger/model/train_heads.py --views crops full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, key, load_rows, low_priority, replay_folds, report, save_json, task_labels  # noqa: E402

TASKS = ["category", "normal", "move:Asuka", "move:Elphelt"]


def tab(rows, cols):
    return np.array([[float(r[c]) if r[c] is not None else np.nan for c in cols] for r in rows])


def fit_tab(X, y, folds, ncls):
    from sklearn.ensemble import HistGradientBoostingClassifier
    proba = np.zeros((len(y), ncls))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=0)
        m.fit(X[tr], y[tr])
        p = np.zeros((te.sum(), ncls))
        p[:, m.classes_] = m.predict_proba(X[te])
        proba[te] = p
    return proba


def fit_head(F, y, folds, ncls, epochs=60, seed=0, frames=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    if frames is not None:
        F = F[:, frames]
    T, D = F.shape[1], F.shape[2]
    proba = np.zeros((len(y), ncls))
    for f in np.unique(folds):
        tr, te = np.nonzero(folds != f)[0], np.nonzero(folds == f)[0]
        mu = F[tr].mean((0, 1), keepdims=True)
        sd = F[tr].std((0, 1), keepdims=True) + 1e-3
        Xtr = torch.tensor((F[tr] - mu) / sd, dtype=torch.float32).cuda()
        Xte = torch.tensor((F[te] - mu) / sd, dtype=torch.float32).cuda()
        ytr = torch.tensor(y[tr]).cuda()
        cnt = np.bincount(y[tr], minlength=ncls).astype(float)
        w = torch.tensor(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0), dtype=torch.float32).cuda()
        net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(D, 256, 3, padding=1), nn.GELU(), nn.Dropout(0.3),
            nn.Conv1d(256, 256, 3, padding=1), nn.GELU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, ncls)).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.CrossEntropyLoss(weight=w / w[w > 0].mean(), label_smoothing=0.1)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(tr), device="cuda")
            for b in range(0, len(tr), 64):
                idx = perm[b:b + 64]
                x = Xtr[idx].transpose(1, 2)
                # temporal jitter: shift by up to 1 frame (alignment is +-1-2 frames)
                s = int(np.random.randint(-1, 2))
                if s:
                    x = torch.roll(x, s, 2)
                loss = lossf(net(x), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            proba[te] = torch.softmax(net(Xte.transpose(1, 2)), 1).cpu().numpy()
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--views", nargs="+", default=["crops"])
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    low_priority()
    rows = load_rows()
    feats = {}
    for v in a.views:
        z = np.load(OUT / f"feat_{a.backbone}_{v}.npz")
        pos = {k: i for i, k in enumerate(z["keys"])}
        feats[v] = z["feats"]
        feats[v + "_pos"] = pos
    rows = [r for r in rows if all(key(r) in feats[v + "_pos"] for v in a.views)]
    F = np.concatenate([feats[v][[feats[v + "_pos"][key(r)] for r in rows]].astype(np.float32) for v in a.views], 2)
    folds_all = replay_folds(rows)
    results = {}
    for task in a.tasks:
        ylab = task_labels(rows, task)
        sel = np.array([l is not None for l in ylab])
        classes = sorted({l for l in ylab if l is not None})
        cid = {c: i for i, c in enumerate(classes)}
        y = np.array([cid[l] for l, s in zip(ylab, sel) if s])
        R = [r for r, s in zip(rows, sel) if s]
        folds = folds_all[sel]
        # replay-grouped folds may leave a fold without a class: fine (pooled OOF report)
        ncls = len(classes)
        res = {"n": int(sel.sum()), "replays": len({r['replay'] for r in R}),
               "class_counts": {c: int((y == i).sum()) for i, c in enumerate(classes)}}
        # majority (per training fold)
        maj = np.zeros((len(y), ncls))
        for f in np.unique(folds):
            maj[folds == f, np.bincount(y[folds != f], minlength=ncls).argmax()] = 1
        res["majority"] = report(y, maj.argmax(1), classes, maj)
        Xdd = tab(R, ["damage", "distance"])
        p = fit_tab(Xdd, y, folds, ncls)
        res["damage_distance"] = report(y, p.argmax(1), classes, p)
        Xlog = tab(R, ["damage", "distance", "attacker_airborne", "defender_airborne", "hits", "hitstop", "attacker_y"])
        p = fit_tab(Xlog, y, folds, ncls)
        res["log_features"] = report(y, p.argmax(1), classes, p)
        p = fit_head(F[sel], y, folds, ncls, epochs=a.epochs)
        res["visual"] = report(y, p.argmax(1), classes, p)
        np.save(OUT / f"oof_{task.replace(':', '_')}_{a.backbone}_{'+'.join(a.views)}{a.tag}.npy", p)
        results[task] = res
        print(f"{task}: n={res['n']} classes={ncls}  majority {res['majority']['accuracy']:.3f}  "
              f"dmg+dist {res['damage_distance']['accuracy']:.3f}  log {res['log_features']['accuracy']:.3f}  "
              f"visual {res['visual']['accuracy']:.3f} (bal {res['visual']['balanced_accuracy']:.3f}, "
              f"top3 {res['visual'].get('top3', 0):.3f})", flush=True)
    save_json(results, OUT / f"results_{a.backbone}_{'+'.join(a.views)}{a.tag}.json")


if __name__ == "__main__":
    main()
