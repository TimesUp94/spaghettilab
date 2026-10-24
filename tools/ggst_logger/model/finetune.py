"""End-to-end fine-tuned ResNet18 (shared per frame, temporal conv late fusion) on the attacker
crops, 5-fold CV grouped by replay. Short runs, small batches, below-normal priority.

    .venv/Scripts/python.exe tools/ggst_logger/model/finetune.py --task category --epochs 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OFFS, OUT, load_rows, low_priority, replay_folds, report, save_json, task_labels  # noqa: E402

FRAMES = [-12, -9, -6, -4, -2, 0]
RES = 160


def load_images(rows):
    import cv2
    idx = [OFFS.index(o) for o in FRAMES]
    X = np.zeros((len(rows), len(FRAMES), RES, RES, 3), np.uint8)
    for i, r in enumerate(rows):
        img = cv2.imread(str(DATA / "crops" / r["replay"] / f"{r['log_row']}_{r['attacker']}.jpg"))[:, :, ::-1]
        w = img.shape[1] // len(OFFS)
        for j, t in enumerate(idx):
            X[i, j] = cv2.resize(img[:, t * w:(t + 1) * w], (RES, RES), interpolation=cv2.INTER_AREA)
    return X


def make_net(ncls):
    import torch.nn as nn
    import torchvision

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            b = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            b.fc = nn.Identity()
            self.b = b
            self.t = nn.Sequential(nn.Conv1d(512, 256, 3, padding=1), nn.GELU(), nn.AdaptiveMaxPool1d(1),
                                   nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, ncls))

        def forward(self, x):            # x [B, T, 3, H, W]
            B, T = x.shape[:2]
            f = self.b(x.flatten(0, 1)).view(B, T, -1)
            return self.t(f.transpose(1, 2))
    return Net()


def run(X, y, folds, ncls, epochs, bs=16, lr=3e-4):
    import torch
    import torch.nn as nn
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)

    def prep(xb, train):
        x = torch.from_numpy(xb).cuda().permute(0, 1, 4, 2, 3).float().div(255)
        if train:      # light augmentation: brightness/contrast jitter + random shift
            B = x.shape[0]
            x = x * (0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device="cuda")) + 0.1 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)
            dx, dy = np.random.randint(-12, 13, 2)
            x = torch.roll(x, (int(dy), int(dx)), (3, 4))
        return ((x.clamp(0, 1) - mean) / std)

    proba = np.zeros((len(y), ncls))
    for f in np.unique(folds):
        t0 = time.time()
        tr, te = np.nonzero(folds != f)[0], np.nonzero(folds == f)[0]
        net = make_net(ncls).cuda()
        cnt = np.bincount(y[tr], minlength=ncls).astype(float)
        w = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(w / w[w > 0].mean(), dtype=torch.float32).cuda(), label_smoothing=0.1)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
        steps = epochs * ((len(tr) + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
        scaler = torch.amp.GradScaler()
        for ep in range(epochs):
            net.train()
            perm = np.random.permutation(tr)
            for b in range(0, len(perm), bs):
                idx = np.sort(perm[b:b + bs])
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lossf(net(prep(X[idx], True)), torch.from_numpy(y[idx]).cuda())
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
        net.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(te), 32):
                out.append(torch.softmax(net(prep(X[te[b:b + 32]], False)).float(), 1).cpu().numpy())
        proba[te] = np.concatenate(out)
        acc = float(np.mean(proba[te].argmax(1) == y[te]))
        print(f"  fold {f}: n_test={len(te)} acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del net
        torch.cuda.empty_cache()
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="category")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    low_priority()
    import torch
    torch.set_num_threads(4)
    rows = load_rows()
    folds_all = replay_folds(rows)
    ylab = task_labels(rows, a.task)
    sel = np.array([l is not None for l in ylab])
    classes = sorted({l for l in ylab if l is not None})
    cid = {c: i for i, c in enumerate(classes)}
    R = [r for r, s in zip(rows, sel) if s]
    y = np.array([cid[l] for l, s in zip(ylab, sel) if s])
    X = load_images(R)
    p = run(X, y, folds_all[sel], len(classes), a.epochs)
    res = report(y, p.argmax(1), classes, p)
    tag = a.task.replace(":", "_").replace("?", "")   # Bedman?: no ? in Windows file names
    np.save(OUT / f"oof_ft_{tag}.npy", p)
    save_json(res, OUT / f"results_ft_{tag}.json")
    print(f"{a.task}: finetuned acc {res['accuracy']:.3f} bal {res['balanced_accuracy']:.3f} top3 {res.get('top3', 0):.3f}")


if __name__ == "__main__":
    main()
