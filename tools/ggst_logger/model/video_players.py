"""Where both players are, and who they are, from a video frame alone.

One network on the small full frame (256x144, the full/ tiles of extract_crops.py), CenterNet-like:
a heatmap of both players' centres (stride 8 on the frame upsampled 2x) and, per cell, the side and
character standing there; the two peaks go to P1/P2 by those (with the replay's characters known,
mirror matches by side alone). A pooled head reads both characters per frame for the replay vote.
Targets are what the crops were cut from: the logged positions through camera.py (1920x1080 pixels) and the replay's characters.

    .venv/Scripts/python.exe tools/ggst_logger/model/video_players.py labels   # positions per tile
    .venv/Scripts/python.exe tools/ggst_logger/model/video_players.py train    # folds 1-4, test fold 0
    .venv/Scripts/python.exe tools/ggst_logger/model/video_players.py final    # every replay

-> <data>/ggst_dataset/model/players_labels.npz, video_players_{test,final}.pt,
   results_video_players.json
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
from common import DATA, OFFS, OUT, TEST_FOLD, low_priority, replay_fold_map, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LABELS = OUT / "players_labels.npz"
W, H = 1920, 1080
IN_W, IN_H = 512, 288           # network input (the 256x144 tile upsampled)


def characters(rows):
    return sorted({r["attacker_char"] for r in rows} | {r["defender_char"] for r in rows})


def make_labels():
    """For every opener with a full/ tile: P1/P2 pixel positions at each of the 13 offsets."""
    import pyarrow.parquet as pq
    sys.path.insert(0, str(ROOT / "tools" / "ggst_logger" / "dataset"))
    import build
    import camera
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        if (DATA / "full" / r["replay"] / f"{r['log_row']}_{r['attacker']}.jpg").exists():
            by[r["replay"]].append(r)
    keys, pos, face = [], [], []
    for k, (stem, rr) in enumerate(sorted(by.items())):
        log = build.load_log(rr[0]["csv"])
        cam = camera.camera(log)
        n = len(log["t"])
        for r in rr:
            p = np.zeros((len(OFFS), 2, 2), np.float32)
            f = np.zeros((len(OFFS), 2), np.int8)
            for j, o in enumerate(OFFS):
                row = min(max(0, r["log_row"] + o), n - 1)
                for s in (1, 2):
                    p[j, s - 1] = camera.to_px(log, cam, row, s)
                    f[j, s - 1] = 1 if log[f"p{s}_face"][row] == 0 else -1     # 1 = faces right
            keys.append(f"{stem}/{r['log_row']}_{r['attacker']}")
            pos.append(p)
            face.append(f)
        if k % 50 == 0:
            print(f"  {k}/{len(by)} replays", flush=True)
    np.savez_compressed(LABELS, keys=np.array(keys), pos=np.stack(pos), face=np.stack(face))
    print(len(keys), "openers labelled")


class Tiles:
    """Dataset of (frame, P1/P2 positions, P1/P2 characters) from the full/ strips."""

    def __init__(self, items, train):
        self.items, self.train = items, train     # items: (path, offset index, pos [2,2], chars [2])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        path, j, pos, ch = self.items[i]
        img = cv2.imread(path)
        w = img.shape[1] // len(OFFS)
        t = img[:, j * w:(j + 1) * w, ::-1]
        t = cv2.resize(t, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        x = t.astype(np.float32) / 255
        if self.train:
            x = np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.05, 0.05), 0, 1)
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        y = np.asarray(pos, np.float32) / [W, H]          # [2 players, (x, y)] in 0..1 of the frame
        return x.transpose(2, 0, 1).astype(np.float32), y.astype(np.float32), np.asarray(ch, np.int64)


def make_net(nchar):
    """ResNet18; stride-8 map (layer2 + upsampled layer3) with a player-centre heatmap and, per
    cell, which side (P1/P2) and which character stands there; pooled layer4 -> both sides'
    characters for the whole frame (the replay vote)."""
    import torch
    import torch.nn as nn
    import torchvision

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            b = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.l2 = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2)
            self.l3, self.l4 = b.layer3, b.layer4
            self.fuse = nn.Sequential(nn.Conv2d(128 + 256, 192, 3, padding=1), nn.BatchNorm2d(192), nn.GELU(),
                                      nn.Conv2d(192, 192, 3, padding=1), nn.BatchNorm2d(192), nn.GELU())
            self.heat = nn.Conv2d(192, 1, 1)
            self.heat.bias.data.fill_(-2.2)
            self.side = nn.Conv2d(192, 2, 1)
            self.cchar = nn.Conv2d(192, nchar, 1)
            self.char = nn.Linear(512, 2 * nchar)
            self.nchar = nchar

        def forward(self, x):
            f2 = self.l2(x)                                    # [B, 128, H/8, W/8]
            f3 = self.l3(f2)
            f = self.fuse(torch.cat([f2, nn.functional.interpolate(f3, size=f2.shape[2:], mode="nearest")], 1))
            g = self.char(self.l4(f3).mean((2, 3))).view(x.shape[0], 2, self.nchar)
            return dict(heat=self.heat(f)[:, 0], side=self.side(f), cchar=self.cchar(f), gchar=g)
    return Net()


def cells(y, hh, ww):
    """Normalised (x, y) [B, 2, 2] -> integer cell indices (clamped) [B, 2, 2]."""
    import torch
    cx = (y[..., 0] * ww).long().clamp(0, ww - 1)
    cy = (y[..., 1] * hh).long().clamp(0, hh - 1)
    return torch.stack([cx, cy], -1)


def losses(out, y, c, sigma=1.5):
    """CenterNet focal loss on the heatmap, CE for side and character at both true centres."""
    import torch
    import torch.nn.functional as F
    heat = out["heat"].float()
    B, hh, ww = heat.shape
    gx = (torch.arange(ww, device=heat.device).float() + 0.5) / ww
    gy = (torch.arange(hh, device=heat.device).float() + 0.5) / hh
    tgt = torch.zeros_like(heat)
    inside = (y[..., 0] >= 0) & (y[..., 0] <= 1) & (y[..., 1] >= -0.1) & (y[..., 1] <= 1.1)
    for k in range(2):
        dx = (gx[None, None, :] - y[:, k, 0, None, None]) * ww
        dy = (gy[None, :, None] - y[:, k, 1, None, None]) * hh
        g = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)) * inside[:, k, None, None]
        tgt = torch.maximum(tgt, g)
    ce = cells(y, hh, ww)
    bi = torch.arange(B, device=heat.device)
    for k in range(2):
        cur = tgt[bi, ce[:, k, 1], ce[:, k, 0]]
        tgt[bi, ce[:, k, 1], ce[:, k, 0]] = torch.where(inside[:, k], torch.ones_like(cur), cur)
    p = torch.sigmoid(heat).clamp(1e-4, 1 - 1e-4)
    pos = tgt.eq(1).float()
    lpos = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    lneg = -(torch.log(1 - p) * p ** 2 * (1 - tgt) ** 4 * (1 - pos)).sum()
    lheat = (lpos + lneg) / pos.sum().clamp(min=1)
    lside = lchar = 0
    for k in range(2):
        m = inside[:, k]
        if m.any():
            sv = out["side"].float()[bi, :, ce[:, k, 1], ce[:, k, 0]]
            cv = out["cchar"].float()[bi, :, ce[:, k, 1], ce[:, k, 0]]
            lside = lside + F.cross_entropy(sv[m], torch.full((int(m.sum()),), k, device=heat.device))
            lchar = lchar + F.cross_entropy(cv[m], c[m, k])
    lg = F.cross_entropy(out["gchar"].float().flatten(0, 1), c.flatten())
    return lheat + lside + 0.5 * lchar + 0.5 * lg


def decode(out, chars=None, window=False):
    """Positions [B, 2, 2] (0..1 of the frame) of P1 and P2, and frame-level character log-probs
    [B, 2, C]. The two strongest heatmap peaks are assigned to P1/P2 by the per-cell side and
    character scores (chars: known (P1, P2) character indices [B, 2], else the frame's own
    vote). One peak only (players overlap): both get it."""
    import torch
    import torch.nn.functional as F
    heat = torch.sigmoid(out["heat"].float())
    B, hh, ww = heat.shape
    nms = (heat == F.max_pool2d(heat[:, None], 5, 1, 2)[:, 0]).float() * heat
    sc, ix = nms.flatten(1).topk(2, 1)
    gl = torch.log_softmax(out["gchar"].float(), -1)
    if chars is None:
        chars = gl.argmax(-1)
    side = torch.log_softmax(out["side"].float(), 1)
    cch = torch.log_softmax(out["cchar"].float(), 1)
    pad = F.pad(heat, (1, 1, 1, 1))
    bi = torch.arange(B, device=heat.device)
    off = torch.tensor([-1.0, 0, 1], device=heat.device)
    pk = []
    for k in range(2):
        cy, cx = ix[:, k] // ww, ix[:, k] % ww
        # sub-cell: heat-weighted centroid of the 3x3 around the peak
        win = torch.stack([pad[bi, cy + dy, cx + dx] for dy in range(3) for dx in range(3)], 1).view(B, 3, 3)
        w = win / win.sum((1, 2), keepdim=True).clamp(min=1e-6)
        ox = (w.sum(1) * off).sum(1)
        oy = (w.sum(2) * off).sum(1)
        xy = torch.stack([(cx.float() + 0.5 + ox) / ww, (cy.float() + 0.5 + oy) / hh], -1)
        s1 = side[bi, 0, cy, cx] + cch[bi, chars[:, 0], cy, cx]
        s2 = side[bi, 1, cy, cx] + cch[bi, chars[:, 1], cy, cx]
        pk.append((xy, s1, s2))
    (a, a1, a2), (b, b1, b2) = pk
    if window:
        # peaks ordered left to right, then one P1/P2 decision for the whole batch (a short window:
        # players seldom change sides in it; per frame swaps 9 % of frames, per window 6 %)
        lr = a[:, 0] > b[:, 0]
        a, b = torch.where(lr[:, None], b, a), torch.where(lr[:, None], a, b)
        a1, b1 = torch.where(lr, b1, a1), torch.where(lr, a1, b1)
        a2, b2 = torch.where(lr, b2, a2), torch.where(lr, a2, b2)
        keep = ((a1 + b2) - (a2 + b1)).sum() >= 0
        keep = keep.expand(B)
    else:
        keep = a1 + b2 >= a2 + b1
    p1 = torch.where(keep[:, None], a, b)
    p2 = torch.where(keep[:, None], b, a)
    single = sc[:, 1] < 0.1 * sc[:, 0]
    p1 = torch.where(single[:, None], a, p1)
    p2 = torch.where(single[:, None], a, p2)
    return torch.stack([p1, p2], 1), gl


def items_for(rows, stems, lab):
    idx = {k: i for i, k in enumerate(lab["keys"])}
    chars = characters(rows)
    cid = {c: i for i, c in enumerate(chars)}
    out = []
    for r in rows:
        k = f"{r['replay']}/{r['log_row']}_{r['attacker']}"
        if r["replay"] not in stems or k not in idx:
            continue
        a = int(r["attacker"][1])
        ch = [0, 0]
        ch[a - 1], ch[2 - a] = cid[r["attacker_char"]], cid[r["defender_char"]]
        path = str(DATA / "full" / r["replay"] / f"{r['log_row']}_{r['attacker']}.jpg")
        for j in range(len(OFFS)):
            out.append((path, j, lab["pos"][idx[k], j], ch))
    return out, chars


def loader(items, train, bs=48):
    import torch
    return torch.utils.data.DataLoader(Tiles(items, train), batch_size=bs, shuffle=train, num_workers=6,
                                       persistent_workers=True, drop_last=train, pin_memory=True)


def fit(items, nchar, epochs, per_epoch=60000, lr=1e-3):
    import torch
    net = make_net(nchar).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    steps = epochs * (per_epoch // 48)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    for ep in range(epochs):
        net.train()
        sub = [items[i] for i in rng.choice(len(items), per_epoch, replace=False)]
        tot, n, t0 = 0.0, 0, time.time()
        for x, y, c in loader(sub, True):
            x, y, c = x.cuda(non_blocking=True), y.cuda(non_blocking=True), c.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(x)
            loss = losses(out, y, c)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            n += 1
        print(f"  epoch {ep + 1}/{epochs} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net


def evaluate(net, items, chars, rows):
    """Errors with the true characters given (the replay vote is right on every test replay)."""
    import torch
    net.eval()
    P, Y, C, CT = [], [], [], []
    with torch.no_grad():
        for x, y, c in loader(items, False, 96):
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(x.cuda())
            pos, gl = decode(out, c.cuda())
            P.append(pos.cpu().numpy())
            Y.append(y.numpy())
            C.append(gl.exp().cpu().numpy())
            CT.append(c.numpy())
    P, Y, C, CT = (np.concatenate(v) for v in (P, Y, C, CT))
    P *= [W, H]
    Y *= [W, H]
    err = np.linalg.norm(P - Y, axis=-1)                      # [N, 2]
    swapped = np.linalg.norm(P - Y[:, ::-1], axis=-1).sum(1) < err.sum(1)
    xerr = np.abs(P[..., 0] - Y[..., 0])
    res = dict(frames=len(P),
               err_px_median=float(np.median(err)), err_px_p90=float(np.percentile(err, 90)),
               within_100=float(np.mean(err <= 100)), within_200=float(np.mean(err <= 200)),
               x_err_median=float(np.median(xerr)), swapped_sides=float(np.mean(swapped)),
               char_acc_frame=float(np.mean(C.argmax(-1) == CT)))
    mirror = CT[:, 0] == CT[:, 1]
    res["swapped_sides_mirror"] = float(np.mean(swapped[mirror])) if mirror.any() else None
    reps = [Path(it[0]).parent.name for it in items]
    by = collections.defaultdict(list)
    for i, r in enumerate(reps):
        by[r].append(i)
    ok = [np.all(np.log(C[ix] + 1e-6).sum(0).argmax(-1) == CT[ix[0]]) for ix in by.values()]
    res["char_acc_replay"] = float(np.mean(ok))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["labels", "train", "final"])
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    low_priority()
    if a.mode == "labels":
        return make_labels()
    import pyarrow.parquet as pq
    import torch
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    lab = dict(np.load(LABELS))
    fold = replay_fold_map()
    stems = set(fold)
    test = {s for s in stems if fold[s] == TEST_FOLD}
    train_items, chars = items_for(rows, stems if a.mode == "final" else stems - test, lab)
    print(len(train_items), "training frames,", len(chars), "characters")
    net = fit(train_items, len(chars), a.epochs)
    tag = "final" if a.mode == "final" else "test"
    torch.save({"state": net.state_dict(), "chars": chars}, OUT / f"video_players_{tag}.pt")
    if a.mode == "train":
        test_items, _ = items_for(rows, test, lab)
        res = evaluate(net, test_items, chars, rows)
        save_json(res, OUT / "results_video_players.json")
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
