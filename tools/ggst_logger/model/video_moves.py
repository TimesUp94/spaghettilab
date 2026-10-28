"""Move recognition on the video-only crops (video_crops.py): one network, shared per-frame backbone
with temporal fusion, and heads for
- category (normal / special / super / throw / projectile / other, plus 'none' for a detection
  that is not a logged opener, so the detector's false alarms can be rejected),
- button x stance of normals,
- counter / punish / neither,
- the exact move: one vocabulary over every character, the logits restricted to the attacker's
  character (the log's at training, the video vote's in the end-to-end test).
A second backbone also sees the whole frame at low resolution (video_scene.py) at the
same offsets, for projectiles, summons and hits away from the attacker crop (run video_scene.py
after video_crops.py). The chain's weights are an ensemble of a ResNet18 (20 epochs) and a
ResNet50 (10 epochs) crop backbone; predictions are averaged over both and over the training
frame shifts (TTA).

    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py train --epochs 20 --name r18 --eval-crops exp_tiny10
    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py train --backbone resnet50 --name r50 --eval-crops exp_tiny10
    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py ensemble video_moves_exp_r18.pt video_moves_exp_r50.pt --name test
    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py final --epochs 20 --name final_r18   # every replay
    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py final --backbone resnet50 --name final_r50
    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py ensemble video_moves_exp_final_r18.pt video_moves_exp_final_r50.pt --name final
    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py train --frac 0.25   # learning curve
    .venv/Scripts/python.exe tools/ggst_logger/model/video_moves.py train --no-scene --name crop   # crop only

-> <data>/ggst_dataset/model/video_moves_{test,final}.pt, results_video_moves.json
   (--frac: a share of the training replays -> video_moves_exp_frac<pct>.pt, results_exp_moves_frac<pct>.json)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAT6, DATA, OUT, TEST_FOLD, collapse_rare, label_category, label_hit_kind, label_normal,  # noqa: E402
                    low_priority, replay_fold_map, report, save_json, subsample)

WIN = list(range(-16, 5))
FRAMES = [-15, -12, -9, -6, -4, -2, 0, 2]
SHIFTS = [-1, 0, 1, 2]
TTA = SHIFTS          # test time: probabilities averaged over the training frame shifts
RES = 192
SCENE = (384, 216)
HEADS = ("category", "normal", "hit_kind", "move")
IGNORE = -100


def items_for(root: Path, rows_by_key, stems):
    """One item per detection: (jpg, labels dict, attacker character (log, if an opener))."""
    out = []
    for s in stems:
        f = root / s / "index.json"
        if not f.exists():
            continue
        idx = json.loads(f.read_text())
        for d in idx["dets"]:
            r = rows_by_key.get(f"{s}/{d['row']}") if d["row"] else None
            out.append(dict(stem=s, file=str(root / s / d["file"]), frame=d["frame"], side=d["side"],
                            score=d["score"], row=r, scene=str(scene_root(root) / s / d["file"])))
    return out


def vocab(items):
    """Class lists per head from the training items (rare moves -> '<char>:(rare)')."""
    rows = [it["row"] for it in items if it["row"] is not None]
    cat = sorted(set(CAT6.values()) | {"none"})
    nor = sorted({l for l in collapse_rare([label_normal(r) for r in rows], 5) if l is not None})
    hk = sorted({l for l in (label_hit_kind(r) for r in rows) if l is not None})
    mv = sorted({l for l in collapse_rare([f"{r['char']}:{r['move']}" for r in rows], 5) if l is not None})
    # every character's rare bucket exists, so any rare move has a class
    mv = sorted(set(mv) | {f"{r['char']}:(rare)" for r in rows})
    return {"category": cat, "normal": nor, "hit_kind": hk, "move": mv}


def targets(it, voc):
    r = it["row"]
    if r is None:
        return {"category": voc["category"].index("none"), "normal": IGNORE, "hit_kind": IGNORE, "move": IGNORE}
    nor = label_normal(r)
    hk = label_hit_kind(r)
    mv = f"{r['char']}:{r['move']}"
    if mv not in voc["move"]:
        mv = f"{r['char']}:(rare)"
    return {"category": voc["category"].index(label_category(r)),
            "normal": voc["normal"].index(nor) if nor in voc["normal"] else IGNORE,
            "hit_kind": voc["hit_kind"].index(hk) if hk is not None else IGNORE,
            "move": voc["move"].index(mv) if mv in voc["move"] else IGNORE}


def scene_root(root: Path) -> Path:
    return root.parent / root.name.replace("vcrops", "vscene")


def char_mask(voc, char):
    return np.array([m.startswith(char + ":") for m in voc["move"]])


class Crops:
    def __init__(self, items, voc, train, chars=None, scene=None, shift=0, flip=False, frames=None, res=None):
        self.items, self.voc, self.train, self.scene = items, voc, train, scene
        self.frames, self.res = frames or FRAMES, res or RES
        self.shift, self.flip = shift, flip   # test-time variants: frame offset, mirrored image
        self.chars = chars           # stem -> (P1 char, P2 char) from the video vote; None = the log's
        self.masks = {}

    def __len__(self):
        return len(self.items)

    def mask(self, char):
        if char not in self.masks:
            self.masks[char] = char_mask(self.voc, char)
        return self.masks[char]

    def __getitem__(self, i):
        import cv2
        it = self.items[i]
        img = cv2.imread(it["file"])[:, :, ::-1]
        w = img.shape[1] // len(WIN)
        sh = int(np.random.choice(SHIFTS)) if self.train else self.shift
        offs = [o + sh for o in self.frames]
        res = self.res
        if self.train and np.random.random() < 0.3:
            # a 30 fps video converted to 60 fps on decode shows every frame twice
            par = np.random.randint(2)
            offs = [o - ((o + par) % 2) for o in offs]
        offs = [min(max(o, WIN[0]), WIN[-1]) for o in offs]
        x = np.stack([cv2.resize(img[:, WIN.index(o) * w:(WIN.index(o) + 1) * w], (res, res),
                                 interpolation=cv2.INTER_AREA) for o in offs])
        if self.train and np.random.random() < 0.3:
            # 720p / compressed sources: lose detail (the crop window is ~1/3 of the frame height)
            k = np.random.uniform(0.45, 0.8)
            lo = int(res * k)
            x = np.stack([cv2.resize(cv2.resize(t, (lo, lo), interpolation=cv2.INTER_AREA), (res, res),
                                     interpolation=cv2.INTER_LINEAR) for t in x])
        xs = np.zeros(0, np.uint8)
        if self.scene:
            sw, sh = self.scene
            img = cv2.imread(it["scene"])[:, :, ::-1]
            w = img.shape[1] // len(WIN)
            xs = np.stack([cv2.resize(img[:, WIN.index(o) * w:(WIN.index(o) + 1) * w], (sw, sh),
                                      interpolation=cv2.INTER_AREA) for o in offs])
        if self.flip:
            x = np.ascontiguousarray(x[:, :, ::-1])
            xs = np.ascontiguousarray(xs[:, :, ::-1]) if xs.size else xs
        t = targets(it, self.voc)
        if self.chars is not None:
            char = self.chars[it["stem"]][it["side"] - 1]
        else:
            char = it["row"]["char"] if it["row"] is not None else None
        m = self.mask(char) if char is not None else np.zeros(len(self.voc["move"]), bool)
        return x, xs, np.array([t[h] for h in HEADS], np.int64), m


def make_net(voc, backbone="resnet18", scene=None):
    import torch
    import torch.nn as nn
    import torchvision

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if backbone == "resnet18":
                b = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                d = 512
            elif backbone == "resnet50":
                b = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
                d = 2048
            b.fc = nn.Identity()
            self.b = b
            self.s = None
            if scene:
                self.s = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                self.s.fc = nn.Identity()
                d += 512
            self.t = nn.Sequential(nn.Conv1d(d, 384, 3, padding=1), nn.GELU(), nn.Conv1d(384, 384, 3, padding=1),
                                   nn.GELU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(0.3))
            self.heads = nn.ModuleDict({h: nn.Linear(384, len(voc[h])) for h in HEADS})

        def forward(self, x, xs=None):   # x [B, T, 3, H, W], xs the scene [B, T, 3, h, w]
            B, T = x.shape[:2]
            f = self.b(x.flatten(0, 1)).view(B, T, -1)
            if self.s is not None:
                f = torch.cat([f, self.s(xs.flatten(0, 1)).view(B, T, -1)], -1)
            z = self.t(f.transpose(1, 2))
            return {h: m(z) for h, m in self.heads.items()}
    return Net()


MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3, 1, 1)


def prep(x, train, shift=16):
    import torch
    if x.ndim < 5:
        return None
    x = x.cuda(non_blocking=True).permute(0, 1, 4, 2, 3).float().div(255)
    if train:
        B = x.shape[0]
        x = x * (0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device="cuda")) + 0.1 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)
        dx, dy = np.random.randint(-shift, shift + 1, 2)
        x = torch.roll(x, (int(dy), int(dx)), (3, 4))
    return (x.clamp(0, 1) - torch.from_numpy(MEAN).cuda()) / torch.from_numpy(STD).cuda()


def loader(ds, train, bs):
    import torch
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train, num_workers=int(os.environ.get("VM_WORKERS", "8")), persistent_workers=False,
                                       drop_last=train, pin_memory=True)


def fit(items, voc, epochs, backbone, bs=24, lr=3e-4, scene=None, frames=None, res=None):
    import torch
    import torch.nn as nn
    net = make_net(voc, backbone, scene).cuda()
    ys = np.array([[targets(it, voc)[h] for h in HEADS] for it in items])
    weights = {}
    for k, h in enumerate(HEADS):
        cnt = np.bincount(ys[:, k][ys[:, k] >= 0], minlength=len(voc[h])).astype(float)
        w = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0)
        # capped: torch applies class weights to the smoothing term too, so a rare class (category
        # 'other') with a huge weight pulls every sample towards itself
        weights[h] = torch.tensor(np.minimum(w / w[w > 0].mean(), 4.0), dtype=torch.float32).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * (len(items) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    ds = Crops(items, voc, True, scene=scene, frames=frames, res=res)
    for ep in range(epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, xs, y, m in loader(ds, True, bs):
            y, m = y.cuda(), m.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(prep(x, True), prep(xs, True, 8))
                loss = 0
                for k, h in enumerate(HEADS):
                    lg = out[h].float()
                    if h == "move":
                        lg = lg.masked_fill(~m, -1e4)
                    loss = loss + nn.functional.cross_entropy(lg, y[:, k], weight=weights[h], ignore_index=IGNORE,
                                                              label_smoothing=0.0 if h == "move" else 0.05) * (1.0 if h != "hit_kind" else 0.5)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            n += 1
        print(f"  epoch {ep + 1}/{epochs} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net


def load(path):
    """(net on the GPU, checkpoint) of saved weights; an ensemble file gives a list of nets."""
    import torch
    ck = torch.load(path, map_location="cpu")
    if "members" in ck:
        return [member(m) for m in ck["members"]], ck
    return member(ck), ck


def member(ck):
    net = make_net(ck["voc"], ck.get("backbone", "resnet18"), ck.get("scene")).cuda()
    net.load_state_dict(ck["state"])
    net.scene, net.frames, net.res = ck.get("scene"), ck.get("frames", FRAMES), ck.get("res", RES)
    return net


def ensemble(paths, out):
    """One file holding several nets trained on the same items (same vocabulary): their
    probabilities are averaged."""
    import torch
    cks = [torch.load(p, map_location="cpu") for p in paths]
    assert all(c["voc"] == cks[0]["voc"] for c in cks)
    torch.save({"members": cks, "voc": cks[0]["voc"], "scene": next((c["scene"] for c in cks if c.get("scene")), None)}, out)


def predict(net, items, voc, chars, shifts=TTA, flip=False):
    """Per item: probabilities per head (move restricted to the given characters), the mean
    over the frame shifts (and the nets of an ensemble)."""
    nets = net if isinstance(net, list) else [net]
    P = [predict_one(n, items, voc, chars, s, flip) for n in nets for s in shifts]
    return {h: np.mean([p[h] for p in P], 0) for h in HEADS}


def predict_one(net, items, voc, chars, shift=0, flip=False):
    import torch
    net.eval()
    ds = Crops(items, voc, False, chars, getattr(net, "scene", None), shift, flip,
               getattr(net, "frames", None), getattr(net, "res", None))
    out = {h: [] for h in HEADS}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, xs, _, m in loader(ds, False, 48):
            o = net(prep(x, False), prep(xs, False))
            for h in HEADS:
                lg = o[h].float()
                if h == "move":
                    lg = lg.masked_fill(~m.cuda(), -1e4)
                out[h].append(torch.softmax(lg, -1).cpu().numpy())
    return {h: np.concatenate(v) for h, v in out.items()}


def video_chars(root, stems, chars_list):
    out = {}
    for s in stems:
        f = root / s / "index.json"
        if f.exists():
            lp = np.array(json.loads(f.read_text())["chars_logp"])
            out[s] = (chars_list[int(lp[0].argmax())], chars_list[int(lp[1].argmax())])
    return out


def end_to_end(items, P, voc, chars, rows, stems):
    """Every logged opener of the test replays (bursts apart: they deal no damage, the video can't
    see them): found by the detector? then category / move right (move with the video's character)?"""
    res = {}
    tru = [r for r in rows if r["replay"] in stems and not r["is_burst"]]
    found = {}
    for i, it in enumerate(items):
        if it["row"] is not None:
            found[f"{it['row']['replay']}/{it['row']['log_row']}_{it['row']['attacker']}"] = i
    cat_none = voc["category"].index("none")
    n = len(tru)
    det = [found.get(f"{r['replay']}/{r['log_row']}_{r['attacker']}") for r in tru]
    kept = [i for i in det if i is not None and P["category"][i].argmax() != cat_none]
    res["openers"] = n
    res["detected"] = sum(i is not None for i in det) / n
    res["detected_and_kept"] = len(kept) / n
    fp = [i for i, it in enumerate(items) if it["row"] is None]
    res["false_detections"] = len(fp)
    res["false_detections_kept"] = sum(P["category"][i].argmax() != cat_none for i in fp)
    res["precision_after_none"] = len(kept) / max(1, len(kept) + res["false_detections_kept"])
    # given detected (whatever the category head says about 'none')
    dd = [(r, i) for r, i in zip(tru, det) if i is not None]
    yc = [voc["category"].index(label_category(r)) for r, _ in dd]
    pc = P["category"][[i for _, i in dd]]
    pc_nonone = pc.copy()
    pc_nonone[:, cat_none] = 0
    res["category"] = report(yc, pc_nonone.argmax(1), voc["category"], pc_nonone)
    char_ok = [chars[r["replay"]][int(r["attacker"][1]) - 1] == r["char"] for r, _ in dd]
    res["video_char_right"] = float(np.mean(char_ok))
    mv = []
    for (r, i), ok in zip(dd, char_ok):
        lab = f"{r['char']}:{r['move']}"
        if lab not in voc["move"]:
            lab = f"{r['char']}:(rare)"
        mv.append((r["char"], lab == voc["move"][int(P["move"][i].argmax())]))
    res["move_acc"] = float(np.mean([ok for _, ok in mv]))
    res["move_acc_by_char"] = {c: round(float(np.mean([ok for cc, ok in mv if cc == c])), 3)
                               for c in sorted({c for c, _ in mv})}
    nr = [(label_normal(r), i) for r, i in dd if label_normal(r) in voc["normal"]]
    res["normal_acc"] = float(np.mean([voc["normal"][int(P["normal"][i].argmax())] == l for l, i in nr]))
    hk = [(label_hit_kind(r), i) for r, i in dd if label_hit_kind(r) is not None]
    res["hit_kind"] = report([voc["hit_kind"].index(l) for l, _ in hk], P["hit_kind"][[i for _, i in hk]].argmax(1),
                             voc["hit_kind"])
    # the whole chain: a logged opener counts when it is found, kept and its move is right
    right = 0
    for (r, i), (_, ok) in zip(dd, mv):
        right += ok and P["category"][i].argmax() != cat_none
    res["chain_move_right_of_all_openers"] = right / n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "final", "ensemble"])
    ap.add_argument("members", nargs="*", help="ensemble: weights in model/ to join into --name")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--crops", default="test")
    ap.add_argument("--frac", type=float, default=1.0, help="share of the training replays (by replay, seed 0)")
    ap.add_argument("--scene", action=argparse.BooleanOptionalAction, default=True, help="the whole-frame stream (video_scene.py)")
    ap.add_argument("--res", type=int, default=RES, help="crop side fed to the backbone (tiles are 224)")
    ap.add_argument("--frames", default=",".join(map(str, FRAMES)), help="frame offsets from the hit, within WIN")
    ap.add_argument("--eval-crops", help="crops dir of the test fold (default: --crops), e.g. exp_tiny10")
    ap.add_argument("--name", help="experiment name -> video_moves_exp_<name>.pt, results_exp_moves_<name>.json")
    a = ap.parse_args()
    if a.mode == "ensemble":
        return ensemble([OUT / m for m in a.members], OUT / f"video_moves_{a.name}.pt")
    low_priority()
    import pyarrow.parquet as pq
    import torch
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    fold = replay_fold_map()
    root = DATA / f"vcrops_{a.crops}"
    stems = sorted(fold)
    test = [s for s in stems if fold[s] == TEST_FOLD]
    train_stems = stems if a.mode == "final" else [s for s in stems if fold[s] != TEST_FOLD]
    train_stems = subsample(train_stems, a.frac)
    items = items_for(root, by_key, train_stems)
    voc = vocab(items)
    print(len(items), "training detections;", {h: len(v) for h, v in voc.items()}, flush=True)
    scene = SCENE if a.scene else None
    frames = [int(f) for f in a.frames.split(",")]
    net = fit(items, voc, a.epochs, a.backbone, scene=scene, frames=frames, res=a.res)
    net.scene, net.frames, net.res = scene, frames, a.res
    tag = "final" if a.mode == "final" else "test"
    suf = "" if a.backbone == "resnet18" else "_" + a.backbone
    if a.frac < 1:
        tag, suf = "exp", f"_frac{round(a.frac * 100)}" + suf
    if a.name:
        tag, suf = "exp", f"_{a.name}"
    exp = tag == "exp"
    torch.save({"state": net.state_dict(), "voc": voc, "backbone": a.backbone, "frames": frames, "res": a.res, "scene": scene},
               OUT / f"video_moves_{tag}{suf}.pt")
    if a.mode == "train":
        troot = DATA / f"vcrops_{a.eval_crops or a.crops}"
        titems = items_for(troot, by_key, test)
        chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
        vch = video_chars(troot, test, chars_list)
        P = predict(net, titems, voc, vch)
        res = end_to_end(titems, P, voc, vch, rows, set(test))
        res["backbone"] = a.backbone
        res["train_replays"] = len(train_stems)
        save_json(res, OUT / (f"results_exp_moves{suf}.json" if exp else f"results_video_moves{suf}.json"))
        brief = {k: v for k, v in res.items() if k not in ("category", "hit_kind", "move_acc_by_char")}
        brief["category_acc"] = res["category"]["accuracy"]
        brief["category_bal"] = res["category"]["balanced_accuracy"]
        brief["hit_kind_acc"] = res["hit_kind"]["accuracy"]
        print(json.dumps(brief, indent=1, default=float))
        print(res["move_acc_by_char"])


if __name__ == "__main__":
    main()
