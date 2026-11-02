"""Fine-tune the last blocks of V-JEPA 2 (ViT-L, facebook/vjepa2-vitl-fpc64-256) on the move items
(ROADMAP point 1, after the frozen comparison in backbone_feats.py).

Same items, labels, vocabulary, losses, character masks and eval as backbone_feats.train (and so as
video_moves): training on the non-test folds of vcrops_test, testing on the test fold's 7,354 items,
scored with backbone_feats.score (video_moves.test_results / end_to_end, banner_reader hit_kind).

What changes against the frozen run:
- The crop stream runs live through V-JEPA 2: embeddings and the first 24-K blocks frozen (fp16, no
  grad), the last K blocks and the final LayerNorm trained (bf16 autocast, fp32 weights). Same clip as
  the frozen extraction (the 21 window frames plus the last again, 256 x 256, mean over space -> 11
  tubelets x 1024).
- The whole-frame stream stays the frozen cached features (DATA/feats/vjepa2_large, scene).
- The head starts from the frozen seed-0 head (model/video_moves_exp_bb_vjepa2_large.pt), so step 0
  is the frozen model.
- Image augmentation like video_moves (brightness / contrast, roll, downscale p 0.3) and a temporal
  shift of the clip (-1 / 0 / +1 frame, the 30 fps parity trick p 0.3).
- Test: shift 0 (as the frozen run) and TTA over shifts -1/0/+1 (decided in advance as the primary
  number, as v2 averages over its shifts); both are reported.

    .venv/Scripts/python.exe tools/ggst_logger/model/exp_bb_ft.py bench --k 4          # speed / memory, a few steps
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_bb_ft.py check                # live frozen features == cached ones
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_bb_ft.py train --k 4 --epochs 3 > DATA/bb_vjepa2_ft.log 2>&1
-> model/video_moves_exp_bb_vjepa2_large_ft.pt (trained blocks + head), model/results_exp_bb_vjepa2_large_ft.json,
   model/exp_bb_P_vjepa2_large_ft.npz (test probabilities, TTA; exp_bb_ensemble.py --tag vjepa2_large_ft uses it)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbone_feats as bf  # noqa: E402
from common import DATA, OUT, TEST_FOLD, device, low_priority, replay_fold_map, save_json  # noqa: E402
import video_moves as vm  # noqa: E402

NAME = "vjepa2_large"
HF = bf.BACKBONES[NAME]["hf"]
SIZE = 256
NWIN = bf.NWIN
TAG = "vjepa2_large_ft"
SHIFTS = (-1, 0, 1)
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3, 1, 1)


class Clips:
    """One item: the crop strip's 21 tiles (shifted by a frame offset, edges repeated) plus the last
    again, 256 x 256 uint8 [22, H, W, 3]; its index (for the cached scene features), targets, mask."""

    def __init__(self, items, voc, train, chars=None, shift=0):
        self.items = [dict(file=it["file"], t=[vm.targets(it, voc)[h] for h in vm.HEADS],
                           char=(chars[it["stem"]][it["side"] - 1] if chars is not None else
                                 (it["row"]["char"] if it["row"] is not None else None))) for it in items]
        self.voc, self.train, self.shift = voc, train, shift
        self.masks = {}

    def __len__(self):
        return len(self.items)

    def mask(self, char):
        if char not in self.masks:
            self.masks[char] = vm.char_mask(self.voc, char)
        return self.masks[char]

    def __getitem__(self, i):
        import cv2
        it = self.items[i]
        img = cv2.imread(it["file"])[:, :, ::-1]
        w = img.shape[1] // NWIN
        sh = int(np.random.choice(SHIFTS)) if self.train else self.shift
        offs = [k + sh for k in range(NWIN)]
        if self.train and np.random.random() < 0.3:
            par = np.random.randint(2)
            offs = [o - ((o + par) % 2) for o in offs]
        offs = [min(max(o, 0), NWIN - 1) for o in offs]
        cache = {}
        for o in set(offs):
            cache[o] = cv2.resize(img[:, o * w:(o + 1) * w], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = np.stack([cache[o] for o in offs] + [cache[offs[-1]]])
        if self.train and np.random.random() < 0.3:
            k = np.random.uniform(0.45, 0.8)
            lo = int(SIZE * k)
            x = np.stack([cv2.resize(cv2.resize(t, (lo, lo), interpolation=cv2.INTER_AREA), (SIZE, SIZE),
                                     interpolation=cv2.INTER_LINEAR) for t in x])
        m = self.mask(it["char"]) if it["char"] is not None else np.zeros(len(self.voc["move"]), bool)
        return x, i, np.array(it["t"], np.int64), m


def prep(x, train):
    """uint8 [B, T, H, W, 3] -> normalised fp32 [B, T, 3, H, W] on the GPU (+ v2's brightness / roll aug)."""
    import torch
    dev = device()
    x = x.to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div(255)
    if train:
        B = x.shape[0]
        x = x * (0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device=dev)) + 0.1 * (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5)
        dx, dy = np.random.randint(-21, 22, 2)
        x = torch.roll(x, (int(dy), int(dx)), (3, 4))
    return (x.clamp(0, 1) - torch.from_numpy(MEAN).to(dev)) / torch.from_numpy(STD).to(dev)


def make_model(voc, k, head_init=True):
    import torch
    import torch.nn as nn
    from transformers import VJEPA2Model

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            enc = VJEPA2Model.from_pretrained(HF, dtype=torch.float32, attn_implementation="sdpa").encoder
            n = len(enc.layer)
            self.k = k
            self.frozen = nn.ModuleList([enc.embeddings, *enc.layer[:n - k]]).half().eval()
            for p in self.frozen.parameters():
                p.requires_grad_(False)
            self.blocks = nn.ModuleList(enc.layer[n - k:])
            self.norm = enc.layernorm
            self.head = bf.make_head(2048, voc)

        def train(self, mode=True):
            super().train(mode)
            self.frozen.eval()
            return self

        def crop_feats(self, x):            # [B, 22, 3, H, W] normalised -> [B, 11, 1024]
            B = x.shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = self.frozen[0](x.half())
                for blk in self.frozen[1:]:
                    h = blk(h, None)[0]
            h = h.float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for blk in self.blocks:
                    h = blk(h, None)[0]
                h = self.norm(h)
            return h.float().view(B, x.shape[1] // 2, -1, h.shape[-1]).mean(2)

        def forward(self, x, scene):
            return self.head(torch.cat([self.crop_feats(x), scene.float()], -1))

    net = FT()
    if head_init:
        ck = torch.load(OUT / f"video_moves_exp_bb_{NAME}.pt", map_location="cpu")
        assert ck["voc"] == voc and not ck["crop_only"]
        net.head.load_state_dict(ck["state"])
    return net.to(device())


def setup():
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    fold = replay_fold_map()
    train_stems = [s for s in sorted(fold) if fold[s] != TEST_FOLD]
    items = vm.items_for(bf.CROPS, by_key, train_stems)
    voc = vm.vocab(items)
    return rows, fold, items, voc


def loader(ds, train, bs, workers):
    import torch
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train, num_workers=workers, drop_last=train,
                                       pin_memory=True, persistent_workers=False, prefetch_factor=4 if workers else None)


def loss_fn(out, y, m, weights):
    import torch.nn as nn
    loss = 0
    for k, h in enumerate(vm.HEADS):
        lg = out[h].float()
        if h == "move":
            lg = lg.masked_fill(~m, -1e4)
        loss = loss + nn.functional.cross_entropy(lg, y[:, k], weight=weights[h], ignore_index=vm.IGNORE,
                                                  label_smoothing=0.0 if h == "move" else 0.05) * (1.0 if h != "hit_kind" else 0.5)
    return loss


def predict(net, titems, voc, vch, S, shift, workers, bs=16):
    import torch
    net.eval()
    ds = Clips(titems, voc, False, chars=vch, shift=shift)
    Sg = torch.from_numpy(S)
    out = {h: [] for h in vm.HEADS}
    t0 = time.time()
    with torch.no_grad():
        for j, (x, idx, _, m) in enumerate(loader(ds, False, bs, workers)):
            o = net(prep(x, False), Sg[idx].to(device()))
            m = m.to(device())
            for h in vm.HEADS:
                lg = o[h].float()
                if h == "move":
                    lg = lg.masked_fill(~m, -1e4)
                out[h].append(torch.softmax(lg, -1).cpu().numpy())
            if j % 100 == 0:
                print(f"  predict shift {shift}: {j * bs}/{len(ds)} ({time.time() - t0:.0f}s)", flush=True)
    return {h: np.concatenate(v) for h, v in out.items()}


def cmd_check(workers):
    """The live frozen path (k blocks untouched) reproduces the cached crop features of a few test items."""
    import torch
    rows, fold, items, voc = setup()
    titems, vch = bf.test_setup(rows, fold)
    sub = titems[:16]
    Tc, _ = bf.load_feats(NAME, sub)
    net = make_model(voc, 4)
    net.eval()
    x = torch.stack([torch.from_numpy(Clips(sub, voc, False, chars=vch)[i][0]) for i in range(len(sub))])
    with torch.no_grad():
        f = net.crop_feats(prep(x, False)).cpu().numpy()
    c = Tc.astype(np.float32)
    cos = (f * c).sum(-1) / (np.linalg.norm(f, axis=-1) * np.linalg.norm(c, axis=-1))
    print(json.dumps(dict(max_abs=float(np.abs(f - c).max()), mean_abs=float(np.abs(f - c).mean()),
                          scale=float(np.abs(c).mean()), cos_min=float(cos.min()), cos_mean=float(cos.mean()))), flush=True)


def cmd_train(k, epochs, bs, accum, lr_bb, lr_head, workers, bench_steps=0):
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    rows, fold, items, voc = setup()
    v2 = torch.load(OUT / "video_moves_test.pt", map_location="cpu")["voc"]
    assert voc == v2
    ys = np.array([[vm.targets(it, voc)[h] for h in vm.HEADS] for it in items], np.int64)
    weights = bf.class_weights(ys, voc)
    _, S = bf.load_feats(NAME, items)
    print(f"{len(items)} training items, scene feats {S.shape}", flush=True)
    net = make_model(voc, k)
    groups = [dict(params=[p for p in list(net.blocks.parameters()) + list(net.norm.parameters())], lr=lr_bb),
              dict(params=list(net.head.parameters()), lr=lr_head)]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    ds = Clips(items, voc, True)
    steps_per_ep = len(ds) // (bs * accum)
    total = bench_steps or epochs * steps_per_ep
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    Sg = torch.from_numpy(S)
    torch.cuda.reset_peak_memory_stats()
    step, t0 = 0, time.time()
    print(f"k={k} blocks trained ({sum(p.numel() for g in groups for p in g['params']) / 1e6:.1f}M params), "
          f"bs {bs} x accum {accum}, {steps_per_ep} steps/epoch, {total} steps", flush=True)
    for ep in range(epochs if not bench_steps else 1):
        net.train()
        tot, n, te = 0.0, 0, time.time()
        opt.zero_grad()
        for j, (x, idx, y, m) in enumerate(loader(ds, True, bs, workers)):
            out = net(prep(x, True), Sg[idx].to(device()))
            loss = loss_fn(out, y.to(device()), m.to(device()), weights) / accum
            loss.backward()
            tot += float(loss.detach()) * accum
            n += 1
            if (j + 1) % accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            step += 1
            if step % 100 == 0 or (bench_steps and step % 5 == 0):
                el = time.time() - te
                print(f"  ep {ep + 1} step {step}/{total} loss {tot / n:.4f} | {el / (n * bs) * 1000:.0f} ms/item, "
                      f"epoch eta {el / (n * bs) * len(ds) / 60:.0f} min, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            if bench_steps and step >= bench_steps:
                return
            if step >= total:
                break
        print(f"epoch {ep + 1}/{epochs} loss {tot / max(1, n):.4f} ({(time.time() - te) / 60:.1f} min)", flush=True)
        torch.save({"blocks": {kk: v.to(torch.bfloat16) for kk, v in net.blocks.state_dict().items()},
                    "norm": net.norm.state_dict(), "head": net.head.state_dict(), "voc": voc, "k": k, "hf": HF,
                    "epoch": ep + 1, "backbone": TAG, "kind": "clip", "crop_only": False},
                   OUT / f"video_moves_exp_bb_{TAG}.pt")
    train_min = (time.time() - t0) / 60
    evaluate(net, rows, fold, voc, workers, dict(k=k, epochs=epochs, bs=bs, accum=accum, lr_bb=lr_bb, lr_head=lr_head,
                                                 train_minutes=round(train_min, 1), train_items=len(items)))


def evaluate(net, rows, fold, voc, workers, info):
    titems, vch = bf.test_setup(rows, fold)
    _, TS = bf.load_feats(NAME, titems)
    t0 = time.time()
    Ps = {sh: predict(net, titems, voc, vch, TS, sh, workers) for sh in (0, -1, 1)}
    info["test_minutes"] = round((time.time() - t0) / 60, 1)
    P_tta = {h: sum(Ps[sh][h] for sh in Ps) / len(Ps) for h in vm.HEADS}
    files = np.array([it["file"] for it in titems])
    np.savez(OUT / f"exp_bb_P_{TAG}.npz", files=files, **P_tta)
    np.savez(OUT / f"exp_bb_P_{TAG}_shift0.npz", files=files, **Ps[0])
    res, ex = bf.score(titems, P_tta, voc, vch, rows, fold)
    s = bf.summary(res, ex)
    res0, ex0 = bf.score(titems, Ps[0], voc, vch, rows, fold)
    s0 = bf.summary(res0, ex0)
    out = dict(backbone=TAG, hf=HF, what="V-JEPA 2 ViT-L, last k blocks + final LayerNorm fine-tuned on the crop stream, "
               "frozen cached whole-frame features, head warm-started from the frozen seed-0 head; summary = TTA over clip shifts -1/0/+1",
               info=info, summary=s, summary_shift0=s0, subsets=res, extras=ex)
    save_json(out, OUT / f"results_exp_bb_{TAG}.json")
    print("TTA", json.dumps({kk: (v["chain"], v["move"], v["category"]) for kk, v in s.items()}), flush=True)
    print("shift0", json.dumps({kk: (v["chain"], v["move"], v["category"]) for kk, v in s0.items()}), flush=True)
    print(json.dumps(s, indent=1, default=float), flush=True)


def cmd_eval(workers):
    """Re-run the eval from the saved checkpoint."""
    import torch
    rows, fold, items, voc = setup()
    ck = torch.load(OUT / f"video_moves_exp_bb_{TAG}.pt", map_location="cpu")
    net = make_model(voc, ck["k"], head_init=False)
    net.blocks.load_state_dict({kk: v.float() for kk, v in ck["blocks"].items()})
    net.norm.load_state_dict(ck["norm"])
    net.head.load_state_dict(ck["head"])
    evaluate(net, rows, fold, voc, workers, dict(k=ck["k"], epoch=ck["epoch"], from_checkpoint=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "check", "train", "eval"])
    ap.add_argument("--k", type=int, default=4, help="last blocks trained")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-bb", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30, help="bench: optimizer steps")
    a = ap.parse_args()
    low_priority()
    if a.mode == "check":
        return cmd_check(a.workers)
    if a.mode == "eval":
        return cmd_eval(a.workers)
    cmd_train(a.k, a.epochs, a.bs, a.accum, a.lr_bb, a.lr_head, a.workers, bench_steps=a.steps if a.mode == "bench" else 0)


if __name__ == "__main__":
    main()
