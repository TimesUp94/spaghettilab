"""Frozen pretrained backbones against the v2 ResNet move nets (ROADMAP point 1), on exactly the
items video_moves.py trains and tests on: the chain's crops (vcrops_test, attacker crop strip) and
whole frames (vscene_test) at its detections, training on the non-test folds, testing on the test
fold, scored with video_moves.test_results (end_to_end + banner_reader counter / punish), for all /
old / new test replays, plus move accuracy by training count of the move (<5, 5-19, 20-49, 50+) and
by category (projectile, special, ...).

1. extract: every tile of every item (the 21 frames of video_moves.WIN, crop and whole frame) through
   the frozen backbone, pooled to one vector per frame (per-frame backbones: CLS / pooled output;
   V-JEPA 2: the clip of the 21 frames (+ the last repeated) -> 11 tubelets, mean over space), fp16,
   one file per replay under <data>/ggst_dataset/feats/<name>/<stem>.npz (resumable: replays already
   written are skipped).
2. train: a small temporal head on the cached features, the same as video_moves.Net's (LayerNorm on
   the input, then video_moves' temporal conv, max-pool, dropout, heads), the same heads, labels,
   vocabulary, class weights, label smoothing and character mask as video_moves.fit, frames sampled
   like video_moves.Crops (FRAMES + a random shift of SHIFTS, the 30 fps parity trick) and averaged
   over SHIFTS at test (TTA). No image augmentation (the features are cached).
3. baseline: the same metrics for the v2 ResNet ensemble (video_moves_test.pt), so the tables line up.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/backbone_feats.py smoke --bb dinov2_base        # ~24 items: shapes, speed, estimates
    .venv/Scripts/python.exe tools/ggst_logger/model/backbone_feats.py extract --bb dinov2_base      # every replay (resumable)
    .venv/Scripts/python.exe tools/ggst_logger/model/backbone_feats.py train --bb dinov2_base --seeds 3
    .venv/Scripts/python.exe tools/ggst_logger/model/backbone_feats.py baseline                     # v2 ResNet ensemble
    .venv/Scripts/python.exe tools/ggst_logger/model/backbone_feats.py table                        # every results_exp_bb_*.json side by side

Backbones: resnet18 (ImageNet, the control: frozen features + new head), dinov2_base, dinov2_large,
siglip2_base, vjepa2_large (BACKBONES).

-> <data>/ggst_dataset/feats/<name>/, model/video_moves_exp_bb_<name>.pt (head), model/results_exp_bb_<name>.json,
   model/results_exp_bb_v2.json (baseline), model/exp_bb_v2_P.npz (its test probabilities)
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
from common import DATA, OUT, TEST_FOLD, device, label_category, low_priority, replay_fold_map, save_json, test_subsets  # noqa: E402
import video_moves as vm  # noqa: E402

FEATS = DATA / "feats"
CROPS = DATA / "vcrops_test"      # the chain's crops: video_moves trains (non-test folds) and tests (test fold) on these
NWIN = len(vm.WIN)
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# crop / scene: (w, h) fed to the backbone (the crop tiles are 224 square, the scene tiles 384 x 216)
BACKBONES = {
    "resnet18": dict(kind="frame", hf=None, crop=(vm.RES, vm.RES), scene=vm.SCENE, norm=IMNET, dim=512, bs=512),
    "dinov2_base": dict(kind="frame", hf="facebook/dinov2-base", crop=(224, 224), scene=(392, 224), norm=IMNET, dim=768, bs=256),
    "dinov2_large": dict(kind="frame", hf="facebook/dinov2-large", crop=(224, 224), scene=(392, 224), norm=IMNET, dim=1024, bs=128),
    "siglip2_base": dict(kind="frame", hf="google/siglip2-base-patch16-224", crop=(224, 224), scene=(384, 224), norm=HALF, dim=768, bs=256),
    # V-JEPA 2's RoPE assumes a square crop_size grid: the whole frame is squashed to 256 x 256
    "vjepa2_large": dict(kind="clip", hf="facebook/vjepa2-vitl-fpc64-256", crop=(256, 256), scene=(256, 256), norm=IMNET, dim=1024, bs=8),
}
BUCKETS = ((50, "50+"), (20, "20-49"), (5, "5-19"), (0, "<5"))


# ---- 1. features ------------------------------------------------------------------------

def load_backbone(name):
    """A function: uint8 tiles [N, H, W, 3] (frame) or clips [B, T, H, W, 3] (clip) -> fp16 [N, D] / [B, T/2, D]."""
    import torch
    spec = BACKBONES[name]
    dev = device()
    mean = torch.tensor(spec["norm"][0], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(spec["norm"][1], device=dev).view(1, 3, 1, 1)

    def norm(x):            # [N, H, W, 3] uint8 -> [N, 3, H, W] normalised
        x = torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(255)
        return ((x - mean) / std).half()

    if name == "resnet18":
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        m = m.to(dev).eval().half()
        f = lambda x: m(norm(x))
    elif name.startswith("dinov2"):
        from transformers import Dinov2Model
        m = Dinov2Model.from_pretrained(spec["hf"], dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
        f = lambda x: m(pixel_values=norm(x)).pooler_output          # the layer-normed CLS token
    elif name.startswith("siglip2"):
        from transformers import SiglipVisionModel
        m = SiglipVisionModel.from_pretrained(spec["hf"], dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
        f = lambda x: m(pixel_values=norm(x), interpolate_pos_encoding=True).pooler_output   # the attention-pool head
    elif name.startswith("vjepa2"):
        from transformers import VJEPA2Model
        m = VJEPA2Model.from_pretrained(spec["hf"], dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()

        def f(x):
            B, T = x.shape[:2]
            v = norm(x.reshape(B * T, *x.shape[2:])).view(B, T, 3, *x.shape[2:4])
            h = m(pixel_values_videos=v, skip_predictor=True).last_hidden_state      # [B, T/2 * gh * gw, D]
            return h.view(B, T // 2, -1, h.shape[-1]).float().mean(2)
    else:
        raise ValueError(name)

    def run(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            return f(x).float().cpu().numpy().astype(np.float16)
    return run


class StemTiles:
    """One replay per sample: its items' crop and whole-frame tiles, resized for the backbone."""

    def __init__(self, stems, spec, limit=None):
        self.stems, self.spec, self.limit = stems, spec, limit

    def __len__(self):
        return len(self.stems)

    @staticmethod
    def tiles(path, size):
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = img[:, :, ::-1]
        w = img.shape[1] // NWIN
        return np.stack([cv2.resize(img[:, k * w:(k + 1) * w], size, interpolation=cv2.INTER_AREA) for k in range(NWIN)])

    def __getitem__(self, i):
        s = self.stems[i]
        dets = json.loads((CROPS / s / "index.json").read_text())["dets"][:self.limit]
        cw, ch = self.spec["crop"]
        sw, sh = self.spec["scene"]
        crop = np.zeros((len(dets), NWIN, ch, cw, 3), np.uint8)
        scene = np.zeros((len(dets), NWIN, sh, sw, 3), np.uint8)
        missing = 0
        for k, d in enumerate(dets):
            c = self.tiles(CROPS / s / d["file"], (cw, ch))
            z = self.tiles(vm.scene_root(CROPS) / s / d["file"], (sw, sh))
            missing += (c is None) + (z is None)
            if c is not None:
                crop[k] = c
            if z is not None:
                scene[k] = z
        return s, [d["file"] for d in dets], crop, scene, missing


def encode(run, spec, arr):
    """[n, 21, H, W, 3] -> [n, 21, D] (frame) or [n, 11, D] (clip: the 21 frames plus the last again)."""
    n = len(arr)
    if spec["kind"] == "frame":
        x = arr.reshape(n * NWIN, *arr.shape[2:])
        out = [run(x[j:j + spec["bs"]]) for j in range(0, len(x), spec["bs"])] if n else [np.zeros((0, spec["dim"]), np.float16)]
        return np.concatenate(out).reshape(n, NWIN, -1)
    x = np.concatenate([arr, arr[:, -1:]], 1)
    out = [run(x[j:j + spec["bs"]]) for j in range(0, n, spec["bs"])] if n else [np.zeros((0, (NWIN + 1) // 2, spec["dim"]), np.float16)]
    return np.concatenate(out)


def extract(name, stems, limit=None, write=True, workers=3):
    import torch
    spec = BACKBONES[name]
    out_dir = FEATS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (out_dir / f"{s}.npz").exists()] if write else list(stems)
    print(f"{name}: {len(stems) - len(todo)} replays already extracted, {len(todo)} to go", flush=True)
    if not todo:
        return {}
    run = load_backbone(name)
    if device() == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dl = torch.utils.data.DataLoader(StemTiles(todo, spec, limit), batch_size=None, shuffle=False,
                                     num_workers=workers, prefetch_factor=2 if workers else None)
    t0, n_items, n_missing, times = time.time(), 0, 0, []
    for k, (s, files, crop, scene, missing) in enumerate(dl):
        t1 = time.time()
        fc, fs = encode(run, spec, crop), encode(run, spec, scene)
        times.append((len(files), time.time() - t1, time.time()))
        n_items += len(files)
        n_missing += missing
        if write:
            tmp = out_dir / f"{s}.tmp.npz"
            np.savez(tmp, files=np.array(files), crop=fc, scene=fs)
            os.replace(tmp, out_dir / f"{s}.npz")
        if k % 25 == 0 or k == len(todo) - 1:
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo)} replays, {n_items} items, {el:.0f}s ({el / max(1, n_items) * 1000:.0f} ms/item), "
                  f"missing tiles {n_missing}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device() == "cuda" else 0
    return dict(items=n_items, seconds=time.time() - t0, times=times, crop_shape=list(fc.shape), scene_shape=list(fs.shape),
                bytes_per_item=int((fc.nbytes + fs.nbytes) / max(1, len(files))), peak_gpu_gb=round(peak, 2), missing=n_missing)


def smoke(name, n_stems=3, per=12):
    """A few items of the first test replays: shapes, ms per item (decode + GPU, after the first replay's
    warm-up), and the estimated time and disk for all items."""
    fold = replay_fold_map()
    test = sorted(s for s in fold if fold[s] == TEST_FOLD)
    st = [s for s in test if (CROPS / s / "index.json").exists()][:n_stems]
    r = extract(name, st, limit=per, write=False, workers=0)
    total = sum(len(json.loads((CROPS / s / "index.json").read_text())["dets"]) for s in sorted(fold) if (CROPS / s / "index.json").exists())
    warm = r["times"][1:]
    gpu_ms = 1000 * sum(t for _, t, _ in warm) / max(1, sum(n for n, _, _ in warm))
    wall_ms = 1000 * (warm[-1][2] - r["times"][0][2]) / max(1, sum(n for n, _, _ in warm))
    est = dict(items_total=total, ms_per_item_gpu=round(gpu_ms, 1), ms_per_item_wall=round(wall_ms, 1),
               est_extract_minutes=round(total * wall_ms / 1000 / 60, 1),
               est_feat_gb=round(total * r["bytes_per_item"] / 2 ** 30, 2))
    out = dict(backbone=name, hf=BACKBONES[name]["hf"], items=r["items"], crop_shape=r["crop_shape"], scene_shape=r["scene_shape"],
               bytes_per_item=r["bytes_per_item"], peak_gpu_gb=r["peak_gpu_gb"], missing_tiles=r["missing"], **est)
    print(json.dumps(out), flush=True)
    save_json(out, OUT / f"results_exp_bb_smoke_{name}.json")
    return out


# ---- 2. head on cached features -----------------------------------------------------------

def load_feats(name, items):
    """crop, scene features of the items, [N, T, D] fp16, in the items' order."""
    cache, C, S = {}, [], []
    for it in items:
        s = it["stem"]
        if s not in cache:
            cache.clear()
            z = np.load(FEATS / name / f"{s}.npz")
            cache[s] = ({f: k for k, f in enumerate(z["files"])}, z["crop"], z["scene"])
        pos, c, z = cache[s]
        k = pos[Path(it["file"]).name]
        C.append(c[k])
        S.append(z[k])
    return np.stack(C), np.stack(S)


def make_head(d, voc):
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1))
            self.t = nn.Sequential(nn.Conv1d(d, 384, 3, padding=1), nn.GELU(), nn.Conv1d(384, 384, 3, padding=1),
                                   nn.GELU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(0.3))
            self.heads = nn.ModuleDict({h: nn.Linear(384, len(voc[h])) for h in vm.HEADS})

        def forward(self, f):       # [B, T, d]
            z = self.t(self.inp(f).transpose(1, 2))
            return {h: m(z) for h, m in self.heads.items()}
    return Head()


def frame_index(B, train, shift=0, frames=None, gen=None):
    """[B, len(FRAMES)] tile indices into WIN, as video_moves.Crops picks them."""
    frames = np.array(frames or vm.FRAMES)
    if train:
        sh = gen.choice(vm.SHIFTS, B)[:, None]
        offs = frames[None] + sh
        par = gen.integers(2, size=(B, 1))
        use = gen.random((B, 1)) < 0.3
        offs = np.where(use, offs - ((offs + par) % 2), offs)
    else:
        offs = np.repeat(frames[None] + shift, B, 0)
    return np.clip(offs, vm.WIN[0], vm.WIN[-1]) - vm.WIN[0]


def class_weights(ys, voc):
    import torch
    w_all = {}
    for k, h in enumerate(vm.HEADS):
        cnt = np.bincount(ys[:, k][ys[:, k] >= 0], minlength=len(voc[h])).astype(float)
        w = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0)
        w_all[h] = torch.tensor(np.minimum(w / w[w > 0].mean(), 4.0), dtype=torch.float32, device=device())
    return w_all


def masks_for(voc, chars):
    cache = {}
    out = np.zeros((len(chars), len(voc["move"])), bool)
    for i, c in enumerate(chars):
        if c is None:
            continue
        if c not in cache:
            cache[c] = vm.char_mask(voc, c)
        out[i] = cache[c]
    return out


def fit_head(F, ys, M, voc, kind, epochs=30, bs=256, lr=1e-3, seed=0):
    """F [N, T, D] fp16 (crop and scene concatenated), ys [N, 4], M [N, n_moves] char masks."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    dev = device()
    Fg = torch.from_numpy(F).to(dev)
    yg = torch.from_numpy(ys).to(dev)
    Mg = torch.from_numpy(M).to(dev)
    net = make_head(F.shape[-1], voc).to(dev)
    weights = class_weights(ys, voc)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * (len(F) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        net.train()
        perm = gen.permutation(len(F))
        tot, n = 0.0, 0
        for j in range(0, len(perm) - bs + 1, bs):
            b = torch.from_numpy(perm[j:j + bs]).to(dev)
            if kind == "frame":
                idx = torch.from_numpy(frame_index(bs, True, gen=gen)).to(dev)
                x = Fg[b[:, None], idx].float()
            else:
                x = Fg[b].float()
            out = net(x)
            loss = 0
            for k, h in enumerate(vm.HEADS):
                lg = out[h].float()
                if h == "move":
                    lg = lg.masked_fill(~Mg[b], -1e4)
                loss = loss + nn.functional.cross_entropy(lg, yg[b, k], weight=weights[h], ignore_index=vm.IGNORE,
                                                          label_smoothing=0.0 if h == "move" else 0.05) * (1.0 if h != "hit_kind" else 0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n += 1
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"  epoch {ep + 1}/{epochs} loss {tot / max(1, n):.4f}", flush=True)
    return net


def predict_head(net, F, M, kind):
    """Probabilities per head, averaged over the frame shifts (frame backbones), move restricted by M."""
    import torch
    dev = device()
    net.eval()
    Fg = torch.from_numpy(F).to(dev)
    Mg = torch.from_numpy(M).to(dev)
    shifts = vm.TTA if kind == "frame" else [0]
    P = {h: 0 for h in vm.HEADS}
    with torch.no_grad():
        for sh in shifts:
            out = {h: [] for h in vm.HEADS}
            for j in range(0, len(F), 1024):
                sl = slice(j, j + 1024)
                if kind == "frame":
                    idx = torch.from_numpy(frame_index(len(F[sl]), False, sh)).to(dev)
                    x = Fg[sl][torch.arange(len(F[sl]), device=dev)[:, None], idx].float()
                else:
                    x = Fg[sl].float()
                o = net(x)
                for h in vm.HEADS:
                    lg = o[h].float()
                    if h == "move":
                        lg = lg.masked_fill(~Mg[sl], -1e4)
                    out[h].append(torch.softmax(lg, -1).cpu().numpy())
            for h in vm.HEADS:
                P[h] = P[h] + np.concatenate(out[h]) / len(shifts)
    return P


# ---- 3. scoring --------------------------------------------------------------------------

def bucket(c):
    return next(name for lo, name in BUCKETS if c >= lo)


def extras(titems, P, voc, vch, rows, stems, train_counts):
    """Per opener of the subset (bursts apart, as end_to_end): status missed / dropped_none / wrong_move /
    right; by training count of the move and by category: n, the chain's right share over all openers
    ('chain_right', as error_report), and the exact move accuracy given detected ('move_acc')."""
    cat_none = voc["category"].index("none")
    st = set(stems)
    tru = [r for r in rows if r["replay"] in st and not r["is_burst"]]
    found = {}
    for i, it in enumerate(titems):
        if it["row"] is not None and it["stem"] in st:
            found[f"{it['row']['replay']}/{it['row']['log_row']}_{it['row']['attacker']}"] = i
    recs = []
    for r in tru:
        i = found.get(f"{r['replay']}/{r['log_row']}_{r['attacker']}")
        lab = f"{r['char']}:{r['move']}"
        if lab not in voc["move"]:
            lab = f"{r['char']}:(rare)"
        if i is None:
            status, mv_ok = "missed", None
        else:
            mv_ok = voc["move"][int(P["move"][i].argmax())] == lab
            status = "dropped_none" if P["category"][i].argmax() == cat_none else ("right" if mv_ok else "wrong_move")
        pc = None
        if i is not None:
            p = P["category"][i].copy()
            p[cat_none] = 0
            pc = voc["category"][int(p.argmax())]
        recs.append(dict(b=bucket(train_counts[f"{r['char']}:{r['move']}"]), cat=label_category(r), status=status,
                         mv_ok=mv_ok, pred_cat=pc))

    def table(keyf, order=None):
        t = collections.defaultdict(list)
        for x in recs:
            t[keyf(x)].append(x)
        out = {}
        for k in (order or sorted(t)):
            xs = t.get(k, [])
            if not xs:
                continue
            det = [x for x in xs if x["mv_ok"] is not None]
            out[k] = dict(n=len(xs), chain_right=round(np.mean([x["status"] == "right" for x in xs]), 4),
                          detected=len(det), move_acc=round(float(np.mean([x["mv_ok"] for x in det])), 4) if det else None,
                          category_recall=round(float(np.mean([x["pred_cat"] == x["cat"] for x in det])), 4) if det else None)
        return out
    res = dict(by_train_count=table(lambda x: x["b"], [b for _, b in reversed(BUCKETS)]), by_category=table(lambda x: x["cat"]))
    det = [x for x in recs if x["pred_cat"] is not None and x["cat"] in ("projectile", "special")]
    res["projectile_special_confusion"] = {c: dict(collections.Counter(x["pred_cat"] for x in det if x["cat"] == c))
                                           for c in ("projectile", "special")}
    return res


def score(titems, P, voc, vch, rows, fold):
    """video_moves.test_results on these probabilities (its predict swapped for them, so the item set,
    the end_to_end metric and the banner_reader counter / punish are the v2 eval's own), plus extras."""
    subs = test_subsets(fold)
    train_counts = collections.Counter(f"{r['char']}:{r['move']}" for r in rows
                                       if r["replay"] in fold and fold[r["replay"]] != TEST_FOLD and not r["is_burst"])
    real = vm.predict

    def given(net, items, voc_, chars, **k):
        assert [it["file"] for it in items] == [it["file"] for it in titems]
        return {h: P[h].copy() for h in vm.HEADS}
    vm.predict = given
    try:
        res = vm.test_results(None, voc, rows, fold, CROPS)
    finally:
        vm.predict = real
    ex = {sub: extras(titems, P, voc, vch, rows, stems, train_counts) for sub, stems in subs.items() if stems}
    return res, ex


def test_setup(rows, fold):
    import torch
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    subs = test_subsets(fold)
    titems = vm.items_for(CROPS, by_key, subs["all"])
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    vch = vm.video_chars(CROPS, subs["all"], chars_list)
    return titems, vch


def summary(res, ex):
    out = {}
    for sub in ("old", "new", "all"):
        if sub not in res:
            continue
        r, e = res[sub], ex[sub]
        out[sub] = dict(chain=round(r["chain_move_right_of_all_openers"], 4), move=round(r["move_acc"], 4),
                        category=round(r["category"]["accuracy"], 4), normal=round(r["normal_acc"], 4),
                        kept=round(r["detected_and_kept"], 4), precision_after_none=round(r["precision_after_none"], 4),
                        hit_kind=round(r["hit_kind"]["accuracy"], 4), hit_kind_movenet=round(r["hit_kind_movenet_acc"], 4),
                        move_by_train_count={k: v["move_acc"] for k, v in e["by_train_count"].items()},
                        chain_by_train_count={k: v["chain_right"] for k, v in e["by_train_count"].items()},
                        move_projectile=e["by_category"].get("projectile", {}).get("move_acc"),
                        move_special=e["by_category"].get("special", {}).get("move_acc"),
                        chain_projectile=e["by_category"].get("projectile", {}).get("chain_right"),
                        chain_special=e["by_category"].get("special", {}).get("chain_right"),
                        catrec_projectile=e["by_category"].get("projectile", {}).get("category_recall"),
                        catrec_special=e["by_category"].get("special", {}).get("category_recall"))
    return out


def baseline(weights):
    import pyarrow.parquet as pq
    os.environ.setdefault("VM_WORKERS", "3")
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    titems, vch = test_setup(rows, fold)
    net, ck = vm.load(OUT / weights)
    cache = OUT / f"exp_bb_v2_P_{Path(weights).stem}.npz"
    if cache.exists():
        z = np.load(cache)
        assert list(z["files"]) == [it["file"] for it in titems]
        P = {h: z[h] for h in vm.HEADS}
    else:
        t0 = time.time()
        P = vm.predict(net, titems, ck["voc"], vch)
        print(f"predicted {len(titems)} items in {time.time() - t0:.0f}s", flush=True)
        np.savez(cache, files=np.array([it["file"] for it in titems]), **{h: P[h] for h in vm.HEADS})
    res, ex = score(titems, P, ck["voc"], vch, rows, fold)
    s = summary(res, ex)
    out = dict(backbone="v2 ResNet18+ResNet50 ensemble (fine-tuned, image aug, TTA)", weights=weights, summary=s, subsets=res, extras=ex)
    save_json(out, OUT / "results_exp_bb_v2.json")
    print(json.dumps(s, indent=1, default=float))


def train(name, epochs, seeds, crop_only=False):
    import pyarrow.parquet as pq
    import torch
    spec = BACKBONES[name]
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    fold = replay_fold_map()
    train_stems = [s for s in sorted(fold) if fold[s] != TEST_FOLD]
    items = vm.items_for(CROPS, by_key, train_stems)
    voc = vm.vocab(items)
    v2 = torch.load(OUT / "video_moves_test.pt", map_location="cpu")["voc"]
    print(len(items), "training items;", {h: len(v) for h, v in voc.items()}, "| same vocabulary as video_moves_test.pt:", voc == v2, flush=True)
    t0 = time.time()
    Fc, Fs = load_feats(name, items)
    F = Fc if crop_only else np.concatenate([Fc, Fs], -1)
    del Fc, Fs
    ys = np.array([[vm.targets(it, voc)[h] for h in vm.HEADS] for it in items], np.int64)
    M = masks_for(voc, [it["row"]["char"] if it["row"] is not None else None for it in items])
    titems, vch = test_setup(rows, fold)
    Tc, Ts = load_feats(name, titems)
    TF = Tc if crop_only else np.concatenate([Tc, Ts], -1)
    del Tc, Ts
    TM = masks_for(voc, [vch[it["stem"]][it["side"] - 1] for it in titems])
    print(f"features loaded in {time.time() - t0:.0f}s: train {F.shape}, test {TF.shape}", flush=True)
    tag = name + ("_croponly" if crop_only else "")
    runs, first = [], None
    for seed in range(seeds):
        t1 = time.time()
        net = fit_head(F, ys, M, voc, spec["kind"], epochs=epochs, seed=seed)
        P = predict_head(net, TF, TM, spec["kind"])
        res, ex = score(titems, P, voc, vch, rows, fold)
        s = summary(res, ex)
        runs.append(dict(seed=seed, train_seconds=round(time.time() - t1), summary=s))
        print(f"seed {seed}:", json.dumps({k: (v["chain"], v["move"], v["category"]) for k, v in s.items()}), flush=True)
        if first is None:
            first = (res, ex, s)
            torch.save({"state": net.state_dict(), "voc": voc, "backbone": name, "kind": spec["kind"], "crop_only": crop_only},
                       OUT / f"video_moves_exp_bb_{tag}.pt")
    res, ex, s = first
    spread = {sub: {m: [r["summary"][sub][m] for r in runs] for m in ("chain", "move", "category")} for sub in s}
    out = dict(backbone=name, hf=spec["hf"], spec={k: v for k, v in spec.items() if k != "norm"}, crop_only=crop_only,
               train_items=len(items), epochs=epochs, seeds=seeds, summary=s, seed_spread=spread, runs=runs,
               subsets=res, extras=ex, train_replays=len(train_stems))
    save_json(out, OUT / f"results_exp_bb_{tag}.json")
    print(json.dumps(s, indent=1, default=float))


def table():
    rows = []
    for f in sorted(OUT.glob("results_exp_bb_*.json")):
        if "smoke" in f.name:
            continue
        r = json.loads(f.read_text())
        for sub in ("old", "new"):
            s = r["summary"][sub]
            rows.append([f.stem.replace("results_exp_bb_", ""), sub, s["chain"], s["move"], s["category"],
                         *[s["move_by_train_count"].get(b) for _, b in reversed(BUCKETS)], s["move_projectile"], s["move_special"]])
    hdr = ["backbone", "sub", "chain", "move", "cat", "<5", "5-19", "20-49", "50+", "proj", "spec"]
    print(" ".join(f"{h:>14}" if i == 0 else f"{h:>7}" for i, h in enumerate(hdr)))
    for r in rows:
        print(" ".join(f"{v:>14}" if i == 0 else (f"{v:>7}" if isinstance(v, str) else f"{(v if v is not None else float('nan')):7.3f}")
                       for i, v in enumerate(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "extract", "train", "baseline", "table"])
    ap.add_argument("--bb", choices=sorted(BACKBONES))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1, help="train: heads with seeds 0..n-1 (the first is saved and reported, all in seed_spread)")
    ap.add_argument("--crop-only", action="store_true", help="train: the crop stream only (no whole frame)")
    ap.add_argument("--stems", choices=["all", "test", "train"], default="all", help="extract: which replays")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--weights", default="video_moves_test.pt", help="baseline: the v2 weights")
    a = ap.parse_args()
    low_priority()
    if a.mode == "table":
        return table()
    if a.mode == "baseline":
        return baseline(a.weights)
    if a.mode == "smoke":
        return smoke(a.bb)
    if a.mode == "extract":
        fold = replay_fold_map()
        stems = [s for s in sorted(fold) if (CROPS / s / "index.json").exists()
                 and (a.stems == "all" or (fold[s] == TEST_FOLD) == (a.stems == "test"))]
        r = extract(a.bb, stems, workers=a.workers)
        r.pop("times", None)
        print(json.dumps(r), flush=True)
        return
    train(a.bb, a.epochs, a.seeds, a.crop_only)


if __name__ == "__main__":
    main()
