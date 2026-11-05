"""Measurement pass, step 1: colour robustness of the v2 move ensemble and of the character vote, on the frozen test
fold's own crops and whole frames (vcrops_test, vscene_test) at the chain's own detections. No model changes.

The transforms act on the input tiles only, after the JPEG decode and resize video_moves.Crops does, on the GPU:
  hue<d>       every pixel's HSV hue + d degrees, crop and whole frame
  bgr / gbr    channel permutations
  sectors      a rough per-material recolour: hue sectors of 60 degrees each get their own random rotation (per replay and
               side, so one "palette" per attacker), pixels with saturation > 0.2 and value > 0.15 only
  *_crop / *_scene   the transform on that stream only, the other untouched
  players_<sectors|hue60|hue180>   the transform only inside a box around each player (340 x 520 px of 1080p, at the chain's
               positions), in both streams, each player with their own palette: the stage, the HUD and most effects stay as they
               are. The nearest thing here to an alternate colour, still with the box's background recoloured and hit sparks
               inside the box too. `--players` runs these, -> results_exp_measure_hue_players.json
Hit detection (HUD, 64x36 stream) is not swept: alternate colours don't recolour the HUD, and a detection the rotated
frames might have lost is not counted. "Fixed vote" keeps the replay's real character mask; "own vote" uses the vote of
the same rotation. The character vote is video_players' gchar on the whole-frame tiles (unmirrored, 256x144 by area),
summed once per frame like video_crops.process, which does it at 256x144 from the source frame and on every frame.

Everything comes from one pass over the tiles, so the data loading is paid once.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_hue.py --limit 300     # check, first 300 items
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_hue.py
-> model/results_exp_measure_hue.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, device, low_priority, replay_fold_map, save_json, test_subsets  # noqa: E402
from exp_measure_common import opener_counts, paired_bootstrap, ratio  # noqa: E402
import video_moves as vm  # noqa: E402

WIN = vm.WIN
TRANSFORMS = ["hue0", "hue30", "hue60", "hue90", "hue120", "hue180", "bgr", "sectors", "hue180_crop", "hue180_scene", "sectors_crop"]
PLAYERS = ["players_sectors", "players_hue60", "players_hue180"]
BOX_W, BOX_H, BOX_UP = 170, 260, 60     # half box around a player, and how far above the player point it is centred (1080p px)
VOTE_EVERY = 3      # every 3rd tile of a window goes to the character vote (GPU memory; the vote sums thousands of frames)


class Tiles:
    """All 21 tiles of an item, resized as video_moves.Crops does, plus its index."""

    def __init__(self, items, res, scene):
        self.items = [dict(file=it["file"], scene=it["scene"]) for it in items]
        self.res, self.scene = res, scene

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        cv2.setNumThreads(1)
        it = self.items[i]
        img = cv2.imread(it["file"])[:, :, ::-1]
        w = img.shape[1] // len(WIN)
        x = np.stack([cv2.resize(img[:, k * w:(k + 1) * w], (self.res, self.res), interpolation=cv2.INTER_AREA) for k in range(len(WIN))])
        img = cv2.imread(it["scene"])[:, :, ::-1]
        w = img.shape[1] // len(WIN)
        xs = np.stack([cv2.resize(img[:, k * w:(k + 1) * w], self.scene, interpolation=cv2.INTER_AREA) for k in range(len(WIN))])
        return i, x, xs


def rgb2hsv(x):
    """[..., 3, H, W] in 0..1 -> h in 0..1, s, v."""
    import torch
    r, g, b = x.unbind(-3)
    mx, mn = x.max(-3).values, x.min(-3).values
    d = mx - mn
    dz = d.clamp(min=1e-6)
    h = torch.where(mx == r, ((g - b) / dz) % 6, torch.where(mx == g, (b - r) / dz + 2, (r - g) / dz + 4)) / 6
    h = torch.where(d <= 0, torch.zeros_like(h), h)
    return h, d / mx.clamp(min=1e-6), mx


def hsv2rgb(h, s, v):
    import torch

    def f(n):
        k = (n + h * 6) % 6
        return v - v * s * torch.minimum(torch.minimum(k, 4 - k), torch.ones_like(k)).clamp(min=0)
    return torch.stack([f(5), f(3), f(1)], -3)


def transform(x, name, off=None):
    """x [B, T, 3, H, W] float 0..1; off [B, 6] rotations (0..1) of the hue sectors for 'sectors'."""
    import torch
    if name == "hue0":
        return x
    if name == "bgr":
        return x[:, :, [2, 1, 0]]
    if name == "gbr":
        return x[:, :, [1, 2, 0]]
    h, s, v = rgb2hsv(x)
    if name.startswith("hue"):
        h = (h + int(name[3:]) / 360) % 1
    elif name == "sectors":
        sec = (h * 6).long().clamp(0, 5)
        o = off[torch.arange(len(x), device=x.device)[:, None, None, None], sec]
        h = torch.where((s > 0.2) & (v > 0.15), (h + o) % 1, h)
    else:
        raise ValueError(name)
    return hsv2rgb(h, s, v)


def box_mask(u, v, hw, hh, H, W, dev):
    """[B, T, H, W] bool: inside the box around (u, v) [B, T] (tile px)."""
    import torch
    xx = torch.arange(W, device=dev, dtype=torch.float32)[None, None, None, :]
    yy = torch.arange(H, device=dev, dtype=torch.float32)[None, None, :, None]
    return ((xx - u[:, :, None, None]).abs() <= hw) & ((yy - v[:, :, None, None]).abs() <= hh)


def player_masks(pos, side, scene_flip, res, scene, dev):
    """Masks per player k: crop [B, T, res, res], scene [B, T, sh, sw], from positions pos [B, T, 2, 2] (1080p px),
    the cuts of video_crops.process and video_scene.process."""
    import torch
    sw, sh = scene
    a = (side - 1)[:, None]
    t = torch.arange(pos.shape[1], device=dev)[None]
    b = torch.arange(pos.shape[0], device=dev)[:, None]
    xa, ya, xd = pos[b, t, a, 0], pos[b, t, a, 1], pos[b, t, 1 - a, 0]
    cx = xa + 0.25 * (xd - xa)
    x0, y0 = cx - 320, ya + 20 - 320
    fc = xd < xa
    k = res / 640
    out = []
    for p in (0, 1):
        u, v = (pos[:, :, p, 0] - x0) * k, (pos[:, :, p, 1] - BOX_UP - y0) * k
        u = torch.where(fc, res - u, u)
        mc = box_mask(u, v, BOX_W * k, BOX_H * k, res, res, dev)
        us, vs = pos[:, :, p, 0] * sw / 1920, (pos[:, :, p, 1] - BOX_UP) * sh / 1080
        us = torch.where(scene_flip[:, None], sw - us, us)
        ms = box_mask(us, vs, BOX_W * sw / 1920, BOX_H * sh / 1080, sh, sw, dev)
        out.append((mc, ms))
    return out


def apply_players(x, xs, name, offs_p, masks):
    import torch
    base = name.split("_")[1]
    for p, (mc, ms) in enumerate(masks):
        x = torch.where(mc[:, :, None], transform(x, base, offs_p[p]), x)
        xs = torch.where(ms[:, :, None], transform(xs, base, offs_p[p]), xs)
    return x, xs


def apply(x, xs, name, off):
    base = name.split("_")[0]
    return (x if name.endswith("_scene") else transform(x, base, off), xs if name.endswith("_crop") else transform(xs, base, off))


def sector_offsets(items):
    """One random palette per (replay, side), stable between runs."""
    return np.stack([np.random.default_rng(zlib.crc32(f"{it['stem']}/{it['side']}".encode())).random(6) for it in items]).astype(np.float32)


def flips(items, root):
    """Was each item's whole-frame strip mirrored (video_scene.process's rule)? And its 21 frames."""
    idx, fl, frs = {}, [], []
    for it in items:
        s = it["stem"]
        if s not in idx:
            idx[s] = {(d["frame"], d["side"]): d for d in json.loads((root / s / "index.json").read_text())["dets"]}
        d = idx[s][(it["frame"], it["side"])]
        p, a = np.array(d["pos"]), it["side"] - 1
        fl.append(np.mean(p[:, 1 - a, 0] < p[:, a, 0]) > 0.5)
        frs.append([max(0, d["frame"] + o) for o in WIN])
    return np.array(fl), np.array(frs)


def positions(items, root):
    """[N, 21, 2, 2] the positions stored with each detection (1080p px, smoothed like the cut)."""
    idx, out = {}, []
    for it in items:
        s = it["stem"]
        if s not in idx:
            idx[s] = {(d["frame"], d["side"]): d for d in json.loads((root / s / "index.json").read_text())["dets"]}
        out.append(np.array(idx[s][(it["frame"], it["side"])]["pos"], np.float32))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N items only (a check; metrics then cover those)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="transforms (default all); hue0 always")
    ap.add_argument("--players", action="store_true", help="the transforms inside the player boxes")
    ap.add_argument("--dump", help="write the first batch's tiles for one transform to this jpg (a look at the masks), then stop")
    a = ap.parse_args()
    low_priority()
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    import backbone_feats as bf
    import video_players as vp
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    subs = test_subsets(fold)
    titems, vch = bf.test_setup(rows, fold)
    n_all = len(titems)
    cached = np.load(OUT / "exp_bb_v2_P_video_moves_test.npz")
    assert list(cached["files"]) == [it["file"] for it in titems]
    if a.limit:
        titems = titems[:a.limit]
    nets, ck = vm.load(OUT / "video_moves_test.pt")
    voc = ck["voc"]
    nets = nets if isinstance(nets, list) else [nets]
    for n in nets:
        assert list(n.frames) == vm.FRAMES and n.res == vm.RES and tuple(n.scene) == vm.SCENE, (n.frames, n.res, n.scene)
        n.eval()
    pck = torch.load(OUT / "video_players_test.pt", map_location=device())
    pnet = vp.make_net(len(pck["chars"])).to(device()).eval()
    pnet.load_state_dict(pck["state"])
    chars_list = pck["chars"]
    dev = device()
    mean, std = torch.from_numpy(vm.MEAN).to(dev)[0, 0], torch.from_numpy(vm.STD).to(dev)[0, 0]
    sel = {sh: torch.tensor([min(max(o + sh, WIN[0]), WIN[-1]) - WIN[0] for o in vm.FRAMES], device=dev) for sh in vm.SHIFTS}
    names = ["hue0"] + PLAYERS if a.players else [t for t in TRANSFORMS if not a.only or t in a.only or t == "hue0"]
    N, K = len(titems), len(nets) * len(vm.SHIFTS)
    store = {t: {h: np.zeros((N, K, len(voc[h])), np.float32) for h in ("category", "normal", "move")} for t in names}
    votes = {t: collections.defaultdict(lambda: 0) for t in names}
    seen = {t: collections.defaultdict(set) for t in names}
    fl, frs = flips(titems, DATA / "vcrops_test")
    offs = torch.from_numpy(sector_offsets(titems)).to(dev)
    offs_p = [torch.from_numpy(sector_offsets([dict(stem=it["stem"], side=p) for it in titems])).to(dev) for p in (1, 2)]
    pos_all, side_all = positions(titems, DATA / "vcrops_test"), torch.as_tensor([it["side"] for it in titems], device=dev)
    pos_all = torch.from_numpy(pos_all).to(dev)
    fl_t = torch.as_tensor(fl, device=dev)
    stems = [it["stem"] for it in titems]
    ds = Tiles(titems, vm.RES, vm.SCENE)
    dl = torch.utils.data.DataLoader(ds, batch_size=24, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=2)
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for idx, x, xs in dl:
            idx = idx.numpy()
            x = x.to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div(255)
            xs = xs.to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div(255)
            if done == 0:    # the conversion itself must not move anything
                h, s, v = rgb2hsv(x)
                print(f"hsv round trip: max diff {float((hsv2rgb(h, s, v) - x).abs().max()):.5f}", flush=True)
            off = offs[torch.as_tensor(idx, device=dev)]
            ti = torch.as_tensor(idx, device=dev)
            masks = player_masks(pos_all[ti], side_all[ti], fl_t[ti], vm.RES, vm.SCENE, dev) if a.players else None
            for t in names:
                if t in PLAYERS:
                    xt, st = apply_players(x, xs, t, [o[ti] for o in offs_p], masks)
                else:
                    xt, st = apply(x, xs, t, off)
                if a.dump and t != "hue0":
                    import cv2
                    im = torch.cat([xt[:6, 8], F.interpolate(st[:6, 8], size=vm.RES, mode="bilinear")], -1)
                    cv2.imwrite(a.dump.replace(".jpg", f"_{t}.jpg"), (im.permute(0, 2, 3, 1).flatten(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
                    continue
                # --- the character vote, from the whole frames
                B = st.shape[0]
                sv = st[:, ::VOTE_EVERY]
                T = sv.shape[1]
                un = torch.where(torch.as_tensor(fl[idx], device=dev)[:, None, None, None, None], sv.flip(-1), sv).flatten(0, 1)
                sm = F.interpolate(un, size=(144, 256), mode="area")
                inp = (F.interpolate(sm, size=(vp.IN_H, vp.IN_W), mode="bilinear", align_corners=False) - mean) / std
                gl = []
                for j in range(0, len(inp), 128):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                        gl.append(torch.log_softmax(pnet(inp[j:j + 128])["gchar"].float(), -1))
                gl = torch.cat(gl).view(B, T, 2, -1).cpu().numpy()
                del un, sm, inp, sv
                for b, i in enumerate(idx):
                    s_ = stems[i]
                    for k, f_ in enumerate(frs[i][::VOTE_EVERY]):
                        if f_ not in seen[t][s_]:
                            seen[t][s_].add(f_)
                            votes[t][s_] = votes[t][s_] + gl[b, k]
                # --- the move nets, every frame shift
                k = 0
                xn, sn = (xt - mean) / std, (st - mean) / std
                for net in nets:
                    for sh in vm.SHIFTS:
                        with vm.amp():
                            o = net(xn[:, sel[sh]], sn[:, sel[sh]])
                        for h in ("category", "normal", "move"):
                            store[t][h][idx, k] = o[h].float().cpu().numpy()
                        k += 1
            if a.dump:
                return
            done += len(idx)
            if done // 24 % 10 == 1:
                print(f"  {done}/{N} items, {time.time() - t0:.0f}s, eta {(N - done) * (time.time() - t0) / done / 60:.1f} min, "
                      f"gpu mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    print(f"pass done: {done} items, {len(names)} transforms, {time.time() - t0:.0f}s", flush=True)

    masks = {c: vm.char_mask(voc, c) for c in {m.split(":")[0] for m in voc["move"]}}

    def probs(t, chars, member=None):
        """The ensemble's probabilities for transform t with the given (P1, P2) per stem. member: one logit column only."""
        M = np.stack([masks.get(chars[it["stem"]][it["side"] - 1], np.zeros(len(voc["move"]), bool)) for it in titems])
        out = {}
        sl = slice(None) if member is None else [member]
        for h in ("category", "normal", "move"):
            v = torch.from_numpy(store[t][h][:, sl])
            if h == "move":
                v = v.masked_fill(~torch.from_numpy(M)[:, None], -1e4)
            out[h] = torch.softmax(v, -1).mean(1).numpy()
        return out

    subs_here = {k: [s for s in v if s in set(stems)] for k, v in subs.items()}
    truth = {}
    for r in pq.read_table(DATA / "transcript" / "transcript_events.parquet", columns=["replay", "side", "char"]).to_pylist():
        truth[(r["replay"], r["side"])] = r["char"]
    light = vm.SHIFTS.index(0)      # the first net (ResNet18), no frame-shift averaging: recognize.py --light
    train_counts = collections.Counter(f"{r['char']}:{r['move']}" for r in rows if r["replay"] in fold and fold[r["replay"]] != 0 and not r["is_burst"])
    res = dict(transforms=names, items=N, items_all=n_all, limit=a.limit, by={})
    base = None
    for t in names:
        vote = {s: (chars_list[int(v[0].argmax())], chars_list[int(v[1].argmax())]) for s, v in votes[t].items()}
        P_fix, P_own, P_lit = probs(t, vch), probs(t, vote), probs(t, vch, light)
        cf, co, cl = (opener_counts(titems, P, voc, rows, set(stems)) for P in (P_fix, P_own, P_lit))
        if base is None:
            base = dict(fix=cf, own=co, light=cl)
            if not a.limit:
                agree = float(np.mean(P_fix["move"].argmax(1) == cached["move"].argmax(1)))
                print(f"hue0 against the saved v2 probabilities: move argmax agrees {agree:.5f}; max |dP| {np.abs(P_fix['move'] - cached['move']).max():.4f}", flush=True)
                res["hue0_agrees_with_saved"] = agree
        d = {}
        for sub, st in subs_here.items():
            if not st or sub == "old_unleaked":
                continue
            sides = [(s, k) for s in st for k in (0, 1)]
            d[sub] = dict(chain=round(ratio(cf, "chain", "n", st), 4), move=round(ratio(cf, "move", "det", st), 4), category=round(ratio(cf, "cat", "det", st), 4),
                          chain_own_vote=round(ratio(co, "chain", "n", st), 4), move_own_vote=round(ratio(co, "move", "det", st), 4),
                          chain_light=round(ratio(cl, "chain", "n", st), 4), move_light=round(ratio(cl, "move", "det", st), 4),
                          vote_sides_right=sum(vote[s][k] == truth.get((s, f"P{k + 1}")) for s, k in sides),
                          vote_replays_right=sum(all(vote[s][k] == truth.get((s, f"P{k + 1}")) for k in (0, 1)) for s in st),
                          vote_sides=len(sides), replays=len(st), vote_same_as_hue0=sum(vote[s][k] == vch[s][k] for s, k in sides))
            if t != "hue0":
                d[sub]["chain_diff"] = paired_bootstrap(base["fix"], cf, "chain", "n", st)
                d[sub]["move_diff"] = paired_bootstrap(base["fix"], cf, "move", "det", st)
                d[sub]["chain_own_diff"] = paired_bootstrap(base["own"], co, "chain", "n", st)
                ex = bf.extras(titems, dict(P_fix), voc, vch, rows, st, train_counts)
                d[sub]["by_category"] = {c: dict(n=v["n"], chain=v["chain_right"], move=v["move_acc"]) for c, v in ex["by_category"].items()}
        res["by"][t] = d
        print(t, json.dumps({k: {x: v[x] for x in ("chain", "move", "category", "chain_own_vote", "chain_light", "vote_sides_right", "vote_sides")}
                             for k, v in d.items()}), flush=True)
    save_json(res, OUT / ("results_exp_measure_hue_check.json" if a.limit else "results_exp_measure_hue_players.json" if a.players else "results_exp_measure_hue.json"))


if __name__ == "__main__":
    main()
