"""Game-state embeddings from the video (ROADMAP point 4): a small tower over a few seconds of the
transcript's frozen V-JEPA 2 tokens + HUD, aligned CLIP-style with the logged game state, then
checked by state probes and situation retrieval on the frozen test fold.

State per 7.5 Hz tubelet (labels, from the game log only; `labels`): the log row shown at the
tubelet's middle video frame (8 t + 4, inputs_labels.py's frame -> row map), per side
- x, y (world units / 1e6), HP (/ 420), facing;
- a state class CLS: knockdown (on the floor: B/F/V Down loop/down/bound, QuickDown, the wake-up
  *Down2Stand), hitstun (any other damage state, throw tech), blockstun (guard states), attacking
  (any move: the inputs labels' `move` state), airborne (y > 0, not one of the above), neutral;
plus the distance |x1 - x2| and, for each side, cornered (|x| >= the wall - CORNER, the wall
of the current stage section: sections are centred on multiples of 1.6 after wallbreaks, see
stage_centre), `live`; x in the state is relative to the section's centre. The log has no
tension or burst gauge, so those are not in the state (the HUD reading of tension is an input).
-> <data>/ggst_dataset/search/state/<stem>.npz, search/state_meta.json

Model (`train`): video tower = per-tubelet projections of the 3 view tokens, the 2 x 2 quadrant
tokens of each player square and the HUD/position aux (video_transcript.load_replay, P1-first
order), a 2-layer transformer over W tubelets (default 16 = 2.1 s) centred on the tubelet, the
centre token -> 128-d unit embedding. State tower = MLP over the state vector -> 128-d. Loss:
symmetric InfoNCE over the batch (learned temperature); `--sup w` adds supervised state heads on the
video embedding (class CE per side, distance / x / HP regression, corner) with weight w. Inputs are
video only; the log only makes the targets.

Evaluation (`eval`, test fold old / new, val for nothing but the epoch): probes fitted on frozen
embeddings of training tubelets (linear softmax / ridge, fitted on the GPU): class per side,
'who is hit' (neither / P1 / P2 in hitstun or knockdown), corner (none / P1 / P2 cornered),
distance, x, HP. Situation retrieval: for query tubelets, the top-k nearest test tubelets of other
replays by cosine; share with the same joint class (cls P1, cls P2), the same corner state, and the
mean |distance difference|, against random picks and against nearest neighbours in the logged
state vector itself (the ceiling of what a state match can give). The same probes and retrieval
on the raw tokens (no tower) are the baseline (`--raw`).

    .venv/Scripts/python.exe tools/ggst_logger/model/state_embed.py labels [--jobs 4]
    .venv/Scripts/python.exe tools/ggst_logger/model/state_embed.py train --name clip [--sup 0] [--epochs 20]
    .venv/Scripts/python.exe tools/ggst_logger/model/state_embed.py eval --name clip     (or --raw)
    .venv/Scripts/python.exe tools/ggst_logger/model/state_embed.py predict --name clip_sup   # per-tubelet state of val/test
    .venv/Scripts/python.exe tools/ggst_logger/model/state_embed.py similar --name clip_sup --stem S --time 1:23.4   # moments like this

-> model/state_embed_exp_se_<name>.pt, model/results_exp_se_<name>.json, search/exp_se_<name>.log,
   search/exp_se_<name>_pred/<stem>.npz (predict: class probabilities per side, corner, distance)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "dataset"))
from common import DATA, OUT, low_priority, save_json, test_subsets, replay_fold_map  # noqa: E402

SE = DATA / "search"
STATE = SE / "state"
TR = DATA / "transcript"
LAB = DATA / "inputs" / "labels"
CLS = ["neutral", "attacking", "blockstun", "hitstun", "knockdown", "airborne"]
KD = re.compile(r"^CmnAct(?:[BFV]Down(?:Loop|Down|Bound|2Stand)|QuickDown)")
CORNER = 0.30          # 1e6 units from the wall (about a character width and a half)
STEP = 8


def log(msg, f=None):
    s = time.strftime("%H:%M:%S ") + msg
    print(s, flush=True)
    if f:
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")


def splits():
    return {k: v for k, v in json.loads((TR / "transcript_splits.json").read_text()).items()}


def split_lists():
    import transcript_eval as te
    sp = splits()
    if not all(isinstance(v, str) for v in sp.values()):
        sp = te.splits(list(sp))
    out = collections.defaultdict(list)
    for s, k in sp.items():
        out[k].append(s)
    return {k: sorted(v) for k, v in out.items()}


# ---- labels ------------------------------------------------------------------------------------

def act_class(acts, move_state):
    import build
    cache = {}
    out = np.zeros(len(acts), np.uint8)
    for i, a in enumerate(acts):
        c = cache.get(a)
        if c is None:
            if KD.match(a):
                c = 4
            elif build.is_damage(a) or a.startswith("CmnActHajikare"):
                c = 3
            elif build.is_guard(a):
                c = 2
            else:
                c = -1
            cache[a] = c
        out[i] = c if c >= 0 else 255
    return out


def label_one(args):
    stem, csv_path = args
    import build
    lab = np.load(LAB / f"{stem}.npz")
    row, live_f, st = lab["row"], lab["live"], lab["state"]
    F = len(row)
    n = len(np.load(TR / "feats" / f"{stem}.npz")["tub_frame"])
    lg = build.load_log(csv_path)
    if "p1_x" not in lg:
        return stem, None
    nr = lg["n"]
    # the row at each tubelet's middle frame; nearest mapped frame within +-4 otherwise
    valid = np.where(row >= 0)[0]
    mid = np.minimum(STEP * np.arange(n) + 4, F - 1)
    j = np.clip(np.searchsorted(valid, mid), 0, len(valid) - 1)
    jm = np.clip(j - 1, 0, len(valid) - 1)
    near = np.where(np.abs(valid[jm] - mid) < np.abs(valid[j] - mid), valid[jm], valid[j])
    okt = np.abs(near - mid) <= 4
    fr = np.where(okt, near, mid)
    r = np.where(okt, row[fr], -1)
    rr = np.clip(r, 0, nr - 1)
    out = dict(row=r.astype(np.int32), live=(okt & live_f[fr]))
    X, Y, H, FC, C = [], [], [], [], []
    for k, s in enumerate((1, 2)):
        X.append(lg[f"p{s}_x"][rr] / 1e6)
        Y.append(lg[f"p{s}_y"][rr] / 1e6)
        H.append(lg[f"p{s}_hp"][rr] / 420.0)
        FC.append(lg[f"p{s}_face"][rr])
        ac = act_class(lg[f"p{s}_act"], None)[rr]
        mv = st[k][fr] == 0                          # inputs_labels STATES[0] = move
        c = np.where(ac != 255, ac, np.where(mv, 1, np.where(Y[-1] > 0, 5, 0))).astype(np.uint8)
        C.append(c)
    out.update(x=np.stack(X).astype(np.float32), y=np.stack(Y).astype(np.float32), hp=np.stack(H).astype(np.float32),
               face=np.stack(FC).astype(np.uint8), cls=np.stack(C))
    out["live"] &= r >= 0
    STATE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STATE / f"{stem}.npz", **out)
    ax = np.abs(out["x"][:, out["live"]]).ravel()
    return stem, dict(n=int(n), live=int(out["live"].sum()), absx_hist=np.histogram(ax, bins=60, range=(0, 3))[0].tolist(),
                      cls=np.bincount(out["cls"][:, out["live"]].ravel(), minlength=len(CLS)).tolist())


def cmd_labels(a):
    import pyarrow.parquet as pq
    import build
    build.set_own_priority(build.BELOW_NORMAL)
    stems = sorted({r["replay"] for r in pq.read_table(TR / "transcript_events.parquet", columns=["replay"]).to_pylist()})
    csv = {o["replay"]: o["csv"] for o in pq.read_table(DATA / "openers.parquet", columns=["replay", "csv"]).to_pylist()}
    t0 = time.time()
    stats = {}
    with ProcessPoolExecutor(max_workers=min(a.jobs, 4)) as ex:
        for k, (st, res) in enumerate(ex.map(label_one, [(s, csv[s]) for s in stems], chunksize=2)):
            stats[st] = res
            if k % 100 == 0:
                print(f"{k}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    good = {k: v for k, v in stats.items() if v}
    hist = np.sum([v["absx_hist"] for v in good.values()], 0)
    cdf = np.cumsum(hist) / hist.sum()
    wall = float(np.searchsorted(cdf, 0.999) + 1) * 0.05
    meta = dict(replays=len(good), skipped=[k for k, v in stats.items() if not v], wall=wall, corner=CORNER, cls=CLS,
                tubelets=int(sum(v["n"] for v in good.values())), live=int(sum(v["live"] for v in good.values())),
                cls_counts=dict(zip(CLS, np.sum([v["cls"] for v in good.values()], 0).tolist())),
                absx_hist_0_3_by_0p05=hist.tolist(), secs=round(time.time() - t0))
    save_json(meta, SE / "state_meta.json")
    print(json.dumps({k: v for k, v in meta.items() if k != "absx_hist_0_3_by_0p05"}, indent=1))


HALF, GRID = 1.515, 1.6    # half the wall-to-wall width; stage sections are centred on multiples of GRID (wallbreaks)


def stage_centre(x, live):
    """Per tubelet the centre of the current stage section (a multiple of GRID; wallbreaks move it): a Viterbi path
    over the GRID multiples, cost per tubelet = 10 x how far either player is outside that section's walls
    (+ a tie-break towards the midpoint), 1 per change of section; non-live tubelets cost nothing. Checked on the
    segments where both walls are touched (hi - lo = 3.03): 99.6% of their tubelets agree; 0.1% of live tubelets
    stay outside the walls afterwards (wallbreak transitions, dropped from live in load_state)."""
    mid = (x[0] + x[1]) / 2
    n = len(mid)
    if not live.any():
        return np.zeros(n, np.float32)
    ks = np.arange(np.floor(min(x[:, live].min(), 0) / GRID) - 1, np.ceil(max(x[:, live].max(), 0) / GRID) + 2)
    cen = ks * GRID
    out = np.maximum(0, np.abs(x[:, :, None] - cen[None, None]) - HALF - 0.015).sum(0)       # [n, K]
    cost = 10 * out + 0.0005 * np.abs(mid[:, None] - cen[None])
    cost[~live] = 0
    K = len(ks)
    acc = cost[0].copy()
    back = np.zeros((n, K), np.int32)
    for t in range(1, n):
        best = acc.argmin()
        stay = acc
        move = acc[best] + 1.0
        back[t] = np.where(stay <= move, np.arange(K), best)
        acc = np.minimum(stay, move) + cost[t]
    path = np.zeros(n, np.int64)
    path[-1] = acc.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return cen[path].astype(np.float32)


def load_state(stem, wall=None):
    """State labels of a replay; x made relative to the stage section's centre (walls at +-HALF), corner =
    within CORNER of a wall and nearer to it than the opponent."""
    z = np.load(STATE / f"{stem}.npz")
    d = {k: z[k] for k in z}
    c = stage_centre(d["x"], d["live"])
    d["x"] = (d["x"] - c[None]).astype(np.float32)
    ax = np.abs(d["x"])
    xo = d["x"][::-1]
    outer = np.where(d["x"] > 0, d["x"] >= xo, d["x"] <= xo)          # nearer to that wall than the opponent
    d["corner"] = (ax >= HALF - CORNER) & outer
    d["dist"] = np.abs(d["x"][0] - d["x"][1])
    d["live"] = d["live"] & (ax.max(0) <= HALF + 0.035)     # wallbreak transitions (0.1%: players flying between sections)
    return d


def state_vector(d):
    """[n, S] float32: per side x, y, hp, facing, corner, class one-hot; distance."""
    parts = []
    for k in (0, 1):
        parts += [d["x"][k][:, None], d["y"][k][:, None] * 2, d["hp"][k][:, None], d["face"][k][:, None].astype(np.float32),
                  d["corner"][k][:, None].astype(np.float32), np.eye(len(CLS), dtype=np.float32)[d["cls"][k]]]
    parts.append(d["dist"][:, None])
    return np.concatenate(parts, 1).astype(np.float32)


def derived(d):
    """Per-tubelet derived targets: who-hit (0 neither, 1 P1 hit, 2 P2 hit; both -> 1), corner state (0 none, 1 P1
    cornered, 2 P2 cornered)."""
    hit = (d["cls"] == 3) | (d["cls"] == 4)
    who = np.where(hit[0], 1, np.where(hit[1], 2, 0))
    cor = np.where(d["corner"][0], 1, np.where(d["corner"][1], 2, 0))
    return who.astype(np.int64), cor.astype(np.int64)


# ---- data on the GPU ---------------------------------------------------------------------------

class Data:
    """All tubelets of some replays: tokens (flat scene | p1 | p2 | p1 grid | p2 grid, fp16), aux (P1 first),
    state vector, labels; on the GPU."""

    def __init__(self, stems, wall, norm=None, dev="cuda", keep_tokens=True):
        import torch
        import video_transcript as vt
        chars = vt.meta()["chars"]
        Vs, As, Ss, L = [], [], [], collections.defaultdict(list)
        self.stems, self.off, self.n = [], [], []
        o = 0
        for st in stems:
            if not (STATE / f"{st}.npz").exists():
                continue
            V, A, _ = vt.load_replay(st, chars, grid=True)
            d = load_state(st, wall)
            n = min(len(V), len(d["row"]))
            if keep_tokens:
                Vs.append(torch.from_numpy(V[:n]).to(dev))
            As.append(A[0, :n])
            Ss.append(state_vector(d)[:n])
            who, cor = derived(d)
            for k, v in (("cls1", d["cls"][0]), ("cls2", d["cls"][1]), ("who", who), ("cor", cor), ("dist", d["dist"]),
                         ("x1", d["x"][0]), ("x2", d["x"][1]), ("hp1", d["hp"][0]), ("hp2", d["hp"][1]), ("live", d["live"])):
                L[k].append(v[:n])
            self.stems.append(st)
            self.off.append(o)
            self.n.append(n)
            o += n
        A = np.concatenate(As)
        if norm is None:
            norm = (A.mean(0), A.std(0) + 1e-3)
        self.norm = norm
        self.V = torch.cat(Vs) if keep_tokens else None
        self.A = torch.from_numpy(((A - norm[0]) / norm[1]).astype(np.float16)).to(dev)
        self.S = torch.from_numpy(np.concatenate(Ss)).to(dev)
        self.lab = {k: np.concatenate(v) for k, v in L.items()}
        self.off, self.n = np.array(self.off), np.array(self.n)
        self.rep = np.repeat(np.arange(len(self.stems)), self.n)
        self.lo = np.repeat(self.off, self.n)
        self.hi = self.lo + np.repeat(self.n, self.n) - 1
        self.live_idx = np.where(self.lab["live"])[0]
        self.dev = dev

    def window(self, idx, W):
        """Token / aux windows [B, W, *] around global tubelet indices idx (clamped inside each replay)."""
        import torch
        rel = np.arange(W) - W // 2
        g = np.clip(idx[:, None] + rel[None], self.lo[idx][:, None], self.hi[idx][:, None])
        gi = torch.as_tensor(g, device=self.dev)
        return self.V[gi], self.A[gi]


# ---- model -------------------------------------------------------------------------------------

def build_nets(n_aux, n_state, d=256, emb=128, W=16, layers=2, heads=4, drop=0.1, in_drop=0.2):
    import torch
    import torch.nn as nn

    class Video(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = dict(n_aux=n_aux, n_state=n_state, d=d, emb=emb, W=W, layers=layers, heads=heads, drop=drop, in_drop=in_drop)
            self.vin = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(in_drop), nn.Linear(1024, d))
            self.gin = nn.Sequential(nn.LayerNorm(192), nn.Dropout(in_drop), nn.Flatten(-2), nn.Linear(4 * 192, d))
            self.vmix = nn.Linear(5 * d, d)
            self.ain = nn.Sequential(nn.Linear(n_aux, d), nn.GELU(), nn.Linear(d, d))
            self.pos = nn.Parameter(torch.zeros(W, d))
            el = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(el, layers)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, emb)
            # supervised heads (used with --sup)
            self.h_cls = nn.Linear(emb, 2 * len(CLS))
            self.h_reg = nn.Linear(emb, 5)          # dist, x1, x2, hp1, hp2
            self.h_cor = nn.Linear(emb, 3)

        def forward(self, V, A):
            B, L = V.shape[:2]
            D = 1024
            v = self.vin(V[..., :3 * D].float().view(B, L, 3, D)).reshape(B, L, -1)
            g = self.gin(V[..., 3 * D:].float().view(B, L, 2, 4, 192)).reshape(B, L, -1)
            x = self.vmix(torch.cat([v, g], -1)) + self.ain(A.float()) + self.pos[None, :L]
            x = self.tr(x)
            z = self.out(self.norm(x[:, L // 2]))
            return z

    class State(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_state, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, emb))

        def forward(self, s):
            return self.net(s)

    return Video(), State()


def sup_loss(net, z, lab, idx, dev):
    import torch
    import torch.nn.functional as F
    t = lambda k, dt=torch.long: torch.as_tensor(lab[k][idx], device=dev, dtype=dt)
    c = net.h_cls(z).view(len(z), 2, len(CLS))
    l = F.cross_entropy(c[:, 0], t("cls1")) + F.cross_entropy(c[:, 1], t("cls2")) + F.cross_entropy(net.h_cor(z), t("cor"))
    reg = torch.stack([t(k, torch.float32) for k in ("dist", "x1", "x2", "hp1", "hp2")], 1)
    return l + F.smooth_l1_loss(net.h_reg(z), reg, beta=0.1) * 4


def embed(net, D, idx, W, bs=4096):
    import torch
    import torch.nn.functional as F
    out = []
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(idx), bs):
            V, A = D.window(idx[i:i + bs], W)
            out.append(F.normalize(net(V, A).float(), dim=-1))
    return torch.cat(out)


def raw_feats(D, idx, bs=8192):
    """Baseline: the centre tubelet's tokens (3 views mean-centred, grid) + aux, no tower."""
    import torch
    out = []
    for i in range(0, len(idx), bs):
        gi = torch.as_tensor(idx[i:i + bs], device=D.dev)
        out.append(torch.cat([D.V[gi].float(), D.A[gi].float()], 1))
    return torch.cat(out)


def cmd_train(a):
    import torch
    import torch.nn.functional as F
    low_priority()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    meta = json.loads((SE / "state_meta.json").read_text())
    sl = split_lists()
    lf = SE / f"exp_se_{a.name}.log"
    log(f"train {a.name}: {vars(a)}", lf)
    tr_stems = sl["train"] if a.frac >= 1 else sorted(rng.permutation(sl["train"])[:int(a.frac * len(sl["train"]))])
    Dtr = Data(tr_stems, meta["wall"])
    Dva = Data(sl["val"], meta["wall"], norm=Dtr.norm)
    log(f"train {len(Dtr.stems)} replays {len(Dtr.live_idx)} live tubelets; val {len(Dva.stems)} / {len(Dva.live_idx)}; "
        f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB", lf)
    video, state = build_nets(Dtr.A.shape[1], Dtr.S.shape[1], W=a.window, emb=a.emb)
    video, state = video.cuda(), state.cuda()
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device="cuda"))
    params = list(video.parameters()) + list(state.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    steps = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()
    # state normalisation (train)
    smu, ssd = Dtr.S[Dtr.live_idx].mean(0), Dtr.S[Dtr.live_idx].std(0) + 1e-3
    best, best_ep, t0 = -1, -1, time.time()
    vq = np.sort(rng.choice(Dva.live_idx, min(20000, len(Dva.live_idx)), replace=False))
    for ep in range(a.epochs):
        video.train()
        state.train()
        tl = 0.0
        for it in range(a.steps):
            idx = rng.choice(Dtr.live_idx, a.batch, replace=False)
            V, A = Dtr.window(idx, a.window)
            s = (Dtr.S[torch.as_tensor(idx, device="cuda")] - smu) / ssd
            s = s + torch.randn_like(s) * a.snoise
            with torch.autocast("cuda", dtype=torch.float16):
                zv = video(V, A)
                zs = state(s)
            zvn, zsn = F.normalize(zv.float(), dim=-1), F.normalize(zs.float(), dim=-1)
            lg = logit_scale.clamp(max=np.log(100)).exp() * zvn @ zsn.T
            y = torch.arange(len(idx), device="cuda")
            loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2 if a.clip > 0 else torch.zeros((), device="cuda")
            loss = loss * a.clip
            if a.sup > 0:
                loss = loss + a.sup * sup_loss(video, zv.float(), Dtr.lab, idx, "cuda")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += float(loss)
        # validation: video->state top-1 within blocks of 1024 val tubelets, and a quick class probe proxy (nearest
        # state-centroid of the sup head when there is one)
        video.eval()
        state.eval()
        with torch.no_grad():
            zv = embed(video, Dva, vq, a.window)
            zs = F.normalize(state((Dva.S[torch.as_tensor(vq, device="cuda")] - smu) / ssd).float(), dim=-1)
            acc, accs = [], []
            for i in range(0, len(vq) - 1023, 1024):
                sim = zv[i:i + 1024] @ zs[i:i + 1024].T
                acc.append(float((sim.argmax(1) == torch.arange(1024, device="cuda")).float().mean()))
                # same joint class of the arg-max state
                j = sim.argmax(1).cpu().numpy() + i
                q = np.arange(i, i + 1024)
                accs.append(float(np.mean((Dva.lab["cls1"][vq[j]] == Dva.lab["cls1"][vq[q]]) & (Dva.lab["cls2"][vq[j]] == Dva.lab["cls2"][vq[q]]))))
            vclass = None
            if a.sup > 0:
                c = video.h_cls(zv).view(len(zv), 2, len(CLS)).argmax(-1).cpu().numpy()
                vclass = float(np.mean([(c[:, 0] == Dva.lab["cls1"][vq]).mean(), (c[:, 1] == Dva.lab["cls2"][vq]).mean()]))
        score = float(np.mean(accs)) if a.clip > 0 else vclass
        log(f"ep {ep + 1} loss {tl / a.steps:.3f} val v->s top1@1024 {np.mean(acc):.3f} joint-class@1 {np.mean(accs):.3f}"
            f" sup-class {vclass} T {1 / float(logit_scale.exp()):.3f} {time.time() - t0:.0f}s", lf)
        if score > best:
            best, best_ep = score, ep + 1
            torch.save(dict(video=video.state_dict(), state=state.state_dict(), cfg=video.cfg, norm=[x.tolist() for x in Dtr.norm],
                            smu=smu.cpu().numpy().tolist(), ssd=ssd.cpu().numpy().tolist(), epoch=ep + 1, args=vars(a),
                            logit_scale=float(logit_scale)), OUT / f"state_embed_exp_se_{a.name}.pt")
    log(f"best epoch {best_ep} val {best:.3f}; {(time.time() - t0) / 60:.1f} min", lf)


def load_net(name):
    import torch
    ck = torch.load(OUT / f"state_embed_exp_se_{name}.pt", map_location="cuda", weights_only=False)
    c = ck["cfg"]
    video, state = build_nets(c["n_aux"], c["n_state"], d=c["d"], emb=c["emb"], W=c["W"], layers=c["layers"], heads=c["heads"])
    video.load_state_dict(ck["video"])
    state.load_state_dict(ck["state"])
    return video.cuda().eval(), state.cuda().eval(), ck


# ---- probes and retrieval ----------------------------------------------------------------------

def fit_softmax(X, y, n, wd=1e-4, epochs=60):
    import torch
    import torch.nn.functional as F
    mu, sd = X.mean(0), X.std(0) + 1e-4
    W = torch.zeros(X.shape[1], n, device=X.device, requires_grad=True)
    b = torch.zeros(n, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    yt = torch.as_tensor(y, device=X.device)
    Xn = (X - mu) / sd
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        for i in range(0, len(X), 8192):
            j = perm[i:i + 8192]
            loss = F.cross_entropy(Xn[j] @ W + b, yt[j]) + wd * (W ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return lambda Z: (((Z - mu) / sd) @ W + b).argmax(1).detach().cpu().numpy()


def fit_ridge(X, Y, lam=1.0):
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xn = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=X.device)], 1).double()
    Yt = torch.as_tensor(Y, device=X.device).double()
    Wt = torch.linalg.solve(Xn.T @ Xn + lam * torch.eye(Xn.shape[1], device=X.device, dtype=torch.float64), Xn.T @ Yt)
    return lambda Z: (torch.cat([(Z - mu) / sd, torch.ones(len(Z), 1, device=Z.device)], 1).double() @ Wt).cpu().numpy()


def fit_mlp(X, lab, idx, epochs=30, hidden=512):
    """One 2-layer MLP probe for every target at once (class heads + the 5 regressions); returns (cls predictors,
    reg predictor) shaped like the linear probes'."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(0)
    mu, sd = X.mean(0), X.std(0) + 1e-4
    heads = (("cls1", len(CLS)), ("cls2", len(CLS)), ("who", 3), ("cor", 3))
    nout = sum(n for _, n in heads) + 5
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, nout)).to(X.device)
    Y = {k: torch.as_tensor(lab[k][idx], device=X.device) for k, _ in heads}
    R = torch.as_tensor(np.stack([lab[k][idx] for k in ("dist", "x1", "x2", "hp1", "hp2")], 1), device=X.device, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xn = (X - mu) / sd
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=X.device)
        for i in range(0, len(X), 4096):
            j = perm[i:i + 4096]
            o = net(Xn[j])
            a, loss = 0, 0
            for k, n in heads:
                loss = loss + F.cross_entropy(o[:, a:a + n], Y[k][j])
                a += n
            loss = loss + 4 * F.smooth_l1_loss(o[:, a:], R[j], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def out(Z):
        with torch.no_grad():
            return net((Z - mu) / sd)
    offs, a = {}, 0
    for k, n in heads:
        offs[k] = (a, n)
        a += n
    cls = {k: (lambda Z, a=a0, n=n0: out(Z)[:, a:a + n].argmax(1).cpu().numpy()) for k, (a0, n0) in offs.items()}
    return cls, (lambda Z: out(Z)[:, a:].cpu().numpy())


def balanced(y, p, n):
    r = [np.mean(p[y == c] == c) for c in range(n) if (y == c).any()]
    return float(np.mean(r))


def probe_metrics(predict_cls, predict_reg, Z, lab, idx):
    out = {}
    for k, n in (("cls1", len(CLS)), ("cls2", len(CLS)), ("who", 3), ("cor", 3)):
        p = predict_cls[k](Z)
        y = lab[k][idx]
        out[k] = dict(acc=float(np.mean(p == y)), balanced=balanced(y, p, n))
        if k == "cls1":
            out["cls_per_class_recall"] = {}
        if k in ("cls1", "cls2"):
            for c in range(n):
                m = y == c
                out.setdefault("cls_per_class_recall", {}).setdefault(CLS[c], []).append(float(np.mean(p[m] == c)) if m.any() else None)
    R = predict_reg(Z)
    for j, k in enumerate(("dist", "x1", "x2", "hp1", "hp2")):
        y = lab[k][idx]
        mae = float(np.mean(np.abs(R[:, j] - y)))
        r2 = float(1 - np.mean((R[:, j] - y) ** 2) / np.var(y))
        out[k] = dict(mae=mae, r2=r2)
    out["cls_mean_acc"] = (out["cls1"]["acc"] + out["cls2"]["acc"]) / 2
    out["cls_mean_balanced"] = (out["cls1"]["balanced"] + out["cls2"]["balanced"]) / 2
    return out


def retrieval(Zq, Zp, lab, q, pool, rep, k=10, rng=None, state_space=False):
    """For each query (global index q), the top-k pool tubelets of other replays by cosine (or by -L2 in the state
    space): share with the same joint class, the same corner state, the same who-is-hit; mean |distance diff|."""
    import torch
    same_j, same_c, same_w, dd = [], [], [], []
    for i in range(0, len(q), 512):
        qi = q[i:i + 512]
        if state_space:
            sim = -torch.cdist(Zq[i:i + 512], Zp)
        else:
            sim = Zq[i:i + 512] @ Zp.T
        sim[torch.as_tensor(rep[qi][:, None] == rep[pool][None], device=sim.device)] = -1e9
        top = sim.topk(k, dim=1).indices.cpu().numpy()
        nb = pool[top]
        same_j.append(((lab["cls1"][nb] == lab["cls1"][qi][:, None]) & (lab["cls2"][nb] == lab["cls2"][qi][:, None])).mean(1))
        same_c.append((lab["cor"][nb] == lab["cor"][qi][:, None]).mean(1))
        same_w.append((lab["who"][nb] == lab["who"][qi][:, None]).mean(1))
        dd.append(np.abs(lab["dist"][nb] - lab["dist"][qi][:, None]).mean(1))
    c = np.concatenate
    return dict(joint_class=float(c(same_j).mean()), corner=float(c(same_c).mean()), who_hit=float(c(same_w).mean()),
                dist_abs_diff=float(c(dd).mean()))


def random_retrieval(lab, q, pool, rep, k, rng):
    nb = pool[rng.integers(0, len(pool), (len(q), k))]
    return dict(joint_class=float(((lab["cls1"][nb] == lab["cls1"][q][:, None]) & (lab["cls2"][nb] == lab["cls2"][q][:, None])).mean()),
                corner=float((lab["cor"][nb] == lab["cor"][q][:, None]).mean()),
                who_hit=float((lab["who"][nb] == lab["who"][q][:, None]).mean()),
                dist_abs_diff=float(np.abs(lab["dist"][nb] - lab["dist"][q][:, None]).mean()))


def cmd_eval(a):
    import torch
    import torch.nn.functional as F
    low_priority()
    rng = np.random.default_rng(0)
    meta = json.loads((SE / "state_meta.json").read_text())
    sl = split_lists()
    name = "raw" if a.raw else a.name
    lf = SE / f"exp_se_{name}.log"
    if a.raw:
        net, W, ck = None, 1, None
        norm = None
    else:
        net, snet, ck = load_net(a.name)
        W = ck["cfg"]["W"]
        norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    t0 = time.time()
    # probe training set: a subsample of training replays' live tubelets
    rs = np.random.default_rng(1)
    pstems = sorted(rs.permutation(sl["train"])[:a.probe_replays])
    Dp = Data(pstems, meta["wall"], norm=norm)
    if norm is None:
        norm = Dp.norm
    pidx = Dp.live_idx if len(Dp.live_idx) <= a.probe_n else np.sort(rs.choice(Dp.live_idx, a.probe_n, replace=False))
    Zp = raw_feats(Dp, pidx) if a.raw else embed(net, Dp, pidx, W)
    if a.raw:   # reduce the raw 4.6k dims with PCA fitted on the probe set (keeps the probes cheap)
        mu = Zp.mean(0)
        U, S, Vt = torch.linalg.svd((Zp - mu)[::4], full_matrices=False)
        P = Vt[:a.raw_pca].T
        proj = lambda Z: (Z - mu) @ P
        Zp = proj(Zp)
    preds = {}
    for k, n in (("cls1", len(CLS)), ("cls2", len(CLS)), ("who", 3), ("cor", 3)):
        preds[k] = fit_softmax(Zp, Dp.lab[k][pidx], n)
    regY = np.stack([Dp.lab[k][pidx] for k in ("dist", "x1", "x2", "hp1", "hp2")], 1)
    reg = fit_ridge(Zp, regY)
    mlp_cls, mlp_reg = fit_mlp(Zp, Dp.lab, pidx)
    probe_train = probe_metrics(preds, reg, Zp, Dp.lab, pidx)
    del Dp
    torch.cuda.empty_cache()
    log(f"eval {name}: probes fitted on {len(pidx)} tubelets of {len(pstems)} training replays, {time.time() - t0:.0f}s", lf)
    fold = replay_fold_map()
    sub = test_subsets(fold)
    Dt = Data(sub["all"], meta["wall"], norm=norm)
    tidx = Dt.live_idx
    Zt = raw_feats(Dt, tidx) if a.raw else embed(net, Dt, tidx, W)
    if a.raw:
        Zt = proj(Zt)
        Zt_r = F.normalize(Zt, dim=-1)
    else:
        Zt_r = Zt
    pos = np.full(len(Dt.lab["live"]), -1)
    pos[tidx] = np.arange(len(tidx))
    res = dict(name=name, probe_train_fit=probe_train, probe_n=int(len(pidx)), probe_replays=len(pstems), window=W,
               ckpt_epoch=None if ck is None else ck["epoch"], args=vars(a))
    # sup-head state reading (no probe) when the net has trained heads
    S_all = Dt.S
    smu = None if ck is None else torch.tensor(ck["smu"], device="cuda")
    for subset in ("old", "new"):
        ss = set(sub[subset])
        m = np.array([Dt.stems[Dt.rep[i]] in ss for i in tidx])
        qi = tidx[m]
        Zs = Zt[torch.as_tensor(np.where(m)[0], device="cuda")]
        r = dict(tubelets=int(m.sum()), probe=probe_metrics(preds, reg, Zs, Dt.lab, qi), probe_mlp=probe_metrics(mlp_cls, mlp_reg, Zs, Dt.lab, qi))
        if ck is not None and ck["args"].get("sup", 0) > 0:
            with torch.no_grad():
                c = net.h_cls(Zs * 0 + Zs).view(len(Zs), 2, len(CLS)).argmax(-1).cpu().numpy()
            r["sup_head_cls_acc"] = [float(np.mean(c[:, 0] == Dt.lab["cls1"][qi])), float(np.mean(c[:, 1] == Dt.lab["cls2"][qi]))]
        # retrieval: queries from this subset, pool = every live test tubelet of other replays
        qs = np.sort(rng.choice(np.where(m)[0], min(a.queries, int(m.sum())), replace=False))
        pool_pos = np.arange(len(tidx))
        rep = Dt.rep
        rr = {}
        for k in (1, 10):
            rr[f"top{k}"] = retrieval(Zt_r[torch.as_tensor(qs, device="cuda")], Zt_r, Dt.lab, tidx[qs], tidx, rep, k=k)
            rr[f"random_k{k}"] = random_retrieval(Dt.lab, tidx[qs], tidx, rep, k, rng) if k == 10 else None
            if smu is not None or a.raw:
                Sn = (S_all - S_all[torch.as_tensor(tidx, device="cuda")].mean(0)) / (S_all[torch.as_tensor(tidx, device="cuda")].std(0) + 1e-3)
                rr[f"state_oracle_top{k}"] = retrieval(Sn[torch.as_tensor(tidx[qs], device="cuda")], Sn[torch.as_tensor(tidx, device="cuda")],
                                                     Dt.lab, tidx[qs], tidx, rep, k=k, state_space=True)
        rr = {k: v for k, v in rr.items() if v is not None}
        r["retrieval"] = rr
        r["queries"] = int(len(qs))
        res[subset] = r
        pm = r["probe_mlp"]
        log(f"{subset} mlp probe: cls {pm['cls_mean_acc']:.3f} (bal {pm['cls_mean_balanced']:.3f}) who {pm['who']['acc']:.3f} corner {pm['cor']['acc']:.3f} "
            f"(bal {pm['cor']['balanced']:.3f}) dist MAE {pm['dist']['mae']:.3f} R2 {pm['dist']['r2']:.3f} x R2 {pm['x1']['r2']:.3f} hp MAE {pm['hp1']['mae']:.3f}", lf)
        log(f"{subset}: cls {r['probe']['cls_mean_acc']:.3f} (bal {r['probe']['cls_mean_balanced']:.3f}) who {r['probe']['who']['acc']:.3f} "
            f"corner {r['probe']['cor']['acc']:.3f} dist MAE {r['probe']['dist']['mae']:.3f} R2 {r['probe']['dist']['r2']:.3f} | "
            f"retr@10 joint {rr['top10']['joint_class']:.3f} (rand {rr['random_k10']['joint_class']:.3f}"
            f"{', oracle %.3f' % rr['state_oracle_top10']['joint_class'] if 'state_oracle_top10' in rr else ''}) "
            f"dist diff {rr['top10']['dist_abs_diff']:.3f}", lf)
    res["secs"] = round(time.time() - t0)
    save_json(res, OUT / f"results_exp_se_{name}.json")


def cmd_predict(a):
    """Per-tubelet state reading of val / test replays from the sup heads (for the search index and habits)."""
    import torch
    low_priority()
    meta = json.loads((SE / "state_meta.json").read_text())
    sl = split_lists()
    net, snet, ck = load_net(a.name)
    W = ck["cfg"]["W"]
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    od = SE / f"exp_se_{a.name}_pred"
    od.mkdir(parents=True, exist_ok=True)
    stems = sl["val"] + sl["test"]
    for i in range(0, len(stems), 40):
        D = Data(stems[i:i + 40], meta["wall"], norm=norm)
        allidx = np.arange(len(D.rep))
        Z = embed(net, D, allidx, W)
        with torch.no_grad():
            pc = torch.softmax(net.h_cls(Z).view(len(Z), 2, len(CLS)).float(), -1).cpu().numpy()
            reg = net.h_reg(Z).float().cpu().numpy()
            cor = torch.softmax(net.h_cor(Z).float(), -1).cpu().numpy()
        Zc = Z.half().cpu().numpy()
        for r, st in enumerate(D.stems):
            o, n = D.off[r], D.n[r]
            np.savez_compressed(od / f"{st}.npz", cls_p=pc[o:o + n].transpose(1, 0, 2).astype(np.float16), reg=reg[o:o + n].astype(np.float16),
                                cor_p=cor[o:o + n].astype(np.float16), emb=Zc[o:o + n])
        del D
        torch.cuda.empty_cache()
        log(f"predict {a.name}: {min(i + 40, len(stems))}/{len(stems)}")


def describe(d, t):
    """One line of the logged state at tubelet t (for the similar-moments demo)."""
    return (f"P1 {CLS[d['cls'][0][t]]:9s} P2 {CLS[d['cls'][1][t]]:9s} dist {d['dist'][t]:.2f} "
            f"x {d['x'][0][t]:+.2f}/{d['x'][1][t]:+.2f}{' P1-corner' if d['corner'][0][t] else ''}{' P2-corner' if d['corner'][1][t] else ''}")


def cmd_similar(a):
    """Moments like (stem, time) among the val / test replays' embeddings of `predict` (other replays only)."""
    od = SE / f"exp_se_{a.name}_pred"
    mm, ss = a.time.split(":") if ":" in a.time else ("0", a.time)
    t = int((float(mm) * 60 + float(ss)) * 60 // STEP)
    q = np.load(od / f"{a.stem}.npz")["emb"][t].astype(np.float32)
    best = []
    for f in sorted(od.glob("*.npz")):
        if f.stem == a.stem:
            continue
        E = np.load(f)["emb"].astype(np.float32)
        sim = E @ q
        for j in np.argsort(-sim)[:a.k]:
            best.append((float(sim[j]), f.stem, int(j)))
    best = sorted(best, reverse=True)
    out, seen = [], set()
    for sc, st, j in best:          # one moment per replay and second
        if (st, j // 8) in seen:
            continue
        seen.add((st, j // 8))
        out.append((sc, st, j))
        if len(out) >= a.k:
            break
    dq = load_state(a.stem)
    print(f"query {a.stem} {a.time}: {describe(dq, t)}")
    for sc, st, j in out:
        ts = j * STEP / 60
        print(f"  {sc:.3f} {st} {int(ts // 60)}:{ts % 60:05.2f}  {describe(load_state(st), j)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("labels")
    p.add_argument("--jobs", type=int, default=4)
    p = sp.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--emb", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--sup", type=float, default=0.0)
    p.add_argument("--snoise", type=float, default=0.05, help="Gaussian noise on the normalised state vector")
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("eval")
    p.add_argument("--name")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--raw_pca", type=int, default=256)
    p.add_argument("--probe_replays", type=int, default=200)
    p.add_argument("--probe_n", type=int, default=150000)
    p.add_argument("--queries", type=int, default=4000)
    p = sp.add_parser("predict")
    p.add_argument("--name", required=True)
    p = sp.add_parser("similar")
    p.add_argument("--name", required=True)
    p.add_argument("--stem", required=True)
    p.add_argument("--time", required=True, help="m:ss.s in the recording")
    p.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    {"similar": cmd_similar, "labels": cmd_labels, "train": cmd_train, "eval": cmd_eval, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
