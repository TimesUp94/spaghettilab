"""Full-match move transcript from the video (ROADMAP point 2): temporal action spotting of every move
start of both players over the whole-match embeddings of transcript_feats.py.

Input per side and per tubelet (8 video frames at 60 fps, 7.5 Hz), all from the video:
- frozen V-JEPA 2 ViT-L tokens of three views: [scene, this side's player square, the other's square]
  (3 x 1024, transcript_feats.py);
- with --grid (improvement round 1): the two player squares' 2 x 2 quadrant tokens as well
  (transcript_feats_grid.py: 4 x 192 PCA dims per square, mine / other), each square through one
  shared projection. The mean token alone keeps little of where in the square something moved,
  and whiffed / interrupted moves (no HUD trace) were the ones the first model missed;
- the CV pipeline's HUD readings at 30 Hz, the 4 rows of the tubelet, this side's block first
  (23 features x 4 + the read flags), standardised on the training replays;
- both players' positions from the player net (2 sampled frames x 2 players x x/y, their distance,
  the mirroring flags) and the side;
- both characters from video_players' vote over the replay (chars_logp argmax, as recognize.py).
The same network runs once per side (shared weights, inputs ordered 'mine' / 'other'), so the two
sides train one model on twice the data.

Model (why): a short temporal conv stem then a local-window transformer (ActionFormer-like: every
tubelet attends to +-WIN tubelets = +-4.3 s with a learned relative position bias per head), heads
per tubelet. Moves are short (median 31 frames = 4 tubelets) and depend on a few seconds of context
(the opponent's action, the hit / block that follows); local attention with relative positions is
length-agnostic, so training on 40 s windows and running a whole match in one pass give the same
computation, and it is cheaper and more parallel than video_openers' BiGRU over 1,000+ steps.

Heads per tubelet (the tubelet that contains the move's start frame is the positive one):
- category: background + normal / special / super / throw / other (class weights capped);
- offset: start frame inside the tubelet, 8-way;
- move: one vocabulary over every character (transcript_eval.move_vocab: < 5 training starts ->
  char:(rare)), logits restricted to the side's character (the vote), weights capped at 4 like
  video_moves;
- normal: button/stance for normals; outcome: counter / hit / block / whiff / interrupted / teched / none.
Decoding: tubelets with P(event) = 1 - P(background) >= thr give an event at 8 t + offset, with the
category / move / normal / outcome argmax; events of one side closer than `merge` frames keep the more
confident. thr and merge are chosen on the validation replays (transcript_splits.json), and the
epoch too; the test fold is scored once with transcript_eval (all / old / new). ensemble --percat
tunes one threshold per predicted category on validation (coordinate ascent from the global one).

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/video_transcript.py train --name base      # train + val tuning + test
    .venv/Scripts/python.exe tools/ggst_logger/model/video_transcript.py eval --name base       # re-score saved weights
    .venv/Scripts/python.exe tools/ggst_logger/model/video_transcript.py ensemble --name ens --members a b c   # average, tune, test
    .venv/Scripts/python.exe tools/ggst_logger/model/video_transcript.py predict --name base --stems a b --out t.parquet

-> <data>/ggst_dataset/model/video_transcript_exp_<name>.pt, results_exp_tr_<name>.json,
   <data>/ggst_dataset/transcript/exp_tr_<name>_{val,test}_preds.parquet, exp_tr_<name>.log
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, low_priority, save_json  # noqa: E402
import transcript_eval as te  # noqa: E402

TR = DATA / "transcript"
FEATS = TR / "feats"
GRIDF = TR / "feats_grid"   # transcript_feats_grid.py: 2x2 quadrant tokens of the player squares (PCA)
STEP = 8                      # video frames per tubelet
CATS = ["bg", "normal", "special", "super", "throw", "other"]
OUTCOMES = ["counter", "hit", "block", "whiff", "interrupted", "teched", "none"]
NORMALS = [f"{b}/{s}" for b in "PKSHD" for s in ("stand", "crouch", "air")]
IGNORE = -100
DEV = "cuda"
CHAIN = {"old": 0.887, "new": 0.855}
VAL_THRS = (0.3, 0.45, 0.6, 0.75)   # per-epoch validation: best of these thresholds (merge 6)


def log(msg, f=None):
    s = time.strftime("%H:%M:%S ") + msg
    print(s, flush=True)
    if f:
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")


# ---- data --------------------------------------------------------------------------------------

def meta():
    return json.loads((FEATS / "meta.json").read_text())


def load_replay(stem, chars, grid=False):
    """Per-replay arrays: views [N, 3, 1024] fp16 (with grid: flat [N, 3 * 1024 + 2 * 4 * PC], the quadrant
    tokens of p1 / p2 appended), aux per side [2, N, A] fp32 (raw), chars voted (2,)."""
    z = np.load(FEATS / f"{stem}.npz")
    n = len(z["tub_frame"])
    V = np.stack([z["scene"], z["p1"], z["p2"]], 1)
    if grid:
        g = np.load(GRIDF / f"{stem}.npz")
        V = np.concatenate([V.reshape(n, -1), g["p1g"].reshape(n, -1), g["p2g"].reshape(n, -1)], 1)
    hud = z["hud"][:4 * n].astype(np.float32)
    ok = z["hud_ok"][:4 * n].astype(np.float32)
    if len(hud) < 4 * n:   # pad (never seen, but safe)
        pad = 4 * n - len(hud)
        hud = np.concatenate([hud, np.repeat(hud[-1:], pad, 0)])
        ok = np.concatenate([ok, np.zeros(pad, np.float32)])
    hud = hud.reshape(n, 4, 23)
    ok = ok.reshape(n, 4)
    pos, flip = z["pos"][:2 * n].astype(np.float32), z["flip"][:2 * n].astype(np.float32)
    if len(pos) < 2 * n:   # an odd number of sampled frames: repeat the last
        pos = np.concatenate([pos, np.repeat(pos[-1:], 2 * n - len(pos), 0)])
        flip = np.concatenate([flip, np.repeat(flip[-1:], 2 * n - len(flip), 0)])
    pos = pos.reshape(n, 2, 2, 2)     # [n, frame, player, xy]
    flip = flip.reshape(n, 2, 2)      # [n, frame, player]
    vote = z["chars_logp"].argmax(1)
    aux = []
    for s in (0, 1):
        o = 1 - s
        h = np.concatenate([hud[:, :, 11 * s:11 * s + 11], hud[:, :, 11 * o:11 * o + 11], hud[:, :, 22:23]], 2).reshape(n, -1)
        p = np.stack([pos[:, :, s, 0] / 1920, pos[:, :, s, 1] / 1080, pos[:, :, o, 0] / 1920, pos[:, :, o, 1] / 1080,
                      (pos[:, :, o, 0] - pos[:, :, s, 0]) / 1920, flip[:, :, s], flip[:, :, o]], 2).reshape(n, -1)
        aux.append(np.concatenate([h, ok, p, np.full((n, 1), s, np.float32)], 1))
    return V, np.stack(aux), vote


def targets(events, n, voc_idx):
    """[2, n, 6] int: category (0 = bg), offset, move, normal, outcome, flag (1 = labelled)."""
    Y = np.full((2, n, 5), IGNORE, np.int64)
    Y[:, :, 0] = 0
    lost = 0
    for r in sorted(events, key=lambda r: r["start_frame"]):
        if not r["in_video"]:
            continue
        s = 0 if r["side"] == "P1" else 1
        t = r["start_frame"] // STEP
        if t < 0 or t >= n:
            continue
        if Y[s, t, 0] != 0:
            lost += 1
            continue
        mv = te.label(r, "move", voc_idx)
        Y[s, t] = [CATS.index(r["category"]), r["start_frame"] - STEP * t, voc_idx.get(mv, IGNORE),
                   NORMALS.index(r["normal"]) if r.get("normal") in NORMALS else IGNORE, OUTCOMES.index(te.outcome_label(r))]
    return Y, lost


class Store:
    """All replays of a split on the GPU, concatenated."""

    def __init__(self, stems, ev_by, voc_idx, chars, norm=None, grid=False):
        Vs, As, Ys, self.off, self.n, self.vote = [], [], [], [], [], []
        self.stems = list(stems)
        self.move_char = None
        o = 0
        self.lost = 0
        for st in self.stems:
            V, A, vote = load_replay(st, chars, grid)
            Y, lost = targets(ev_by.get(st, []), len(V), voc_idx)
            self.lost += lost
            Vs.append(V)
            As.append(A)
            Ys.append(Y)
            self.off.append(o)
            self.n.append(len(V))
            self.vote.append(vote)
            o += len(V)
        A = np.concatenate(As, 1)
        if norm is None:
            mu, sd = A.reshape(-1, A.shape[-1]).mean(0), A.reshape(-1, A.shape[-1]).std(0) + 1e-3
            norm = (mu, sd)
        self.norm = norm
        self.grid = grid
        self.V = torch.from_numpy(np.concatenate(Vs)).to(DEV)                                 # [T, 3, 1024] fp16 (grid: flat)
        self.A = torch.from_numpy(((A - norm[0]) / norm[1]).astype(np.float16)).to(DEV)        # [2, T, A]
        self.Y = torch.from_numpy(np.concatenate(Ys, 1)).to(DEV)                              # [2, T, 5]
        self.vote = np.stack(self.vote)                                                      # [R, 2]
        self.off = np.array(self.off)
        self.n = np.array(self.n)

    def side(self, r, s, a=0, b=None):
        """(views [L, 3, 1024] ordered scene / mine / other, aux [L, A], Y [L, 5]) of replay r, side s."""
        o = self.off[r]
        b = self.n[r] if b is None else b
        V = self.V[o + a:o + b]
        if self.grid:   # flat: scene | p1 | p2 | p1 grid | p2 grid -> scene | mine | other | mine grid | other grid
            D, Gd = 1024, (V.shape[1] - 3072) // 2
            parts = [V[:, :D], V[:, D:2 * D], V[:, 2 * D:3 * D], V[:, 3 * D:3 * D + Gd], V[:, 3 * D + Gd:]]
            if s == 1:
                parts = [parts[0], parts[2], parts[1], parts[4], parts[3]]
            return torch.cat(parts, 1), self.A[s, o + a:o + b], self.Y[s, o + a:o + b]
        V = V[:, [0, 1, 2]] if s == 0 else V[:, [0, 2, 1]]
        return V, self.A[s, o + a:o + b], self.Y[s, o + a:o + b]


# ---- model -------------------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, heads, win, drop):
        super().__init__()
        self.h, self.win = heads, win
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * win + 1))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def bias(self, L, pad):
        i = torch.arange(L, device=self.rel.device)
        d = i[None, :] - i[:, None]
        b = self.rel[:, (d.clamp(-self.win, self.win) + self.win)]                 # [h, L, L]
        b = b.masked_fill((d.abs() > self.win)[None], float("-inf"))
        b = b[None]                                                                # [1, h, L, L]
        if pad is not None:                                                        # pad [B, L] True = padding
            # padding is never a key, except for itself (a padded query keeps one finite key, no NaN)
            b = b.masked_fill(pad[:, None, None, :] & (d != 0)[None, None], float("-inf"))
        return b

    def forward(self, x, pad=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias(L, pad).to(q.dtype))
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))


class Net(nn.Module):
    def __init__(self, n_aux, n_move, n_char, move_char, d=384, layers=4, heads=6, win=32, drop=0.1, in_drop=0.2, grid=0):
        super().__init__()
        self.cfg = dict(n_aux=n_aux, n_move=n_move, n_char=n_char, d=d, layers=layers, heads=heads, win=win, drop=drop, in_drop=in_drop,
                        grid=grid)
        self.grid = grid          # 0, or the flat size of one player's quadrant tokens (4 * PC)
        self.vin = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(in_drop), nn.Linear(1024, d))     # per view, then mixed
        if grid:   # per player square: the 4 quadrants (LayerNorm per quadrant), shared by mine / other
            self.gin = nn.Sequential(nn.LayerNorm(grid // 4), nn.Dropout(in_drop), nn.Flatten(-2), nn.Linear(grid, d))
        self.vmix = nn.Linear((5 if grid else 3) * d, d)
        self.ain = nn.Sequential(nn.Linear(n_aux, d), nn.GELU(), nn.Linear(d, d))
        self.cme, self.cop = nn.Embedding(n_char, d), nn.Embedding(n_char, d)
        self.stem = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2, groups=d) for _ in range(2)])
        self.stem_pw = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.stem_n = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.blocks = nn.ModuleList([Block(d, heads, win, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.h_cat, self.h_off = nn.Linear(d, len(CATS)), nn.Linear(d, STEP)
        self.h_move, self.h_norm, self.h_out = nn.Linear(d, n_move), nn.Linear(d, len(NORMALS)), nn.Linear(d, len(OUTCOMES))
        self.register_buffer("move_char", torch.as_tensor(move_char))               # [n_move] char index

    def forward(self, V, A, cme, cop, pad=None):
        """V [B, L, 3, 1024], A [B, L, n_aux], cme / cop [B] char index; returns dict of logits [B, L, *]."""
        B, L = V.shape[:2]
        if self.grid:
            D, Gd = 1024, self.grid
            v = self.vin(V[..., :3 * D].float().view(B, L, 3, D)).reshape(B, L, -1)
            g = self.gin(V[..., 3 * D:].float().view(B, L, 2, 4, Gd // 4)).reshape(B, L, -1)
            v = torch.cat([v, g], -1)
        else:
            v = self.vin(V.float()).reshape(B, L, -1)
        x = self.vmix(v) + self.ain(A.float()) + (self.cme(cme) + self.cop(cop))[:, None]
        if pad is not None:
            x = x.masked_fill(pad[..., None], 0.0)
        for c, pw, n in zip(self.stem, self.stem_pw, self.stem_n):
            x = x + pw(F.gelu(c(n(x).transpose(1, 2)).transpose(1, 2)))
            if pad is not None:
                x = x.masked_fill(pad[..., None], 0.0)
        for b in self.blocks:
            x = b(x, pad)
        x = self.norm(x)
        mv = self.h_move(x).float()
        mv = mv.masked_fill((self.move_char[None, :] != cme[:, None])[:, None, :], -1e4)
        return dict(cat=self.h_cat(x).float(), off=self.h_off(x).float(), move=mv,
                    normal=self.h_norm(x).float(), outcome=self.h_out(x).float())


# ---- training ----------------------------------------------------------------------------------

def class_weights(y, n, cap=4.0):
    """sqrt inverse frequency, scaled so the average training sample weighs 1, capped."""
    c = np.bincount(y[y >= 0], minlength=n).astype(float)
    w = np.where(c > 0, (c.sum() / np.maximum(c, 1)) ** 0.5, 0)
    w = w / ((c * w).sum() / c.sum())
    return torch.tensor(np.minimum(w, cap), dtype=torch.float32, device=DEV)


def batch(store, idx, L, rng, train=True):
    """Windows of L tubelets (random start; zero-padded when the replay is shorter), one per (replay, side)."""
    Vb, Ab, Yb, pad, cme, cop = [], [], [], [], [], []
    mc = store.move_char
    for r, s in idx:
        n = store.n[r]
        a = int(rng.integers(0, max(1, n - L + 1))) if train else 0
        b = min(n, a + L)
        V, A, Y = store.side(r, s, a, b)
        m = b - a
        if m < L:
            V = F.pad(V, (0, 0) * (V.dim() - 1) + (0, L - m))
            A = F.pad(A, (0, 0, 0, L - m))
            Y = F.pad(Y, (0, 0, 0, L - m), value=IGNORE)
        if mc is not None:
            mv = Y[:, 2]
            bad = (mv >= 0) & (mc[mv.clamp(min=0)] != int(store.vote[r, s]))
            Y = Y.clone()
            Y[bad, 2] = IGNORE
        Vb.append(V)
        Ab.append(A)
        Yb.append(Y)
        p = torch.zeros(L, dtype=torch.bool, device=DEV)
        p[m:] = True
        pad.append(p)
        cme.append(store.vote[r, s])
        cop.append(store.vote[r, 1 - s])
    t = lambda x: torch.as_tensor(np.array(x), device=DEV)
    return torch.stack(Vb), torch.stack(Ab), torch.stack(Yb), torch.stack(pad), t(cme), t(cop)


def loss_fn(out, Y, W, bg_w):
    cat = Y[..., 0]
    l_cat = F.cross_entropy(out["cat"].flatten(0, 1), cat.flatten(), weight=W["cat"], ignore_index=IGNORE, label_smoothing=0.02)
    pos = (cat > 0)
    tot = l_cat
    for h, k, w in (("off", 1, None), ("move", 2, W["move"]), ("normal", 3, W["normal"]), ("outcome", 4, W["outcome"])):
        y = torch.where(pos, Y[..., k], torch.full_like(Y[..., k], IGNORE))
        if (y >= 0).any():
            tot = tot + F.cross_entropy(out[h][pos], y[pos], weight=w, ignore_index=IGNORE,
                                        label_smoothing=0.0 if h in ("move", "off") else 0.05) * (0.5 if h in ("normal",) else 1.0)
    return tot


@torch.no_grad()
def infer(net, store, r):
    """Per side: probabilities [N, *] for one whole replay (one pass; local attention). net may be a
    list (an ensemble: probabilities averaged)."""
    nets = net if isinstance(net, (list, tuple)) else [net]
    res = []
    for s in (0, 1):
        V, A, _ = store.side(r, s)
        cme = torch.as_tensor([store.vote[r, s]], device=DEV)
        cop = torch.as_tensor([store.vote[r, 1 - s]], device=DEV)
        acc = None
        for m in nets:
            m.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                out = m(V[None], A[None], cme, cop)
            pr = {k: F.softmax(v[0].float(), -1) for k, v in out.items()}
            acc = pr if acc is None else {k: acc[k] + pr[k] for k in acc}
        res.append({k: (v / len(nets)).cpu().numpy() for k, v in acc.items()})
    return res


def candidates(net, store, voc, chars, min_p=0.05):
    """Every tubelet with P(event) >= min_p of every replay / side, as prediction dicts."""
    out = []
    for r, st in enumerate(store.stems):
        for s, P in enumerate(infer(net, store, r)):
            pe = 1 - P["cat"][:, 0]
            for t in np.nonzero(pe >= min_p)[0]:
                cat = CATS[1 + int(P["cat"][t, 1:].argmax())]
                mv = voc[int(P["move"][t].argmax())]
                out.append(dict(replay=st, side=f"P{s + 1}", start_frame=int(STEP * t + P["off"][t].argmax()), score=float(pe[t]),
                                category=cat, char_move=mv, char=chars[store.vote[r, s]],
                                normal=NORMALS[int(P["normal"][t].argmax())] if cat == "normal" else None,
                                outcome=OUTCOMES[int(P["outcome"][t].argmax())],
                                p_cat=float(P["cat"][t, 1:].max() / max(pe[t], 1e-6)), p_move=float(P["move"][t].max()),
                                p_out=float(P["outcome"][t].max())))
    return out


def candidates_by_segment(net, store, voc, chars, min_p, segments):
    """candidates() with each stretch of the one replay in store under its own characters. segments: [(first 60 fps frame, end frame or None,
    (P1 name, P2 name))]; a name the net doesn't know keeps the replay's vote."""
    base = store.vote.copy()
    out = []
    for a, b, names in segments:
        store.vote = base.copy()
        for s in (0, 1):
            if names[s] in chars:
                store.vote[0, s] = chars.index(names[s])
        out += [c for c in candidates(net, store, voc, chars, min_p) if c["start_frame"] >= a and (b is None or c["start_frame"] < b)]
    store.vote = base
    return out


def thr_min(thr):
    return min(thr.values()) if isinstance(thr, dict) else thr


def decode(cands, thr, merge):
    """Threshold (one value, or a dict per predicted category), then per side keep the most
    confident of events closer than `merge` frames."""
    by = collections.defaultdict(list)
    per = isinstance(thr, dict)
    for c in cands:
        if c["score"] >= (thr[c["category"]] if per else thr):
            by[(c["replay"], c["side"])].append(c)
    out = []
    for lst in by.values():
        lst.sort(key=lambda c: -c["score"])
        kept = []
        for c in lst:
            if merge and any(abs(c["start_frame"] - k["start_frame"]) < merge for k in kept):
                continue
            kept.append(c)
        out += kept
    return out


def quick_f1(truth, preds, vocab, k=4, levels=("category", "move")):
    return {lv: te.score_level(truth, preds, lv, k, vocab)["micro"]["f1"] for lv in levels}


def tune(cands, truth, vocab, logf):
    best = None
    grid = []
    for thr in (0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        for merge in (4, 6, 8):
            f = quick_f1(truth, decode(cands, thr, merge), vocab)
            sc = (f["category"] + f["move"]) / 2
            grid.append(dict(thr=thr, merge=merge, **f, mean=sc))
            if best is None or sc > best[0]:
                best = (sc, thr, merge)
    log(f"tune: thr {best[1]} merge {best[2]} -> {best[0]:.4f}", logf)
    return best[1], best[2], grid


def tune_percat(cands, truth, vocab, thr, merge, logf, rounds=2):
    """Coordinate ascent of one threshold per predicted category, from the global one (val only)."""
    th = {c: thr for c in CATS[1:]}
    f = quick_f1(truth, decode(cands, th, merge), vocab)
    best = (f["category"] + f["move"]) / 2
    for _ in range(rounds):
        for c in CATS[1:]:
            for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
                tt = dict(th, **{c: t})
                f = quick_f1(truth, decode(cands, tt, merge), vocab)
                sc = (f["category"] + f["move"]) / 2
                if sc > best + 1e-5:
                    best, th = sc, tt
    log(f"tune per category: {th} -> {best:.4f}", logf)
    return th, best


def summarise(ev):
    """The headline numbers of transcript_eval.evaluate."""
    out = {}
    for k, d in ev.items():
        o = {}
        for lv in te.LEVELS:
            v = d[lv]
            o[lv] = dict(micro={x: round(y, 4) for x, y in v["micro"].items()},
                         macro={x: (round(y, 4) if isinstance(y, float) else y) for x, y in v["macro"].items()},
                         n_true=v["n_true"], n_pred=v["n_pred"])
        o["outcome_given_category"] = d["outcome_given_category"]
        o["openers_recovered"] = d["openers_recovered"]
        out[k] = o
    return out


def full_eval(preds, cands, truth, vocab, subsets):
    res = {}
    for name, reps in subsets.items():
        reps = set(reps)
        t = [r for r in truth if r["replay"] in reps]
        p = [r for r in preds if r["replay"] in reps]
        pa = [r for r in cands if r["replay"] in reps]
        ev = te.evaluate(t, p, vocab)
        d = summarise(ev)
        # mAP from every candidate (no threshold), per level at each k
        d["mAP_all_candidates"] = {f"k{k}": {lv: te.score_level(t, decode(pa, 0.0, 6), lv, k, vocab)["macro"]["mAP"] for lv in te.LEVELS}
                                   for k in te.TOLS}
        d["per_side_micro_f1_k4"] = {side: {lv: round(te.score_level([x for x in t if x["side"] == side], [x for x in p if x["side"] == side],
                                                                   lv, 4, vocab)["micro"]["f1"], 4) for lv in te.LEVELS} for side in ("P1", "P2")}
        cat = te.score_level(t, p, "category", 4, vocab)["per_class"]
        d["per_category_k4"] = {c: {x: (round(v[x], 4) if isinstance(v[x], float) else v[x]) for x in ("n_true", "n_pred", "precision", "recall", "f1", "ap")}
                                for c, v in cat.items()}
        res[name] = d
    return res


def setup():
    rows = te.load_events()
    sp = json.loads((TR / "transcript_splits.json").read_text())
    train_rows = [r for r in rows if sp.get(r["replay"]) == "train"]
    vocab = te.move_vocab(train_rows)
    voc_idx = {m: i for i, m in enumerate(vocab)}
    chars = meta()["chars"]
    ev_by = collections.defaultdict(list)
    for r in rows:
        ev_by[r["replay"]].append(r)
    return rows, sp, vocab, voc_idx, chars, ev_by


def cmd_train(a):
    low_priority()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    logf = TR / f"exp_tr_{a.name}.log"
    rows, sp, vocab, voc_idx, chars, ev_by = setup()
    stems = {s: sorted(k for k, v in sp.items() if v == s) for s in ("train", "val", "test")}
    if a.frac < 1:
        stems["train"] = stems["train"][:max(1, int(len(stems["train"]) * a.frac))]
    if a.grid_subset:   # pilot: only the training replays that already have grid tokens (same set with and without --grid)
        stems["train"] = [x for x in stems["train"] if (GRIDF / f"{x}.npz").exists()]
    t0 = time.time()
    tr = Store(stems["train"], ev_by, voc_idx, chars, grid=a.grid)
    va = Store(stems["val"], ev_by, voc_idx, chars, tr.norm, grid=a.grid)
    log(f"loaded train {len(tr.stems)} ({int(tr.n.sum())} tubelets, {tr.lost} events lost to shared tubelets) "
        f"val {len(va.stems)} in {time.time() - t0:.0f}s; vocab {len(vocab)}", logf)
    move_char = np.array([chars.index(m.split(":", 1)[0]) for m in vocab])
    tr.move_char = torch.as_tensor(move_char, device=DEV)
    net = Net(tr.A.shape[-1], len(vocab), len(chars), move_char, d=a.d, layers=a.layers, heads=a.heads, win=a.win,
              drop=a.drop, in_drop=a.in_drop, grid=(tr.V.shape[1] - 3072) // 2 if a.grid else 0).to(DEV)
    Yc = tr.Y.cpu().numpy().reshape(-1, 5)
    cw = class_weights(Yc[Yc[:, 0] > 0, 0] - 1, len(CATS) - 1)     # event classes among themselves (mean 1)
    W = dict(cat=torch.cat([torch.ones(1, device=DEV), cw]), move=class_weights(Yc[Yc[:, 0] > 0, 2], len(vocab)),
             normal=class_weights(Yc[Yc[:, 0] > 0, 3], len(NORMALS)), outcome=class_weights(Yc[Yc[:, 0] > 0, 4], len(OUTCOMES)))
    W["cat"][0] = a.bg_w   # background: plain weight; the threshold is tuned on val
    log(f"cat weights {W['cat'].cpu().numpy().round(2).tolist()} outcome {W['outcome'].cpu().numpy().round(2).tolist()}", logf)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
    pairs = [(r, s) for r in range(len(tr.stems)) for s in (0, 1)]
    per_ep = max(1, int(len(pairs) * a.reps) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * per_ep, pct_start=0.1)
    val_truth = [r for r in rows if sp.get(r["replay"]) == "val"]
    best = (-1, None, -1)
    hist = []
    for ep in range(a.epochs):
        net.train()
        t1, tot = time.time(), 0.0
        for it in range(per_ep):
            idx = [pairs[i] for i in rng.integers(0, len(pairs), a.bs)]
            V, A, Y, pad, cme, cop = batch(tr, idx, a.L, rng)
            if a.view_drop > 0:   # drop whole view tokens (scene / mine / other [/ mine grid / other grid]) at random
                if a.grid:
                    Gd = (V.shape[-1] - 3072) // 2
                    keep = (torch.rand(V.shape[:2] + (5,), device=DEV) >= a.view_drop).to(V.dtype)
                    V = V * torch.repeat_interleave(keep, torch.tensor([1024, 1024, 1024, Gd, Gd], device=DEV), dim=-1)
                else:
                    V = V * (torch.rand(V.shape[:3], device=DEV) >= a.view_drop)[..., None].to(V.dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(V, A, cme, cop, pad)
            loss = loss_fn(out, Y, W, a.bg_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if ep + 1 < a.epochs * 0.5:   # the first half: no validation (the best epoch has always been late)
            log(f"ep {ep + 1}/{a.epochs} loss {tot / per_ep:.4f} ({time.time() - t1:.0f}s)", logf)
            continue
        cands = candidates(net, va, vocab, chars, min(VAL_THRS))
        sc, f, vt_ = max((((g["category"] + g["move"]) / 2, g, t) for t in VAL_THRS
                          for g in [quick_f1(val_truth, decode(cands, t, 6), vocab)]), key=lambda x: x[0])
        hist.append(dict(epoch=ep + 1, loss=tot / per_ep, thr=vt_, **f))
        log(f"ep {ep + 1}/{a.epochs} loss {tot / per_ep:.4f} val@{vt_} cat {f['category']:.4f} move {f['move']:.4f} ({time.time() - t1:.0f}s)", logf)
        if sc > best[0]:
            best = (sc, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep + 1)
    net.load_state_dict(best[1])
    log(f"best epoch {best[2]} ({best[0]:.4f})", logf)
    ck = dict(state=net.state_dict(), cfg=net.cfg, vocab=vocab, chars=chars, norm=[x.tolist() for x in tr.norm],
              move_char=move_char.tolist(), epoch=best[2], args=vars(a), hist=hist)
    del tr
    torch.cuda.empty_cache()
    thr, merge, grid = tune(candidates(net, va, vocab, chars, 0.05), val_truth, vocab, logf)
    ck.update(thr=thr, merge=merge, tune=grid)
    path = OUT / f"video_transcript_exp_{a.name}.pt"
    torch.save(ck, path)
    log(f"saved {path}", logf)
    del va
    torch.cuda.empty_cache()
    if not a.no_test:
        evaluate_test(net, ck, a.name, rows, sp, ev_by, voc_idx, logf, train_s=time.time() - t0)


def evaluate_test(net, ck, name, rows, sp, ev_by, voc_idx, logf, train_s=None):
    vocab, chars = ck["vocab"], ck["chars"]
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    stems = sorted(k for k, v in sp.items() if v == "test")
    te_store = Store(stems, ev_by, voc_idx, chars, norm, grid=bool(ck_grid(ck)))
    cands = candidates(net, te_store, vocab, chars, 0.05)
    preds = decode(cands, ck["thr"], ck["merge"])
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["replay", "side", "start_frame", "score", "category", "char_move", "char", "normal", "outcome", "p_cat", "p_move"]
    pq.write_table(pa.Table.from_pylist([{c: p[c] for c in cols} for p in preds]), TR / f"exp_tr_{name}_test_preds.parquet")
    truth = [r for r in rows if sp.get(r["replay"]) == "test"]
    from common import replay_fold_map, test_subsets
    sub = test_subsets(replay_fold_map())
    res = full_eval(preds, cands, truth, vocab, {k: sub[k] for k in ("all", "old", "new")})
    vote_ok = {}
    for r, st in enumerate(te_store.stems):
        for s in (0, 1):
            tc = next((x["char"] for x in ev_by[st] if x["side"] == f"P{s + 1}"), None)
            vote_ok[(st, s)] = tc == chars[te_store.vote[r, s]]
    out = dict(name=name, weights=f"video_transcript_exp_{name}.pt", epoch=ck["epoch"], thr=ck["thr"], merge=ck["merge"],
               args=ck["args"], hist=ck["hist"], tune=ck.get("tune"), train_seconds=train_s,
               char_vote_sides_right=f"{sum(vote_ok.values())}/{len(vote_ok)}",
               chain_reference=CHAIN, test=res)
    save_json(out, OUT / f"results_exp_tr_{name}.json")
    for s in ("old", "new"):
        d = res[s]
        log(f"[{s}] " + " | ".join(f"k{k}: " + " ".join(f"{lv[:3]} {d[f'k{k}'][lv]['micro']['f1']:.3f}" for lv in te.LEVELS)
                                 + f" op {d[f'k{k}']['openers_recovered']['frac']:.3f}" for k in te.TOLS) + f" (chain {CHAIN[s]})", logf)


def ck_grid(ck):
    return (ck["members"][0]["cfg"] if "members" in ck else ck["cfg"]).get("grid", 0)


def load_net(name):
    """(net, checkpoint); an ensemble file gives a list of nets."""
    ck = torch.load(OUT / f"video_transcript_exp_{name}.pt", map_location=DEV, weights_only=False)

    def build(state, c):
        net = Net(c["n_aux"], c["n_move"], c["n_char"], np.array(ck["move_char"]), d=c["d"], layers=c["layers"],
                  heads=c["heads"], win=c["win"], drop=c["drop"], in_drop=c["in_drop"], grid=c.get("grid", 0)).to(DEV)
        net.load_state_dict(state)
        return net.eval()
    if "members" in ck:
        return [build(m["state"], m["cfg"]) for m in ck["members"]], ck
    return build(ck["state"], ck["cfg"]), ck


def cmd_eval(a):
    rows, sp, vocab, voc_idx, chars, ev_by = setup()
    net, ck = load_net(a.name)
    evaluate_test(net, ck, a.name, rows, sp, ev_by, {m: i for i, m in enumerate(ck["vocab"])}, TR / f"exp_tr_{a.name}.log")


def cmd_ensemble(a):
    """Average the probabilities of trained members (same split and vocabulary), tune thr / merge on
    val, score the test fold; the weights file holds every member."""
    logf = TR / f"exp_tr_{a.name}.log"
    rows, sp, vocab, voc_idx, chars, ev_by = setup()
    loaded = [load_net(m) for m in a.members]
    nets = [n for n, _ in loaded]
    ck0 = loaded[0][1]
    norm = tuple(np.array(x, np.float32) for x in ck0["norm"])
    va = Store(sorted(k for k, v in sp.items() if v == "val"), ev_by, voc_idx, chars, norm, grid=bool(ck_grid(ck0)))
    val_truth = [r for r in rows if sp.get(r["replay"]) == "val"]
    vc = candidates(nets, va, vocab, chars, 0.05)
    thr, merge, grid = tune(vc, val_truth, vocab, logf)
    if a.percat:
        thr, _ = tune_percat(vc, val_truth, vocab, thr, merge, logf)
    del va
    torch.cuda.empty_cache()
    ck = dict(members=[dict(state=c["state"], cfg=c["cfg"]) for _, c in loaded], vocab=vocab, chars=chars, norm=ck0["norm"],
              move_char=ck0["move_char"], epoch=[c["epoch"] for _, c in loaded], args=dict(members=a.members),
              hist=None, thr=thr, merge=merge, tune=grid)
    torch.save(ck, OUT / f"video_transcript_exp_{a.name}.pt")
    evaluate_test(nets, ck, a.name, rows, sp, ev_by, voc_idx, logf)


def cmd_predict(a):
    """Transcript of replays that have transcript_feats.py files (no labels needed)."""
    net, ck = load_net(a.name)
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    st = Store(a.stems, {}, {m: i for i, m in enumerate(ck["vocab"])}, ck["chars"], norm, grid=bool(ck_grid(ck)))
    preds = decode(candidates(net, st, ck["vocab"], ck["chars"], thr_min(ck["thr"])), ck["thr"], ck["merge"])
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(sorted(preds, key=lambda p: (p["replay"], p["start_frame"]))), a.out)
    print(f"{len(preds)} events -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("train", "eval", "predict", "ensemble"))
    ap.add_argument("--members", nargs="*", help="ensemble: trained names to average")
    ap.add_argument("--no-test", action="store_true", help="train and tune only (no test-fold scoring)")
    ap.add_argument("--view-drop", type=float, default=0.0)
    ap.add_argument("--grid-subset", action="store_true", help="train only on replays with grid tokens (pilot)")
    ap.add_argument("--percat", action="store_true", help="ensemble: one threshold per predicted category (tuned on val)")
    ap.add_argument("--grid", action="store_true", help="add the player squares' 2x2 quadrant tokens (transcript_feats_grid.py)")
    ap.add_argument("--name", default="base")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--L", type=int, default=320)
    ap.add_argument("--reps", type=float, default=4.0, help="windows per (replay, side) per epoch")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--in-drop", type=float, default=0.2)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--bg-w", type=float, default=1.0)
    ap.add_argument("--val-thr", type=float, default=0.3, help="threshold for the per-epoch validation F1")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stems", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "predict": cmd_predict, "ensemble": cmd_ensemble}[a.cmd](a)


if __name__ == "__main__":
    main()
