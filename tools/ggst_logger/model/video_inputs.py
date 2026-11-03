"""Inputs from video (ROADMAP point 3, inverse dynamics): per video frame (60 fps) and side, the held
direction (numpad 1-9 relative to the facing) and the button presses (P/K/S/H/D), from the video only.

Labels: dataset/inputs_labels.py (from the logs: p*_rec + p*_face), metric: model/inputs_eval.py.

Inputs per side and per tubelet (8 video frames, 7.5 Hz), all from the video:
- the transcript's inputs (video_transcript.py --grid): V-JEPA 2 tokens of scene / mine / other square
  + the squares' 2x2 quadrant tokens, the HUD at 30 Hz, the player net's positions, the voted characters;
- the transcript's move posteriors per tubelet (P(category) incl. background, start offset, normal,
  outcome, and the 8 most likely moves with their probabilities). The move fixes most of a special's
  motion (236K), so they are the strongest single input. Which transcript gives them:
  * validation and test replays: the final transcript `ens5grid` (5 members, trained on the train
    split only, so these replays are held out for it);
  * training replays: out-of-fold. The train split is cut into OOF_K folds by a hash of the name; a
    single transcript net with ens5grid's member setup (grid, 40 epochs, last epoch) is trained on the
    other folds and predicts the held-out one (`oof`). The training-time posteriors of ens5grid on its
    own training replays would be over-confident (it saw their labels), so they are not used. The
    OOF nets are a bit weaker than ens5grid (3/4 of the data, one member), so the inputs head sees
    slightly noisier posteriors in training than at test time (conservative).

Model: the transcript's trunk (per-view projections, conv stem, local-window transformer, d 384, +-32
tubelets) on [video tokens + HUD + posteriors], then each tubelet is expanded to its 8 frames (a linear
layer to 8 x 64 channels, i.e. a learned position inside the tubelet) and two temporal convolutions at
60 fps (kernel 9) give per frame: direction (9-way softmax) and the five buttons' press (sigmoid; the
target is the press frame, spread +-1 frame at 0.5 weight). Loss on live frames only (inputs_labels).
Decoding presses: per button, local maxima over +-2 frames with p >= thr (thr chosen on validation).

--cmd-head: an event head for "the input given the move": for every logged move start with a command
(inputs_eval.event_cmds), the trunk token of its tubelet, the mean frame features of the 8 frames up to
the start and of the 4 after, and an embedding of the move give the command (motion + button, 236K,
[4]6S, 2H ...) and the plain direction + button at the press (2K, 6P), each restricted to the classes
seen for that move in training (classes with < CMD_MIN examples -> "other"). Trained at the true start
+-4 frames (the transcript's start error); scored at the start of the transcript's matched prediction
with the transcript's move (command_head_eval), against the move prior (inputs_eval.move_prior).

--frame-feats (stage 2): per-frame DINOv2-small CLS tokens of both player squares at 60 fps
(inputs_frame_feats.py; val, test and 250 training replays) added to the 60 fps stream before the
temporal convolutions; trains on those 250 training replays only. --ff-subset trains the stage-1 model
on the same 250 replays (the control); --init warm-starts from a stage-1 model.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/video_inputs.py oof            # OOF transcript nets (train split)
    .venv/Scripts/python.exe tools/ggst_logger/model/video_inputs.py post           # posteriors for every replay
    .venv/Scripts/python.exe tools/ggst_logger/model/video_inputs.py train --name base [--no-post]
    .venv/Scripts/python.exe tools/ggst_logger/model/video_inputs.py eval --name base

-> <data>/ggst_dataset/inputs/post/<stem>.npz, inputs/exp_in_<name>_{val,test}/<stem>.npz (probabilities),
   model/video_inputs_exp_in_<name>.pt, model/results_exp_in_<name>.json, inputs/exp_in_<name>.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, low_priority, save_json  # noqa: E402
import video_transcript as vt  # noqa: E402

INP = DATA / "inputs"
POST = INP / "post"
LAB = INP / "labels"
FFD = INP / "frame_feats"
OOF_K = 4
TOPM = 8
STEP = vt.STEP
DEV = "cuda"
OOF_ARGS = dict(epochs=40, bs=32, L=320, reps=4.0, lr=5e-4, d=384, layers=4, heads=6, drop=0.25, in_drop=0.4, win=32,
                bg_w=0.15, view_drop=0.15)


def log(msg, f=None):
    vt.log(msg, f)


def oof_fold(stem: str) -> int:
    return int(hashlib.md5(("oof:" + stem).encode()).hexdigest(), 16) % OOF_K


def splits():
    sp = json.loads((vt.TR / "transcript_splits.json").read_text())
    return {s: sorted(k for k, v in sp.items() if v == s) for s in ("train", "val", "test")}


# ---- OOF transcript nets ----------------------------------------------------------------------

def train_transcript(stems, name, seed, logf):
    """One transcript net (ens5grid member setup) on `stems`, fixed epochs, last epoch kept."""
    a = argparse.Namespace(**OOF_ARGS)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows, sp, vocab, voc_idx, chars, ev_by = vt.setup()
    tr = vt.Store(stems, ev_by, voc_idx, chars, grid=True)
    move_char = np.array([chars.index(m.split(":", 1)[0]) for m in vocab])
    tr.move_char = torch.as_tensor(move_char, device=DEV)
    net = vt.Net(tr.A.shape[-1], len(vocab), len(chars), move_char, d=a.d, layers=a.layers, heads=a.heads, win=a.win,
                 drop=a.drop, in_drop=a.in_drop, grid=(tr.V.shape[1] - 3072) // 2).to(DEV)
    Yc = tr.Y.cpu().numpy().reshape(-1, 5)
    cw = vt.class_weights(Yc[Yc[:, 0] > 0, 0] - 1, len(vt.CATS) - 1)
    W = dict(cat=torch.cat([torch.ones(1, device=DEV), cw]), move=vt.class_weights(Yc[Yc[:, 0] > 0, 2], len(vocab)),
             normal=vt.class_weights(Yc[Yc[:, 0] > 0, 3], len(vt.NORMALS)), outcome=vt.class_weights(Yc[Yc[:, 0] > 0, 4], len(vt.OUTCOMES)))
    W["cat"][0] = a.bg_w
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
    pairs = [(r, s) for r in range(len(tr.stems)) for s in (0, 1)]
    per_ep = max(1, int(len(pairs) * a.reps) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * per_ep, pct_start=0.1)
    for ep in range(a.epochs):
        net.train()
        t1, tot = time.time(), 0.0
        for _ in range(per_ep):
            idx = [pairs[i] for i in rng.integers(0, len(pairs), a.bs)]
            V, A, Y, pad, cme, cop = vt.batch(tr, idx, a.L, rng)
            Gd = (V.shape[-1] - 3072) // 2
            keep = (torch.rand(V.shape[:2] + (5,), device=DEV) >= a.view_drop).to(V.dtype)
            V = V * torch.repeat_interleave(keep, torch.tensor([1024, 1024, 1024, Gd, Gd], device=DEV), dim=-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(V, A, cme, cop, pad)
            loss = vt.loss_fn(out, Y, W, a.bg_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        log(f"{name} ep {ep + 1}/{a.epochs} loss {tot / per_ep:.4f} ({time.time() - t1:.0f}s)", logf)
    ck = dict(state=net.state_dict(), cfg=net.cfg, vocab=vocab, chars=chars, norm=[x.tolist() for x in tr.norm],
              move_char=move_char.tolist(), epoch=a.epochs, args=dict(OOF_ARGS, stems=len(stems), seed=seed))
    del tr
    torch.cuda.empty_cache()
    return net, ck


def cmd_oof(a):
    low_priority()
    logf = INP / "exp_in_oof.log"
    INP.mkdir(parents=True, exist_ok=True)
    tr = splits()["train"]
    for k in a.folds or range(OOF_K):
        path = OUT / f"video_transcript_exp_in_oof{k}.pt"
        if path.exists():
            log(f"fold {k}: exists", logf)
            continue
        stems = [s for s in tr if oof_fold(s) != k]
        t0 = time.time()
        log(f"fold {k}: training on {len(stems)} replays, held out {sum(oof_fold(s) == k for s in tr)}", logf)
        net, ck = train_transcript(stems, f"oof{k}", seed=100 + k, logf=logf)
        torch.save(ck, path)
        log(f"fold {k}: saved {path} in {time.time() - t0:.0f}s", logf)
        del net
        torch.cuda.empty_cache()


# ---- posteriors --------------------------------------------------------------------------------

def load_tnet(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)

    def build(state, c):
        net = vt.Net(c["n_aux"], c["n_move"], c["n_char"], np.array(ck["move_char"]), d=c["d"], layers=c["layers"],
                     heads=c["heads"], win=c["win"], drop=c["drop"], in_drop=c["in_drop"], grid=c.get("grid", 0)).to(DEV)
        net.load_state_dict(state)
        return net.eval()
    nets = [build(m["state"], m["cfg"]) for m in ck["members"]] if "members" in ck else [build(ck["state"], ck["cfg"])]
    return nets, ck


def write_post(nets, ck, stems, src, logf):
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    voc_idx = {m: i for i, m in enumerate(ck["vocab"])}
    for i in range(0, len(stems), 64):           # in chunks (GPU memory)
        chunk = stems[i:i + 64]
        st = vt.Store(chunk, {}, voc_idx, ck["chars"], norm, grid=True)
        for r, stem in enumerate(st.stems):
            P = vt.infer(nets, st, r)
            out = {}
            for k in ("cat", "off", "normal", "outcome"):
                out[k] = np.stack([P[s][k] for s in (0, 1)]).astype(np.float16)
            mv = np.stack([P[s]["move"] for s in (0, 1)])                 # [2, N, n_move]
            idx = np.argsort(-mv, -1)[..., :TOPM]
            out["mv_idx"] = idx.astype(np.int16)
            out["mv_p"] = np.take_along_axis(mv, idx, -1).astype(np.float16)
            np.savez_compressed(POST / f"{stem}.npz", src=np.array(src), **out)
        del st
        torch.cuda.empty_cache()
        log(f"post {src}: {min(i + 64, len(stems))}/{len(stems)}", logf)


def cmd_post(a):
    low_priority()
    logf = INP / "exp_in_post.log"
    POST.mkdir(parents=True, exist_ok=True)
    sp = splits()
    nets, ck = load_tnet(OUT / "video_transcript_exp_ens5grid.pt")
    assert ck["vocab"] == vt.setup()[2], "ens5grid vocabulary differs from the train split's"
    write_post(nets, ck, sp["val"] + sp["test"], "ens5grid", logf)
    # the transcript's own validation events (inputs_eval's transcript-timed command reading on val)
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    va = vt.Store(sp["val"], {}, {m: i for i, m in enumerate(ck["vocab"])}, ck["chars"], norm, grid=True)
    preds = vt.decode(vt.candidates(nets, va, ck["vocab"], ck["chars"], vt.thr_min(ck["thr"])), ck["thr"], ck["merge"])
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(preds), INP / "exp_in_ens5grid_val_preds.parquet")
    del va, nets
    torch.cuda.empty_cache()
    for k in range(OOF_K):
        nets, ck = load_tnet(OUT / f"video_transcript_exp_in_oof{k}.pt")
        write_post(nets, ck, [s for s in sp["train"] if oof_fold(s) == k], f"oof{k}", logf)
        del nets
        torch.cuda.empty_cache()


# ---- inputs model ------------------------------------------------------------------------------

def post_feats(stem, n):
    """[2, n, 36] dense posteriors (cat, offset, normal, outcome) and the move top-M (idx [2, n, M], p)."""
    z = np.load(POST / f"{stem}.npz")
    d = np.concatenate([z["cat"], z["off"], z["normal"], z["outcome"]], -1).astype(np.float32)[:, :n]
    return d, z["mv_idx"].astype(np.int64)[:, :n], z["mv_p"].astype(np.float32)[:, :n]


CMD_MIN = 20      # commands / direction+button seen fewer times in training -> "other"


def cmd_vocab(train_stems, move_vocab):
    """Event head vocabularies from the training events (inputs_eval.event_cmds): commands and plain
    direction + button inputs with >= CMD_MIN examples (+ "other"), and per transcript move (char:move
    with the (rare) folding) the masks of the classes seen for it in training."""
    import collections
    import inputs_eval as ie
    import transcript_eval as te
    voc = set(move_vocab)
    mv_idx = {m: i for i, m in enumerate(move_vocab)}
    tr = set(train_stems)
    rows = [r for r in ie.event_cmds() if r["replay"] in tr]
    out = {}
    for key in ("cmd", "db"):
        c = collections.Counter(r[key] for r in rows)
        names = ["other"] + sorted(k for k, v in c.items() if v >= CMD_MIN)
        idx = {k: i for i, k in enumerate(names)}
        mask = np.zeros((len(move_vocab), len(names)), bool)
        mask[:, 0] = True
        for r in rows:
            mask[mv_idx[te.label(r, "move", voc)], idx.get(r[key], 0)] = True
        out[key] = dict(names=names, mask=mask)
    return out


def event_targets(stems, move_vocab, cv):
    """{stem: [(side 0/1, start frame, move idx, cmd idx, db idx)]} of the events with a true command."""
    import collections
    import inputs_eval as ie
    import transcript_eval as te
    voc = set(move_vocab)
    mv_idx = {m: i for i, m in enumerate(move_vocab)}
    ci = {k: i for i, k in enumerate(cv["cmd"]["names"])}
    di = {k: i for i, k in enumerate(cv["db"]["names"])}
    ss = set(stems)
    out = collections.defaultdict(list)
    for r in ie.event_cmds():
        if r["replay"] in ss:
            out[r["replay"]].append((0 if r["side"] == "P1" else 1, r["start_frame"], mv_idx[te.label(r, "move", voc)],
                                     ci.get(r["cmd"], 0), di.get(r["db"], 0)))
    return out


class IStore:
    """A split on the GPU: the transcript's inputs (vt.Store), posteriors, per-frame labels."""

    def __init__(self, stems, norm=None, labels=True, ff=False):
        rows, sp, vocab, voc_idx, chars, ev_by = vt.setup()
        self.vocab, self.chars = vocab, chars
        self.T = vt.Store(stems, {}, voc_idx, chars, norm, grid=True)
        self.stems = self.T.stems
        Ps, Mi, Mp, D, Pr, E = [], [], [], [], [], []
        for st, n in zip(self.stems, self.T.n):
            if ff:     # per-frame DINOv2-small of both squares (inputs_frame_feats.py), [n * 8, 2, 384]
                f = np.load(FFD / f"{st}.npz")["f"][:n * STEP]
                if len(f) < n * STEP:
                    f = np.concatenate([f, np.repeat(f[-1:], n * STEP - len(f), 0)])
                E.append(f)
            d, mi, mp = post_feats(st, n)
            Ps.append(d)
            Mi.append(mi)
            Mp.append(mp)
            if labels:
                z = np.load(LAB / f"{st}.npz")
                Fn = n * STEP
                dr = z["dir"][:, :Fn].astype(np.int64) - 1                       # -1 = unknown
                ok = z["live"][:Fn]
                dr[:, ~ok] = -1
                pr = np.stack([(z["press"][:, :Fn] >> k) & 1 for k in range(5)], -1).astype(np.uint8)
                pr[:, ~ok] = 255                                                  # ignored
                if dr.shape[1] < Fn:
                    padn = Fn - dr.shape[1]
                    dr = np.concatenate([dr, np.full((2, padn), -1)], 1)
                    pr = np.concatenate([pr, np.full((2, padn, 5), 255, np.uint8)], 1)
                D.append(dr)
                Pr.append(pr)
        self.P = torch.from_numpy(np.concatenate(Ps, 1)).to(DEV).half()           # [2, T, 36]
        self.Mi = torch.from_numpy(np.concatenate(Mi, 1)).to(DEV)                 # [2, T, M]
        self.Mp = torch.from_numpy(np.concatenate(Mp, 1)).to(DEV)
        if labels:
            self.D = torch.from_numpy(np.concatenate(D, 1)).to(DEV)              # [2, T * 8]
            self.Pr = torch.from_numpy(np.concatenate(Pr, 1)).to(DEV)            # [2, T * 8, 5]
        self.labels = labels
        self.E = torch.from_numpy(np.concatenate(E)).to(DEV) if ff else None     # [T * 8, 2, 384] fp16
        self.ev = None              # event_targets per replay index (set by the caller for the command head)

    def set_events(self, cv):
        et = event_targets(self.stems, self.vocab, cv)
        self.ev = [et.get(st, []) for st in self.stems]

    def side(self, r, s, a=0, b=None):
        V, A, _ = self.T.side(r, s, a, b)
        o = self.T.off[r]
        b = self.T.n[r] if b is None else b
        sl = slice(o + a, o + b)
        P = torch.cat([self.P[s, sl], self.P[1 - s, sl]], -1)
        M = (self.Mi[s, sl], self.Mp[s, sl], self.Mi[1 - s, sl], self.Mp[1 - s, sl])
        Y = E = None
        fs = slice(STEP * (o + a), STEP * (o + b))
        if self.labels:
            Y = (self.D[s, fs], self.Pr[s, fs])
        if self.E is not None:
            E = torch.cat([self.E[fs, s], self.E[fs, 1 - s]], -1)              # mine | other
        return V, A, P, M, Y, E


class InNet(nn.Module):
    def __init__(self, n_aux, n_post, n_move, n_char, grid, d=384, layers=4, heads=6, win=32, drop=0.1, in_drop=0.2, c=96,
                 use_post=True, ff=0, n_cmd=0, n_db=0):
        super().__init__()
        self.cfg = dict(n_aux=n_aux, n_post=n_post, n_move=n_move, n_char=n_char, grid=grid, d=d, layers=layers, heads=heads, win=win,
                        drop=drop, in_drop=in_drop, c=c, use_post=use_post, ff=ff, n_cmd=n_cmd, n_db=n_db)
        self.ff = ff
        self.n_cmd = n_cmd
        if n_cmd:   # event command head: trunk token + frames around the start + the move
            self.eemb = nn.Embedding(n_move, d)
            self.ehead = nn.Sequential(nn.LayerNorm(d + 2 * c + d), nn.Linear(d + 2 * c + d, 384), nn.GELU(), nn.Dropout(drop))
            self.h_cmd, self.h_db = nn.Linear(384, n_cmd), nn.Linear(384, n_db)
        if ff:     # per-frame features (mine | other) into the 60 fps stream
            self.fin = nn.Sequential(nn.LayerNorm(ff), nn.Dropout(in_drop), nn.Linear(ff, c), nn.GELU(), nn.Linear(c, c))
        self.grid, self.use_post = grid, use_post
        self.vin = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(in_drop), nn.Linear(1024, d))
        self.gin = nn.Sequential(nn.LayerNorm(grid // 4), nn.Dropout(in_drop), nn.Flatten(-2), nn.Linear(grid, d))
        self.vmix = nn.Linear(5 * d, d)
        self.ain = nn.Sequential(nn.Linear(n_aux, d), nn.GELU(), nn.Linear(d, d))
        self.cme, self.cop = nn.Embedding(n_char, d), nn.Embedding(n_char, d)
        if use_post:
            self.pin = nn.Sequential(nn.Linear(n_post, d), nn.GELU(), nn.Linear(d, d))
            self.memb = nn.Embedding(n_move, d)
            self.mmix = nn.Linear(2 * d, d)
        self.stem = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2, groups=d) for _ in range(2)])
        self.stem_pw = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.stem_n = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.blocks = nn.ModuleList([vt.Block(d, heads, win, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.c = c
        self.up = nn.Linear(d, STEP * c)          # tubelet -> its 8 frames (learned position inside the tubelet)
        self.fconv = nn.ModuleList([nn.Conv1d(c, c, 9, padding=4) for _ in range(2)])
        self.fn = nn.ModuleList([nn.LayerNorm(c) for _ in range(2)])
        self.h_dir, self.h_press = nn.Linear(c, 9), nn.Linear(c, 5)

    def trunk(self, V, A, P, M, cme, cop, pad=None):
        B, L = V.shape[:2]
        D, Gd = 1024, self.grid
        v = self.vin(V[..., :3 * D].float().view(B, L, 3, D)).reshape(B, L, -1)
        g = self.gin(V[..., 3 * D:3 * D + 2 * Gd].float().view(B, L, 2, 4, Gd // 4)).reshape(B, L, -1)
        x = self.vmix(torch.cat([v, g], -1)) + self.ain(A.float()) + (self.cme(cme) + self.cop(cop))[:, None]
        if self.use_post:
            mi, mp, oi, op = M
            em = (self.memb(mi) * mp[..., None]).sum(-2)
            eo = (self.memb(oi) * op[..., None]).sum(-2)
            x = x + self.pin(P.float()) + self.mmix(torch.cat([em, eo], -1))
        if pad is not None:
            x = x.masked_fill(pad[..., None], 0.0)
        for c, pw, n in zip(self.stem, self.stem_pw, self.stem_n):
            x = x + pw(F.gelu(c(n(x).transpose(1, 2)).transpose(1, 2)))
            if pad is not None:
                x = x.masked_fill(pad[..., None], 0.0)
        for b in self.blocks:
            x = b(x, pad)
        return self.norm(x)

    def frames(self, x, extra=None):
        B, L = x.shape[:2]
        f = self.up(x).view(B, L * STEP, self.c)
        if self.ff and extra is not None:
            f = f + self.fin(extra.float())
        for cv, n in zip(self.fconv, self.fn):
            f = f + F.gelu(cv(n(f).transpose(1, 2)).transpose(1, 2))
        return f

    def forward(self, V, A, P, M, cme, cop, pad=None, extra=None, feats=False):
        x = self.trunk(V, A, P, M, cme, cop, pad)
        f = self.frames(x, extra)
        if feats:
            return self.h_dir(f).float(), self.h_press(f).float(), x, f
        return self.h_dir(f).float(), self.h_press(f).float()

    def event_logits(self, x, f, b, fr, mv):
        """Command / direction+button logits for events (batch index b, frame fr in the window, move mv)."""
        L = x.shape[1]
        t = (fr // STEP).clamp(0, L - 1)
        Fn = f.shape[1]
        win = torch.stack([f[b, (fr + o).clamp(0, Fn - 1)] for o in range(-8, 1)], 1)    # the 8 frames up to the start
        aft = torch.stack([f[b, (fr + o).clamp(0, Fn - 1)] for o in range(1, 5)], 1)
        h = self.ehead(torch.cat([x[b, t], win.mean(1), aft.mean(1), self.eemb(mv)], -1).float())
        return self.h_cmd(h).float(), self.h_db(h).float()


def ibatch(st, idx, L, rng, train=True):
    Vb, Ab, Pb, Mb, Db, Rb, pad, cme, cop, Eb, EV = [], [], [], [[], [], [], []], [], [], [], [], [], [], []
    for r, s in idx:
        n = st.T.n[r]
        a = int(rng.integers(0, max(1, n - L + 1))) if train else 0
        b = min(n, a + L)
        V, A, P, M, (D, R), E = st.side(r, s, a, b)
        m = b - a
        if st.ev is not None:
            bi = len(Vb)
            for (es, fr, mv, ci, di) in st.ev[r]:
                if es == s and STEP * a <= fr < STEP * b:
                    jit = int(rng.integers(-4, 5)) if train else 0    # the transcript's start is good to +-4
                    EV.append((bi, fr - STEP * a + jit, mv, ci, di))
        if E is not None:
            if m < L:
                E = F.pad(E, (0, 0, 0, STEP * (L - m)))
            Eb.append(E)
        if m < L:
            V = F.pad(V, (0, 0, 0, L - m))
            A = F.pad(A, (0, 0, 0, L - m))
            P = F.pad(P, (0, 0, 0, L - m))
            M = tuple(F.pad(x, (0, 0, 0, L - m)) for x in M)
            D = F.pad(D, (0, STEP * (L - m)), value=-1)
            R = F.pad(R, (0, 0, 0, STEP * (L - m)), value=255)
        Vb.append(V)
        Ab.append(A)
        Pb.append(P)
        for k in range(4):
            Mb[k].append(M[k])
        Db.append(D)
        Rb.append(R)
        p = torch.zeros(L, dtype=torch.bool, device=DEV)
        p[m:] = True
        pad.append(p)
        cme.append(st.T.vote[r, s])
        cop.append(st.T.vote[r, 1 - s])
    t = lambda x: torch.as_tensor(np.array(x), device=DEV)
    return (torch.stack(Vb), torch.stack(Ab), torch.stack(Pb), tuple(torch.stack(x) for x in Mb), torch.stack(Db),
            torch.stack(Rb), torch.stack(pad), t(cme), t(cop), torch.stack(Eb) if Eb else None,
            torch.as_tensor(np.array(EV, np.int64).reshape(-1, 5), device=DEV) if st.ev is not None else None)


def iloss(dlog, plog, D, R, pos_w, spread):
    l_dir = F.cross_entropy(dlog.flatten(0, 1), D.flatten(), ignore_index=-1)
    ok = R != 255
    y = (R == 1).float() * ok
    if spread > 0:     # +-1 frame at `spread` (the press frame itself 1)
        sh = torch.zeros_like(y)
        sh[:, 1:] = torch.maximum(sh[:, 1:], y[:, :-1])
        sh[:, :-1] = torch.maximum(sh[:, :-1], y[:, 1:])
        y = torch.maximum(y, sh * spread)
    w = torch.where(y > 0, pos_w, 1.0) * ok
    l_p = (F.binary_cross_entropy_with_logits(plog, y, reduction="none") * w).sum() / ok.sum().clamp(min=1)
    return l_dir + l_p, float(l_dir), float(l_p)


@torch.no_grad()
def iinfer(nets, st, r):
    """[2, F, 9] dir probabilities, [2, F, 5] press probabilities for replay r (nets averaged)."""
    nets = nets if isinstance(nets, (list, tuple)) else [nets]
    D, P = [], []
    for s in (0, 1):
        V, A, Pp, M, _, E = st.side(r, s)
        cme = torch.as_tensor([st.T.vote[r, s]], device=DEV)
        cop = torch.as_tensor([st.T.vote[r, 1 - s]], device=DEV)
        dd = pp = 0
        for m in nets:
            m.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dl, pl = m(V[None], A[None], Pp[None], tuple(x[None] for x in M), cme, cop, extra=None if E is None else E[None])
            dd = dd + F.softmax(dl[0].float(), -1)
            pp = pp + torch.sigmoid(pl[0].float())
        D.append((dd / len(nets)).cpu().numpy())
        P.append((pp / len(nets)).cpu().numpy())
    return np.stack(D), np.stack(P)


def write_preds(nets, st, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r, stem in enumerate(st.stems):
        d, p = iinfer(nets, st, r)
        np.savez_compressed(out_dir / f"{stem}.npz", dir=d.astype(np.float16), press=p.astype(np.float16))


def quick_val(nets, st, thr=0.5):
    """(dir acc, press micro F1 at k=2) on a labelled store, in memory."""
    import inputs_eval as ie
    acc = ie.Acc()
    ev = ie.events_by_replay(set(st.stems))
    for r, stem in enumerate(st.stems):
        d, p = iinfer(nets, st, r)
        acc.add(ie.replay_scores(ie.load_label(stem), d, p, thr, ev.get(stem, [])))
    return acc.direction()["acc"], acc.presses()["k2"]["micro"]["f1"]


def tune_thr(pdir, stems, logf):
    import inputs_eval as ie
    ev = ie.events_by_replay(set(stems))
    labs = {s: ie.load_label(s) for s in stems}
    preds = {s: ie.load_pred(pdir, s) for s in stems}
    best, grid = None, []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        acc = ie.Acc()
        for s in stems:
            acc.add(ie.replay_scores(labs[s], *preds[s], thr, ev.get(s, [])))
        f = acc.presses()["k2"]["micro"]["f1"]
        grid.append(dict(thr=thr, f1_k2=f))
        if best is None or f > best[0]:
            best = (f, thr)
    log(f"press threshold {best[1]} (val F1@2 {best[0]:.4f})", logf)
    return best[1], grid


def cmd_train(a):
    low_priority()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    logf = INP / f"exp_in_{a.name}.log"
    sp = splits()
    t0 = time.time()
    trs = sp["train"][:int(len(sp["train"]) * a.frac)] if a.frac < 1 else sp["train"]
    if a.ff_subset or a.frame_feats:   # the training replays that have per-frame features
        import inputs_frame_feats as iff
        keep = set(iff.selected()["train"])
        trs = [x for x in trs if x in keep]
    tr = IStore(trs, ff=a.frame_feats)
    va = IStore(sp["val"], tr.T.norm, ff=a.frame_feats)
    log(f"loaded train {len(tr.stems)} val {len(va.stems)} in {time.time() - t0:.0f}s; args {vars(a)}", logf)
    grid = (tr.T.V.shape[1] - 3072) // 2
    cv = None
    if a.cmd_head:
        cv = cmd_vocab(sp["train"], tr.vocab)
        tr.set_events(cv)
        CM = torch.as_tensor(cv["cmd"]["mask"], device=DEV)
        DM = torch.as_tensor(cv["db"]["mask"], device=DEV)
        log(f"command head: {len(cv['cmd']['names'])} commands, {len(cv['db']['names'])} direction+button classes", logf)
    net = InNet(tr.T.A.shape[-1], tr.P.shape[-1] * 2, len(tr.vocab), len(tr.chars), grid, d=a.d, layers=a.layers, win=a.win,
                drop=a.drop, in_drop=a.in_drop, c=a.c, use_post=not a.no_post, ff=2 * 384 if a.frame_feats else 0,
                n_cmd=len(cv["cmd"]["names"]) if cv else 0, n_db=len(cv["db"]["names"]) if cv else 0).to(DEV)
    if a.init:     # warm start (e.g. the stage-1 model; the frame-feature branch starts new)
        st0 = torch.load(OUT / f"video_inputs_exp_in_{a.init}.pt", map_location=DEV, weights_only=False)["state"]
        miss = net.load_state_dict(st0, strict=False)
        log(f"init from {a.init}: missing {miss.missing_keys}", logf)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
    pairs = [(r, s) for r in range(len(tr.stems)) for s in (0, 1)]
    per_ep = max(1, int(len(pairs) * a.reps) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * per_ep, pct_start=0.1)
    best, hist = (-1, None, -1), []
    for ep in range(a.epochs):
        net.train()
        t1, tot, td, tp = time.time(), 0.0, 0.0, 0.0
        for _ in range(per_ep):
            idx = [pairs[i] for i in rng.integers(0, len(pairs), a.bs)]
            V, A, P, M, D, R, pad, cme, cop, E, EV = ibatch(tr, idx, a.L, rng)
            if E is not None and a.ff_drop > 0:    # whole frames without the per-frame features
                E = E * (torch.rand(E.shape[:2], device=DEV) >= a.ff_drop)[..., None].to(E.dtype)
            if a.view_drop > 0:
                keep = (torch.rand(V.shape[:2] + (5,), device=DEV) >= a.view_drop).to(V.dtype)
                V = V * torch.repeat_interleave(keep, torch.tensor([1024, 1024, 1024, grid, grid], device=DEV), dim=-1)
            if a.post_drop > 0 and not a.no_post:   # tubelets without the transcript's posteriors (robustness)
                kp = (torch.rand(P.shape[:2], device=DEV) >= a.post_drop)
                P = P * kp[..., None].to(P.dtype)
                M = (M[0], M[1] * kp[..., None], M[2], M[3] * kp[..., None])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dl, pl, xx, ff_ = net(V, A, P, M, cme, cop, pad, extra=E, feats=True)
                if EV is not None and len(EV):
                    cl, bl = net.event_logits(xx, ff_, EV[:, 0], EV[:, 1], EV[:, 2])
            loss, l1, l2 = iloss(dl, pl, D, R, a.pos_w, a.spread)
            if EV is not None and len(EV):
                cl = cl.masked_fill(~CM[EV[:, 2]], -1e4)
                bl = bl.masked_fill(~DM[EV[:, 2]], -1e4)
                loss = loss + a.cmd_w * (F.cross_entropy(cl, EV[:, 3]) + F.cross_entropy(bl, EV[:, 4]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            td += l1
            tp += l2
        msg = f"ep {ep + 1}/{a.epochs} loss {tot / per_ep:.4f} (dir {td / per_ep:.4f} press {tp / per_ep:.4f}) {time.time() - t1:.0f}s"
        if ep + 1 >= a.epochs * 0.6 and ((ep + 1) % 2 == 0 or ep + 1 == a.epochs):
            da, pf = quick_val(net, va)
            sc = (da + pf) / 2
            hist.append(dict(epoch=ep + 1, loss=tot / per_ep, val_dir_acc=da, val_press_f1_k2=pf))
            msg += f" | val dir {da:.4f} press F1@2 {pf:.4f}"
            if sc > best[0]:
                best = (sc, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep + 1)
        log(msg, logf)
    net.load_state_dict(best[1])
    log(f"best epoch {best[2]} ({best[0]:.4f}); trained in {time.time() - t0:.0f}s", logf)
    ck = dict(state=net.state_dict(), cfg=net.cfg, norm=[x.tolist() for x in tr.T.norm], vocab=tr.vocab, chars=tr.chars,
              epoch=best[2], args=vars(a), hist=hist, train_seconds=time.time() - t0,
              cmd_vocab=None if cv is None else {k: dict(names=v["names"], mask=v["mask"].tolist()) for k, v in cv.items()})
    del tr
    torch.cuda.empty_cache()
    vdir = INP / f"exp_in_{a.name}_val"
    write_preds(net, va, vdir)
    del va
    torch.cuda.empty_cache()
    thr, tgrid = tune_thr(vdir, splits()["val"], logf)
    ck.update(thr=thr, tune=tgrid)
    torch.save(ck, OUT / f"video_inputs_exp_in_{a.name}.pt")
    evaluate_test(net, ck, a.name, logf)


def load_inet(name):
    ck = torch.load(OUT / f"video_inputs_exp_in_{name}.pt", map_location=DEV, weights_only=False)
    c = ck["cfg"]
    net = InNet(c["n_aux"], c["n_post"], c["n_move"], c["n_char"], c["grid"], d=c["d"], layers=c["layers"], heads=c["heads"],
                win=c["win"], drop=c["drop"], in_drop=c["in_drop"], c=c["c"], use_post=c["use_post"], ff=c.get("ff", 0),
                n_cmd=c.get("n_cmd", 0), n_db=c.get("n_db", 0)).to(DEV)
    net.load_state_dict(ck["state"])
    return net.eval(), ck


def evaluate_test(net, ck, name, logf):
    import inputs_eval as ie
    sp = splits()
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    t0 = time.time()
    tdir = INP / f"exp_in_{name}_test"
    for i in range(0, len(sp["test"]), 110):
        st = IStore(sp["test"][i:i + 110], norm, labels=False, ff=bool(ck["cfg"].get("ff", 0)))
        write_preds(net, st, tdir)
        del st
        torch.cuda.empty_cache()
    infer_s = time.time() - t0
    val = ie.evaluate(INP / f"exp_in_{name}_val", sp["val"], ck["thr"], preds_path=INP / "exp_in_ens5grid_val_preds.parquet")
    res = ie.evaluate_subsets(tdir, ck["thr"], names=("old", "new", "all"))
    out = dict(name=name, weights=f"video_inputs_exp_in_{name}.pt", epoch=ck["epoch"], thr=ck["thr"], args=ck["args"],
               hist=ck["hist"], tune=ck.get("tune"), train_seconds=ck.get("train_seconds"), test_infer_seconds=infer_s,
               posteriors=("none" if ck["args"].get("no_post") else
                           "val/test: exp_tr_ens5grid; train: out-of-fold single transcript nets (video_transcript_exp_in_oof0-3.pt)"),
               val=val, test=res)
    if ck.get("cmd_vocab"):
        out["command_head"] = command_head_eval(net, ck, logf)
    save_json(out, OUT / f"results_exp_in_{name}.json")
    log("val:\n" + ie.headline({"val": val}), logf)
    log("test:\n" + ie.headline(res), logf)


@torch.no_grad()
def command_head_eval(net, ck, logf):
    """The event head's command and direction+button, read at the transcript's matched start (move level,
    +-4 frames) with the transcript's move, and at the true start; against the move prior, same events."""
    import collections
    import inputs_eval as ie
    import transcript_eval as te
    from common import replay_fold_map, test_subsets
    cv = {k: dict(names=v["names"], mask=torch.as_tensor(np.array(v["mask"]), device=DEV)) for k, v in ck["cmd_vocab"].items()}
    voc = set(ck["vocab"])
    mv_idx = {m: i for i, m in enumerate(ck["vocab"])}
    prior = ie.move_prior(splits()["train"])
    sp = splits()
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    tcmd = collections.defaultdict(dict)
    for r in ie.event_cmds():
        tcmd[r["replay"]][(r["side"], r["start_frame"])] = r
    res = {}
    for split, stems, preds in (("val", sp["val"], INP / "exp_in_ens5grid_val_preds.parquet"),
                                ("test", sp["test"], vt.TR / "exp_tr_ens5grid_test_preds.parquet")):
        matched, ev = ie.transcript_matches(stems, preds)
        rows = []
        for i0 in range(0, len(stems), 110):
            st = IStore(stems[i0:i0 + 110], norm, labels=False, ff=bool(ck["cfg"].get("ff", 0)))
            for r, stem in enumerate(st.stems):
                for s in (0, 1):
                    V, A, Pp, M, _, E = st.side(r, s)
                    cme = torch.as_tensor([st.T.vote[r, s]], device=DEV)
                    cop = torch.as_tensor([st.T.vote[r, 1 - s]], device=DEV)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, _, x, f = net(V[None], A[None], Pp[None], tuple(q[None] for q in M), cme, cop,
                                         extra=None if E is None else E[None], feats=True)
                    items = []
                    for i, e in enumerate(ev.get(stem, [])):
                        t = tcmd[stem].get((e["side"], e["start_frame"]))
                        if e["side"] != f"P{s + 1}" or t is None:
                            continue
                        mv = mv_idx[te.label(e, "move", voc)]
                        mf = matched.get((stem, i))
                        items.append((t, e, mv, e["start_frame"], mf))
                    if not items:
                        continue
                    mvs = torch.as_tensor([it[2] for it in items], device=DEV)
                    b0 = torch.zeros(len(items), dtype=torch.long, device=DEV)
                    out = {}
                    for key, frs in (("oracle", [it[3] for it in items]), ("tr", [it[4] if it[4] is not None else it[3] for it in items])):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            cl, bl = net.event_logits(x, f, b0, torch.as_tensor(frs, device=DEV), mvs)
                        cl = cl.masked_fill(~cv["cmd"]["mask"][mvs], -1e4).argmax(-1).tolist()
                        bl = bl.masked_fill(~cv["db"]["mask"][mvs], -1e4).argmax(-1).tolist()
                        out[key] = (cl, bl)
                    for j, (t, e, mv, _, mf) in enumerate(items):
                        rows.append(dict(replay=stem, cat=e["category"], matched=mf is not None, true=t["cmd"], tdb=t["db"],
                                         o=cv["cmd"]["names"][out["oracle"][0][j]], odb=cv["db"]["names"][out["oracle"][1][j]],
                                         x=cv["cmd"]["names"][out["tr"][0][j]], xdb=cv["db"]["names"][out["tr"][1][j]],
                                         p=prior["cmd"].get(e["char_move"]), pdb=prior["db"].get(e["char_move"])))
            del st
            torch.cuda.empty_cache()

        def summ(rr):
            m = [r for r in rr if r["matched"]]
            if not m:
                return None
            acc = lambda a_, b_, L: sum(r[a_] == r[b_] for r in L) / len(L)
            mot = lambda a_, L: sum(ie.motion_of(r[a_]) == ie.motion_of(r["true"]) for r in L) / len(L)
            return dict(n=len(rr), matched=len(m), head_oracle_cmd_acc=acc("o", "true", rr), head_oracle_dirbtn_acc=acc("odb", "tdb", rr),
                        head_transcript_cmd_acc=acc("x", "true", m), head_transcript_dirbtn_acc=acc("xdb", "tdb", m),
                        head_transcript_motion_acc=mot("x", m),
                        move_prior_cmd_acc=acc("p", "true", m), move_prior_dirbtn_acc=acc("pdb", "tdb", m), move_prior_motion_acc=mot("p", m))
        subsets = {"val": sp["val"]} if split == "val" else {k: v for k, v in test_subsets(replay_fold_map()).items() if k in ("old", "new", "all")}
        for nm, reps in subsets.items():
            reps = set(reps)
            rr = [r for r in rows if r["replay"] in reps]
            d = {"all": summ(rr)}
            for c in ("normal", "special", "super", "throw", "other"):
                d[c] = summ([r for r in rr if r["cat"] == c])
            sps = [r for r in rr if r["cat"] in ("special", "super") and ie.motion_of(r["true"]) in ie.MOTIONS + ["[4]6", "[2]8"]]
            d["special_super_with_motion"] = summ(sps)
            res[nm] = d
            a_ = d["all"]
            ss_ = d["special_super_with_motion"] or {}
            log(f"[{nm}] command head: cmd tr {a_['head_transcript_cmd_acc']:.3f} (prior {a_['move_prior_cmd_acc']:.3f}) "
                f"dirbtn tr {a_['head_transcript_dirbtn_acc']:.3f} (prior {a_['move_prior_dirbtn_acc']:.3f}) "
                f"oracle cmd {a_['head_oracle_cmd_acc']:.3f} | sp/su motion tr {ss_.get('head_transcript_motion_acc', 0):.3f} "
                f"(prior {ss_.get('move_prior_motion_acc', 0):.3f})", logf)
    return res


def cmd_eval(a):
    net, ck = load_inet(a.name)
    evaluate_test(net, ck, a.name, INP / f"exp_in_{a.name}.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("oof", "post", "train", "eval"))
    ap.add_argument("--folds", nargs="*", type=int)
    ap.add_argument("--name", default="base")
    ap.add_argument("--no-post", action="store_true", help="ablation: without the transcript's posteriors")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--reps", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--c", type=int, default=96)
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--in-drop", type=float, default=0.4)
    ap.add_argument("--view-drop", type=float, default=0.15)
    ap.add_argument("--post-drop", type=float, default=0.1)
    ap.add_argument("--pos-w", type=float, default=4.0)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-feats", action="store_true", help="stage 2: per-frame DINOv2 of the squares (inputs_frame_feats.py)")
    ap.add_argument("--ff-subset", action="store_true", help="train only on the replays with per-frame features (control)")
    ap.add_argument("--ff-drop", type=float, default=0.1)
    ap.add_argument("--init", help="warm start from video_inputs_exp_in_<init>.pt")
    ap.add_argument("--cmd-head", action="store_true", help="event command head (command / direction+button given the move)")
    ap.add_argument("--cmd-w", type=float, default=0.5)
    a = ap.parse_args()
    {"oof": cmd_oof, "post": cmd_post, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
