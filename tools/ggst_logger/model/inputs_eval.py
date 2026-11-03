"""Inputs from video (ROADMAP point 3): the metric.

Labels: dataset/inputs_labels.py -> ggst_dataset/inputs/labels/<stem>.npz (per video frame and side:
dir 1-9 relative to the facing, press / hold bits P K S H D, state, live). Predictions: one npz per
replay with `dir` [2, F, 9] (probabilities of numpad 1-9) and `press` [2, F, 5] (per-frame press
probabilities of P K S H D), F = the label's frame count.

Scored on live frames (both players in play) of the test fold, reported for old / new (and all):
- direction: accuracy per frame (argmax), balanced accuracy (mean over the 9 directions), accuracy
  on the frames whose true direction is not 5, the vertical (up / neutral / down) and horizontal
  (back / neutral / forward) parts alone, and per 7.5 Hz step (the tubelet's most common true
  direction against the argmax of the mean probability);
- per state (the log's action: move, idle, walk_f, walk_b, dash, guard, jump, hit): direction
  accuracy, i.e. "inputs without a visible move" are the non-move states;
- button presses as spotting: per side and button, local maxima of the press probability over
  +-NMS frames with p >= thr are predicted presses; a prediction matches an unmatched true press of
  the same side and button within +-k frames (k = 1, 2, 4), greedy by |dt| then score. P / R / F1
  micro over the buttons, per button, AP; `any` ignores the button (timing alone). Recall is also
  split by whether a move of that side starts within 0..10 frames after the press (the log's
  transcript events): presses that start a visible move vs presses without one (mashing, buffered,
  kara, dash and movement presses);
- commands (motion + button, read like verify_inputs.read_command on the per-frame streams: the
  latest press up to 8 frames before the move start, the longest motion in the 24 frames before
  it, charges [4]6 / [2]8, else the held direction + button), for every transcript event (the
  log's moves) whose true command was found:
  * `oracle`: read at the true move start from the predicted streams;
  * `transcript`: read at the start of the transcript's (exp_tr_ens5grid) prediction matched to the
    event at the move level within +-4 frames, i.e. input accuracy given a correctly spotted move;
  * `move_prior`: the most common true command of that move in the training replays (what the
    move alone gives, no input model), on the same matched events;
  also the plain input at the press (`dirbtn`: held direction + button, 2K / 6P / 5H) for the same
  three; per category; for specials / supers also the motion alone (236, 214, 623, 41236, 63214, 632146,
  [4]6 ...) as recognition (`motion_acc`: same motion read) and the press timing error (|dt| of the
  read press against the true press, share within 1 / 2 / 4 frames).

    .venv/Scripts/python.exe tools/ggst_logger/model/inputs_eval.py stats
    .venv/Scripts/python.exe tools/ggst_logger/model/inputs_eval.py score <pred_dir> [--thr 0.5] [--subset old]
    .venv/Scripts/python.exe tools/ggst_logger/model/inputs_eval.py selftest      # labels as predictions (+ jitter)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
from common import DATA, replay_fold_map, test_subsets  # noqa: E402

INP = DATA / "inputs"
LAB = INP / "labels"
TR = DATA / "transcript"
STATES = ["move", "idle", "walk_f", "walk_b", "dash", "guard", "jump", "hit"]
BTN = "PKSHD"
TOLS = (1, 2, 4)
NMS = 2
STEP = 8
PRESS_BEFORE, MOTION_WINDOW, AFTER_PRESS, CHARGE, CHARGE_WINDOW = 8, 24, 3, 30, 90
MOTIONS = ["632146", "236236", "214214", "41236", "63214", "21478", "23698", "47896", "69874",
           "87412", "89632", "623", "421", "236", "214", "22"]
TR_SHIFT = 2           # transcript-timed reading at the predicted start + this; chosen on val with the labels as
                      # predictions (0/1/2/3/4 -> 0.78/0.89/0.92/0.91/0.89 of the commands read right)
MOVE_GAP = 10          # a press "starts a visible move" when a move of that side starts within 0..10 frames


# ---- loading -----------------------------------------------------------------------------------

def load_label(stem):
    z = np.load(LAB / f"{stem}.npz")
    return {k: z[k] for k in z}


def load_pred(pdir, stem):
    z = np.load(Path(pdir) / f"{stem}.npz")
    return z["dir"].astype(np.float32), z["press"].astype(np.float32)


def splits():
    sp = json.loads((TR / "transcript_splits.json").read_text())
    return {s: sorted(k for k, v in sp.items() if v == s) for s in ("train", "val", "test")}


def events_by_replay(stems=None):
    import pyarrow.parquet as pq
    by = collections.defaultdict(list)
    for r in pq.read_table(TR / "transcript_events.parquet",
                           columns=["replay", "side", "start_frame", "char_move", "category", "in_video", "kind"]).to_pylist():
        if stems is None or r["replay"] in stems:
            by[r["replay"]].append(r)
    return by


# ---- press decoding / matching -------------------------------------------------------------------

def peaks(p, thr, nms=NMS):
    """Frames where p >= thr and p is the maximum over +-nms (first of a plateau)."""
    n = len(p)
    if n == 0:
        return np.zeros(0, int)
    pad = np.concatenate([np.full(nms, -1.0), p, np.full(nms, -1.0)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * nms + 1)
    mx = win.max(1)
    cand = np.nonzero((p >= thr) & (p >= mx))[0]
    if len(cand) == 0:
        return cand
    keep = [cand[0]]
    for c in cand[1:]:
        if c - keep[-1] > nms or p[c] > p[keep[-1]]:
            keep.append(c)
    return np.array(keep)


def match(t, p, ps, k):
    """Greedy one-to-one matching of true frames t to predicted frames p (scores ps) within k;
    returns (matched true mask, matched pred mask, dt of the pairs)."""
    t, p = np.asarray(t), np.asarray(p)
    mt, mp = np.zeros(len(t), bool), np.zeros(len(p), bool)
    if len(t) == 0 or len(p) == 0:
        return mt, mp, []
    pairs = []
    lo = np.searchsorted(p, t - k)
    hi = np.searchsorted(p, t + k, side="right")
    for i in range(len(t)):
        for j in range(lo[i], hi[i]):
            pairs.append((abs(int(p[j]) - int(t[i])), -ps[j], i, j))
    pairs.sort()
    dts = []
    for d, _, i, j in pairs:
        if mt[i] or mp[j]:
            continue
        mt[i] = mp[j] = True
        dts.append(int(p[j]) - int(t[i]))
    return mt, mp, dts


def ap(scores, hits, n_true):
    if n_true == 0:
        return None
    o = np.argsort(-np.asarray(scores, float), kind="stable")
    h = np.asarray(hits, bool)[o]
    tp = np.cumsum(h)
    return float(np.sum((tp / np.arange(1, len(h) + 1))[h]) / n_true) if len(h) else 0.0


# ---- command reading -----------------------------------------------------------------------------

def has_motion(seq: str, m: str) -> bool:
    """seq: direction changes (consecutive duplicates removed)."""
    if m == "22":
        return re.search(r"[123][^123]+[123]", seq) is not None
    it = iter(seq)
    return all(any(c == d for d in it) for c in m)


def dedup(s: str) -> str:
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def read_cmd(dirs: str, press: np.ndarray, t: int):
    """verify_inputs.read_command on frame streams: dirs = one numpad digit per frame ('0' = unknown,
    read as 5), press = press bits per frame. Returns (top reading, press frame) or (None, None)."""
    b = None
    for x in range(t, max(0, t - PRESS_BEFORE) - 1, -1):
        if x < len(press) and press[x]:
            b = x
            break
    if b is None:
        return None, None
    pr = [BTN[k] for k in range(5) if press[b] >> k & 1]
    btn = "/".join(pr)
    lo = max(0, b - MOTION_WINDOW)
    d = dirs[lo:b + 1].replace("0", "5")
    seq = dedup(d)
    for m in MOTIONS:
        if has_motion(seq, m):
            return m + btn, b
    held = dirs[max(0, b - CHARGE_WINDOW):b + 1].replace("0", "5")
    for ch, rel, name in (("147", "369", "[4]6"), ("123", "789", "[2]8")):
        k = len(held) - 1
        while k >= 0 and held[k] in rel:
            k -= 1
        if k < len(held) - 1 and sum(c in ch for c in held[:k + 1]) >= CHARGE:
            return name + btn, b
    return d[-1].replace("5", "") + btn, b


def motion_of(cmd):
    return None if cmd is None else cmd.rstrip("PKSHD/")


# ---- per replay ----------------------------------------------------------------------------------

def dir_string(d: np.ndarray) -> str:
    return "".join(map(str, d.tolist()))


def pred_presses(press_p, thr):
    """[(frame, score)] per (side, button), and the per-frame bit stream of the decoded presses."""
    out = {}
    bits = np.zeros(press_p.shape[:2], np.uint8)
    for s in (0, 1):
        for k in range(5):
            f = peaks(press_p[s, :, k], thr)
            out[(s, k)] = (f, press_p[s, f, k])
            bits[s, f] |= np.uint8(1 << k)
    return out, bits


def replay_scores(lab, dp, pp, thr, events):
    """Accumulators of one replay (see Acc)."""
    F = lab["dir"].shape[1]
    dp, pp = dp[:, :F], pp[:, :F]
    if dp.shape[1] < F:     # pad (never expected)
        dp = np.concatenate([dp, np.zeros((2, F - dp.shape[1], 9), np.float32)], 1)
        pp = np.concatenate([pp, np.zeros((2, F - pp.shape[1], 5), np.float32)], 1)
    live = lab["live"]
    res = dict(dir=[], tub=[], press={}, any={}, cmd=[], streams=None)
    pdir = dp.argmax(-1) + 1
    for s in (0, 1):
        m = live & (lab["dir"][s] > 0)
        res["dir"].append((lab["dir"][s][m], pdir[s][m], lab["state"][s][m]))
        # per tubelet
        nt = F // STEP
        td = lab["dir"][s][:nt * STEP].reshape(nt, STEP)
        tm = m[:nt * STEP].reshape(nt, STEP)
        pm = dp[s, :nt * STEP].reshape(nt, STEP, 9)
        for i in np.nonzero(tm.sum(1) >= STEP // 2)[0]:
            res["tub"].append((np.bincount(td[i][tm[i]], minlength=10)[1:].argmax() + 1, pm[i][tm[i]].mean(0).argmax() + 1))
    pr, bits = pred_presses(pp, thr)
    starts = {s: np.array(sorted(e["start_frame"] for e in events if e["side"] == f"P{s + 1}")) for s in (0, 1)}
    for s in (0, 1):
        any_t, any_p, any_s = [], [], []
        for k in range(5):
            t = np.nonzero(((lab["press"][s] >> k) & 1).astype(bool) & live)[0]
            f, sc = pr[(s, k)]
            keep = live[f]
            f, sc = f[keep], sc[keep]
            st = starts[s]
            j = np.searchsorted(st, t)
            vis = (j < len(st)) & (st[np.minimum(j, len(st) - 1)] - t <= MOVE_GAP) if len(st) else np.zeros(len(t), bool)
            res["press"][(s, k)] = (t, f, sc, vis)
            any_t.append(t)
            any_p.append(f)
            any_s.append(sc)
        t = np.unique(np.concatenate(any_t))
        fa, sa = np.concatenate(any_p), np.concatenate(any_s)
        o = np.lexsort((-sa, fa))
        fa, sa = fa[o], sa[o]
        first = np.ones(len(fa), bool)
        first[1:] = fa[1:] != fa[:-1]          # one prediction per frame (the best button's score)
        res["any"][s] = (t, fa[first], sa[first])
    res["streams"] = (pdir, bits)
    return res


class Acc:
    """Pools replay_scores over a set of replays."""

    def __init__(self):
        self.td, self.pd, self.st, self.tub = [], [], [], []
        self.press = collections.defaultdict(list)
        self.any = []

    def add(self, r):
        for t, p, s in r["dir"]:
            self.td.append(t)
            self.pd.append(p)
            self.st.append(s)
        self.tub += r["tub"]
        for (s, k), v in r["press"].items():
            self.press[k].append(v)
        self.any += list(r["any"].values())

    def direction(self):
        t, p, s = np.concatenate(self.td).astype(int), np.concatenate(self.pd).astype(int), np.concatenate(self.st)
        vert = lambda d: (d - 1) // 3          # 0 down, 1 neutral, 2 up
        hor = lambda d: (d - 1) % 3            # 0 back, 1 neutral, 2 forward
        per_dir = {str(d): float((p[t == d] == d).mean()) for d in range(1, 10) if (t == d).any()}
        tub = np.array(self.tub) if self.tub else np.zeros((0, 2), int)
        return dict(frames=int(len(t)), acc=float((t == p).mean()), balanced_acc=float(np.mean(list(per_dir.values()))),
                    acc_not5=float((t[t != 5] == p[t != 5]).mean()), vertical_acc=float((vert(t) == vert(p)).mean()),
                    horizontal_acc=float((hor(t) == hor(p)).mean()), per_dir_recall=per_dir,
                    true_share={str(d): float((t == d).mean()) for d in range(1, 10)},
                    step_acc=float((tub[:, 0] == tub[:, 1]).mean()) if len(tub) else None, steps=int(len(tub)),
                    per_state={STATES[k]: dict(frames=int((s == k).sum()), acc=float((t[s == k] == p[s == k]).mean()),
                                               acc_not5=float((t[(s == k) & (t != 5)] == p[(s == k) & (t != 5)]).mean())
                                               if ((s == k) & (t != 5)).any() else None)
                               for k in range(len(STATES)) if (s == k).any()},
                    no_move=dict(frames=int((s != 0).sum()), acc=float((t[s != 0] == p[s != 0]).mean()),
                                 acc_not5=float((t[(s != 0) & (t != 5)] == p[(s != 0) & (t != 5)]).mean())))

    def presses(self):
        out = {}
        f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
        for k in TOLS:
            tot = dict(tp=0, nt=0, np=0)
            per = {}
            vis = dict(vis_tp=0, vis_n=0, inv_tp=0, inv_n=0)
            dts = []
            for b in range(5):
                tp = nt = npred = 0
                scores, hits = [], []
                for t, f, sc, v in self.press[b]:
                    mt, mp, dt = match(t, f, sc, k)
                    tp += int(mt.sum())
                    nt += len(t)
                    npred += len(f)
                    scores += list(sc)
                    hits += list(mp)
                    vis["vis_tp"] += int(mt[v].sum())
                    vis["vis_n"] += int(v.sum())
                    vis["inv_tp"] += int(mt[~v].sum())
                    vis["inv_n"] += int((~v).sum())
                    dts += dt
                P, R = tp / max(1, npred), tp / max(1, nt)
                per[BTN[b]] = dict(n_true=nt, n_pred=npred, precision=P, recall=R, f1=f1(P, R), ap=ap(scores, hits, nt))
                tot["tp"] += tp
                tot["nt"] += nt
                tot["np"] += npred
            P, R = tot["tp"] / max(1, tot["np"]), tot["tp"] / max(1, tot["nt"])
            atp = ant = anp = 0
            for t, f, sc in self.any:
                mt, mp, _ = match(t, f, sc, k)
                atp += int(mt.sum())
                ant += len(t)
                anp += len(f)
            aP, aR = atp / max(1, anp), atp / max(1, ant)
            dts = np.abs(np.array(dts)) if dts else np.zeros(0)
            out[f"k{k}"] = dict(micro=dict(precision=P, recall=R, f1=f1(P, R), n_true=tot["nt"], n_pred=tot["np"]),
                                macro_f1=float(np.mean([v["f1"] for v in per.values()])),
                                mAP=float(np.mean([v["ap"] for v in per.values() if v["ap"] is not None])),
                                per_button=per,
                                any_button=dict(precision=aP, recall=aR, f1=f1(aP, aR)),
                                recall_visible_move=vis["vis_tp"] / max(1, vis["vis_n"]), n_visible_move=vis["vis_n"],
                                recall_no_visible_move=vis["inv_tp"] / max(1, vis["inv_n"]), n_no_visible_move=vis["inv_n"],
                                mean_abs_dt_matched=float(dts.mean()) if len(dts) else None)
        return out


# ---- commands --------------------------------------------------------------------------------------

def db_at(dirs: str, press: np.ndarray, b):
    """The plain input at a press frame: held direction + button(s) (2K, 6P, 5S/H); None without one."""
    if b is None:
        return None
    return dirs[b].replace("0", "5") + "/".join(BTN[k] for k in range(5) if press[b] >> k & 1)


def true_cmds(lab, events):
    """Per event (index -> (cmd, press frame)) read from the label streams at the true start."""
    out = {}
    for s in (0, 1):
        ds = dir_string(lab["dir"][s])
        pr = lab["press"][s]
        for i, e in enumerate(events):
            if e["side"] == f"P{s + 1}" and e["in_video"] and 0 <= e["start_frame"] < len(pr) and lab["live"][e["start_frame"]]:
                out[i] = read_cmd(ds, pr, e["start_frame"])
    return out


def move_prior(train_stems):
    """{'cmd': char_move -> most common true command, 'db': ... most common direction + button at the
    press} over the training replays, cached."""
    path = INP / "inputs_move_prior.json"
    if path.exists():
        return json.loads(path.read_text())
    ev = events_by_replay(set(train_stems))
    C = collections.defaultdict(collections.Counter)
    Cd = collections.defaultdict(collections.Counter)
    for st in train_stems:
        lab = load_label(st)
        es = ev.get(st, [])
        for i, (c, b) in true_cmds(lab, es).items():
            if c is not None:
                s = 0 if es[i]["side"] == "P1" else 1
                C[es[i]["char_move"]][c] += 1
                Cd[es[i]["char_move"]][db_at(dir_string(lab["dir"][s]), lab["press"][s], b)] += 1
    pri = dict(cmd={m: c.most_common(1)[0][0] for m, c in C.items()}, db={m: c.most_common(1)[0][0] for m, c in Cd.items()})
    path.write_text(json.dumps(pri, indent=0))
    return pri


def event_cmds():
    """Every transcript event's true command and direction + button (read from the label streams at its
    start), all replays: inputs/inputs_event_cmds.parquet (cached). Used to train the event command
    head of video_inputs.py."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = INP / "inputs_event_cmds.parquet"
    if path.exists():
        return pq.read_table(path).to_pylist()
    ev = events_by_replay()
    rows = []
    for st in sorted(ev):
        lab = load_label(st)
        es = ev[st]
        for i, (c, b) in true_cmds(lab, es).items():
            if c is None:
                continue
            e = es[i]
            s = 0 if e["side"] == "P1" else 1
            rows.append(dict(replay=st, side=e["side"], start_frame=e["start_frame"], char_move=e["char_move"],
                             category=e["category"], cmd=c, db=db_at(dir_string(lab["dir"][s]), lab["press"][s], b), press_frame=b))
    pq.write_table(pa.Table.from_pylist(rows), path)
    return rows


def transcript_matches(stems, preds_path, k=4):
    """{(replay, event index): predicted start frame} for the events matched at the move level by the
    transcript's test predictions."""
    import pyarrow.parquet as pq
    import transcript_eval as te
    ev = events_by_replay(set(stems))
    P = collections.defaultdict(list)
    for p in pq.read_table(preds_path).to_pylist():
        if p["replay"] in stems:
            P[(p["replay"], p["side"], p["char_move"])].append((p["start_frame"], p["score"]))
    out = {}
    sp = json.loads((TR / "transcript_splits.json").read_text())
    vocab = set(te.move_vocab([r for r in te.load_events() if sp.get(r["replay"]) == "train"]))   # (rare) folding
    for st in stems:
        T = collections.defaultdict(list)
        for i, e in enumerate(ev.get(st, [])):
            if e["in_video"]:
                T[(e["side"], te.label(e, "move", vocab))].append(i)
        for (side, mv), idx in T.items():
            p = P.get((st, side, mv), [])
            for a, b in te.match([ev[st][i]["start_frame"] for i in idx], p, k):
                out[(st, idx[a])] = p[b][0]
    return out, ev


def command_scores(stems, pdir_fn, ev, matched, prior):
    """Command accuracy: oracle timing, transcript timing, move prior (see the module doc)."""
    rows = []
    for st in stems:
        lab = load_label(st)
        es = ev.get(st, [])
        tc = true_cmds(lab, es)
        pdir, bits = pdir_fn(st)
        tds = [dir_string(lab["dir"][s]) for s in (0, 1)]
        pds = [dir_string(pdir[s]) for s in (0, 1)]
        for i, (c, tb) in tc.items():
            if c is None:
                continue
            e = es[i]
            s = 0 if e["side"] == "P1" else 1
            ds = pds[s]
            oc, ob = read_cmd(ds, bits[s], e["start_frame"])
            mf = matched.get((st, i))
            if mf is not None:
                xc, xb = read_cmd(ds, bits[s], int(mf) + TR_SHIFT)
            else:
                xc = xb = None
            rows.append(dict(cat=e["category"], true=c, tb=tb, oracle=oc, ob=ob, matched=mf is not None, tr=xc, xb=xb,
                             prior=prior["cmd"].get(e["char_move"]), tdb=db_at(tds[s], lab["press"][s], tb),
                             odb=db_at(ds, bits[s], ob), xdb=db_at(ds, bits[s], xb), pdb=prior["db"].get(e["char_move"])))
    def summ(rr):
        if not rr:
            return None
        n = len(rr)
        m = [r for r in rr if r["matched"]]
        mot = lambda r, k: motion_of(r[k]) == motion_of(r["true"])
        dt = lambda key, kb: np.array([abs(r[kb] - r["tb"]) for r in m if r[kb] is not None]) if m else np.zeros(0)
        d_tr = dt("tr", "xb")
        return dict(n=n, oracle_cmd_acc=sum(r["oracle"] == r["true"] for r in rr) / n,
                    oracle_motion_acc=sum(mot(r, "oracle") for r in rr) / n,
                    matched=len(m), matched_frac=len(m) / n,
                    transcript_cmd_acc=sum(r["tr"] == r["true"] for r in m) / max(1, len(m)),
                    transcript_motion_acc=sum(mot(r, "tr") for r in m) / max(1, len(m)),
                    move_prior_cmd_acc=sum(r["prior"] == r["true"] for r in m) / max(1, len(m)),
                    move_prior_motion_acc=sum(motion_of(r["prior"]) == motion_of(r["true"]) for r in m) / max(1, len(m)),
                    oracle_dirbtn_acc=sum(r["odb"] == r["tdb"] for r in rr) / n,
                    transcript_dirbtn_acc=sum(r["xdb"] == r["tdb"] for r in m) / max(1, len(m)),
                    move_prior_dirbtn_acc=sum(r["pdb"] == r["tdb"] for r in m) / max(1, len(m)),
                    transcript_press_dt=dict(read=int(len(d_tr)), le1=float((d_tr <= 1).mean()) if len(d_tr) else None,
                                             le2=float((d_tr <= 2).mean()) if len(d_tr) else None,
                                             le4=float((d_tr <= 4).mean()) if len(d_tr) else None,
                                             mean=float(d_tr.mean()) if len(d_tr) else None))
    out = {"all": summ(rows)}
    for c in ("normal", "special", "super", "throw", "other"):
        out[c] = summ([r for r in rows if r["cat"] == c])
    sp = [r for r in rows if r["cat"] in ("special", "super") and motion_of(r["true"]) in MOTIONS + ["[4]6", "[2]8"]]
    out["special_super_with_motion"] = summ(sp)
    out["per_motion"] = {m: summ([r for r in sp if motion_of(r["true"]) == m])
                         for m, _ in collections.Counter(motion_of(r["true"]) for r in sp).most_common(10)}
    return out


# ---- driver ----------------------------------------------------------------------------------------

def evaluate(pdir, stems, thr, with_cmd=True, preds_path=None):
    """Every metric on `stems` for the predictions in pdir."""
    ev = events_by_replay(set(stems))
    acc = Acc()
    streams = {}
    for st in stems:
        lab = load_label(st)
        dp, pp = load_pred(pdir, st)
        r = replay_scores(lab, dp, pp, thr, ev.get(st, []))
        acc.add(r)
        streams[st] = r["streams"]
    out = dict(direction=acc.direction(), presses=acc.presses())
    if with_cmd:
        prior = move_prior(splits()["train"])
        matched, _ = transcript_matches(stems, preds_path or TR / "exp_tr_ens5grid_test_preds.parquet")
        out["commands"] = command_scores(stems, lambda st: streams[st], ev, matched, prior)
    return out


def evaluate_subsets(pdir, thr, preds_path=None, names=("old", "new")):
    sub = test_subsets(replay_fold_map())
    return {k: evaluate(pdir, sub[k], thr, preds_path=preds_path) for k in names}


def headline(res):
    """One line per subset."""
    lines = []
    for k, d in res.items():
        di, pr = d["direction"], d["presses"]
        c = d.get("commands", {})
        ss = c.get("special_super_with_motion") or {}
        a = c.get("all") or {}
        lines.append(f"[{k}] dir {di['acc']:.3f} bal {di['balanced_acc']:.3f} not5 {di['acc_not5']:.3f} step {di['step_acc']:.3f} "
                     f"nomove {di['no_move']['acc']:.3f} | press F1 k1 {pr['k1']['micro']['f1']:.3f} k2 {pr['k2']['micro']['f1']:.3f} "
                     f"k4 {pr['k4']['micro']['f1']:.3f} (any k2 {pr['k2']['any_button']['f1']:.3f}; vis R {pr['k2']['recall_visible_move']:.3f} "
                     f"novis R {pr['k2']['recall_no_visible_move']:.3f}) | cmd oracle {a.get('oracle_cmd_acc', 0):.3f} "
                     f"tr {a.get('transcript_cmd_acc', 0):.3f} prior {a.get('move_prior_cmd_acc', 0):.3f} | "
                     f"motion sp oracle {ss.get('oracle_motion_acc', 0):.3f} tr {ss.get('transcript_motion_acc', 0):.3f} "
                     f"prior {ss.get('move_prior_motion_acc', 0):.3f} | dirbtn oracle {a.get('oracle_dirbtn_acc', 0):.3f} "
                     f"tr {a.get('transcript_dirbtn_acc', 0):.3f} prior {a.get('move_prior_dirbtn_acc', 0):.3f}")
    return "\n".join(lines)


def write_label_preds(stems, out_dir, jitter=0, drop=0.0, seed=0):
    """Selftest: labels as predictions (one-hot dir, presses moved by +-jitter, a share dropped)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for st in stems:
        lab = load_label(st)
        F = lab["dir"].shape[1]
        d = np.zeros((2, F, 9), np.float16)
        for s in (0, 1):
            dd = np.where(lab["dir"][s] > 0, lab["dir"][s], 5).astype(int)
            d[s, np.arange(F), dd - 1] = 1
        p = np.zeros((2, F, 5), np.float16)
        for s in (0, 1):
            for k in range(5):
                f = np.nonzero((lab["press"][s] >> k) & 1)[0]
                f = f[rng.random(len(f)) >= drop]
                if jitter:
                    f = np.clip(f + rng.integers(-jitter, jitter + 1, len(f)), 0, F - 1)
                p[s, f, k] = 1
        np.savez_compressed(out_dir / f"{st}.npz", dir=d, press=p)


def write_transcript_baseline(stems, preds_path, out_dir):
    """Inputs from the transcript alone (no input model): direction 5 on every frame; one press per
    transcript event on its start frame (press and move start share the frame for 68% of the logged
    moves), the button = the last button of the move's most common command in training (normals:
    the predicted button), score = the event's score."""
    import pyarrow.parquet as pq
    prior = move_prior(splits()["train"])
    by = collections.defaultdict(list)
    for p in pq.read_table(preds_path).to_pylist():
        by[p["replay"]].append(p)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for st in stems:
        F = load_label(st)["dir"].shape[1]
        d = np.zeros((2, F, 9), np.float16)
        d[:, :, 4] = 1
        pp = np.zeros((2, F, 5), np.float32)
        for p in by.get(st, []):
            c = prior["cmd"].get(p["char_move"]) or ""
            b = (p.get("normal") or "")[:1] if p["category"] == "normal" and p.get("normal") else (c[-1:] if c[-1:] in BTN else "")
            f = int(p["start_frame"])
            if b and 0 <= f < F:
                s = 0 if p["side"] == "P1" else 1
                pp[s, f, BTN.index(b)] = max(pp[s, f, BTN.index(b)], p["score"])
        np.savez_compressed(out_dir / f"{st}.npz", dir=d, press=pp.astype(np.float16))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("cmd", choices=("stats", "score", "selftest", "baseline"))
    ap_.add_argument("pred_dir", nargs="?")
    ap_.add_argument("--thr", type=float, default=0.5)
    ap_.add_argument("--subset", default=None)
    a = ap_.parse_args()
    if a.cmd == "stats":
        sp = splits()
        print({k: len(v) for k, v in sp.items()})
        print(json.dumps({k: v for k, v in json.loads((INP / "inputs_labels_stats.json").read_text()).items() if k != "per_replay"}, indent=1)[:4000])
    elif a.cmd == "score":
        names = (a.subset,) if a.subset else ("old", "new")
        res = evaluate_subsets(a.pred_dir, a.thr, names=names)
        print(headline(res))
        print(json.dumps(res, indent=1, default=float)[:6000])
    elif a.cmd == "baseline":
        # transcript-only inputs on val (threshold) and test -> results_exp_in_transcript_only.json
        from common import OUT, save_json
        sp = splits()
        write_transcript_baseline(sp["val"], INP / "exp_in_ens5grid_val_preds.parquet", INP / "exp_in_transcript_only_val")
        write_transcript_baseline(sp["test"], TR / "exp_tr_ens5grid_test_preds.parquet", INP / "exp_in_transcript_only_test")
        # every event above the transcript's threshold is a press already: thr only cuts by score
        best = max(((evaluate(INP / "exp_in_transcript_only_val", sp["val"], t, with_cmd=False)["presses"]["k2"]["micro"]["f1"], t)
                    for t in (0.1, 0.65, 0.75, 0.85)))
        val = evaluate(INP / "exp_in_transcript_only_val", sp["val"], best[1], preds_path=INP / "exp_in_ens5grid_val_preds.parquet")
        res = evaluate_subsets(INP / "exp_in_transcript_only_test", best[1], names=("old", "new", "all"))
        save_json(dict(name="transcript_only", thr=best[1], doc=write_transcript_baseline.__doc__, val=val, test=res),
                  OUT / "results_exp_in_transcript_only.json")
        print(headline({"val": val}))
        print(headline(res))
    else:
        sub = test_subsets(replay_fold_map())
        stems = sub["old"][:12]
        for jit, drop in ((0, 0.0), (2, 0.0), (0, 0.3)):
            d = INP / "debug" / f"selftest_j{jit}_d{int(drop * 10)}"
            write_label_preds(stems, d, jit, drop)
            res = {"old12": evaluate(d, stems, 0.5)}
            print(f"jitter {jit} drop {drop}:\n" + headline(res))


if __name__ == "__main__":
    main()
