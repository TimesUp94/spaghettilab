"""Searchable event index over the match transcript (ROADMAP point 4, first part), its evaluation
against the same queries over the logged events, and per-character habit groups.

Index (`build`): one row per event of either side of a replay, from
- `pred`: the transcript's predictions (video only; default transcript/exp_tr_ens5grid_test_preds.parquet),
  with `opener` = a combo start of the opener detector (model/detections.json, cross-fitted: each
  replay's detections come from the fold net that did not train on it) of the same side within
  [start - 4, start + 40] frames, and optionally the state tower's per-tubelet state reading
  (`--state <exp_se name>`: search/exp_se_<name>_pred/, knockdown / hitstun ... per side);
- `log`: the logged events (transcript_events.parquet, in_video) with the same fields, moves outside
  the transcript's vocabulary folded into char:(rare) like transcript_eval, `opener` = is_opener,
  `punish` from the log, and the logged state per tubelet (search/state/, state_embed.py labels).
Columns: replay, side, frame (60 fps video frame of the move start), time_s, char, opp_char, category,
move (char_move without the char), char_move, normal, outcome (counter / hit / block / whiff /
interrupted / teched / none), score, opener, punish (log only), the previous / next event of the
same side and of the opponent (move, outcome, frame gap), own / opponent state class at the start
and whether this is the side's first event after its own knockdown ended (`after_kd`, frames since
wake-up) -> search/search_index_<source>_<split>.parquet.

Query helper (`query()` / CLI `query`): filters on any column (`char`, `opp_char`, `move` ("5H" or
"Sol:5H"), `outcome` (one or a list, `connected` = hit or counter), `category`, `side`, `opener`),
plus `after=` (an event filter the previous opponent / own event must satisfy within `within`
frames) and `after_kd` (first action after wake-up). E.g.
    query(ix, char="Sol", move="5H", outcome="whiff")                       # every whiffed 5H of Sol
    query(ix, char="Ky", opp_char="Sol", outcome="counter")                 # every time Sol got counter-hit by Ky
    query(ix, side="P1", after_kd=True)                                     # what P1 does after a knockdown
    query(ix, char="Sol", after=dict(who="own", outcome="block", within=40))  # Sol's next move after a blocked one

Evaluation (`eval`): query families generated from the test fold's logged events (every key with
>= MIN_N logged results); each query runs on the predicted and on the logged index; a returned
moment matches a logged one of the same replay and side within +-K frames (one-to-one, greedy by
|dt|). Per family: micro precision / recall / F1 over all returned moments, mean per-query F1, the
median relative error of the result count (|n_pred - n_log| / n_log, what a stats page would show)
-> model/results_exp_se_index[_<state>].json. Families:
- move_outcome: (char, move, outcome in whiff / block / connected / counter)  "every whiffed 5H of Sol";
- move: (char, move);
- counter_by: (victim char, attacker char), attacker events with outcome counter;
- opener: (char, move) of combo openers;
- after_block: (char), the side's next own event within 40 frames after one of its moves was blocked;
- after_kd: (char), the side's first own event within 60 frames of waking up from a knockdown (needs
  the state reading for `pred`).
Habits (`habits`): per character in the test fold, groups of what follows a knockdown (every wake-up
of the state stream: the waking side's first own event within [-8, 45] frames of it, else 'no move') and a blocked string (>= 2 consecutive own
blocked moves: the attacker's next action within 60 frames, else 'stops'), by category then
move; predicted vs logged shares, total variation distance, and example moments -> model/
results_exp_se_habits.json.

    .venv/Scripts/python.exe tools/ggst_logger/model/search_index.py build [--state clip_sup]
    .venv/Scripts/python.exe tools/ggst_logger/model/search_index.py query --char Sol --move 5H --outcome whiff [--source log]
    .venv/Scripts/python.exe tools/ggst_logger/model/search_index.py eval [--state clip_sup]
    .venv/Scripts/python.exe tools/ggst_logger/model/search_index.py habits --state clip_sup
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, replay_fold_map, save_json, test_subsets  # noqa: E402
import transcript_eval as te  # noqa: E402

TR = DATA / "transcript"
SE = DATA / "search"
PRED = TR / "exp_tr_ens5grid_test_preds.parquet"
FPS = 60.0
STEP = 8
K = 8
MIN_N = 5
CLS = ["neutral", "attacking", "blockstun", "hitstun", "knockdown", "airborne"]
FAMILIES = ("move_outcome", "move", "counter_by", "opener", "after_block", "after_kd")


# ---- build -------------------------------------------------------------------------------------

def _state_runs(cls_side):
    """Wake-up frames: the video frame after each run (>= 2 tubelets) of knockdown ends, when the side gets up
    (the next tubelet is not hitstun / knockdown: an OTG hit or a re-launch is not a wake-up)."""
    cls_side = np.asarray(cls_side)
    kd = cls_side == 4
    out = []
    i, n = 0, len(kd)
    while i < n:
        if kd[i]:
            j = i
            while j < n and kd[j]:
                j += 1
            if j - i >= 2 and j < n and cls_side[j] != 3:
                out.append(STEP * j)
            i = j
        else:
            i += 1
    return out


def state_streams(source, stems, state_name=None):
    """{stem: cls [2, n] int} per tubelet: the log's (source log) or the state tower's argmax (pred)."""
    out = {}
    for st in stems:
        if source == "log":
            p = SE / "state" / f"{st}.npz"
            if p.exists():
                z = np.load(p)
                c = z["cls"].astype(np.int64)
                c[:, ~z["live"]] = 0
                out[st] = c
        elif state_name:
            p = SE / f"exp_se_{state_name}_pred" / f"{st}.npz"
            if p.exists():
                c = np.load(p)["cls_p"].astype(np.float32)
                # 3-tap median-ish smoothing of the probabilities, then argmax
                cs = c.copy()
                cs[:, 1:-1] = (c[:, :-2] + c[:, 1:-1] + c[:, 2:]) / 3
                out[st] = cs.argmax(-1)
    return out


def _link(rows, states):
    """Adds prev / next own and opponent events, the state at the start and after_kd to rows of one replay."""
    by = {s: sorted([r for r in rows if r["side"] == s], key=lambda r: r["frame"]) for s in ("P1", "P2")}
    for s, o in (("P1", "P2"), ("P2", "P1")):
        own, opp = by[s], by[o]
        of = np.array([r["frame"] for r in opp]) if opp else np.zeros(0, int)
        k = 0 if s == "P1" else 1
        cls = states[k] if states is not None else None
        wake = _state_runs(cls) if cls is not None else []
        wi = 0
        for i, r in enumerate(own):
            p, nx = (own[i - 1] if i else None), (own[i + 1] if i + 1 < len(own) else None)
            r["prev_own"], r["prev_own_outcome"], r["prev_own_dt"] = (p["move"], p["outcome"], r["frame"] - p["frame"]) if p else (None, None, None)
            r["next_own"], r["next_own_outcome"], r["next_own_dt"] = (nx["move"], nx["outcome"], nx["frame"] - r["frame"]) if nx else (None, None, None)
            j = int(np.searchsorted(of, r["frame"], side="left"))
            po, no = (opp[j - 1] if j else None), (opp[j] if j < len(opp) else None)
            r["prev_opp"], r["prev_opp_outcome"], r["prev_opp_dt"] = (po["move"], po["outcome"], r["frame"] - po["frame"]) if po else (None, None, None)
            r["next_opp"], r["next_opp_outcome"], r["next_opp_dt"] = (no["move"], no["outcome"], no["frame"] - r["frame"]) if no else (None, None, None)
            t = min(r["frame"] // STEP, (cls.shape[0] if cls is not None else 1) - 1)
            r["state_own"] = CLS[int(cls[t])] if cls is not None and t >= 0 else None
            r["state_opp"] = CLS[int(states[1 - k][t])] if cls is not None and t >= 0 else None
            # first own event after a wake-up (within 60 frames, none of this side's events in between)
            r["after_kd"] = None
            while wi < len(wake) and wake[wi] < r["frame"] - 60:
                wi += 1
            if wi < len(wake) and wake[wi] - 8 <= r["frame"] <= wake[wi] + 60 and (p is None or p["frame"] < wake[wi] - 8):
                r["after_kd"] = int(r["frame"] - wake[wi])
                wi += 1
        for r in own:
            r.setdefault("after_kd", None)
    return [r for s in ("P1", "P2") for r in by[s]]


def build(source, stems, state_name=None, pred_path=PRED):
    import pyarrow.parquet as pq
    stems = set(stems)
    rows = collections.defaultdict(list)
    if source == "log":
        sp = json.loads((TR / "transcript_splits.json").read_text())
        vocab = set(te.move_vocab([r for r in te.load_events() if sp.get(r["replay"]) == "train"]))
        for e in te.load_events():
            if e["replay"] not in stems or not e["in_video"]:
                continue
            cm = te.label(e, "move", vocab)
            rows[e["replay"]].append(dict(replay=e["replay"], side=e["side"], frame=int(e["start_frame"]), char=e["char"], opp_char=e["opp_char"],
                                          category=e["category"], char_move=cm, move=cm.split(":", 1)[1], normal=e.get("normal"),
                                          outcome=te.outcome_label(e), score=1.0, opener=bool(e["is_opener"]), punish=bool(e.get("punish"))))
    else:
        dets = json.loads((OUT / "detections.json").read_text())
        P = pq.read_table(pred_path).to_pylist()
        chars = collections.defaultdict(collections.Counter)
        for p in P:
            chars[(p["replay"], p["side"])][p["char"]] += 1
        for p in P:
            if p["replay"] not in stems:
                continue
            o = "P2" if p["side"] == "P1" else "P1"
            oc = chars[(p["replay"], o)].most_common(1)[0][0] if chars[(p["replay"], o)] else None
            cm = p["char_move"] or f"{p['char']}:(rare)"
            rows[p["replay"]].append(dict(replay=p["replay"], side=p["side"], frame=int(p["start_frame"]), char=p["char"], opp_char=oc,
                                          category=p["category"], char_move=cm, move=cm.split(":", 1)[1], normal=p.get("normal"),
                                          outcome=p["outcome"], score=float(p["score"]), opener=False, punish=None))
        # openers: a detection of the same side within [start - 4, start + 40]; each detection labels one event (the nearest
        # start before it)
        for st, rs in rows.items():
            for d in dets.get(st, []):
                if d["score"] < 0.1:
                    continue
                s = f"P{d['side']}"
                cand = [r for r in rs if r["side"] == s and -4 <= d["frame"] - r["frame"] <= 40]
                if cand:
                    min(cand, key=lambda r: (d["frame"] - r["frame"]) if d["frame"] >= r["frame"] else 99)["opener"] = True
    states = state_streams(source, stems, state_name)
    out = []
    for st in sorted(rows):
        out += _link(rows[st], states.get(st))
    for r in out:
        r["time_s"] = round(r["frame"] / FPS, 2)
    return out


def save_index(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def load_index(source="pred", split="test", state=None):
    import pyarrow.parquet as pq
    p = SE / f"search_index_{source}_{split}{'_' + state if state and source == 'pred' else ''}.parquet"
    return pq.read_table(p).to_pylist()


# ---- query -------------------------------------------------------------------------------------

def _ok(r, k, v):
    if v is None:
        return True
    x = r.get(k)
    if k == "move" and isinstance(v, str) and ":" in v:
        return r["char_move"] == v
    if k == "outcome":
        vs = {v} if isinstance(v, str) else set(v)
        if "connected" in vs:
            vs |= {"hit", "counter"}
        return x in vs
    if isinstance(v, (list, tuple, set)):
        return x in v
    return x == v


def query(ix, after=None, after_kd=None, **filt):
    """Events matching every filter. after=dict(who='own'|'opp', within=frames, **event filter on that previous
    event's move / outcome); after_kd=True: the side's first event after waking up from a knockdown."""
    out = []
    for r in ix:
        if not all(_ok(r, k, v) for k, v in filt.items()):
            continue
        if after_kd and r.get("after_kd") is None:
            continue
        if after:
            who = after.get("who", "opp")
            dt = r.get(f"prev_{who}_dt")
            if dt is None or dt > after.get("within", 40):
                continue
            if "move" in after and r.get(f"prev_{who}") != after["move"]:
                continue
            if "outcome" in after and not _ok({"outcome": r.get(f"prev_{who}_outcome")}, "outcome", after["outcome"]):
                continue
        out.append(r)
    return out


def fmt(r):
    t = r["time_s"]
    return (f"{r['replay']} {r['side']} {int(t // 60)}:{t % 60:05.2f} {r['char']} {r['move']} [{r['category']}, {r['outcome']}]"
            f"{' opener' if r.get('opener') else ''}  prev opp: {r.get('prev_opp')} ({r.get('prev_opp_outcome')}, {r.get('prev_opp_dt')}f)"
            f"  next own: {r.get('next_own')} ({r.get('next_own_dt')}f)")


# ---- evaluation --------------------------------------------------------------------------------

def match_moments(true, pred, k=K):
    """true / pred: lists of (replay, side, frame); one-to-one within k, greedy by |dt|. Returns TP."""
    T = collections.defaultdict(list)
    Pm = collections.defaultdict(list)
    for t in true:
        T[t[:2]].append(t[2])
    for p in pred:
        Pm[p[:2]].append(p[2])
    tp = 0
    for key, ps in Pm.items():
        ts = T.get(key)
        if ts:
            tp += len(te.match(ts, [(p, 0.0) for p in ps], k))
    return tp


def family_queries(fam, logix):
    """(query name, kwargs) with >= MIN_N logged results."""
    c = collections.Counter()
    for r in logix:
        if r["move"] == "(rare)" and fam in ("move_outcome", "move", "opener"):
            continue
        if fam == "move_outcome":
            for oc in ("whiff", "block", "connected", "counter"):
                if _ok(r, "outcome", oc):
                    c[(r["char"], r["char_move"], oc)] += 1
        elif fam == "move":
            c[(r["char"], r["char_move"])] += 1
        elif fam == "counter_by" and r["outcome"] == "counter":
            c[(r["opp_char"], r["char"])] += 1
        elif fam == "opener" and r["opener"]:
            c[(r["char"], r["char_move"])] += 1
        elif fam == "after_block" and r.get("prev_own_outcome") == "block" and (r.get("prev_own_dt") or 99) <= 40:
            c[(r["char"],)] += 1
        elif fam == "after_kd" and r.get("after_kd") is not None:
            c[(r["char"],)] += 1
    out = []
    for key, n in c.items():
        if n < MIN_N:
            continue
        if fam == "move_outcome":
            out.append((f"{key[1]} {key[2]}", dict(char=key[0], move=key[1], outcome=key[2])))
        elif fam == "move":
            out.append((key[1], dict(char=key[0], move=key[1])))
        elif fam == "counter_by":
            out.append((f"{key[0]} counter-hit by {key[1]}", dict(opp_char=key[0], char=key[1], outcome="counter")))
        elif fam == "opener":
            out.append((f"{key[1]} opener", dict(char=key[0], move=key[1], opener=True)))
        elif fam == "after_block":
            out.append((f"{key[0]} after own blocked move", dict(char=key[0], after=dict(who="own", outcome="block", within=40))))
        elif fam == "after_kd":
            out.append((f"{key[0]} after knockdown", dict(char=key[0], after_kd=True)))
    return sorted(out)


def eval_family(fam, predix, logix, k=K):
    qs = family_queries(fam, logix)
    tp = npred = ntrue = 0
    f1s, cerr, per = [], [], []
    by_l, by_p = collections.defaultdict(list), collections.defaultdict(list)   # every query filters on char
    for r in logix:
        by_l[r["char"]].append(r)
    for r in predix:
        by_p[r["char"]].append(r)
    sub = collections.defaultdict(lambda: [0, 0, 0])
    for name, kw in qs:
        T = [(r["replay"], r["side"], r["frame"]) for r in query(by_l[kw["char"]], **kw)]
        Pr = [(r["replay"], r["side"], r["frame"]) for r in query(by_p[kw["char"]], **kw)]
        m = match_moments(T, Pr, k)
        tp, npred, ntrue = tp + m, npred + len(Pr), ntrue + len(T)
        if fam == "move_outcome":
            g = sub[kw["outcome"]]
            g[0], g[1], g[2] = g[0] + m, g[1] + len(Pr), g[2] + len(T)
        p = m / len(Pr) if Pr else 0.0
        rc = m / len(T)
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        f1s.append(f)
        cerr.append(abs(len(Pr) - len(T)) / len(T))
        per.append((name, len(T), len(Pr), round(f, 3)))
    if not qs:
        return dict(queries=0)
    P, R = tp / max(1, npred), tp / max(1, ntrue)
    by_outcome = {o: dict(precision=g[0] / max(1, g[1]), recall=g[0] / max(1, g[2]), f1=2 * g[0] / max(1, g[1] + g[2]), true=g[2])
                  for o, g in sub.items()}
    return dict(queries=len(qs), by_outcome=by_outcome or None, true_moments=ntrue, pred_moments=npred, precision=P, recall=R, f1=2 * P * R / max(1e-9, P + R),
                mean_query_f1=float(np.mean(f1s)), share_queries_f1_ge_0p8=float(np.mean(np.array(f1s) >= 0.8)),
                median_count_rel_err=float(np.median(cerr)),
                examples=sorted(per, key=lambda x: -x[1])[:8], worst=sorted([x for x in per if x[1] >= 15], key=lambda x: x[3])[:5])


def cmd_build(a):
    fold = replay_fold_map()
    sub = test_subsets(fold)
    for source in ("log", "pred"):
        rows = build(source, sub["all"], a.state)
        p = SE / f"search_index_{source}_test{'_' + a.state if a.state and source == 'pred' else ''}.parquet"
        save_index(rows, p)
        print(f"{source}: {len(rows)} events -> {p}")


def cmd_query(a):
    ix = load_index(a.source, "test", a.state)
    kw = {k: getattr(a, k) for k in ("char", "opp_char", "move", "category", "side") if getattr(a, k)}
    if a.outcome:
        kw["outcome"] = a.outcome.split(",")
    if a.opener:
        kw["opener"] = True
    after = None
    if a.after_outcome or a.after_move:
        after = dict(who=a.after_who, within=a.within)
        if a.after_outcome:
            after["outcome"] = a.after_outcome.split(",")
        if a.after_move:
            after["move"] = a.after_move
    res = query(ix, after=after, after_kd=a.after_kd or None, **kw)
    print(f"{len(res)} events")
    for r in res[:a.limit]:
        print("  " + fmt(r))
    if res:
        nx = collections.Counter(r["move"] for r in res)
        print("moves:", nx.most_common(10))


def cmd_eval(a):
    fold = replay_fold_map()
    sub = test_subsets(fold)
    res = dict(k=K, min_n=MIN_N, state=a.state, pred=str(PRED.name))
    logix = load_index("log", "test")
    predix = load_index("pred", "test", a.state)
    for subset in ("old", "new"):
        ss = set(sub[subset])
        L = [r for r in logix if r["replay"] in ss]
        P = [r for r in predix if r["replay"] in ss]
        res[subset] = {}
        for fam in FAMILIES:
            if fam == "after_kd" and not a.state:
                continue
            res[subset][fam] = {f"k{k}": eval_family(fam, P, L, k) for k in (4, 8, 15)}
            r8 = res[subset][fam]["k8"]
            if r8.get("queries"):
                print(f"{subset} {fam}: {r8['queries']} queries, P {r8['precision']:.3f} R {r8['recall']:.3f} F1 {r8['f1']:.3f} "
                      f"mean query F1 {r8['mean_query_f1']:.3f} count err {r8['median_count_rel_err']:.3f}  (k4 F1 {res[subset][fam]['k4']['f1']:.3f})")
    save_json(res, OUT / f"results_exp_se_index{'_' + a.state if a.state else ''}.json")


# ---- habits ------------------------------------------------------------------------------------

def habit_events(ix, kind, states=None):
    """(char, replay, side, frame, option category, option move) per situation. wakeup: every wake-up of the
    state streams ({stem: cls [2, n]}), option = the side's first own event within [-8, 45] frames, else 'no move'."""
    by = collections.defaultdict(list)
    for r in ix:
        by[(r["replay"], r["side"])].append(r)
    out = []
    if kind == "wakeup":
        chars = {(r["replay"], r["side"]): r["char"] for r in ix}
        for st, cls in (states or {}).items():
            for k, s in enumerate(("P1", "P2")):
                if (st, s) not in chars:
                    continue
                fr = np.array([r["frame"] for r in sorted(by[(st, s)], key=lambda r: r["frame"])])
                rs = sorted(by[(st, s)], key=lambda r: r["frame"])
                for w in _state_runs(cls[k]):
                    j = int(np.searchsorted(fr, w - 8))
                    nx = rs[j] if j < len(rs) and rs[j]["frame"] <= w + 45 else None
                    if nx:
                        out.append((chars[(st, s)], st, s, w, nx["category"], nx["move"]))
                        continue
                    seg = cls[k][w // STEP: (w + 45) // STEP + 1]   # no move: what the state stream shows instead
                    opt = "blocks" if (seg == 2).any() else "jumps" if (seg == 5).any() else "gets hit" if (seg == 3).any() else "waits"
                    out.append((chars[(st, s)], st, s, w, "no move", f"({opt})"))
        return out
    # blocked strings: >= 2 consecutive own events blocked, each within 40 frames of the previous; the next own action
    for (st, s), rs in by.items():
        rs.sort(key=lambda r: r["frame"])
        i = 0
        while i < len(rs):
            j = i
            while j < len(rs) and rs[j]["outcome"] == "block" and (j == i or rs[j]["frame"] - rs[j - 1]["frame"] <= 40):
                j += 1
            if j - i >= 2:
                last = rs[j - 1]
                nx = rs[j] if j < len(rs) and rs[j]["frame"] - last["frame"] <= 60 else None
                out.append((last["char"], st, s, last["frame"], nx["category"] if nx else "stops", nx["move"] if nx else "stops"))
                i = j
            else:
                i += 1
    return out


def _sit_match(L, P, k):
    """Situations (wake-ups / blocked-string ends) found: one-to-one match of predicted to logged moments (same replay,
    side) within k frames; precision, recall, and the option agreement on matched pairs."""
    T = collections.defaultdict(list)
    for i, x in enumerate(L):
        T[(x[1], x[2])].append(i)
    Pm = collections.defaultdict(list)
    for j, x in enumerate(P):
        Pm[(x[1], x[2])].append(j)
    tp = agree = 0
    for key, ps in Pm.items():
        ts = T.get(key, [])
        for ti, pj in te.match([L[i][3] for i in ts], [(P[j][3], 0.0) for j in ps], k):
            tp += 1
            agree += L[ts[ti]][5] == P[ps[pj]][5]
    return dict(precision=tp / max(1, len(P)), recall=tp / max(1, len(L)), option_agrees_when_matched=agree / max(1, tp))


def cmd_habits(a):
    fold = replay_fold_map()
    sub = test_subsets(fold)
    logix = load_index("log", "test")
    predix = load_index("pred", "test", a.state)
    res = dict(state=a.state, situations={})
    stems = sorted({r["replay"] for r in logix})
    st_log, st_pred = state_streams("log", stems), state_streams("pred", stems, a.state)
    for kind in ("wakeup", "blocked_string"):
        L, P = habit_events(logix, kind, st_log), habit_events(predix, kind, st_pred)
        per = {}
        tvs, tvs_cat, ws, floors = [], [], [], []
        chars = sorted({x[0] for x in L})
        for ch in chars:
            l = [x for x in L if x[0] == ch]
            p = [x for x in P if x[0] == ch]
            if len(l) < 8:
                continue
            lc, pc = collections.Counter(x[4] for x in l), collections.Counter(x[4] for x in p)
            lm, pm = collections.Counter(x[5] for x in l), collections.Counter(x[5] for x in p)
            tv = lambda A, B: 0.5 * sum(abs(A[k] / max(1, sum(A.values())) - B[k] / max(1, sum(B.values()))) for k in set(A) | set(B))
            t_cat, t_mv = tv(lc, pc), tv(lm, pm)
            tvs.append(t_mv)
            tvs_cat.append(t_cat)
            ws.append(len(l))
            ex = collections.defaultdict(list)
            for x in p:
                if len(ex[x[5]]) < 3:
                    t = x[3] / FPS
                    ex[x[5]].append(f"{x[1]} {x[2]} {int(t // 60)}:{t % 60:04.1f}")
            # noise floor: TV between the logged situations of two random halves of the character's replays
            reps = sorted({x[1] for x in l})
            fl = []
            for seed in range(20):
                h = set(np.random.default_rng(seed).permutation(reps)[:len(reps) // 2])
                A_, B_ = [x for x in l if x[1] in h], [x for x in l if x[1] not in h]
                if A_ and B_:
                    fl.append(tv(collections.Counter(x[5] for x in A_), collections.Counter(x[5] for x in B_)))
            floors.append(float(np.mean(fl)) if fl else np.nan)
            per[ch] = dict(n_log=len(l), n_pred=len(p), replays=len({x[1] for x in l}), tv_move_split_half_log=round(floors[-1], 3),
                           log_top=[(m, round(c / len(l), 2)) for m, c in lm.most_common(4)],
                           pred_top=[(m, round(c / max(1, len(p)), 2)) for m, c in pm.most_common(4)],
                           log_cat={k: round(v / len(l), 2) for k, v in lc.most_common()},
                           pred_cat={k: round(v / max(1, len(p)), 2) for k, v in pc.most_common()},
                           tv_category=round(t_cat, 3), tv_move=round(t_mv, 3),
                           top1_agrees=bool(lm and pm and lm.most_common(1)[0][0] == pm.most_common(1)[0][0]),
                           examples={m: ex[m] for m, _ in pm.most_common(3)})
        w = np.array(ws, float)
        res["situations"][kind] = dict(chars=len(per), log_situations=len(L), pred_situations=len(P),
                                       mean_tv_category=float(np.average(tvs_cat, weights=w)) if per else None,
                                       mean_tv_move=float(np.average(tvs, weights=w)) if per else None,
                                       top1_option_agrees=float(np.mean([v["top1_agrees"] for v in per.values()])) if per else None,
                                       mean_tv_move_split_half_log=float(np.nanmean(np.where(np.isnan(floors), np.nan, floors) * w) / np.mean(w[~np.isnan(floors)])) if per else None,
                                       situation_match_k16=_sit_match(L, P, 16),
                                       per_char=per)
        s = res["situations"][kind]
        print(f"{kind}: {s['log_situations']} logged / {s['pred_situations']} predicted situations, {s['chars']} chars with >= 8; "
              f"TV category {s['mean_tv_category']:.3f}, move {s['mean_tv_move']:.3f} (split-half floor of the log {s['mean_tv_move_split_half_log']:.3f}); "
              f"top option agrees {s['top1_option_agrees']:.2f}; situations found {s['situation_match_k16']}")
        for ch in list(per)[:40]:
            v = per[ch]
            print(f"  {ch:12s} n {v['n_log']:3d}/{v['n_pred']:3d} log {v['log_top'][:3]}  pred {v['pred_top'][:3]}")
    # old / new split of the overall TV
    for subset in ("old", "new"):
        ss = set(sub[subset])
        for kind in ("wakeup", "blocked_string"):
            L = habit_events([r for r in logix if r["replay"] in ss], kind, {k: v for k, v in st_log.items() if k in ss})
            P = habit_events([r for r in predix if r["replay"] in ss], kind, {k: v for k, v in st_pred.items() if k in ss})
            res["situations"][kind][f"{subset}_n_log_pred"] = [len(L), len(P)]
            lc, pc = collections.Counter(x[4] for x in L), collections.Counter(x[4] for x in P)
            res["situations"][kind][f"{subset}_overall_category_share"] = dict(
                log={k: round(v / max(1, len(L)), 3) for k, v in lc.most_common()}, pred={k: round(v / max(1, len(P)), 3) for k, v in pc.most_common()})
    save_json(res, OUT / f"results_exp_se_habits{'_' + a.state if a.state else ''}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--state")
    p = sp.add_parser("query")
    p.add_argument("--source", default="pred", choices=("pred", "log"))
    p.add_argument("--state")
    for k in ("char", "opp_char", "move", "category", "side", "outcome", "after_outcome", "after_move"):
        p.add_argument(f"--{k}")
    p.add_argument("--after_who", default="opp", choices=("own", "opp"))
    p.add_argument("--within", type=int, default=40)
    p.add_argument("--after_kd", action="store_true")
    p.add_argument("--opener", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p = sp.add_parser("eval")
    p.add_argument("--state")
    p = sp.add_parser("habits")
    p.add_argument("--state")
    a = ap.parse_args()
    {"build": cmd_build, "query": cmd_query, "eval": cmd_eval, "habits": cmd_habits}[a.cmd](a)


if __name__ == "__main__":
    main()
