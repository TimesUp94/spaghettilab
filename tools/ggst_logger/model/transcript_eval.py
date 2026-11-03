"""Full-match transcript (ROADMAP point 2): task taxonomy, splits and the action-spotting metric.

Labels: dataset/transcript_labels.py -> ggst_dataset/transcript/transcript_events.parquet (one row
per move start of either player, 60 fps video frames).

Taxonomy (per event; a model predicts events per player side = P1 / P2 of the video):
- `category`: CAT6 of the move itself (normal, special, super, throw, other = burst / Roman
  Cancel / misc). `projectile` never labels a move start: it is a property of a contact (an
  object of an earlier special hit), kept in the opener columns; a fireball's start is `special`
  with `proj` True.
- `normal`: button/stance (P/K/S/H/D x stand/crouch/air) for normals and command normals; None
  for everything else (not scored at this level).
- `move`: `char:move` (openers.parquet naming), moves with < MIN_COUNT starts in the training
  replays collapsed into `char:(rare)` (like video_moves.vocab, which uses 5).
- `outcome`: counter (a hit whose linked opener shows a counter hit), hit, block, whiff,
  interrupted (the mover was hit during the move before any contact of its own: stuffed,
  counter hit, whiff punished), teched (throws), none (Roman Cancel). `counter_hit` None (banner
  unclear) counts as hit.

Splits: the frozen replay folds (common.replay_fold_map: fold 0 = test, reported for its `old`
and `new` subsets); VAL_FRAC of the other replays (hash of the name, stable as the dataset grows)
is the validation split for tuning; the rest trains.

Metric (action spotting, per level L and tolerance k in TOLS video frames at 60 fps):
- within one replay and one side, a prediction (frame, class) matches an unmatched true event of
  the same class with |frame - true start_frame| <= k; pairs are taken greedily by smallest
  distance (higher score first on ties). TP = matched pairs; precision = TP / predictions,
  recall = TP / true events, F1.
- micro: pooled over classes; macro: mean over the classes with >= 1 true event in the scored
  set (recall/F1) or >= 1 prediction (precision); classes are those of the level (for `move`,
  the training vocabulary incl. the `(rare)` buckets; any move outside it, true or predicted,
  counts as its character's `(rare)`).
- with scores, average precision per class (precision-recall over the score ranking, matched
  at k) and its mean (mAP), as in SoccerNet / ActionFormer spotting.
- `outcome_given_category`: share of the category-level matches (k) whose predicted outcome is
  right (outcome as an attribute of a found move, besides outcome-as-class spotting).
- `openers_recovered@k`: share of the test openers (openers.parquet rows, linked to their dense
  event) whose event is matched at the move level within k: the dense model's counterpart of the
  current chain's "all openers found with the right move" (which is scored at the hit frame;
  here the anchor is the move start, so the two are close but not identical).

Usage:
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_eval.py stats      # class counts, splits
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_eval.py selftest   # metric on jittered / dropped truth
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_eval.py score preds.parquet [--subset old]
preds.parquet: replay, side, start_frame, category, [normal], [char_move], [outcome], [score].
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, TEST_FOLD, replay_fold_map, test_subsets  # noqa: E402

TR = DATA / "transcript"
EVENTS = TR / "transcript_events.parquet"
TOLS = (2, 4, 8)
MIN_COUNT = 5
VAL_FRAC = 0.1
LEVELS = ("category", "normal", "move", "outcome")


def load_events():
    import pyarrow.parquet as pq
    return pq.read_table(EVENTS).to_pylist()


def outcome_label(r) -> str:
    if r["outcome"] == "hit" and r.get("counter_hit"):
        return "counter"
    return r["outcome"]


def splits(replays) -> dict[str, str]:
    """replay -> 'train' | 'val' | 'test' (test = frozen fold 0; val = VAL_FRAC of the rest by name hash)."""
    fold = replay_fold_map()
    out = {}
    for s in replays:
        if fold.get(s, -1) == TEST_FOLD:
            out[s] = "test"
        else:
            h = int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 2 ** 32
            out[s] = "val" if h < VAL_FRAC else "train"
    return out


def move_vocab(train_rows, min_count=MIN_COUNT) -> list[str]:
    c = collections.Counter(r["char_move"] for r in train_rows)
    keep = {m for m, n in c.items() if n >= min_count}
    return sorted(keep | {f"{r['char']}:(rare)" for r in train_rows})


def label(r, level, vocab=None):
    if level == "category":
        return r["category"]
    if level == "normal":
        return r.get("normal")
    if level == "outcome":
        return outcome_label(r) if "counter_hit" in r else r.get("outcome")
    if level == "move":
        m = r.get("char_move")
        if m is None:
            return None
        if vocab is not None and m not in vocab:
            return m.split(":", 1)[0] + ":(rare)"
        return m
    raise ValueError(level)


def match(true, pred, k):
    """Greedy one-to-one matching of (frame) lists within k; returns [(ti, pi)]. pred entries:
    (frame, score)."""
    pairs = []
    for i, t in enumerate(true):
        for j, (p, s) in enumerate(pred):
            d = abs(p - t)
            if d <= k:
                pairs.append((d, -s, i, j))
    pairs.sort()
    used_t, used_p, out = set(), set(), []
    for _, _, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        out.append((i, j))
    return out


def average_precision(scores, hits, n_true) -> float | None:
    if n_true == 0:
        return None
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    h = np.asarray(hits, bool)[order]
    tp = np.cumsum(h)
    prec = tp / np.arange(1, len(h) + 1)
    return float(np.sum(prec[h]) / n_true) if len(h) else 0.0


def score_level(truth, preds, level, k, vocab=None) -> dict:
    """truth / preds: lists of dicts with replay, side, start_frame, the level's label (+ score)."""
    T = collections.defaultdict(list)
    P = collections.defaultdict(list)
    for r in truth:
        c = label(r, level, vocab)
        if c is not None:
            T[(r["replay"], r["side"], c)].append(r["start_frame"])
    for r in preds:
        c = label(r, level, vocab)
        if c is not None:
            P[(r["replay"], r["side"], c)].append((r["start_frame"], float(r.get("score", 1.0))))
    per = collections.defaultdict(lambda: dict(tp=0, n_true=0, n_pred=0, scores=[], hits=[]))
    for key in set(T) | set(P):
        c = key[2]
        t, p = T.get(key, []), P.get(key, [])
        m = match(t, p, k)
        hit_p = {j for _, j in m}
        d = per[c]
        d["tp"] += len(m)
        d["n_true"] += len(t)
        d["n_pred"] += len(p)
        d["scores"] += [s for _, s in p]
        d["hits"] += [j in hit_p for j in range(len(p))]
    tp = sum(d["tp"] for d in per.values())
    nt = sum(d["n_true"] for d in per.values())
    npred = sum(d["n_pred"] for d in per.values())
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    mp, mr = tp / max(1, npred), tp / max(1, nt)
    cls = {}
    for c, d in per.items():
        p = d["tp"] / d["n_pred"] if d["n_pred"] else None
        r = d["tp"] / d["n_true"] if d["n_true"] else None
        cls[c] = dict(n_true=d["n_true"], n_pred=d["n_pred"], tp=d["tp"], precision=p, recall=r,
                      f1=f1(p or 0, r or 0) if d["n_true"] else None,
                      ap=average_precision(d["scores"], d["hits"], d["n_true"]))
    have = [v for v in cls.values() if v["n_true"]]
    return dict(level=level, k=k, n_true=nt, n_pred=npred, tp=tp,
                micro=dict(precision=mp, recall=mr, f1=f1(mp, mr)),
                macro=dict(precision=float(np.mean([v["precision"] for v in cls.values() if v["precision"] is not None])) if cls else None,
                           recall=float(np.mean([v["recall"] for v in have])) if have else None,
                           f1=float(np.mean([v["f1"] for v in have])) if have else None,
                           mAP=float(np.mean([v["ap"] for v in have])) if have else None,
                           n_classes=len(have)),
                per_class=cls)


def outcome_given_category(truth, preds, k) -> dict:
    T = collections.defaultdict(list)
    P = collections.defaultdict(list)
    for r in truth:
        T[(r["replay"], r["side"], r["category"])].append(r)
    for r in preds:
        P[(r["replay"], r["side"], r["category"])].append(r)
    n = right = 0
    for key, t in T.items():
        p = P.get(key, [])
        for i, j in match([x["start_frame"] for x in t], [(x["start_frame"], float(x.get("score", 1.0))) for x in p], k):
            if p[j].get("outcome") is None:
                continue
            n += 1
            right += label(t[i], "outcome") == label(p[j], "outcome")
    return dict(matched=n, accuracy=right / n if n else None)


def openers_recovered(truth, preds, k, vocab) -> dict:
    """Test openers whose dense event is matched at the move level within k."""
    T = collections.defaultdict(list)
    P = collections.defaultdict(list)
    for r in truth:
        T[(r["replay"], r["side"], label(r, "move", vocab))].append(r)
    for r in preds:
        P[(r["replay"], r["side"], label(r, "move", vocab))].append(r)
    n = sum(r["is_opener"] for r in truth)
    got = 0
    for key, t in T.items():
        p = P.get(key, [])
        for i, _ in match([x["start_frame"] for x in t], [(x["start_frame"], float(x.get("score", 1.0))) for x in p], k):
            got += bool(t[i]["is_opener"])
    return dict(openers=n, recovered=got, frac=got / n if n else None)


def evaluate(truth, preds, vocab) -> dict:
    out = {}
    for k in TOLS:
        out[f"k{k}"] = {lv: {kk: v for kk, v in score_level(truth, preds, lv, k, vocab).items() if kk != "per_class"}
                        for lv in LEVELS}
        out[f"k{k}"]["outcome_given_category"] = outcome_given_category(truth, preds, k)
        out[f"k{k}"]["openers_recovered"] = openers_recovered(truth, preds, k, vocab)
    return out


def subsets(rows):
    fold = replay_fold_map()
    sub = test_subsets(fold)
    return {name: [r for r in rows if r["replay"] in set(sub[name])] for name in ("all", "old", "new")}


def cmd_stats():
    rows = load_events()
    sp = splits({r["replay"] for r in rows})
    tr = [r for r in rows if sp[r["replay"]] == "train"]
    voc = move_vocab(tr)
    c = collections.Counter(r["char_move"] for r in tr)
    sub = subsets([r for r in rows if sp[r["replay"]] == "test"])
    res = dict(
        replays={s: sum(v == s for v in sp.values()) for s in ("train", "val", "test")},
        test_subsets={k: dict(replays=len({r["replay"] for r in v}), events=len(v),
                              openers=sum(r["is_opener"] for r in v)) for k, v in sub.items()},
        events={s: sum(sp[r["replay"]] == s for r in rows) for s in ("train", "val", "test")},
        levels=dict(
            category=dict(collections.Counter(r["category"] for r in tr)),
            normal=dict(collections.Counter(r["normal"] for r in tr if r["normal"])),
            outcome=dict(collections.Counter(outcome_label(r) for r in tr)),
            move=dict(classes_raw=len(c), vocab=len(voc), ge5=sum(n >= 5 for n in c.values()),
                      ge20=sum(n >= 20 for n in c.values()), ge100=sum(n >= 100 for n in c.values()),
                      rare_events=sum(n for m, n in c.items() if n < MIN_COUNT))),
    )
    (TR / "transcript_splits.json").write_text(json.dumps(sp, indent=0))
    (TR / "transcript_taxonomy.json").write_text(json.dumps(dict(res, move_vocab=voc), indent=1))
    print(json.dumps(res, indent=1))


def cmd_selftest():
    """The metric on perfect, jittered and thinned copies of the test truth."""
    rows = load_events()
    sp = splits({r["replay"] for r in rows})
    voc = move_vocab([r for r in rows if sp[r["replay"]] == "train"])
    test = [r for r in rows if sp[r["replay"]] == "test"]
    rng = np.random.default_rng(0)
    out = {}
    for name, jit, keep in (("perfect", 0, 1.0), ("jitter3", 3, 1.0), ("jitter3_keep80", 3, 0.8)):
        preds = [dict(r, start_frame=r["start_frame"] + int(rng.integers(-jit, jit + 1)), score=float(rng.random()))
                 for r in test if rng.random() < keep]
        ev = evaluate(test, preds, voc)
        out[name] = {k: {lv: (v["micro"]["f1"], v["macro"]["f1"]) if isinstance(v, dict) and "micro" in v else v
                         for lv, v in d.items()} for k, d in ev.items()}
    print(json.dumps(out, indent=1, default=float))


def cmd_score(path, subset):
    import pyarrow.parquet as pq
    rows = load_events()
    sp = splits({r["replay"] for r in rows})
    voc = move_vocab([r for r in rows if sp[r["replay"]] == "train"])
    test = subsets([r for r in rows if sp[r["replay"]] == "test"])[subset]
    reps = {r["replay"] for r in test}
    preds = [r for r in pq.read_table(path).to_pylist() if r["replay"] in reps]
    print(json.dumps(evaluate(test, preds, voc), indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stats", "selftest", "score"))
    ap.add_argument("preds", nargs="?")
    ap.add_argument("--subset", default="all", choices=("all", "old", "new"))
    a = ap.parse_args()
    if a.cmd == "stats":
        cmd_stats()
    elif a.cmd == "selftest":
        cmd_selftest()
    else:
        cmd_score(a.preds, a.subset)


if __name__ == "__main__":
    main()
