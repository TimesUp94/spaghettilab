"""Shared helpers: dataset rows, task labels, replay-grouped folds, metrics."""
from __future__ import annotations

import collections
import ctypes
import json
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("GGST_DATA", r"G:\Projects\replanal-data")) / "ggst_dataset"
OUT = DATA / "model"
OFFS = list(range(-12, 1))


def low_priority():
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)   # below normal
    except Exception:
        pass


def key(r) -> str:
    return f"{r['replay']}/{r['log_row']}_{r['attacker']}"


def load_rows(with_crops=True) -> list[dict]:
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    if with_crops:
        rows = [r for r in rows if (DATA / "crops" / r["replay"] / f"{r['log_row']}_{r['attacker']}.jpg").exists()]
    return rows


# ---- task labels ------------------------------------------------------------------------

CAT6 = {"normal": "normal", "command normal": "normal", "special": "special", "super": "super",
        "throw": "throw", "projectile": "projectile", "burst": "other", "other": "other"}


def label_category(r):
    return CAT6[r["category"]]


def label_normal(r):
    """Button x stance for normals / command normals (None for everything else)."""
    if r["category"] not in ("normal", "command normal") or not r.get("button") or not r.get("stance"):
        return None
    return f"{r['button']}/{r['stance']}"


def label_hit_kind(r):
    """Counter hit (small or large), punish or neither (dataset build.mark_counter_hits); None
    when unknown (banner word unclear, REVERSAL, no video pass). The model frames end at the
    hit, before the banner shows: this is the situation (defender attacking, recovering or
    neutral), not the banner."""
    if r.get("counter_hit"):
        return "counter"
    if r.get("punish"):
        return "punish"
    return "none" if r.get("counter_hit") is False and r.get("punish") is False else None


def label_move(char, min_count=5):
    def f(r, _counts={}):
        if r["char"] != char:
            return None
        return r["move"]
    return f


def collapse_rare(labels, min_count=5, other="(rare)"):
    c = collections.Counter(l for l in labels if l is not None)
    return [None if l is None else (l if c[l] >= min_count else other) for l in labels]


def task_labels(rows, task):
    if task == "category":
        y = [label_category(r) for r in rows]
    elif task == "hit_kind":
        y = [label_hit_kind(r) for r in rows]
    elif task == "normal":
        y = collapse_rare([label_normal(r) for r in rows], 5)
    elif task.startswith("move:"):
        ch = task.split(":", 1)[1]
        y = collapse_rare([r["move"] if r["char"] == ch else None for r in rows], 5)
    else:
        raise ValueError(task)
    return y


# ---- folds --------------------------------------------------------------------------------

def replay_folds(rows, k=5, seed=0):
    """Fold id per row, grouping by replay (no replay in two folds). Replays are assigned
    greedily by size so folds are balanced."""
    rng = np.random.default_rng(seed)
    reps = collections.Counter(r["replay"] for r in rows)
    names = list(reps)
    rng.shuffle(names)
    names.sort(key=lambda n: -reps[n])
    load = [0] * k
    fold_of = {}
    for n in names:
        f = int(np.argmin(load))
        fold_of[n] = f
        load[f] += reps[n]
    return np.array([fold_of[r["replay"]] for r in rows])


# The video-only chain (openers, players, characters, moves) is measured end to end on the replays of
# TEST_FOLD; every stage trains on the other folds for that, and on everything for the final weights.
TEST_FOLD = 0


def replay_fold_map() -> dict[str, int]:
    """Replay -> fold over every opener row, so all video-only stages share one split."""
    rows = load_rows(with_crops=False)
    return dict(zip((r["replay"] for r in rows), (int(f) for f in replay_folds(rows))))


# ---- metrics ------------------------------------------------------------------------------

def report(y_true, y_pred, classes, proba=None, topk=(1, 3)):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {"n": int(len(y_true)), "accuracy": float(np.mean(y_true == y_pred))}
    per = {}
    for i, c in enumerate(classes):
        m = y_true == i
        tp = int(np.sum((y_pred == i) & m))
        per[c] = {"n": int(m.sum()), "recall": tp / m.sum() if m.sum() else None,
                  "precision": tp / max(1, int(np.sum(y_pred == i)))}
    out["per_class"] = per
    rec = [v["recall"] for v in per.values() if v["recall"] is not None]
    out["balanced_accuracy"] = float(np.mean(rec)) if rec else None
    if proba is not None:
        order = np.argsort(-proba, 1)
        for k in topk:
            out[f"top{k}"] = float(np.mean([y_true[i] in order[i, :k] for i in range(len(y_true))]))
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    out["confusion"] = cm.tolist()
    out["classes"] = list(classes)
    # biggest off-diagonal confusions
    off = [(int(cm[i, j]), classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
    out["top_confusions"] = sorted(off, reverse=True)[:10]
    return out


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
