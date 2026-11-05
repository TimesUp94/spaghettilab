"""Shared by the measurement pass (exp_measure_*.py): per-replay counts of the opener chain and a paired
bootstrap over replays. Nothing here trains or changes a model; results go to model/results_exp_measure_*.json,
logs and temporary files under <data>/ggst_dataset/measure/.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, label_category  # noqa: E402

MEASURE = DATA / "measure"


def opener_counts(items, P, voc, rows, stems):
    """Per replay, what video_moves.end_to_end counts: n openers (bursts apart), det found, chain = found + kept +
    right move, move = right move given found, cat = right category given found (the 'none' class excluded)."""
    st = set(stems)
    cat_none = voc["category"].index("none")
    found = {}
    for i, it in enumerate(items):
        if it["row"] is not None:
            found[f"{it['row']['replay']}/{it['row']['log_row']}_{it['row']['attacker']}"] = i
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["replay"] not in st or r["is_burst"]:
            continue
        c = out[r["replay"]]
        c["n"] += 1
        i = found.get(f"{r['replay']}/{r['log_row']}_{r['attacker']}")
        if i is None:
            continue
        c["det"] += 1
        lab = f"{r['char']}:{r['move']}"
        if lab not in voc["move"]:
            lab = f"{r['char']}:(rare)"
        ok = voc["move"][int(P["move"][i].argmax())] == lab
        c["move"] += bool(ok)
        c["chain"] += bool(ok and P["category"][i].argmax() != cat_none)
        pc = P["category"][i].copy()
        pc[cat_none] = 0
        c["cat"] += voc["category"][int(pc.argmax())] == label_category(r)
    return {s: dict(v) for s, v in out.items()}


def ratio(counts, num, den, stems=None):
    stems = list(counts) if stems is None else [s for s in stems if s in counts]
    return sum(counts[s].get(num, 0) for s in stems) / max(1, sum(counts[s].get(den, 0) for s in stems))


def paired_bootstrap(a, b, num, den, stems=None, n=4000, seed=0):
    """b - a of sum(num) / sum(den), resampling replays: {a, b, diff, lo, hi} (95 %)."""
    stems = sorted(set(a) & set(b)) if stems is None else sorted(s for s in stems if s in a and s in b)
    A = np.array([[a[s].get(num, 0), a[s].get(den, 0)] for s in stems], float)
    B = np.array([[b[s].get(num, 0), b[s].get(den, 0)] for s in stems], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stems), (n, len(stems)))
    ra = A[idx, 0].sum(1) / np.maximum(A[idx, 1].sum(1), 1)
    rb = B[idx, 0].sum(1) / np.maximum(B[idx, 1].sum(1), 1)
    d = rb - ra
    va, vb = A[:, 0].sum() / max(A[:, 1].sum(), 1), B[:, 0].sum() / max(B[:, 1].sum(), 1)
    return dict(replays=len(stems), a=round(float(va), 4), b=round(float(vb), 4), diff=round(float(vb - va), 4),
                lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4))
