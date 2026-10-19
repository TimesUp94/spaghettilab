"""Before/after health quality metrics on set 17 (two analysis DBs, in-round frames only).

Usage: python health_compare.py BEFORE.db AFTER.db [REPLAY_ID]
"""
from __future__ import annotations

import json
import sqlite3
import sys

import numpy as np

FPS = 30.0
ROUNDS_17 = [(12, 38), (42, 67), (72, 135), (148, 190), (194, 241), (248, 277), (289, 319), (324, 389)]
# Hand-checked moments from the baseline review (time window s, side, true HP range) -- see
# output/embed_test/health_baseline/m*_bars.png.  Pending (red) damage counts as lost HP.
CHECKS = [
    ("m1 P2 low sliver", (118.0, 119.5), 2, (0.02, 0.06)),
    ("m2 P1 survives on ~2%", (183.0, 183.6), 1, (0.0, 0.05)),
    ("m2 P2 ~0.55", (183.0, 183.6), 2, (0.50, 0.60)),
    ("m3 P2 under blade (fill 0.19, red to 0.35)", (29.5, 30.3), 2, (0.15, 0.22)),
    ("m4 P1 ~0.24 after flash", (349.6, 351.0), 1, (0.20, 0.28)),
    ("m6 P1 ~0.76 under slash VFX", (199.8, 202.4), 1, (0.72, 0.80)),
    ("m7 both ~0.2 visible (P1)", (374.0, 384.0), 1, (0.0, 0.28)),
    ("m7 both ~0.2 visible (P2)", (374.0, 384.0), 2, (0.17, 0.25)),
    ("m8 P2 static 0.68", (159.0, 161.0), 2, (0.64, 0.72)),
    ("m9 P1 ~0.14 after wallbreak", (264.5, 266.0), 1, (0.10, 0.18)),
]


def load(db, rid):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute("select frame_number,p1_health_pct,p2_health_pct from frame_data where replay_id=? "
                     "order by frame_number", (rid,)).fetchall()
    ev = c.execute("select frame_start,target_side,damage_pct from damage_events where replay_id=?", (rid,)).fetchall()
    n = rows[-1][0] + 1
    hp = np.full((n, 2), np.nan)
    for f, a, b in rows:
        hp[f] = [np.nan if a is None else a, np.nan if b is None else b]
    return hp, ev


def metrics(hp, ev):
    n = len(hp)
    inr = np.zeros(n, bool)
    rid = np.full(n, -1)
    for k, (a, b) in enumerate(ROUNDS_17):
        inr[int(a * FPS):int(b * FPS)] = True
        rid[int(a * FPS):int(b * FPS)] = k
    out = {}
    for s in (0, 1):
        x = hp[:, s]
        m = {}
        m["missing_pct"] = float(np.isnan(x[inr]).mean() * 100)
        valid = np.flatnonzero(inr & ~np.isnan(x))
        up = 0
        for i, j in zip(valid[:-1], valid[1:]):
            if rid[i] == rid[j] and j - i <= 90 and x[j] - x[i] > 0.05:
                up += 1
        m["upward_jumps"] = up
        # spikes: value differs from its round's centred 31-frame median by > 0.10
        sp = 0
        dips_recover = 0
        fz = 0
        for k in range(len(ROUNDS_17)):
            idx = np.flatnonzero((rid == k) & ~np.isnan(x))
            v = x[idx]
            for t in range(len(v)):
                lo, hi = max(0, t - 15), t + 16
                if abs(v[t] - np.median(v[lo:hi])) > 0.10:
                    sp += 1
            # false zero: <= 0.01 while a later reading in the same round is > 0.05
            later_max = np.maximum.accumulate(v[::-1])[::-1]
            fz += int(np.sum((v <= 0.01) & (np.concatenate([later_max[1:], [0]]) > 0.05)))
            # dips that recover: drop >= 0.12 then back within 3 s
            for t in range(len(v)):
                w = v[t + 1:t + 91]
                if len(w) and w.min() < v[t] - 0.12 and w[-1] > w.min() + 0.10:
                    dips_recover += 1
                    break
        m["spike_frames"] = sp
        m["false_zero_frames"] = fz
        m["rounds_with_recovering_dip"] = dips_recover
        out[f"P{s + 1}"] = m
    in_ev = [e for e in ev if inr[min(e[0], n - 1)]]
    out["damage_events"] = len(ev)
    out["damage_events_in_round"] = len(in_ev)
    return out


def checks(hp):
    res = []
    for name, (t0, t1), side, (lo, hi) in CHECKS:
        x = hp[int(t0 * FPS):int(t1 * FPS), side - 1]
        cov = float((~np.isnan(x)).mean())
        med = float(np.nanmedian(x)) if cov > 0 else float("nan")
        ok = cov > 0 and lo <= med <= hi
        res.append(dict(check=name, coverage=round(cov, 2), median=round(med, 3), ok=bool(ok)))
    return res


def main():
    before, after = sys.argv[1], sys.argv[2]
    rid = sys.argv[3] if len(sys.argv) > 3 else "17_DIHLScarecrow_vs_QSalkazko"
    out = {}
    for label, db in (("before", before), ("after", after)):
        hp, ev = load(db, rid)
        out[label] = dict(metrics=metrics(hp, ev), checks=checks(hp))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
