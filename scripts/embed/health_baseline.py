"""Characterise baseline health-reading quality for one replay in an experiment DB.

Reads frame_data + damage_events from an analysis.db (never the user's output/analysis.db
unless pointed there explicitly), the raw widths from health_raw.py, and a hand-made list of
in-round intervals (seconds). Produces a plot, an anomaly table (CSV/JSON) and crops of the
worst moments.

Usage:
  .venv/Scripts/python.exe scripts/embed/health_baseline.py --db output/exp17/analysis.db \
     --video ggstsets/17_DIHLScarecrow_vs_QSalkazko.mp4 \
     --raw output/embed_test/health_baseline/raw_widths.npz --out output/embed_test/health_baseline
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

FPS = 30.0
# Set 17 in-round intervals (s), read by eye from the ROUND TIME digits (99 -> last legible
# value before KO / round-end).  Super cinematics inside rounds hide the HUD and are kept.
ROUNDS_17 = [(12, 38), (42, 67), (72, 135), (148, 190), (194, 241), (248, 277), (289, 319),
             (324, 389)]


def load(db, replay_id):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute("select frame_number,p1_health_pct,p2_health_pct,timer_value from frame_data "
                     "where replay_id=? order by frame_number", (replay_id,)).fetchall()
    ev = c.execute("select frame_start,frame_end,target_side,damage_pct,pre_health_pct,post_health_pct "
                   "from damage_events where replay_id=? order by frame_start", (replay_id,)).fetchall()
    fn = np.array([r[0] for r in rows])
    hp = np.array([[np.nan if r[1] is None else r[1], np.nan if r[2] is None else r[2]] for r in rows])
    timer = np.array([np.nan if r[3] is None else r[3] for r in rows])
    return fn, hp, timer, ev


def in_round_mask(n, rounds):
    m = np.zeros(n, bool)
    rid = np.full(n, -1)
    for i, (a, b) in enumerate(rounds):
        m[int(a * FPS):int(b * FPS)] = True
        rid[int(a * FPS):int(b * FPS)] = i
    return m, rid


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def find_anomalies(hp, inr, rid):
    out = []
    n = len(hp)
    for side in (0, 1):
        s = hp[:, side]
        # 1) dropouts: NaN inside a round
        for a, b in runs(inr & np.isnan(s)):
            if b - a >= 6:
                out.append(dict(kind="dropout", side=side + 1, f0=int(a), f1=int(b), mag=float((b - a) / FPS)))
        # 2) upward jumps inside a round (HP cannot rise mid-round; Faust's Donut heals a little)
        valid = np.where(inr & ~np.isnan(s))[0]
        for k in range(1, len(valid)):
            i, j = valid[k - 1], valid[k]
            if rid[i] != rid[j] or j - i > 90:
                continue
            d = s[j] - s[i]
            if d > 0.05:
                out.append(dict(kind="upjump", side=side + 1, f0=int(i), f1=int(j), mag=float(d),
                                pre=float(s[i]), post=float(s[j])))
        # 3) dips that recover: value falls >=0.12 below the level before, then returns within 3 s
        for k in range(len(valid)):
            i = valid[k]
            pre = s[i]
            w = valid[(valid > i) & (valid <= i + 90) & (rid[valid] == rid[i])]
            if len(w) < 3:
                continue
            low_idx = w[np.argmin(s[w])]
            if pre - s[low_idx] < 0.12:
                continue
            after = w[w > low_idx]
            if len(after) and s[after].max() >= s[low_idx] + 0.10 and s[after].max() >= pre - 0.04:
                out.append(dict(kind="dip_recover", side=side + 1, f0=int(i), f1=int(low_idx),
                                mag=float(pre - s[low_idx]), pre=float(pre), post=float(s[low_idx])))
    # merge overlapping same-kind/side anomalies (dip_recover fires on consecutive i)
    out.sort(key=lambda a: (a["kind"], a["side"], a["f0"]))
    merged = []
    for a in out:
        if merged and merged[-1]["kind"] == a["kind"] and merged[-1]["side"] == a["side"] \
                and a["f0"] <= merged[-1]["f1"] + 15:
            m = merged[-1]
            m["f1"] = max(m["f1"], a["f1"])
            m["mag"] = max(m["mag"], a["mag"])
        else:
            merged.append(dict(a))
    merged.sort(key=lambda a: a["f0"])
    return merged


def plot(fn, hp, raw, inr, rounds, anoms, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = fn / FPS
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    for side, ax in enumerate(axes):
        rw = raw["p1"] / raw["p1_max"] if side == 0 else raw["p2"] / raw["p2_max"]
        ax.plot(t, rw, color="0.75", lw=0.6, label="raw anchored width (primary band)")
        ax.plot(t, hp[:, side], color="C0" if side == 0 else "C3", lw=1.1, label="stored health_pct")
        for a, b in rounds:
            ax.axvspan(a, b, color="C2", alpha=0.07)
        for an in anoms:
            if an["side"] == side + 1:
                c = {"dropout": "k", "upjump": "m", "dip_recover": "orange"}[an["kind"]]
                ax.axvline(an["f0"] / FPS, color=c, lw=0.5, alpha=0.6)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel(f"P{side + 1} HP ({'Faust' if side == 0 else 'Anji'})")
        ax.legend(loc="lower left", fontsize=8)
    axes[1].set_xlabel("time (s)   green = in-round, magenta = up-jump, orange = dip+recover, black = dropout")
    fig.tight_layout()
    fig.savefig(out, dpi=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rid_name = Path(args.video).stem
    fn, hp, timer, ev = load(args.db, rid_name)
    raw = dict(np.load(args.raw))
    inr, rid = in_round_mask(len(fn), ROUNDS_17)
    anoms = find_anomalies(hp, inr, rid)
    plot(fn, hp, raw, inr, ROUNDS_17, anoms, out / "hp_baseline.png")

    n_in = int(inr.sum())
    stats = {"frames": int(len(fn)), "in_round_frames": n_in}
    for side in (0, 1):
        s = hp[:, side]
        stats[f"p{side + 1}_nan_in_round_pct"] = float(np.isnan(s[inr]).mean() * 100)
    stats["timer_nan_in_round_pct"] = float(np.isnan(timer[inr]).mean() * 100)
    stats["damage_events"] = len(ev)
    stats["damage_events_in_round"] = int(sum(inr[min(e[0], len(inr) - 1)] for e in ev))
    kinds = {}
    for a in anoms:
        kinds[f"{a['kind']}_p{a['side']}"] = kinds.get(f"{a['kind']}_p{a['side']}", 0) + 1
    stats["anomaly_counts"] = kinds
    (out / "anomalies.json").write_text(json.dumps({"stats": stats, "anomalies": anoms}, indent=1))
    print(json.dumps(stats, indent=1))
    worst = sorted([a for a in anoms if a["kind"] != "dropout"], key=lambda a: -a["mag"])[:20]
    for a in worst:
        print(f"{a['kind']:12s} P{a['side']} t={a['f0'] / FPS:7.2f}-{a['f1'] / FPS:7.2f}s mag={a['mag']:.2f}"
              f" pre={a.get('pre', 0):.2f} post={a.get('post', 0):.2f}")


if __name__ == "__main__":
    main()
