"""Pick candidate frames for human move/box labeling.

~40 "hit" frames: a few frames before the onset of a confirmed HP drop (the moment an attack
connects; GGST hitstop keeps the attacker in its active pose there) and ~20 "neutral" frames
with no HP change for +-2 s. All inside hand-marked round intervals.

HP drops come from the RAW per-frame widths (health_raw.py), not from the baseline
damage_events, because ~half of those are artefacts of the extractor's mid-round state resets
(see health_baseline/report.md). A drop counts only if the 5-frame running median falls by
>= 2.5% of the bar AND is still lower 1 s later (rules out occlusion flicker). DB damage events
that the raw signal confirms are preferred when they coincide.

Usage:
  .venv/Scripts/python.exe scripts/embed/pick_label_frames.py VIDEO RAW.npz OUT_DIR [--db DB]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health_baseline import FPS, ROUNDS_17, in_round_mask  # noqa: E402

PRE_FRAMES = 2       # frames before the drop onset to grab (raw-width fallback)
HIT_OFFSET = 1       # frames after the pending-red onset (hitstop: attacker still in active pose)
N_HIT, N_NEUTRAL = 40, 20
MIN_GAP = int(1.2 * FPS)


def running_median(x, k=5):
    pad = k // 2
    xp = np.pad(x.astype(float), pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def find_drops(w, mx, inr):
    """Return [(onset_frame, drop_pct)] of sustained drops for one side."""
    m = running_median(np.where(w >= 50, w, np.nan_to_num(np.nan)), 5)
    valid = (w >= 50)
    out = []
    i = 10
    n = len(w)
    while i < n - 40:
        if not (inr[i] and valid[i - 3:i + 1].all()):
            i += 1
            continue
        before = np.median(w[i - 8:i][valid[i - 8:i]]) if valid[i - 8:i].sum() >= 4 else None
        if before is None:
            i += 1
            continue
        drop = (before - m[i + 2]) / mx
        if drop >= 0.025:
            later = w[i + 25:i + 40][valid[i + 25:i + 40]]
            if len(later) >= 5 and (before - np.median(later)) / mx >= 0.02:
                # onset: first frame where the raw width is below before - 1%
                j = i
                while j > i - 6 and w[j - 1] < before - 0.01 * mx:
                    j -= 1
                out.append((j, float((before - np.median(later)) / mx)))
                i += 20
                continue
        i += 1
    return out


def find_hit_onsets(red, inr):
    """Frames where a dark-red pending-damage segment appears on a bar that had none for the
    previous 10 frames and stays for >= 6 of the next 8 (an attack/combo starting to connect)."""
    out = []
    n = len(red)
    for i in range(10, n - 8):
        if inr[i] and red[i] >= 8 and (red[i - 10:i] < 4).all() and (red[i:i + 8] >= 8).sum() >= 6:
            out.append((i, int(red[i:i + 60].max())))
    return out


def spread_pick(cands, n, min_gap):
    """Greedy pick: highest score first, enforce min frame gap, then sort by time."""
    chosen = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if all(abs(c["frame"] - o["frame"]) >= min_gap for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda c: c["frame"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("raw")
    ap.add_argument("out_dir")
    ap.add_argument("--db", default=None)
    ap.add_argument("--pending", default=None, help="pending_red.npy from health_raw.py --pending")
    args = ap.parse_args()
    raw = np.load(args.raw)
    n = len(raw["p1"])
    inr, rid = in_round_mask(n, ROUNDS_17)
    # DB events (optional, only used as a tie-break bonus)
    ev_frames = {1: [], 2: []}
    if args.db:
        c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        for f0, side in c.execute("select frame_start,target_side from damage_events where replay_id=?",
                                  (Path(args.video).stem,)):
            ev_frames[side].append(f0)

    hits = []
    pend = np.load(args.pending) if args.pending else None
    for side, key, mx, victim, attacker in ((1, "p1", 713, "Faust", "Anji Mito"),
                                           (2, "p2", 730, "Anji Mito", "Faust")):
        if pend is not None:
            cands = [(f, red / mx) for f, red in find_hit_onsets(pend[:, 0 if side == 1 else 2], inr)]
            src = "pending-red segment appears"
        else:
            cands = find_drops(raw[key], mx, inr)
            src = "HP drops"
        for onset, dmg in cands:
            f = onset + HIT_OFFSET if pend is not None else onset - PRE_FRAMES
            bonus = 0.05 if any(abs(f - e) <= 15 for e in ev_frames[side]) else 0.0
            hits.append({"frame": int(f), "score": dmg + bonus, "round": int(rid[f]),
                         "reason": f"hit: P{side} ({victim}) {src} at frame {onset} (~{dmg:.0%} pending)"
                                   f" -> {attacker} attack connecting"
                                   + (" [DB event agrees]" if bonus else "")})
    print(f"{len(hits)} hit-onset candidates")
    # per-round quota so all rounds are represented
    picked = []
    rounds = sorted(set(h["round"] for h in hits))
    quota = {r: max(2, round(N_HIT * sum(h["round"] == r for h in hits) / len(hits))) for r in rounds}
    for r in rounds:
        picked += spread_pick([h for h in hits if h["round"] == r], quota[r], MIN_GAP)
    picked = spread_pick(picked, N_HIT, MIN_GAP)  # trim to N_HIT, best first

    # neutral: both bars mostly readable, no pending-red on either bar, HP stable over +-1 s
    neutral = []
    w1, w2 = raw["p1"], raw["p2"]
    for f in range(60, n - 60, 15):
        if not inr[f] or not inr[f - 60] or not inr[f + 60]:
            continue
        seg1, seg2 = w1[f - 30:f + 30], w2[f - 30:f + 30]
        if (seg1 < 50).mean() > 0.4 or (seg2 < 50).mean() > 0.4:
            continue
        if pend is not None and (pend[f - 20:f + 20, 0].max() >= 8 or pend[f - 20:f + 20, 2].max() >= 8):
            continue
        a1, b1 = w1[f - 30:f], w1[f:f + 30]
        a2, b2 = w2[f - 30:f], w2[f:f + 30]
        if min((a1 >= 50).sum(), (b1 >= 50).sum(), (a2 >= 50).sum(), (b2 >= 50).sum()) < 8:
            continue
        if (abs(np.median(a1[a1 >= 50]) - np.median(b1[b1 >= 50])) / 713 > 0.02
                or abs(np.median(a2[a2 >= 50]) - np.median(b2[b2 >= 50])) / 730 > 0.02):
            continue
        if any(abs(f - p["frame"]) < 45 for p in picked):
            continue
        neutral.append({"frame": f, "score": 0.0, "round": int(rid[f]),
                        "reason": "neutral: no pending damage on either bar for +-0.7 s, HP stable +-1 s"})
    # even spread over time
    # spread over time: seeded random order, >= 3 s apart
    rng = np.random.default_rng(17)
    for c in neutral:
        c["score"] = float(rng.random())
    neutral = spread_pick(neutral, N_NEUTRAL, int(2.5 * FPS))
    print(f"{len(picked)} hit frames, {len(neutral)} neutral frames")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    frames = []
    for p in sorted(picked + neutral, key=lambda p: p["frame"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
        ok, img = cap.read()
        if not ok:
            print("could not read", p["frame"])
            continue
        name = f"f{p['frame']:05d}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append({"frame": p["frame"], "time_s": round(p["frame"] / FPS, 3), "file": name,
                       "round": p["round"] + 1, "kind": "hit" if p["reason"].startswith("hit") else "neutral",
                       "reason": p["reason"]})
    (out / "frames.json").write_text(json.dumps(frames, indent=1))
    print(f"wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
