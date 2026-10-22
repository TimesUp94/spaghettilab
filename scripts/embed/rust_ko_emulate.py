"""Emulate the first stages of lib.rs winner detection (normalisation, junk filter, KO counting)
on given round spans, to compare DBs when cargo is unavailable.  NOT a full port: boundaries are
supplied (true round spans), and only the KO-count stage + a final-HP readout are reported.
Usage: python rust_ko_emulate.py DB REPLAY_ID
"""
import sqlite3
import sys

import numpy as np

# HP is the filled fraction of the bar (full = 1.0); thresholds as in lib.rs.
# Set 17 true rounds: (start_s, end_s, true winner) -- end = next round's reset.
ROUNDS = [(12, 40.5, "P1"), (40.5, 71.0, "P2"), (71.0, 148.5, "P2"), (148.5, 193.0, "P2"),
          (193.0, 246.5, "P1"), (246.5, 289.6, "P2"), (289.7, 322.6, "P2"), (322.6, 393.0, "P2")]


def rolling_median(a, w):
    n = len(a); out = np.full(n, np.nan); h = w // 2
    for i in range(n):
        s = a[max(0, i - h):i + h + 1]; s = s[~np.isnan(s)]
        if len(s) >= w // 3:
            out[i] = np.median(s)
    return out


def main():
    db, rid = sys.argv[1], sys.argv[2]
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute("select frame_number,p1_health_pct,p2_health_pct from frame_data where replay_id=? order by timestamp_ms", (rid,)).fetchall()
    p1 = np.array([np.nan if r[1] is None else min(max(r[1], 0), 1) for r in rows])
    p2 = np.array([np.nan if r[2] is None else min(max(r[2], 0), 1) for r in rows])
    j1, j2 = rolling_median(p1, 150), rolling_median(p2, 150)
    junk = ~np.isnan(j1) & ~np.isnan(j2) & (j1 < 0.13125) & (j2 < 0.13125)
    print(f"junk frames NaN-ed: {junk.sum()}")
    p1[junk] = np.nan; p2[junk] = np.nan
    ok = 0
    for s_t, e_t, truth in ROUNDS:
        s, e = int(s_t * 30), min(int(e_t * 30), len(p1))
        ss = s + int((e - s) * 0.4)
        k1 = int(np.sum(~np.isnan(p1[ss:e]) & (p1[ss:e] < 0.13125) & ~np.isnan(p2[ss:e]) & (p2[ss:e] < 0.7875)))
        k2 = int(np.sum(~np.isnan(p2[ss:e]) & (p2[ss:e] < 0.13125) & ~np.isnan(p1[ss:e]) & (p1[ss:e] < 0.7875)))
        w = None
        if k1 >= 3 and k1 >= 3 * k2 + 1:
            w = "P2"
        elif k2 >= 3 and k2 >= 3 * k1 + 1:
            w = "P1"
        both = np.flatnonzero(~np.isnan(p1[s:e]) & ~np.isnan(p2[s:e]))
        last = (round(float(p1[s + both[-1]]), 3), round(float(p2[s + both[-1]]), 3)) if len(both) else None
        mins = (round(float(np.nanmin(p1[s:e])), 3), round(float(np.nanmin(p2[s:e])), 3))
        ok += w == truth
        print(f"{s_t:6.1f}-{e_t:6.1f} truth={truth} ko_count p1={k1:4d} p2={k2:4d} -> {w or 'undecided (later stages)'}  min={mins} last_both={last}")
    print(f"KO-count stage correct: {ok}/{len(ROUNDS)}")


if __name__ == "__main__":
    main()
