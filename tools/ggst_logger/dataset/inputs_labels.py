"""Per-frame input labels from the game logs (ROADMAP point 3, inputs from video).

For every replay of the transcript (transcript_events.parquet, duplicates already dropped) and each
side, per video frame (60 fps, the same game -> video mapping build.py / transcript_labels.py use,
video frame = rate * t(row) + c):
- `dir`: the held direction in numpad notation (1-9) relative to the player's facing (p*_rec's
  screen-relative U/D/L/R bits turned with p*_face, like verify_inputs.numpad: 6 = towards the
  facing side); 0 = no log row on this frame;
- `press`: the buttons pressed on this frame (rising edges of P/K/S/H/D: bits 1/2/4/8/16), from
  every log row that falls on the frame (a frame can take two rows when the video dropped one);
- `hold`: the buttons held (same bits);
- `state`: what the player is doing from the log's action (STATES): move (an attack start ...
  its end, the transcript's moves incl. Roman Cancel and unmapped actions), idle (stand, crouch,
  turn), walk_f, walk_b, dash (ground / air, forward / back, dash stop), guard (any guard state),
  jump (pre-jump, jump, landing), hit (damage, knockdown, wakeup, tech, other common states);
- `live`: both players in play (neither in a round / match intro or win state: CmnActEntry*,
  CmnActLockWait, CmnActRoundWin, CmnActMatchWin, WinAction*); scoring uses live frames only;
- `row`: the log row shown on the frame (-1: none).

Plus one row per transcript event (inputs_motions.parquet) with the logged command that started it,
read like verify_inputs.read_command at the event's log row: `cmd` = the top reading (motion +
button, e.g. 236K, [4]6S, 2H, 6P; None when no press was found within 8 rows before the start),
`motion` = its motion part ('' for none), `press_row` / `press_frame` (row and video frame of the
button press), `cands` = every candidate reading ('|'-joined; a jump-cancelled 236 rolled into
6236 also reads as 623).

Writes $GGST_DATA/ggst_dataset/inputs/labels/<stem>.npz (dir, press, hold, state [2, F] uint8;
live [F] bool; row [F] int32), inputs/inputs_motions.parquet and inputs/inputs_labels_stats.json.
Labels only: the models see the video (model/video_inputs.py), the metric is model/inputs_eval.py.

    .venv/Scripts/python.exe tools/ggst_logger/dataset/inputs_labels.py [--jobs 4] [--only STEM ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402
from build import is_damage, is_guard, run_dir  # noqa: E402
from moves import resolve_character  # noqa: E402
import transcript_labels as tl  # noqa: E402
import verify_inputs as vi  # noqa: E402

DATA = build.OUT
INP = DATA / "inputs"
LAB = INP / "labels"
STATES = ["move", "idle", "walk_f", "walk_b", "dash", "guard", "jump", "hit"]
BTN_BITS = [0x10, 0x20, 0x40, 0x80, 0x100]           # P K S H D in p*_rec
BTN_NAMES = "PKSHD"
NOT_LIVE = ("CmnActEntry", "CmnActLockWait", "CmnActRoundWin", "CmnActMatchWin", "WinAction")
IDLE = ("CmnActStand", "CmnActCrouch", "CmnActStand2Crouch", "CmnActCrouch2Stand", "CmnActStandTurn",
        "CmnActCrouchTurn", "CmnActAirTurn")
DASH = ("CmnActFDash", "CmnActBDash", "CmnActAirFDash", "CmnActAirBDash", "CmnActFDashStop")
JUMP = ("CmnActJumpPre", "CmnActJump", "CmnActJumpLanding", "CmnActLandingStiff")


def dir_numpad(rec: np.ndarray, face: np.ndarray) -> np.ndarray:
    """p*_rec bits + p*_face (0 = facing right) -> numpad 1-9 relative to the facing (vectorised numpad())."""
    rec = rec.astype(np.int64)
    up, dn, lt, rt = (rec & vi.U) > 0, (rec & vi.D) > 0, (rec & vi.L) > 0, (rec & vi.R) > 0
    fwd = np.where(face == 0, rt, lt)
    back = np.where(face == 0, lt, rt)
    v = np.where(dn, 1, np.where(up, 7, 4))
    h = np.where(fwd, 1, np.where(back, -1, 0))
    return (v + 1 + h).astype(np.uint8)


def buttons(rec: np.ndarray) -> np.ndarray:
    rec = rec.astype(np.int64)
    out = np.zeros(len(rec), np.uint8)
    for k, b in enumerate(BTN_BITS):
        out |= (((rec & b) > 0).astype(np.uint8) << k)
    return out


def states(acts: np.ndarray, char: str) -> np.ndarray:
    cache = {}
    out = np.zeros(len(acts), np.uint8)
    for i, a in enumerate(acts):
        s = cache.get(a)
        if s is None:
            kind, _ = tl.act_kind(char, a)
            if not a.startswith(("CmnAct", "WSB_", "WinAction")) or a == "CmnActRomanCancel":
                s = "move"          # every character action (attacks, stances, specials' parts) and RC
            elif a.startswith(IDLE) and not is_guard(a):
                s = "idle"
            elif a == "CmnActFWalk":
                s = "walk_f"
            elif a == "CmnActBWalk":
                s = "walk_b"
            elif a.startswith(DASH):
                s = "dash"
            elif is_guard(a):
                s = "guard"
            elif a.startswith(JUMP):
                s = "jump"
            elif kind == "move":
                s = "move"          # common attacks (throws, burst ...)
            else:
                s = "hit"
            cache[a] = s
        out[i] = STATES.index(s)
    return out


def process(args) -> dict:
    stem, csv_path, nframes, events = args
    log = build.load_log(csv_path)
    al = json.loads((run_dir(stem) / "openers.json").read_text())["summary"]["alignment"]
    n = log["n"]
    vf = al["rate"] * log["t"] + al["c"]
    F = int(nframes) if nframes else int(np.ceil(vf[-1])) + 1
    chars = {s: resolve_character(log[f"p{s}_act"], log.get(f"p{s}_char"), log.get(f"p{s}_sprite"))[0] for s in (1, 2)}
    # rows -> frames: the first row within half a frame of f
    fr = np.arange(F)
    row = np.searchsorted(vf, fr - 0.5)
    ok = (row < n) & (np.abs(vf[np.minimum(row, n - 1)] - fr) <= 0.5)
    row = np.where(ok, row, -1).astype(np.int32)
    rf = np.rint(vf).astype(np.int64)               # frame of each row
    rin = (rf >= 0) & (rf < F)
    D = np.zeros((2, F), np.uint8)
    P = np.zeros((2, F), np.uint8)
    H = np.zeros((2, F), np.uint8)
    S = np.zeros((2, F), np.uint8)
    live_row = np.ones(n, bool)
    for s in (1, 2):
        live_row &= np.array([not a.startswith(NOT_LIVE) for a in log[f"p{s}_act"]])
    for k, s in enumerate((1, 2)):
        rec = log[f"p{s}_rec"].astype(np.int64)
        face = log[f"p{s}_face"].astype(np.int64)
        d = dir_numpad(rec, face)
        b = buttons(rec)
        pr = b & ~np.concatenate([[0], b[:-1]]).astype(np.uint8)
        st = states(log[f"p{s}_act"], chars[s])
        D[k, ok] = d[row[ok]]
        H[k, ok] = b[row[ok]]
        S[k, ok] = st[row[ok]]
        np.bitwise_or.at(P[k], rf[rin], pr[rin])
    live = np.zeros(F, bool)
    live[ok] = live_row[row[ok]]
    LAB.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LAB / f"{stem}.npz", dir=D, press=P, hold=H, state=S, live=live, row=row)
    # logged command per transcript event
    mot = []
    recs = {s: [int(x) for x in log[f"p{s}_rec"]] for s in (1, 2)}
    faces = {s: [int(x) for x in log[f"p{s}_face"]] for s in (1, 2)}
    for e in events:
        s = int(e["side"][1])
        t = int(e["log_row"])
        c = vi.read_command(recs[s], faces[s], t)
        pb = None
        if c is not None:
            for bb in range(t, max(0, t - vi.PRESS_BEFORE) - 1, -1):
                if any(recs[s][bb] & bit and not (bb and recs[s][bb - 1] & bit) for bit, _ in vi.BUTTONS):
                    pb = bb
                    break
        cmd = c[0] if c else None
        m = None if cmd is None else cmd.rstrip("PKSHD/")
        mot.append(dict(replay=stem, side=e["side"], log_row=t, start_frame=e["start_frame"], char_move=e["char_move"],
                        category=e["category"], kind=e["kind"], cmd=cmd, motion=m, cands=None if c is None else "|".join(c),
                        press_row=pb, press_frame=None if pb is None else int(round(vf[pb]))))
    sm = dict(frames=F, live=int(live.sum()), rows=n, rows_mapped=int(ok.sum()),
              dir=np.bincount(D[:, live].ravel(), minlength=10).tolist(),
              press=[int(((P[:, live] >> k) & 1).sum()) for k in range(5)],
              state=np.bincount(S[:, live].ravel(), minlength=len(STATES)).tolist(),
              double_rows=int(len(rf[rin]) - len(np.unique(rf[rin]))))
    return dict(stem=stem, motions=mot, stats=sm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    build.set_own_priority(build.BELOW_NORMAL)
    import pyarrow as pa
    import pyarrow.parquet as pq
    ev = collections.defaultdict(list)
    for r in pq.read_table(DATA / "transcript" / "transcript_events.parquet",
                           columns=["replay", "side", "log_row", "start_frame", "char_move", "category", "kind"]).to_pylist():
        ev[r["replay"]].append(r)
    csv = {o["replay"]: o["csv"] for o in pq.read_table(DATA / "openers.parquet", columns=["replay", "csv"]).to_pylist()}
    nf = tl.manifest_frames()
    stems = sorted(ev)
    if a.only:
        stems = [s for s in stems if s in a.only]
    INP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mot, stats = [], {}
    with ProcessPoolExecutor(max_workers=min(a.jobs, 4)) as ex:
        for k, res in enumerate(ex.map(process, [(s, csv[s], nf.get(s), ev[s]) for s in stems], chunksize=2)):
            mot += res["motions"]
            stats[res["stem"]] = res["stats"]
            if k % 100 == 0:
                print(f"{k}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    suf = "" if not a.only else "_partial"
    pq.write_table(pa.Table.from_pylist(mot), INP / f"inputs_motions{suf}.parquet", compression="zstd")
    tot = lambda k: np.sum([v[k] for v in stats.values()], 0).tolist()
    C = collections.Counter
    summ = dict(replays=len(stats), frames=int(sum(v["frames"] for v in stats.values())),
                live_frames=int(sum(v["live"] for v in stats.values())),
                frames_with_two_rows=int(sum(v["double_rows"] for v in stats.values())),
                dir_counts_live={str(d): c for d, c in enumerate(tot("dir")) if d},
                press_counts_live=dict(zip(BTN_NAMES, tot("press"))),
                state_counts_live=dict(zip(STATES, tot("state"))),
                events=len(mot), events_with_cmd=sum(m["cmd"] is not None for m in mot),
                cmd_by_category={c: dict(C((m["motion"] or "-") if m["cmd"] else "none" for m in mot if m["category"] == c).most_common(12))
                                 for c in sorted({m["category"] for m in mot})},
                per_replay=stats)
    (INP / f"inputs_labels_stats{suf}.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_replay"}, indent=1, default=float))
    print(f"{len(stems)} replays, {time.time() - t0:.0f}s -> {INP}")


if __name__ == "__main__":
    main()
