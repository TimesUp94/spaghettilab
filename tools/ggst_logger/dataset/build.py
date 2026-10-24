"""Combo-opener dataset from game-logged replays (logger CSV = ground truth, video = CV input).

For every complete entry of the batch manifest (tools/ggst_logger/drive/batch.py):

1. run the app's CV pipeline on the video (scripts/analyze_replay.py) unless already done;
2. align game frames to video frames: qpc clock model (v = rate * t_qpc + c), coarse
   offset by matching the logged HP (hp / max) to the pipeline's raw health readings,
   refined (and drift-checked) on isolated HP drop edges; residuals are reported;
3. extract hits / combos / openers from the game log (defender action states + HP);
4. label every opener (attacker action, its start, previous action, counter-hit guess,
   facing, distance, airborne flags, damage, combo length, hits) and match it against the
   pipeline's own damage events and hit-counter combos;
5. write ggst_dataset/openers.parquet (+ .csv), a 5-frame strip per opener, and
   per-replay summaries (summary.csv / summary.json).

Per-replay results are cached in runs/<stem>/openers.json and alignment.json; a rerun only
processes new manifest entries (use --force to redo the analysis, --force-cv to rerun the
CV pipeline too).

Usage (from the repo root, project venv):
    .venv/Scripts/python.exe tools/ggst_logger/dataset/build.py
    .venv/Scripts/python.exe tools/ggst_logger/dataset/build.py --jobs 2 --force
"""
from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
DATA = Path(os.environ.get("GGST_DATA", r"G:\Projects\replanal-data"))
MANIFEST = DATA / "ggst_rec" / "manifest.jsonl"
OUT = DATA / "ggst_dataset"

BELOW_NORMAL = 0x00004000
IDLE = 0x00000040

# --------------------------------------------------------------------------------------
# Game-state vocabulary
# --------------------------------------------------------------------------------------

# Defender states that mean "being hit / combo'd" (hitstun, launch, knockdown, throw lock,
# wall splat, wallbreak cinematic). Wake-up / tech states (*2Stand, QuickDown) are *not*
# damage: a hit after them is a new opener (meaty / okizeme).
DAMAGE_PREFIXES = (
    "CmnActNokezori", "CmnActFloatDamage", "CmnActBlowoff", "CmnActKirimomi",
    "CmnActBDownUpper", "CmnActBDownLoop", "CmnActBDownDown", "CmnActBDownBound",
    "CmnActFDownUpper", "CmnActFDownLoop", "CmnActFDownDown", "CmnActFDownBound",
    "CmnActVDown", "CmnActWallHaritsuki", "CmnActWallBound", "CmnActJitabata",
    "CmnActKorogari", "CmnActLockWait", "CmnActExDamage", "CmnActGroundBound",
    "CmnActHizakuzure", "CmnActFrozen", "CmnActHarden", "WSB_Slave",
)
THROW_TECH = "CmnActHajikare"   # both players after a teched throw

# Character identification and move naming live in moves.py (CHAR_CODES, CHAR_SIGNATURES,
# per-character move tables, categories).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moves import (CHAR_CODES, CHAR_SIGNATURES, COMMON_PREFIXES, BTN, classify,  # noqa: E402,F401
                   infer_character, move_info, notation, resolve_character)


def is_damage(act: str) -> bool:
    return act.startswith(DAMAGE_PREFIXES)


def is_guard(act: str) -> bool:
    return act.startswith("CmnAct") and ("Guard" in act or "FaultlessPressure" in act)


def is_common(act: str) -> bool:
    return act.startswith("CmnAct") or act.startswith("WSB_") or act.startswith("WinAction")


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------

def read_manifest() -> list[dict]:
    out = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue          # line being written right now
        if Path(e["video"]).exists() and Path(e["csv"]).exists():
            out.append(e)
    return out


def load_log(path: str) -> dict:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head, rows = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
    cols = list(zip(*rows))
    d = {}
    for k, col in zip(head, cols):
        if k.endswith(("_act", "_char", "_sprite")):  # p*_char: name or id; p*_sprite: e.g. sol000_11
            d[k] = np.array(col)
        else:
            d[k] = np.array(col, dtype=np.float64)
    d["t"] = (d["qpc"] - d["qpc"][0]) / d["qpc_freq"][0]
    d["n"] = len(rows)
    return d


def run_dir(stem: str) -> Path:
    return OUT / "runs" / stem


def cv_done(stem: str) -> bool:
    rd = run_dir(stem)
    return (rd / "frames" / f"{stem}_frames.parquet").exists() and (rd / f"{stem}_combos.json").exists()


def run_pipeline(entry: dict, priority: int) -> tuple[str, float, int]:
    stem = Path(entry["video"]).stem
    rd = run_dir(stem)
    rd.mkdir(parents=True, exist_ok=True)
    db = rd / "analysis.db"
    if db.exists():
        db.unlink()           # storage appends rows; start from a clean per-run DB
    t0 = time.time()
    with open(rd / "pipeline.log", "w") as log:
        p = subprocess.run(
            [str(PY), str(ROOT / "scripts" / "analyze_replay.py"), entry["video"],
             "--config", str(ROOT / "config" / "default.yaml"), "--output", str(rd)],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), creationflags=priority)
    return stem, time.time() - t0, p.returncode


def load_cv(stem: str) -> dict:
    import pyarrow.parquet as pq
    rd = run_dir(stem)
    t = pq.read_table(rd / "frames" / f"{stem}_frames.parquet").to_pydict()
    f = lambda k: np.array([np.nan if x is None else x for x in t[k]], dtype=np.float64)
    cv = {"vf": np.array(t["frame_number"], dtype=np.int64)}
    for k in ("p1_health_raw", "p2_health_raw", "p1_health_pct", "p2_health_pct", "timer_value"):
        cv[k] = f(k)
    con = sqlite3.connect(f"file:{rd / 'analysis.db'}?mode=ro", uri=True)
    cv["damage_events"] = [dict(zip(("frame_start", "frame_end", "target_side", "damage_pct"), r))
                           for r in con.execute("select frame_start, frame_end, target_side, damage_pct "
                                                "from damage_events where replay_id=? order by frame_start", (stem,))]
    con.close()
    cv["combos"] = json.loads((rd / f"{stem}_combos.json").read_text())
    return cv


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------

def _log_at(log_t: np.ndarray, series: np.ndarray, vf: np.ndarray, rate: float, c: float) -> np.ndarray:
    """Logged value shown at video frames vf under v = rate * t + c (latest row at or before)."""
    i = np.searchsorted(log_t, (vf - c) / rate, side="right") - 1
    out = series[np.clip(i, 0, len(series) - 1)].astype(float)
    out[(i < 0) | (i >= len(series))] = np.nan
    return out


def hp_edges(log: dict, hpmax: dict, min_drop=0.03, iso=20) -> list[tuple[int, int, float]]:
    """Isolated HP drops (row, side, drop fraction): no other HP change on that side within iso rows."""
    out = []
    for s in (1, 2):
        hp = log[f"p{s}_hp"]
        ch = np.nonzero(np.diff(hp) != 0)[0] + 1
        for k, i in enumerate(ch):
            drop = (hp[i - 1] - hp[i]) / hpmax[s]
            if drop < min_drop:
                continue
            prev_ok = k == 0 or i - ch[k - 1] > iso
            next_ok = k == len(ch) - 1 or ch[k + 1] - i > iso
            if prev_ok and next_ok:
                out.append((int(i), s, float(drop)))
    return out


def align(log: dict, cv: dict, hpmax: dict) -> dict:
    vf = cv["vf"]
    t = log["t"]
    H = {s: cv[f"p{s}_health_raw"] for s in (1, 2)}
    L = {s: log[f"p{s}_hp"] / hpmax[s] for s in (1, 2)}
    rate = 60.0

    # 1) coarse offset: HP-level agreement over the whole video
    def level_cost(c):
        errs = [np.abs(_log_at(t, L[s], vf, rate, c) - H[s]) for s in (1, 2)]
        e = np.concatenate(errs)
        e = e[np.isfinite(e)]
        return np.mean(np.minimum(e, 0.2)) if len(e) > 100 else np.inf
    cands = np.arange(-900, 900, 2.0)
    costs = np.array([level_cost(c) for c in cands])
    c0 = float(cands[np.argmin(costs)])
    fine = np.arange(c0 - 4, c0 + 4.01, 0.5)
    c0 = float(fine[np.argmin([level_cost(c) for c in fine])])

    # 2) edges: first pipeline sample showing each isolated logged HP drop
    edges = hp_edges(log, hpmax)

    def measure(rate, c):
        res = []
        for i, s, drop in edges:
            pred = rate * t[i] + c
            pre_m = (vf >= pred - 24) & (vf <= pred - 6)
            if pre_m.sum() < 3:
                continue
            pre = np.nanmedian(H[s][pre_m])
            win = np.nonzero((vf >= pred - 12) & (vf <= pred + 12))[0]
            hit = [j for j in win if np.isfinite(H[s][j]) and H[s][j] < pre - 0.5 * drop]
            # require it to stay down for 2 samples (not a flash)
            hit = [j for j in hit if j + 1 < len(vf) and np.isfinite(H[s][j + 1]) and H[s][j + 1] < pre - 0.5 * drop]
            if hit:
                res.append((float(t[i]), float(vf[hit[0]] - pred), i, s))
        return res

    res = measure(rate, c0)
    r = np.array([x[1] for x in res])
    tt = np.array([x[0] for x in res])
    # sampling every 2nd frame: the first sample at/after the true edge is 0 or 1 frame late
    keep = np.abs(r - np.median(r)) <= 3
    c1 = c0 + float(np.mean(r[keep])) - 0.5
    slope_ppm = 0.0
    if keep.sum() >= 8 and np.ptp(tt[keep]) > 30:
        A = np.vstack([tt[keep], np.ones(keep.sum())]).T
        (b, a), *_ = np.linalg.lstsq(A, r[keep], rcond=None)
        drift = b * np.ptp(tt[keep])
        if abs(drift) > 0.75:         # more than ~1 frame over the video: use the fitted rate
            rate = 60.0 + b
            c1 = c0 + a - 0.5
            slope_ppm = b / 60.0 * 1e6
    res2 = measure(rate, c1)
    r2 = np.array([x[1] for x in res2]) - 0.5   # centre of the 2-frame sampling bin
    ok = np.abs(r2) <= 3
    return {
        "rate": rate, "c": c1, "c_coarse": c0, "drift_ppm": slope_ppm,
        "level_cost": float(level_cost(c1)),
        "n_edges": int(len(edges)), "n_edges_found": int(len(r2)),
        "n_edges_outliers": int((~ok).sum()),
        "resid_median": float(np.median(r2[ok])) if ok.any() else None,
        "resid_mad": float(np.median(np.abs(r2[ok] - np.median(r2[ok])))) if ok.any() else None,
        "resid_abs_p90": float(np.percentile(np.abs(r2[ok]), 90)) if ok.any() else None,
        "within_1": float(np.mean(np.abs(r2) <= 1.0)) if len(r2) else None,
        "residuals": [round(float(x), 2) for x in r2],
    }


def game_to_video(log: dict, al: dict, i: int) -> float:
    return al["rate"] * log["t"][i] + al["c"]


# --------------------------------------------------------------------------------------
# Ground-truth hits, combos and openers
# --------------------------------------------------------------------------------------

def run_start(acts: np.ndarray, i: int) -> int:
    j = i
    while j > 0 and acts[j - 1] == acts[i]:
        j -= 1
    return j


def hitstop_len(log: dict, h: int, max_len=40) -> int:
    """Frames after the hit frame during which neither player moves (hitstop freeze)."""
    n = 0
    for k in range(h + 2, min(h + 2 + max_len, log["n"])):
        same = all(log[f"p{s}_{a}"][k] == log[f"p{s}_{a}"][k - 1] for s in (1, 2) for a in ("x", "y"))
        if not same:
            break
        n += 1
    return n


def freeze_len(log: dict, side: int, h: int, max_len=60) -> int:
    """Frames after the hit frame during which the given player does not move."""
    n = 0
    for k in range(h + 2, min(h + 2 + max_len, log["n"])):
        if log[f"p{side}_x"][k] != log[f"p{side}_x"][k - 1] or log[f"p{side}_y"][k] != log[f"p{side}_y"][k - 1]:
            break
        n += 1
    return n


def extract_openers(log: dict, hpmax: dict, n_quiet: int) -> tuple[list[dict], dict]:
    n = log["n"]
    rnd = log["round"]
    out = []
    stats = collections.Counter()
    for dside in (1, 2):
        aside = 3 - dside
        dact, aact = log[f"p{dside}_act"], log[f"p{aside}_act"]
        hp = log[f"p{dside}_hp"]
        dmg = np.array([is_damage(a) for a in dact])
        grd = np.array([is_guard(a) for a in dact])
        # vocabulary check: states in which this player loses HP but that are neither damage
        # nor guard (chip) states -> probably a hitstun state missing from DAMAGE_PREFIXES
        for k in np.nonzero(np.diff(hp) < 0)[0] + 1:
            nxt = dact[min(k + 1, n - 1)]
            if not dmg[min(k + 1, n - 1)] and not grd[min(k + 1, n - 1)] and nxt.startswith("CmnAct"):
                stats[f"unknown_state_with_hp_loss:{nxt}"] += 1
        # a hit lands on row h when the defender is in a damage state on h+1 but not on h,
        # or when HP drops on h and the defender is (or becomes) in a damage state.
        i = 1
        while i < n - 1:
            entry = dmg[i + 1] and not dmg[i]
            hp_drop = hp[i] < hp[i - 1] and not dmg[i] and (dmg[i + 1] or (i + 2 < n and dmg[i + 2]))
            if not (entry or hp_drop):
                i += 1
                continue
            h = i      # HP drops on row h, the defender's hit state starts on h+1
            # combo span: until the defender leaves damage states (or the round changes)
            e = h + 1
            while e < n and dmg[e] and rnd[e] == rnd[h]:
                e += 1
            damage = int(hp[h - 1] - hp[h - 1:min(e + 1, n)].min())
            seq = dact[h + 1:e]
            teched = e < n and dact[e].startswith(THROW_TECH)
            # hits inside the combo: starts of HP-drop runs
            hits = max(1, sum(1 for k in range(h, min(e + 1, n)) if hp[k] < hp[k - 1] and not hp[k - 1] < hp[k - 2]))
            only_lock = all(a.startswith("CmnActLockWait") for a in seq)
            if teched or (damage == 0 and only_lock):
                stats["throw_teched_or_0dmg_lock"] += 1
                i = e
                continue
            if damage == 0:
                stats["zero_damage_hit"] += 1
            lo = max(0, h - n_quiet)
            quiet = not dmg[lo:h + 1].any()
            if not quiet:
                stats["not_quiet"] += 1
                i = e
                continue
            last_dmg = np.nonzero(dmg[:h + 1])[0]
            last_grd = np.nonzero(grd[max(0, h - 120):h + 1])[0]
            a_act = aact[h]
            a_start = run_start(aact, h)
            prev_i = a_start - 1
            prev_act = aact[prev_i] if prev_i >= 0 else ""
            prev_start = run_start(aact, prev_i) if prev_i >= 0 else -1
            # most recent attacker action that is an attack (non-common) within 120 frames
            last_atk, last_atk_start = "", -1
            for k in range(h, max(0, h - 120), -1):
                if not is_common(aact[k]):
                    last_atk, last_atk_start = aact[k], run_start(aact, k)
                    break
            # move = the action that dealt the hit; "Exe"/continuation actions resolve to their start
            move_act = a_act
            if a_act in ("ThrowExe", "AirThrowExe") and prev_act.startswith("NmlAtk"):
                move_act = prev_act
            elif a_act == "CmnActRomanCancel" and h - a_start <= 3 and prev_act:
                move_act = prev_act      # RC'd on hit: the hit came from the cancelled move
            projectile_likely = (is_common(a_act) and a_act not in ("CmnActBurst", "CmnActRomanCancel")
                                 or (h - a_start) > 45 and "Ultimate" not in a_act and not a_act.endswith("ThrowExe"))
            rc_recent = any(aact[k] == "CmnActRomanCancel" for k in range(max(0, h - 30), h + 1))
            dx = abs(log[f"p1_x"][h] - log[f"p2_x"][h])
            row = dict(
                game_frame=int(log["game_frame"][h]), log_row=int(h), round=int(rnd[h]),
                attacker=f"P{aside}", defender=f"P{dside}",
                attacker_act=a_act, attacker_act_start=int(log["game_frame"][a_start]),
                attacker_act_age=int(h - a_start),
                attacker_prev_act=prev_act, attacker_prev_act_start=int(log["game_frame"][prev_start]) if prev_start >= 0 else None,
                move_act=move_act, move=notation(move_act),
                # for likely projectiles the hitting action is the attacker's last attack, not its current state
                move_guess=notation(last_atk) if projectile_likely and last_atk else notation(move_act),
                last_attack_act=last_atk, last_attack_age=int(h - last_atk_start) if last_atk_start >= 0 else None,
                projectile_likely=bool(projectile_likely),
                is_throw=move_act in ("NmlAtkThrow", "NmlAtkAirThrow") or a_act.endswith("ThrowExe"),
                is_super="Ultimate" in a_act or "Ultimate" in last_atk,
                rc_recent=bool(rc_recent),
                defender_act=dact[h], defender_act_after=dact[h + 1],
                defender_attacking=not is_common(dact[h]) and not is_guard(dact[h]),
                frames_since_guard=int(h - (max(0, h - 120) + last_grd[-1])) if len(last_grd) else None,
                frames_since_damage=int(h - last_dmg[-1]) if len(last_dmg) else None,
                attacker_facing_right=bool(log[f"p{aside}_face"][h] == 0),
                distance=float(dx),
                attacker_x=float(log[f"p{aside}_x"][h]), defender_x=float(log[f"p{dside}_x"][h]),
                attacker_y=float(log[f"p{aside}_y"][h]), defender_y=float(log[f"p{dside}_y"][h]),
                attacker_airborne=bool(log[f"p{aside}_y"][h] > 0),
                defender_airborne=bool(log[f"p{dside}_y"][h] > 0),
                hp_before=int(hp[h - 1]), damage=damage, damage_frac=damage / hpmax[dside],
                combo_frames=int(e - h), combo_end_game_frame=int(log["game_frame"][min(e, n - 1)]),
                hits=hits, hitstop=hitstop_len(log, h),
                first_damage_row=next((int(k) for k in range(h, min(e + 1, n)) if hp[k] < hp[k - 1]), None),
                defender_freeze=freeze_len(log, dside, h), attacker_freeze=freeze_len(log, aside, h),
                ko=bool(hp[h:min(e + 1, n)].min() <= 0),
                is_burst=a_act == "CmnActBurst",
                attacker_hp=int(log[f"p{aside}_hp"][h]),
            )
            out.append(row)
            stats["openers"] += 1
            i = e
    out.sort(key=lambda r: r["log_row"])
    return out, dict(stats)


def mark_counter_hits(rows: list[dict]) -> None:
    """Counter hit = small-CH banner (yellow "COUNTER" box on the attacker's HUD side, measured
    in the video pass) or large CH (giant "COUNTER" text and a long hitstop freeze, which the
    banner box misses).

    Large CH from the log: per (attacker char, move) the normal hitstop is the most common
    hitstop among hits without a banner (>= 3 samples; else the move's minimum); a hitstop at
    least 10 frames longer is a large CH (5H: 14 normal, 30-31 on large CH). Trades (both
    players hit on the same frame) are flagged and never counted as CH."""
    key = lambda r: (r["attacker_char"], r["move_act"])
    by = collections.defaultdict(list)
    for r in rows:
        if not r.get("counter_banner"):
            by[key(r)].append(r["hitstop"])
    at = {(r["replay"], r["log_row"], r["attacker"]) for r in rows}
    for r in rows:
        other = "P1" if r["attacker"] == "P2" else "P2"
        r["trade"] = any((r["replay"], r["log_row"] + d, other) in at for d in (-1, 0, 1))
        hs = by.get(key(r), [])
        normal = (collections.Counter(hs).most_common(1)[0][0] if len(hs) >= 3 else min(hs)) if hs else None
        r["hitstop_normal"] = None if normal is None else int(normal)
        large = (not r["is_throw"] and not r["trade"]
                 and (r["hitstop"] >= normal + 10 if normal is not None else r["hitstop"] >= 25))
        r["counter_hit_large"] = bool(large)
        banner = r.get("counter_banner")
        if banner is None and not large:
            r["counter_hit"] = None            # no video pass: unknown unless large
        else:
            r["counter_hit"] = bool((banner or large) and not r["trade"])
        r["counter_type"] = ("large" if large else "small") if r["counter_hit"] else None


# --------------------------------------------------------------------------------------
# CV matching
# --------------------------------------------------------------------------------------

def cv_openers(cv: dict) -> list[tuple[int, int, str]]:
    """(video frame, defender side, source) for every CV damage event / hit-counter combo."""
    ev = [(int(d["frame_start"]), int(d["target_side"]), "damage") for d in cv["damage_events"]]
    ev += [(int(c["frame_start"]), 2 if c["attacker"] == "P1" else 1, "combo") for c in cv["combos"]]
    return sorted(ev)


def match_cv(rows: list[dict], cv: dict, tol=10) -> dict:
    ev = cv_openers(cv)
    for r in rows:
        ds = int(r["defender"][1])
        # the HP drop is what the CV sees: throws (and some hits) deal damage after the hit frame
        v = r["first_damage_video_frame"] if r["first_damage_video_frame"] is not None else r["video_frame"]
        dm = [e for e in ev if e[1] == ds and abs(e[0] - v) <= tol and e[2] == "damage"]
        cm = [e for e in ev if e[1] == ds and abs(e[0] - v) <= tol and e[2] == "combo"]
        r["cv_damage_dt"] = min((e[0] - v for e in dm), key=abs) if dm else None
        r["cv_combo_dt"] = min((e[0] - v for e in cm), key=abs) if cm else None
        r["cv_detected"] = bool(dm or cm)
        # combo-counter combos start at the 2nd hit: also note any combo starting inside the truth combo
        span_end = v + r["combo_frames"]
        cc = [c for c in cv["combos"] if (2 if c["attacker"] == "P1" else 1) == ds and v - tol <= c["frame_start"] <= span_end]
        r["cv_combo_in_span"] = bool(cc)
        r["cv_combo_max_hits"] = int(cc[0]["max_hits"]) if cc else None
        r["cv_combo_damage"] = float(cc[0]["damage"]) if cc and cc[0]["damage"] is not None else None
    # precision over CV events
    tp = fp = tp_len = 0
    for vfr, ds, src in ev:
        near = any(int(r["defender"][1]) == ds and min(abs(r["video_frame"] - vfr), abs((r["first_damage_video_frame"] or r["video_frame"]) - vfr)) <= tol
                   for r in rows)
        inside = any(int(r["defender"][1]) == ds and r["video_frame"] - tol <= vfr <= r["video_frame"] + r["combo_frames"] + tol
                     for r in rows)
        tp += near
        fp += not near
        tp_len += inside
    n = len(rows)
    det = sum(r["cv_detected"] for r in rows)
    return dict(
        cv_events=len(ev), cv_damage_events=sum(e[2] == "damage" for e in ev),
        cv_combos=sum(e[2] == "combo" for e in ev),
        recall=det / n if n else None,
        recall_damage=sum(r["cv_damage_dt"] is not None for r in rows) / n if n else None,
        recall_combo_start=sum(r["cv_combo_dt"] is not None for r in rows) / n if n else None,
        precision=tp / len(ev) if ev else None,
        precision_in_combo=tp_len / len(ev) if ev else None,
        recall_excl_burst=(lambda sel: sum(r["cv_detected"] for r in sel) / len(sel) if sel else None)(
            [r for r in rows if not r["is_burst"]]),
        recall_multi_hit=(sum(r["cv_detected"] for r in rows if r["hits"] >= 2) / max(1, sum(r["hits"] >= 2 for r in rows))),
        recall_by_damage={k: (lambda sel: round(sum(r["cv_detected"] for r in sel) / len(sel), 3) if sel else None)(
            [r for r in rows if lo <= r["damage_frac"] < hi]) for k, (lo, hi) in
            {"<3%": (0, .03), "3-10%": (.03, .10), "10-25%": (.10, .25), ">=25%": (.25, 9)}.items()},
    )


# --------------------------------------------------------------------------------------
# Strips
# --------------------------------------------------------------------------------------

STRIP_OFFS = (-12, -6, -3, 0, 3)
BANNER_OFFS = tuple(range(2, 41, 2))      # frames after the hit searched for the COUNTER banner (large CH: after the long hitstop)
BANNER_BASE = -3
COUNTER_BANNER_THR = 0.12
# HUD box of the yellow "COUNTER" banner, on the attacker's side (1920x1080)
BANNER_BOX = {1: (90, 200, 420, 340), 2: (1500, 200, 1830, 340)}


def yellow_frac(img: np.ndarray, side: int) -> float:
    import cv2
    x1, y1, x2, y2 = BANNER_BOX[side]
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 160)
    return float(m.mean())


def video_pass(video: str, rows: list[dict], stem: str, tile_w=480) -> None:
    """One sequential decode: strip tiles + COUNTER-banner score per opener; writes the strips."""
    import cv2
    sdir = OUT / "strips" / stem
    sdir.mkdir(parents=True, exist_ok=True)
    for old in sdir.glob("*.jpg"):
        old.unlink()              # names depend on labels: drop strips of a previous build
    want = collections.defaultdict(list)
    for k, r in enumerate(rows):
        for o in set(STRIP_OFFS) | set(BANNER_OFFS) | {BANNER_BASE}:
            want[r["video_frame"] + o].append((k, o))
    tiles = collections.defaultdict(dict)
    banner = collections.defaultdict(dict)
    cap = cv2.VideoCapture(video)
    last = max(want) if want else -1
    th = tile_w * 9 // 16
    f = 0
    while f <= last:
        if f in want:
            ok, img = cap.read()
            if not ok:
                break
            t = cv2.resize(img, (tile_w, th), interpolation=cv2.INTER_AREA)
            for k, o in want[f]:
                if o in BANNER_OFFS or o == BANNER_BASE:
                    banner[k][o] = yellow_frac(img, int(rows[k]["attacker"][1]))
                if o in STRIP_OFFS:
                    tt = t.copy()
                    cv2.putText(tt, f"{o:+d}", (8, th - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(tt, f"{o:+d}", (8, th - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    tiles[k][o] = tt
        else:
            if not cap.grab():
                break
        f += 1
    cap.release()
    for k, r in enumerate(rows):
        b = banner[k]
        if BANNER_BASE in b and any(o in b for o in BANNER_OFFS):
            r["counter_banner_score"] = round(max(b[o] for o in BANNER_OFFS if o in b) - b[BANNER_BASE], 4)
            r["counter_banner"] = bool(r["counter_banner_score"] >= COUNTER_BANNER_THR)
    for k, r in enumerate(rows):
        if len(tiles[k]) != len(STRIP_OFFS):
            continue
        row_img = np.hstack([tiles[k][o] for o in STRIP_OFFS])
        hdr = np.zeros((64, row_img.shape[1], 3), np.uint8)
        l1 = (f"{stem} g{r['game_frame']} v{r['video_frame']}  {r['attacker']} {r['attacker_char']} -> "
              f"{r['move']} [{r['attacker_act']} age {r['attacker_act_age']}f]  dmg {r['damage']} hits {r['hits']}"
              f"  hitstop {r['hitstop']}  CHbanner {r.get('counter_banner_score')}"
              f"{'  PROJ?' if r['projectile_likely'] else ''}")
        l2 = (f"prev {r['attacker_prev_act']}  def {r['defender_act']}->{r['defender_act_after']}  "
              f"dist {r['distance'] / 1000:.0f}  air A{int(r['attacker_airborne'])}/D{int(r['defender_airborne'])}  "
              f"CV dmg {r['cv_damage_dt']} combo {r['cv_combo_dt']}")
        cv2.putText(hdr, l1, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(hdr, l2, (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
        name = f"g{r['game_frame']:05d}_{r['attacker']}_{re.sub(r'[^A-Za-z0-9.]+', '_', r['move'])}.jpg"
        r["strip"] = str(sdir / name)
        cv2.imwrite(r["strip"], np.vstack([hdr, row_img]), [cv2.IMWRITE_JPEG_QUALITY, 80])


# --------------------------------------------------------------------------------------
# Per replay
# --------------------------------------------------------------------------------------

def process_replay(entry: dict, args) -> dict:
    stem = Path(entry["video"]).stem
    rd = run_dir(stem)
    cache = rd / "openers.json"
    if cache.exists() and not args.force:
        return json.loads(cache.read_text())
    log = load_log(entry["csv"])
    cv = load_cv(stem)
    hpmax = {s: float(log[f"p{s}_hp"].max()) for s in (1, 2)}
    chars = {s: resolve_character(log[f"p{s}_act"], log.get(f"p{s}_char"), log.get(f"p{s}_sprite"))[0] for s in (1, 2)}
    al = align(log, cv, hpmax)
    rows, stats = extract_openers(log, hpmax, args.quiet)
    for r in rows:
        r["replay"] = stem
        r["manifest_index"] = entry.get("index")
        r["video"] = entry["video"]
        r["csv"] = entry["csv"]
        r["attacker_char"] = chars[int(r["attacker"][1])]
        r["defender_char"] = chars[int(r["defender"][1])]
        vfl = game_to_video(log, al, r["log_row"])
        r["video_frame_f"] = round(float(vfl), 2)
        r["video_frame"] = int(round(vfl))
        fd = r.pop("first_damage_row")
        r["first_damage_game_frame"] = None if fd is None else int(log["game_frame"][fd])
        r["first_damage_video_frame"] = None if fd is None else int(round(game_to_video(log, al, fd)))
        r["attacker_act_start_video"] = int(round(game_to_video(log, al, r["log_row"] - r["attacker_act_age"])))
    m = match_cv(rows, cv, args.tol)
    rounds = sorted({int(x) for x in np.unique(log["round"])})
    summ = dict(replay=stem, index=entry.get("index"), video=entry["video"], csv=entry["csv"],
                p1_char=chars[1], p2_char=chars[2], hp_max=hpmax, rounds=len(rounds),
                game_frames=int(log["n"]), openers=len(rows), opener_stats=stats,
                alignment={k: v for k, v in al.items() if k != "residuals"}, cv=m)
    res = dict(summary=summ, rows=rows, residuals=al["residuals"])
    cache.write_text(json.dumps(res, indent=1, default=float))
    return res


def relabel(entry: dict, res: dict) -> None:
    """Characters (logger sprite codes, char column, else action-name inference) and clean move / category
    (moves.classify) for a processed replay, cached or fresh: naming changes need no --force."""
    log = load_log(entry["csv"])
    chars, src = {}, {}
    for s in (1, 2):
        chars[s], src[s] = resolve_character(log[f"p{s}_act"], log.get(f"p{s}_char"), log.get(f"p{s}_sprite"))
    summ = res["summary"]
    summ["p1_char"], summ["p2_char"] = chars[1], chars[2]
    summ["char_source"] = {"p1": src[1], "p2": src[2]}
    for r in res["rows"]:
        side = int(r["attacker"][1])
        # last real attack (not a common / non-attack state such as Asuka's Draw, Leo's backturn
        # walk): the source of a projectile hit while the attacker does something else
        acts = log[f"p{side}_act"]
        r["last_real_attack_act"], r["last_real_attack_age"] = "", None
        for k in range(r["log_row"], max(0, r["log_row"] - 240), -1):
            mi = move_info(chars[side], acts[k])
            if mi and "nonattack" not in mi[3] and mi[2] not in ("other",) or (mi is None and not is_common(acts[k])):
                r["last_real_attack_act"], r["last_real_attack_age"] = acts[k], int(r["log_row"] - run_start(acts, k))
                break
        r["attacker_char"] = chars[int(r["attacker"][1])]
        r["defender_char"] = chars[int(r["defender"][1])]
        r["char"] = r["attacker_char"]
        r["char_source"] = src[int(r["attacker"][1])]
        r["move_guess"] = notation(r["last_attack_act"], r["char"]) if r["projectile_likely"] and r["last_attack_act"]             else notation(r["move_act"], r["char"])
        r.update(classify(r))


def set_own_priority(cls: int) -> None:
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), cls)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2, help="parallel CV pipeline runs (default 2)")
    ap.add_argument("--idle", action="store_true", help="run children at Idle instead of BelowNormal priority")
    ap.add_argument("--force", action="store_true", help="redo alignment/openers/strips (keeps CV runs)")
    ap.add_argument("--force-cv", action="store_true", help="rerun the CV pipeline too")
    ap.add_argument("--no-strips", action="store_true")
    ap.add_argument("--no-cv", action="store_true",
                    help="never start the CV pipeline (replays without CV output are skipped)")
    ap.add_argument("--only", nargs="*", help="replay stems to process")
    ap.add_argument("--quiet", type=int, default=1,
                    help="opener: defender in no damage state in the preceding N frames (default 1)")
    ap.add_argument("--tol", type=int, default=10, help="CV match window in video frames (default 10)")
    args = ap.parse_args()
    prio = IDLE if args.idle else BELOW_NORMAL
    set_own_priority(prio)

    entries = read_manifest()
    if args.only:
        entries = [e for e in entries if Path(e["video"]).stem in args.only]
    print(f"{len(entries)} complete manifest entries")
    todo = [] if args.no_cv else [e for e in entries if args.force_cv or not cv_done(Path(e["video"]).stem)]
    if todo:
        print(f"running CV pipeline on {len(todo)} video(s), {args.jobs} at a time")
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            for stem, dt, rc in ex.map(lambda e: run_pipeline(e, prio), todo):
                print(f"  {stem}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {dt / 60:.1f} min")

    results = []
    for e in entries:
        stem = Path(e["video"]).stem
        if not cv_done(stem):
            print(f"  {stem}: no CV output, skipped")
            continue
        fresh = args.force or not (run_dir(stem) / "openers.json").exists()
        res = process_replay(e, args)
        relabel(e, res)
        results.append((e, res, fresh))

    # counter-hit guess needs all replays (hitstop per move)
    rows = [r for _, res, _ in results for r in res["rows"]]
    mark_counter_hits(rows)
    if not args.no_strips:
        for e, res, fresh in results:
            if fresh or any("strip" not in r for r in res["rows"]):
                t0 = time.time()
                video_pass(e["video"], res["rows"], res["summary"]["replay"])
                print(f"  strips {res['summary']['replay']}: {len(res['rows'])} in {time.time() - t0:.0f}s")
                (run_dir(res["summary"]["replay"]) / "openers.json").write_text(json.dumps(res, indent=1, default=float))
    mark_counter_hits(rows)       # again, with the banner scores of the video pass
    write_outputs(results, rows)


def write_outputs(results, rows) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    flat = [{k: (json.dumps(r[k]) if isinstance(r.get(k), (dict, list)) else r.get(k)) for k in cols} for r in rows]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(flat), OUT / "openers.parquet")
    except Exception as ex:      # pragma: no cover
        print("parquet not written:", ex)
    with open(OUT / "openers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(flat)

    summ = [res["summary"] for _, res, _ in results]
    for s in summ:
        rr = [r for r in rows if r["replay"] == s["replay"]]
        s["counter_hits"] = sum(bool(r.get("counter_hit")) for r in rr)
    scols = ["replay", "index", "p1_char", "p2_char", "rounds", "game_frames", "openers", "counter_hits",
             "align_c", "align_rate", "align_edges", "align_resid_median", "align_resid_mad", "align_within_1",
             "cv_events", "recall", "recall_damage", "recall_combo_start", "precision", "precision_in_combo"]
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(scols)
        for s in summ:
            a, c = s["alignment"], s["cv"]
            w.writerow([s["replay"], s["index"], s["p1_char"], s["p2_char"], s["rounds"], s["game_frames"], s["openers"],
                        s["counter_hits"], round(a["c"], 2), round(a["rate"], 4), a["n_edges_found"],
                        a["resid_median"], a["resid_mad"], a["within_1"], c["cv_events"],
                        _r(c["recall"]), _r(c["recall_damage"]), _r(c["recall_combo_start"]), _r(c["precision"]),
                        _r(c["precision_in_combo"])])
    # overall
    n = len(rows)
    allres = [x for _, res, _ in results for x in res["residuals"]]
    tot_ev = sum(s["cv"]["cv_events"] for s in summ)
    overall = dict(
        replays=len(summ), openers=n,
        alignment=dict(edges=len(allres),
                       abs_median=_r(float(np.median(np.abs(allres)))) if allres else None,
                       within_1=_r(float(np.mean(np.abs(allres) <= 1))) if allres else None,
                       within_2=_r(float(np.mean(np.abs(allres) <= 2))) if allres else None),
        cv=dict(recall=_r(sum(r["cv_detected"] for r in rows) / n) if n else None,
                recall_damage=_r(sum(r["cv_damage_dt"] is not None for r in rows) / n) if n else None,
                recall_combo_start=_r(sum(r["cv_combo_dt"] is not None for r in rows) / n) if n else None,
                recall_excl_burst=(lambda sel: sum(r["cv_detected"] for r in sel) / len(sel) if sel else None)(
            [r for r in rows if not r["is_burst"]]),
        recall_multi_hit=_r(sum(r["cv_detected"] for r in rows if r["hits"] >= 2) / max(1, sum(r["hits"] >= 2 for r in rows))),
                precision=_r(sum(s["cv"]["precision"] * s["cv"]["cv_events"] for s in summ if s["cv"]["precision"] is not None) / tot_ev) if tot_ev else None,
                precision_in_combo=_r(sum(s["cv"]["precision_in_combo"] * s["cv"]["cv_events"] for s in summ if s["cv"]["precision_in_combo"] is not None) / tot_ev) if tot_ev else None),
        moves_by_char={ch: collections.Counter(r["move"] for r in rows if r["attacker_char"] == ch).most_common(15)
                       for ch in sorted({r["attacker_char"] for r in rows})},
        openers_by_char=dict(collections.Counter(r["attacker_char"] for r in rows).most_common()),
        openers_by_category=dict(collections.Counter(r["category"] for r in rows).most_common()),
        char_category={ch: dict(collections.Counter(r["category"] for r in rows if r["attacker_char"] == ch))
                       for ch in sorted({r["attacker_char"] for r in rows})},
        unmapped_moves=sorted({f'{r["attacker_char"]}:{r["move_src_act"]}' for r in rows if not r["mapped"]}),
        flags=dict(projectile_likely=sum(r["projectile_likely"] for r in rows), throws=sum(r["is_throw"] for r in rows),
                   supers=sum(r["is_super"] for r in rows), rc_recent=sum(r["rc_recent"] for r in rows),
                   counter_hit=sum(bool(r.get("counter_hit")) for r in rows),
                   after_guard_20f=sum(1 for r in rows if r["frames_since_guard"] is not None and r["frames_since_guard"] <= 20)),
        per_replay=summ,
    )
    (OUT / "summary.json").write_text(json.dumps(overall, indent=1, default=float))
    print(json.dumps({k: v for k, v in overall.items() if k != "per_replay"}, indent=1, default=float))
    print(f"-> {OUT / 'openers.parquet'} ({n} rows), {OUT / 'summary.csv'}, {OUT / 'summary.json'}")


def _r(x, k=3):
    return None if x is None else round(float(x), k)


if __name__ == "__main__":
    main()
