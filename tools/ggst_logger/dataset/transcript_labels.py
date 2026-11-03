"""Dense move transcript labels from the game logs (ROADMAP point 2).

One row per move start of either player in every replay of the v2 opener dataset (the replays of
openers.parquet, duplicates already dropped), with the video frame (60 fps, the same game -> video
mapping build.py uses for the openers), the move named like openers.parquet (char:move, CAT6
category, button/stance), its outcome (hit / block / whiff / teched / interrupted) and a link to
the opener row when one of its contacts opened a combo (counter hit / punish from there).

Writes $GGST_DATA/ggst_dataset/transcript/transcript_events.parquet, transcript_opener_links.parquet
(every opener row against its dense event) and transcript_stats.json (counts and the opener check).
Labels only: models see the video. Taxonomy, splits and the metric: model/transcript_eval.py.

    .venv/Scripts/python.exe tools/ggst_logger/dataset/transcript_labels.py [--jobs 4] [--only STEM ...]
    .venv/Scripts/python.exe tools/ggst_logger/dataset/transcript_labels.py --heuristic-check   # G2 logs: contact attribution without the hit-by field vs with it

What the log has (three generations, column sets):
- G0 (99 replays): act, face, x, y, hp, inputs. G1 (467): + sprite (character code + animation
  cel). G2 (524): + the pool objects and each player's hit-by field (the last player / pool
  object that touched them, set on the contact row, for hits and blocks alike, sticky).

Move starts (per player):
- a new action (p*_act changes) that is an attack: moves.move_info maps it and it is not
  `nonattack` (walks, dashes, Asuka's card draw, Leo's backturn ...), plus Roman Cancel (category
  other) and character actions moves.py does not know (mapped False; 6 in 60 sampled replays);
- the same normal / command normal / special again without another action in between (5P 5P,
  mashed 2K): the sprite returns to the action's first cel. Needs the sprite (G1, G2); in G0 such
  a restart stays part of the first event (`restart_detect` False);
- continuations are not new moves: an action that follows a move directly and has the same
  notation (NmlAtkThrow -> ThrowExe, Bloodsucking_Universe -> ...Exe), the same name plus an
  exe/Exe suffix, or no table entry (I-No Megalo_ThrowExe) extends that move's event;
- moves seen only as an object (G2): Happy Chaos's shots (ShootingObj) while he rolls / dashes in
  At the Ready (moves.OBJECT_MOVES) are synthetic events (kind `object`) on the row the shot
  appears. In G0/G1 they are not in the transcript.

Contacts on the defender (row h = the row before the defender's new state, like build.py's hit row):
- hit: the defender enters a damage state (build.DAMAGE_PREFIXES) on h+1, or a new HP-drop run
  starts on h while the defender is (or becomes) in a damage state;
- block: the defender is in a guard state on h+1 and one of: the hit-by field set on h (G2), chip
  damage on h, both players frozen (x, y, sprite) from h+1 for 10-17 rows after a change (a direct
  block's blockstop: 11-16 on the G2 logs), or (with a sprite) the defender alone frozen for 6+
  rows after its own sprite changed (a projectile's blockstop freezes only the defender). With a
  sprite, the defender's guard cel must be >= 3 (proximity guard shows cels 0-2, never a logged
  contact). Measured on 40 G2 replays against the hit-by field's changes: direct blocks found
  ~98%, object blocks ~84% (a repeated object block with the defender already in its blockstun
  pose is invisible without the field);
- teched: both players enter CmnActHajikare (throw tech) on h+1, credited to the thrower(s).
Hits on a KO's own row count (the round counter moves on that row).
A contact is credited to a move: with the hit-by field (G2) a direct contact to the attacker's
move at h, an object's to the move that made it (build.hit_source + the moves.classify rules:
summon commands, Asuka's spells); without it (G0, G1, an empty field, or no such move near the
object's start) heuristic_move. `--heuristic-check` runs every G2 replay both ways: the move
outcomes agree on 98.7% of the events.

Outcome of a move: hit if any credited contact hit, else teched, else block, else interrupted (the
mover was hit before any contact of the move), else whiff; Roman Cancel: none. counter_hit /
punish come from the linked opener (banner + log, build.mark_counter_hits); a non-opener hit is a
hit inside a combo (no counter hit possible) and gets counter_hit False.
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
from build import is_common, is_damage, is_guard, log_objects, run_dir  # noqa: E402
from moves import (OBJECT_MOVES, move_info, resolve_character, spell_info)  # noqa: E402

DATA = build.OUT                                   # $GGST_DATA/ggst_dataset
TR = DATA / "transcript"
CAT6 = {"normal": "normal", "command normal": "normal", "special": "special", "super": "super",
        "throw": "throw", "projectile": "projectile", "burst": "other", "other": "other"}

BOTH_STOP = (10, 17)  # rows both players stay frozen from a direct block (blockstop 11-16 on the G2 logs)
ALONE_STOP = 6        # rows the defender alone stays frozen from a projectile block
MIN_CEL = 3           # blockstun guard cel (proximity guard shows cels 0-2, never a contact on G2)
OBJ_GAP = 90         # the move behind an object ended at most this many rows before it appeared
PROJ_BACK = 240       # rows back for the projectile move behind a contact without the hit-by field
TRAIL = 3             # a direct contact this many rows after the move ended still counts (hit row trails)
RESTART_CATS = ("normal", "command normal", "special")
EXE_SUFFIX = ("Exe", "_exe", "_Exe", "exe")


# --------------------------------------------------------------------------------------
# Move segmentation
# --------------------------------------------------------------------------------------

def act_kind(char: str, act: str):
    """('move', move_info) | ('rc', info) | ('unmapped', None) | (None, info) for a non-move."""
    if act == "CmnActRomanCancel":
        return "rc", move_info(char, act)
    mi = move_info(char, act)
    if mi is None:
        return (None, None) if is_common(act) else ("unmapped", None)
    if "nonattack" in mi[3] or act.startswith("WorldSideBreak"):
        return None, mi
    return "move", mi


def continues(prev_act: str, prev_mi, act: str, kind: str, mi) -> bool:
    """`act` right after `prev_act` (a move) is the same move going on."""
    if kind == "unmapped":
        return True
    if kind != "move" or prev_mi is None:
        return False
    if act.startswith(prev_act) and act[len(prev_act):] in EXE_SUFFIX:
        return True
    if act in ("ThrowExe", "AirThrowExe") and prev_mi[2] == "throw":
        return True
    return bool(mi[0]) and mi[0] == prev_mi[0] and mi[2] == prev_mi[2]


def segment(log: dict, side: int, char: str) -> tuple[list[dict], bool]:
    """Move events of one player: dicts with start (row), end (row, exclusive), acts, mi, kind,
    restart. Second value: whether same-action restarts could be detected (sprite logged)."""
    acts = log[f"p{side}_act"]
    spr = log.get(f"p{side}_sprite")
    n = log["n"]
    events: list[dict] = []
    cur = None                 # the open event (its last segment ended at the current row)
    run0 = 0                   # first row of the current action run (sprite restart reference)
    for i in range(n):
        new_act = i == 0 or acts[i] != acts[i - 1]
        if new_act:
            run0 = i
        restart = False
        if not new_act and spr is not None and i > run0 and spr[i] == spr[run0] and spr[i - 1] != spr[run0]:
            kind, mi = act_kind(char, acts[i])
            restart = kind == "move" and mi[2] in RESTART_CATS
            if restart:
                run0 = i
        if not (new_act or restart):
            continue
        if cur is not None:
            cur["end"] = i
        kind, mi = act_kind(char, acts[i])
        if kind is None:
            cur = None
            continue
        if (new_act and cur is not None
                and continues(cur["acts"][-1], cur["mi_last"], acts[i], kind, mi)):
            cur["acts"].append(acts[i])
            if mi is not None:
                cur["mi_last"] = mi
            continue
        cur = dict(start=i, end=n, acts=[acts[i]], mi=mi, mi_last=mi, kind=kind, restart=restart)
        events.append(cur)
    return events, spr is not None        # an event's end: the boundary that closed it (else n)


# --------------------------------------------------------------------------------------
# Contacts
# --------------------------------------------------------------------------------------

def still_run(log: dict, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(changed, run): changed[j] = row j differs from j-1 in the given columns; run[j] = rows
    from j on equal to row j (j included)."""
    n = log["n"]
    same = np.ones(n, bool)
    same[0] = False
    for c in cols:
        v = log[c]
        same[1:] &= v[1:] == v[:-1]
    run = np.ones(n, np.int32)
    for j in range(n - 2, -1, -1):
        if same[j + 1]:
            run[j] = run[j + 1] + 1
    return ~same, run


def cel(sprite: str) -> int:
    """Animation cel of a sprite name (kyk043_03 -> 3), -1 when there is none."""
    t = sprite.rsplit("_", 1)
    return int(t[1]) if len(t) == 2 and t[1].isdigit() else -1


def contacts(log: dict, dside: int, use_hitby: bool) -> list[tuple[int, str]]:
    """(row h, 'hit' | 'block') contacts on player dside (h = the row before the new state)."""
    n = log["n"]
    dact = log[f"p{dside}_act"]
    hp = log[f"p{dside}_hp"]
    dmg = np.array([is_damage(a) for a in dact])
    grd = np.array([is_guard(a) for a in dact])
    sprite = "p1_sprite" in log
    pos = ["p1_x", "p1_y", "p2_x", "p2_y"] + (["p1_sprite", "p2_sprite"] if sprite else [])
    b_ch, b_run = still_run(log, pos)
    if sprite:
        dspr = log[f"p{dside}_sprite"]
        a_ch, a_run = still_run(log, [f"p{dside}_x", f"p{dside}_y", f"p{dside}_sprite"])
    hb = log.get(f"p{dside}_hitby") if use_hitby else None
    out = []
    last = {"hit": -9, "block": -9}
    for h in range(2, n - 1):
        # (the round counter moves on a KO's own row: no skip on round changes; a round start
        # raises HP, which is no drop)
        drop = hp[h] < hp[h - 1]
        new_drop = drop and not hp[h - 1] < hp[h - 2]
        if (dmg[h + 1] and not dmg[h]) or (new_drop and (dmg[h] or dmg[h + 1])):
            if h - last["hit"] > 1:
                out.append((h, "hit"))
            last["hit"] = h
            continue
        if not grd[h + 1]:
            continue
        pose = not sprite or cel(dspr[h + 1]) >= MIN_CEL      # proximity guard shows cels 0-2
        both = pose and b_ch[h + 1] and BOTH_STOP[0] <= b_run[h + 1] <= BOTH_STOP[1]
        alone = sprite and pose and dspr[h + 1] != dspr[h] and a_run[h + 1] >= ALONE_STOP
        ev = new_drop or (hb is not None and hb[h] != hb[h - 1] and hb[h] != "") or both or alone
        if ev and h - last["block"] > 3:
            out.append((h, "block"))
        if ev:
            last["block"] = h
    return out


def techs(log: dict) -> list[int]:
    a1, a2 = log["p1_act"], log["p2_act"]
    tt = np.array([x.startswith(build.THROW_TECH) and y.startswith(build.THROW_TECH) for x, y in zip(a1, a2)])
    return [int(h) for h in np.nonzero(tt[1:] & ~tt[:-1])[0]]          # tt[h+1] and not tt[h]


# --------------------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------------------

def event_at(evs: list[dict], row: int, trail: int = 0):
    """The event whose rows contain `row` (or ended at most `trail` rows before it)."""
    best = None
    for e in evs:
        if e["start"] > row:
            break
        if row < e["end"] + trail:
            best = e
    return best


def object_move(log: dict, dside: int, aside: int, h: int, char: str, evs: list[dict]):
    """The attacker's event that made the pool object which touched dside on row h (G2)."""
    src = build.hit_source(log, dside, aside, h)
    if src is None:
        return None, None
    spawn = OBJECT_MOVES.get((char, src["hit_object"])) or src["hit_object_spawn_act"]
    cmd = next((c for c in src["hit_object_cmd_acts"] or []
                if (m := move_info(char, c)) and "command" in m[3]), None)
    cands = ([cmd] if cmd else []) + [spawn] + list(src["hit_object_prior_acts"] or [])
    act = next((c for c in cands if (m := move_info(char, c)) and "nonattack" not in m[3]), None)
    if char == "Asuka" and spell_info(src["hit_object"]):
        act = next((c for c in cands if c in ("Magic", "Magic_air")), "Magic")
    if act is None:
        return None, src
    # the latest event of that action that started before the contact (a command: before h;
    # the spawn move: at or before the object's first row)
    limit = h if act == cmd else src["hit_object_spawn_row"] + 1
    best = None
    for e in evs:
        if e["start"] > limit:
            break
        if act in e["acts"]:
            best = e
    if best is not None and best["end"] < limit - OBJ_GAP:
        best = None              # no such move near the object's start (Happy Chaos fires while rolling)
    return best, src


def heuristic_move(evs: list[dict], h: int):
    """Without the hit-by field: the attacker's move at h (direct); when the attacker is in no move,
    a move that ended within TRAIL rows (the hit row trails), else its latest projectile move
    within PROJ_BACK rows, else its latest move there. ('direct' | 'object', event).
    On G2 against the hit-by field (object contacts): right for 85% of the contacts while the
    attacker is in another move (credited to that move), 78% while it is in none (projectile
    first: 61% with any latest attack first)."""
    e = event_at(evs, h)
    if e is not None and e["kind"] != "rc":
        return "direct", e
    e = event_at(evs, h, TRAIL)
    if e is not None and e["kind"] != "rc":
        return "direct", e
    att = [x for x in evs if h - PROJ_BACK <= x["start"] <= h and x["kind"] != "rc"]
    proj = [x for x in att if x["mi"] is not None and "proj" in x["mi"][3]]
    if proj or att:
        return "object", (proj or att)[-1]
    return None, None


def attribute(log: dict, evs: dict, chars: dict, use_hitby: bool) -> collections.Counter:
    """Fill every event's `contacts` [(row, kind, how)]; returns attribution stats."""
    st = collections.Counter()
    for side in (1, 2):
        for e in evs[side]:
            e["contacts"] = []
            e["hit_on_mover"] = []
    for dside in (1, 2):
        aside = 3 - dside
        hb = log.get(f"p{dside}_hitby") if use_hitby else None
        for h, kind in contacts(log, dside, use_hitby):
            # the defender's own move, if any, was interrupted by a hit
            if kind == "hit":
                own = event_at(evs[dside], h, 1)
                if own is not None:
                    own["hit_on_mover"].append(h)
            ev, how = None, None
            if hb is not None and hb[h] == f"P{aside}":
                ev = event_at(evs[aside], h, TRAIL)
                how = "hitby_direct"
                if ev is None:
                    how, ev = (lambda r: ("hitby_direct_heur", r[1]))(heuristic_move(evs[aside], h))
            elif hb is not None and hb[h].startswith("o"):
                ev, src = object_move(log, dside, aside, h, chars[aside], evs[aside])
                how = "hitby_object"
                if ev is None:
                    st["object_unresolved"] += 1
                    how, ev = (lambda r: ("hitby_object_heur", r[1]))(heuristic_move(evs[aside], h))
            else:
                via, ev = heuristic_move(evs[aside], h)
                how = f"heur_{via}"
            if ev is None:
                st[f"unattributed_{kind}"] += 1
                continue
            st[f"{kind}:{how}"] += 1
            ev["contacts"].append((h, kind, how))
    for h in techs(log):
        for side in (1, 2):
            e = event_at(evs[side], h, TRAIL)
            if e is not None and e["mi"] is not None and e["mi"][2] == "throw":
                e["contacts"].append((h, "teched", "tech"))
                st["teched"] += 1
    return st


def add_object_moves(log: dict, side: int, char: str, evs: list[dict]) -> list[dict]:
    """Moves seen only as an object (G2): Happy Chaos fires (ShootingObj) while rolling, dashing or
    walking in At the Ready, his action staying Roll / dash (moves.OBJECT_MOVES). A new such object
    while he is in no move is a synthetic move event on the row it appears."""
    want = {o: act for (ch, o), act in OBJECT_MOVES.items() if ch == char}
    if not want or "objs" not in log:
        return evs
    add = []
    prev = {}
    for k in range(log["n"]):
        cur = log_objects(log, k)
        for slot, (owner, oact) in cur.items():
            if owner == f"P{side}" and oact in want and prev.get(slot) != (owner, oact):
                e = event_at(evs, k)
                if e is None or e["kind"] == "rc":
                    mi = move_info(char, want[oact])
                    add.append(dict(start=k, end=k + 1, acts=[want[oact]], mi=mi, mi_last=mi, kind="object",
                                    restart=False))
        prev = cur
    return sorted(evs + add, key=lambda e: e["start"]) if add else evs


def outcome(e: dict) -> str:
    if e["kind"] == "rc":
        return "none"
    kinds = {k for _, k, _ in e["contacts"]}
    if "hit" in kinds:
        return "hit"
    if "teched" in kinds:
        return "teched"
    if "block" in kinds:
        return "block"
    if e["hit_on_mover"]:
        return "interrupted"
    return "whiff"


# --------------------------------------------------------------------------------------
# Per replay
# --------------------------------------------------------------------------------------

def spell_name(log: dict, side: int, e: dict) -> str | None:
    """Asuka's cast: the spell object (AttackMagic_NN) of this player appearing in the cast."""
    if "objs" not in log:
        return None
    before = set(log_objects(log, max(0, e["start"] - 1)).items())      # already out (the bookmark timer)
    for k in range(e["start"], min(log["n"], e["start"] + 40)):
        for o, (owner, act) in log_objects(log, k).items():
            if (owner == f"P{side}" and (o, (owner, act)) not in before and "TimeCount" not in act
                    and (sp := spell_info(act))):
                return sp[0]
    return None


def move_fields(char: str, e: dict, log: dict, side: int) -> dict:
    act = e["acts"][0]
    mi = e["mi"]
    if mi is None:            # unmapped first action
        mi = ("", act, "other", {"unmapped"})
    nota, name, cat, flags = mi
    if char == "Asuka" and act in ("Magic", "Magic_air"):
        sp = spell_name(log, side, e)
        if sp:
            name = sp
            nota = ("j." if act == "Magic_air" else "") + sp
    btn = next((f[4:] for f in flags if f.startswith("btn:")), None)
    stance = next((f[7:] for f in flags if f.startswith("stance:")), None)
    move = nota or name or act
    return dict(move=move, move_name=name, char_move=f"{char}:{move}", category=CAT6[cat], category_raw=cat,
                button=btn, stance=stance,
                normal=f"{btn}/{stance}" if cat in ("normal", "command normal") and btn and stance else None,
                proj="proj" in flags, mapped="unmapped" not in flags, move_flags=" ".join(sorted(f for f in flags if ":" not in f)) or None)


def manifest_frames() -> dict:
    out = {}
    for line in build.MANIFEST.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[Path(e["video"]).stem] = e.get("frames")
    return out


def process(args) -> dict:
    stem, csv_path, openers, nframes, use_hitby = args
    log = build.load_log(csv_path)
    summ = json.loads((run_dir(stem) / "openers.json").read_text())["summary"]
    al = summ["alignment"]
    gen = 2 if "p1_hitby" in log else (1 if "p1_sprite" in log else 0)
    chars = {s: resolve_character(log[f"p{s}_act"], log.get(f"p{s}_char"), log.get(f"p{s}_sprite"))[0] for s in (1, 2)}
    evs, restart_ok = {}, False
    for s in (1, 2):
        evs[s], restart_ok = segment(log, s, chars[s])
        evs[s] = add_object_moves(log, s, chars[s], evs[s])
    st = attribute(log, evs, chars, use_hitby)
    vf = lambda row: al["rate"] * log["t"][min(row, log["n"] - 1)] + al["c"]
    # opener link: the event credited with the opener's hit row
    by_hit = {}
    for s in (1, 2):
        for e in evs[s]:
            for h, k, _ in e["contacts"]:
                if k == "hit":
                    by_hit.setdefault((s, h), e)
    links = []
    for o in openers:
        s = int(o["attacker"][1])
        e = by_hit.get((s, o["log_row"]))
        if e is None:      # the hit row of the dense contacts can differ by a row
            e = next((by_hit[(s, o["log_row"] + d)] for d in (1, -1, 2, -2) if (s, o["log_row"] + d) in by_hit), None)
        links.append((o, e))
        if e is not None and "opener" not in e:
            e["opener"] = o
    rows = []
    for s in (1, 2):
        for e in evs[s]:
            f0 = vf(e["start"])
            o = e.get("opener")
            out = outcome(e)
            c_rows = sorted(h for h, _, _ in e["contacts"])
            hows = {how for _, _, how in e["contacts"]}
            rows.append(dict(
                replay=stem, log_gen=gen, side=f"P{s}", char=chars[s], opp_char=chars[3 - s],
                round=int(log["round"][e["start"]]), log_row=int(e["start"]), game_frame=int(log["game_frame"][e["start"]]),
                start_frame=int(round(f0)), start_frame_f=round(float(f0), 2),
                end_frame=int(round(vf(e["end"]))), dur_rows=int(e["end"] - e["start"]),
                in_video=bool(0 <= round(f0) < (nframes or 10 ** 9)),
                move_act=e["acts"][0], acts="|".join(e["acts"]), kind=e["kind"], restart=bool(e["restart"]),
                **move_fields(chars[s], e, log, s),
                outcome=out, n_hit=sum(k == "hit" for _, k, _ in e["contacts"]),
                n_block=sum(k == "block" for _, k, _ in e["contacts"]),
                first_contact_frame=int(round(vf(c_rows[0]))) if c_rows else None,
                first_contact_row=int(c_rows[0]) if c_rows else None,
                contact_via=("object" if all("object" in h for h in hows) else "direct" if not any("object" in h for h in hows) else "mixed") if hows else None,
                contact_attr=("hitby" if all(h.startswith("hitby") and not h.endswith("heur") for h in hows) else "heuristic") if hows else None,
                is_opener=o is not None, opener_log_row=None if o is None else int(o["log_row"]),
                opener_move=None if o is None else o["move"],
                # only an opener can be a counter hit / punish (later hits land on a defender in hitstun);
                # an opener's may be None (banner unclear)
                counter_hit=o.get("counter_hit") if o is not None and out == "hit" else False,
                punish=o.get("punish") if o is not None and out == "hit" else False,
                opener_category=None if o is None else o["category"],
                airborne=bool(log[f"p{s}_y"][e["start"]] > 0),
                distance=float(abs(log["p1_x"][e["start"]] - log["p2_x"][e["start"]])),
                restart_detect=restart_ok,
            ))
    rows.sort(key=lambda r: (r["log_row"], r["side"]))
    # validation of the opener link: same move, start within 1-2 rows of the opener's own move start
    acts = {s: log[f"p{s}_act"] for s in (1, 2)}
    val = []
    for o, e in links:
        s = int(o["attacker"][1])
        if o["move_src"] == "action":
            o_start = o["log_row"] - o["attacker_act_age"]
        elif o["move_src"] == "object" and o.get("hit_object_spawn_row") is not None:
            r = int(o["hit_object_spawn_row"])
            k = next((j for j in range(r, max(-1, r - 90), -1) if acts[s][j] == o["move_src_act"]), r)
            o_start = build.run_start(acts[s], k)
        else:
            o_start = o["log_row"] - (o.get("last_real_attack_age") or o["attacker_act_age"])
        ev_move = None if e is None else move_fields(chars[s], e, log, s)["move"]
        val.append(dict(replay=stem, log_row=o["log_row"], side=o["attacker"], char=o["char"], move=o["move"],
                        category=o["category"], move_src=o["move_src"], linked=e is not None,
                        ev_move=ev_move, ev_acts=None if e is None else "|".join(e["acts"]),
                        same_move=e is not None and ev_move == o["move"],
                        d_start=None if e is None else int(e["start"] - o_start),
                        ev_restart=None if e is None else bool(e["restart"]), gen=gen))
    return dict(stem=stem, rows=rows, val=val, stats=dict(st), gen=gen)


# --------------------------------------------------------------------------------------

def load_openers() -> dict:
    import pyarrow.parquet as pq
    cols = ["replay", "csv", "log_row", "attacker", "char", "move", "move_src", "move_src_act", "category",
            "attacker_act_age", "hit_object_spawn_row", "last_real_attack_age", "counter_hit", "punish"]
    by = collections.defaultdict(list)
    for r in pq.read_table(DATA / "openers.parquet", columns=cols).to_pylist():
        by[r["replay"]].append(r)
    return by


def heuristic_check(jobs: int, limit: int | None):
    """G2 logs: outcomes with the hit-by field vs the heuristic contact detection/attribution."""
    by = load_openers()
    stems = sorted(s for s, rr in by.items() if open(rr[0]["csv"]).readline().strip().endswith("p2_hitby"))[:limit]
    nf = manifest_frames()
    conf = collections.Counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        A = list(ex.map(process, [(s, by[s][0]["csv"], by[s], nf.get(s), True) for s in stems], chunksize=2))
        B = list(ex.map(process, [(s, by[s][0]["csv"], by[s], nf.get(s), False) for s in stems], chunksize=2))
    for a, b in zip(A, B):
        ka = {(r["side"], r["log_row"]): r for r in a["rows"]}
        for r in b["rows"]:
            t = ka.get((r["side"], r["log_row"]))
            if t is not None:
                conf[(t["outcome"], r["outcome"])] += 1
    tot = sum(conf.values())
    agree = sum(v for (x, y), v in conf.items() if x == y)
    res = dict(replays=len(stems), events=tot, agreement=agree / tot,
               confusion={f"{x}->{y}": v for (x, y), v in sorted(conf.items(), key=lambda kv: -kv[1])})
    out = TR / "transcript_heuristic_check.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--heuristic-check", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    build.set_own_priority(build.BELOW_NORMAL)
    TR.mkdir(parents=True, exist_ok=True)
    if a.heuristic_check:
        heuristic_check(min(a.jobs, 4), a.limit)
        return
    by = load_openers()
    stems = sorted(by)
    if a.only:
        stems = [s for s in stems if s in a.only]
    if a.limit:
        stems = stems[:a.limit]
    nf = manifest_frames()
    t0 = time.time()
    rows, val, stats, gens = [], [], collections.Counter(), collections.Counter()
    with ProcessPoolExecutor(max_workers=min(a.jobs, 4)) as ex:
        for k, res in enumerate(ex.map(process, [(s, by[s][0]["csv"], by[s], nf.get(s), True) for s in stems], chunksize=2)):
            rows += res["rows"]
            val += res["val"]
            stats.update(res["stats"])
            gens[res["gen"]] += 1
            if k % 100 == 0:
                print(f"{k}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    full = not a.only and not a.limit
    suf = "" if full else "_partial"
    pq.write_table(pa.Table.from_pylist(rows), TR / f"transcript_events{suf}.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(val), TR / f"transcript_opener_links{suf}.parquet", compression="zstd")
    summary = summarize(rows, val, stats, gens)
    (TR / f"transcript_stats{suf}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("moves_per_char", "mismatch_examples")}, indent=1, default=float))
    print(f"{len(rows)} events, {len(stems)} replays, {time.time() - t0:.0f}s -> {TR}")


def summarize(rows, val, stats, gens) -> dict:
    C = collections.Counter
    n = len(rows)
    reps = C(r["replay"] for r in rows)
    lk = [v for v in val if v["linked"]]
    near = lambda v, k: v["d_start"] is not None and abs(v["d_start"]) <= k
    mism = C((v["char"], v["move"], v["ev_move"]) for v in lk if not v["same_move"])
    unl = C((v["char"], v["move"], v["move_src"]) for v in val if not v["linked"])
    far = C((v["move_src"], "restart" if v["ev_restart"] else "act", "early" if v["d_start"] < 0 else "late")
            for v in lk if v["same_move"] and not near(v, 2))
    move_counts = C(r["char_move"] for r in rows)
    return dict(
        replays=len(reps), events=n, log_generations=dict(gens),
        events_per_replay=dict(mean=n / max(1, len(reps)), median=float(np.median(list(reps.values()))),
                               min=min(reps.values()), max=max(reps.values())),
        per_category=dict(C(r["category"] for r in rows).most_common()),
        per_outcome=dict(C(r["outcome"] for r in rows).most_common()),
        outcome_by_category={c: dict(C(r["outcome"] for r in rows if r["category"] == c).most_common())
                             for c in sorted({r["category"] for r in rows})},
        kinds=dict(C(r["kind"] for r in rows)), restarts=sum(r["restart"] for r in rows),
        unmapped=dict(C(r["char_move"] for r in rows if not r["mapped"]).most_common(20)),
        in_video=sum(r["in_video"] for r in rows),
        counter_hit_events=sum(bool(r["counter_hit"]) for r in rows),
        classes=dict(char_move=len(move_counts), char_move_ge20=sum(v >= 20 for v in move_counts.values()),
                     char_move_ge5=sum(v >= 5 for v in move_counts.values()),
                     normal=len({r["normal"] for r in rows if r["normal"]})),
        attribution=dict(stats),
        opener_check=dict(
            openers=len(val), linked=len(lk), linked_frac=len(lk) / max(1, len(val)),
            same_move=sum(v["same_move"] for v in lk), same_move_frac=sum(v["same_move"] for v in lk) / max(1, len(val)),
            start_within_1=sum(near(v, 1) for v in lk) / max(1, len(val)),
            start_within_2=sum(near(v, 2) for v in lk) / max(1, len(val)),
            match_move_and_start2=sum(v["same_move"] and near(v, 2) for v in lk) / max(1, len(val)),
            by_gen={g: dict(n=sum(v["gen"] == g for v in val),
                            match=sum(v["same_move"] and near(v, 2) for v in lk if v["gen"] == g) / max(1, sum(v["gen"] == g for v in val)))
                    for g in (0, 1, 2)},
            far_start_reasons={"/".join(k): v for k, v in far.most_common(10)},
            top_move_mismatches=[list(k) + [c] for k, c in mism.most_common(25)],
            top_unlinked=[list(k) + [c] for k, c in unl.most_common(15)]),
        moves_per_char={ch: C(r["move"] for r in rows if r["char"] == ch).most_common(12) for ch in sorted({r["char"] for r in rows})},
    )


if __name__ == "__main__":
    main()
