"""Per-move hitbox database from the logger's box columns (GGST_LOG_BOXES=1). Research data, stays local.

  python hitbox_db.py build     # logs -> $GGST_DATA/ggst_hitboxes (moves/<char>.json, summary.csv, ...)
  python hitbox_db.py verify    # do logged hits happen where the boxes overlap?
  python hitbox_db.py export    # db -> dustloop_hitboxes/gamedata/ for compare.html

Log format: see BoxRecord in ../Main.cs. Conventions:
- An instance of a move starts on the row where the action appears with action_frames <= 2 and ends
  when the action changes or action_frames restarts. Frame numbers are the game's action_frames: frame
  1 is the first frame of the move, hitstop repeats a number and is collapsed. Dustloop's startup N =
  first frame with an active hit box.
- Forward reach is the front of the hit box in game units minus the character's standing pushbox
  centre, behind reach the distance behind that centre. Moved reach adds the player's forward movement since
  the frame before the move: median of starts at rest when there are some, else of all starts.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re
import statistics
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moves as MV  # noqa: E402

DATA = os.environ.get("GGST_DATA", r"G:\Projects\replanal-data")
LOGS = os.path.join(DATA, "ggst_boxlogs")
OUT = os.path.join(DATA, "ggst_hitboxes")
DUST = os.path.join(DATA, "dustloop_hitboxes")
PX_PER_UNIT = 0.001518         # normalised px of Dustloop's images per game unit (our calibration)
MAX_FRAMES = 300
MAX_DX_SAMPLES = 400
HIT_TYPES = {1, 18}            # hit, follow-up hit
F_ACTIVE, F_PUSH, F_CH, F_STRIKE, F_THROW, F_WAKE, F_CINE, F_AIR, F_LANDED = 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 0x100
REST = {"CmnActStand", "CmnActCrouch", "CmnActStand2Crouch", "CmnActCrouch2Stand", "CmnActStandTurn",
        "CmnActCrouchTurn", "CmnActMidGuardEnd", "CmnActHighGuardEnd", "CmnActCrouchGuardEnd", ""}
BASELINE = {"CmnActStand", "CmnActCrouch", "CmnActFWalk", "CmnActBWalk", "CmnActFDash", "CmnActBDash",
            "CmnActJump", "CmnActJumpPre", "CmnActJumpLanding"}
# logger / moves.py character -> Dustloop page name
DL_NAME = {"Sol": "Sol Badguy", "Ky": "Ky Kiske", "Axl": "Axl Low", "Chipp": "Chipp Zanuff",
           "Millia": "Millia Rage", "Ramlethal": "Ramlethal Valentine", "Leo": "Leo Whitefang",
           "Anji": "Anji Mito", "Goldlewis": "Goldlewis Dickinson", "Sin": "Sin Kiske",
           "Bedman?": "Bedman", "Asuka": "Asuka R", "Elphelt": "Elphelt Valentine",
           "Dizzy": "Queen Dizzy", "Jam": "Jam Kuradoberi"}


# ---- parsing --------------------------------------------------------------------------------

def parse_record(rec):
    """(state fields, boxes string) of one entity record, or None."""
    head, sep, boxes = rec.partition(";")
    if not sep:
        return None
    p = head.split(":")
    if len(p) != 14:
        return None
    try:
        return [int(v, 16) if i == 5 else int(v) for i, v in enumerate(p)], boxes
    except ValueError:
        return None


def parse_boxes(s):
    """Box string -> [(type, back, bottom, front, top)]."""
    out = []
    if s:
        for b in s.split("+"):
            v = b.split(":")
            if len(v) == 5:
                out.append(tuple(int(x) for x in v))
    return out


class Expander:
    """Undoes the '=' delta coding per entity key."""
    def __init__(self):
        self.last = {}

    def __call__(self, key, s):
        if s == "=":
            return self.last.get(key, "")
        self.last[key] = s
        return s


def split_boxes(boxes):
    """(hit, hurt, other): tuples of (back, bottom, front, top); other keeps the type, zero-size points dropped."""
    hit, hurt, other = [], [], []
    for t, b, d, f, u in boxes:
        if f == b or u == d:
            continue
        if t in HIT_TYPES:
            hit.append((b, d, f, u))
        elif t == 0:
            hurt.append((b, d, f, u))
        else:
            other.append((t, b, d, f, u))
    return tuple(sorted(hit)), tuple(sorted(hurt)), tuple(sorted(other))


def rows_of(path):
    with open(path, encoding="ascii", errors="ignore", newline="") as f:
        head = f.readline().strip().split(",")
        if "p1_boxes" not in head:
            return
        ix = {k: i for i, k in enumerate(head)}
        n = len(head)
        for line in f:
            c = line.rstrip("\r\n").split(",")
            if len(c) == n:
                yield ix, c


def character_of(sprites):
    if not sprites:
        return None
    code = collections.Counter(s[:3] for s in sprites if len(s) >= 3).most_common(1)
    if not code:
        return None
    c = code[0][0]
    return MV.CHAR_CODES.get(c) or MV.CHAR_CODES.get(c.upper()) or MV.CHAR_CODES.get(c.lower()) or c


class Inst:
    __slots__ = ("act", "start_frames", "rows", "x0", "face", "rest", "last_atime", "connected", "owner_act", "owner_at")

    def __init__(self, act, x0, face, rest):
        self.act, self.x0, self.face, self.rest = act, x0, face, rest
        self.rows = {}
        self.last_atime = -1
        self.connected = False
        self.owner_act, self.owner_at = "", 0


def new_stat():
    return {"n": 0, "n_rest": 0, "frames": collections.defaultdict(collections.Counter),
            "active_whiff": collections.Counter(), "active_hit": collections.Counter(),
            "dx": collections.defaultdict(list), "dx_rest": collections.defaultdict(list),
            "dy": collections.defaultdict(list), "spawned_by": collections.Counter(), "sources": collections.Counter()}


def close_inst(inst, stats, key, source):
    """Reduce a finished instance into the stats of its key."""
    if inst is None or not inst.rows or min(inst.rows) > 2:
        return
    has_hit = any(r[0] for r in inst.rows.values())
    st = stats[key]
    st["n"] += 1
    st["n_rest"] += bool(inst.rest)
    st["sources"][source] += 1
    active = []
    for f, r in inst.rows.items():
        if f > MAX_FRAMES:
            continue
        hit, hurt, other, push, a, ch, si, ti, thr, dx, dy = r
        st["frames"][f][(hit, hurt, other, push, a, ch, si, ti, thr)] += 1
        for name, v in (("dx", dx), ("dy", dy)):
            if len(st[name][f]) < MAX_DX_SAMPLES:
                st[name][f].append(v)
        if inst.rest and len(st["dx_rest"][f]) < MAX_DX_SAMPLES:
            st["dx_rest"][f].append(dx)
        if a and (hit or thr >= 0):
            active.append(f)
    if has_hit or any(r[8] >= 0 for r in inst.rows.values()):
        st["active_hit" if inst.connected else "active_whiff"][tuple(sorted(active))] += 1
    if inst.owner_act:
        st["spawned_by"][(inst.owner_act, min(inst.owner_at, 10_000))] += 1


def keep_action(char, act, has_box_hit):
    if act in BASELINE or has_box_hit:
        return True
    mi = MV.move_info(char or "", act)
    return bool(mi and mi[0] and "nonattack" not in mi[3])


def scan(path):
    """One log -> stats per (kind, character, action), box type counts, characters seen."""
    stats = collections.defaultdict(new_stat)
    types = collections.Counter()
    exp = Expander()
    sides = {1: {"inst": None, "prev": None, "prevprev": None}, 2: {"inst": None, "prev": None, "prevprev": None}}
    obj_inst = {}
    sprites = {1: [], 2: []}
    source = "training" if os.path.basename(path) in training_logs() else "replay"
    n_rows = 0
    obj_owner = {}
    pending = []                                    # (side key, inst): moves closed before the character is known
    for ix, c in rows_of(path) or ():
        n_rows += 1
        attached = collections.defaultdict(list)
        obj_recs = {}
        for item in filter(None, c[ix["obj_boxes"]].split("|")):
            key, _, rec = item.partition("~")
            parsed = parse_record(rec)
            if not parsed:
                continue
            state, s = parsed
            boxes = parse_boxes(exp(key, s))
            for b in boxes:
                types[(b[0], b[3] != b[1] and b[4] != b[2])] += 1
            m = re.fullmatch(r"(P[12]|o\d+|x\d+)a\d+", key)
            if m:
                attached[m.group(1)].append((state, boxes))
            else:
                obj_recs[key] = (state, boxes)

        owner_now = {}
        for side in (1, 2):
            p = f"p{side}"
            act = c[ix[p + "_act"]]
            if len(sprites[side]) < 200:
                sprites[side].append(c[ix[p + "_sprite"]])
            parsed = parse_record(c[ix[p + "_boxes"]])
            sd = sides[side]
            if not parsed:
                continue
            state, s = parsed
            x, y, face, at, hitstop, flags = state[:6]
            thr = state[9]
            push = tuple(state[10:14])
            boxes = parse_boxes(exp(f"P{side}", s))
            for b in boxes:
                types[(b[0], b[3] != b[1] and b[4] != b[2])] += 1
            sgn = 1 if face == 0 else -1
            for st2, bx in attached.get(f"P{side}", ()):
                ax, ay, af = st2[:3]
                asgn = 1 if af == 0 else -1
                for t, b0, d0, f0, u0 in bx:
                    # attached entity's frame -> this player's frame
                    w0 = ax + asgn * b0 - x
                    w1 = ax + asgn * f0 - x
                    lo, hi = sorted((sgn * w0, sgn * w1))
                    boxes = boxes + [(t, lo, d0 + ay - y, hi, u0 + ay - y)]
            hit, hurt, other = split_boxes(boxes)
            inst = sd["inst"]
            if inst is None or inst.act != act or at < inst.last_atime:
                if inst is not None:
                    pending.append((side, inst))
                prev = sd["prev"]
                rest = bool(prev and prev[0] in REST and (sd["prevprev"] is None or sd["prevprev"][1] == prev[1]))
                inst = Inst(act, prev[1] if prev else x, face, rest)
                inst.rows = {}
                sd["inst"] = inst
            if at not in inst.rows:
                s_inst = 1 if inst.face == 0 else -1
                inst.rows[at] = (hit, hurt, other, push, bool(flags & F_ACTIVE), bool(flags & F_CH), bool(flags & F_STRIKE),
                                 bool(flags & F_THROW), thr, s_inst * (x - inst.x0), y)
            if flags & F_LANDED:
                inst.connected = True
            inst.last_atime = at
            sd["prevprev"] = sd["prev"]
            sd["prev"] = (act, x)
            owner_now[f"P{side}"] = (act, at)

        # objects
        names = {}
        for it in filter(None, c[ix["objs"]].split("|")):
            v = it.split("/")
            if len(v) >= 5:
                names["o" + v[0]] = (v[1], v[2])
        seen = set()
        for key, (state, boxes) in obj_recs.items():
            nm = names.get(key)
            if not nm:
                continue
            owner, act = nm
            x, y, face, at, hitstop, flags = state[:6]
            thr = state[9]
            hit, hurt, other = split_boxes(boxes)
            inst = obj_inst.get(key)
            if inst is None or inst.act != act or at < inst.last_atime:
                if inst is not None:
                    pending.append((key, inst))
                inst = Inst(act, x, face, False)
                o = owner_now.get(owner)
                inst.owner_act, inst.owner_at = o if o else ("", 0)
                obj_inst[key] = inst
                inst.face = face
            if at not in inst.rows:
                inst.rows[at] = (hit, hurt, other, tuple(state[10:14]), bool(flags & F_ACTIVE), False, False, False, thr,
                                 (1 if inst.face == 0 else -1) * (x - inst.x0), y)
            if flags & F_LANDED:
                inst.connected = True
            inst.last_atime = at
            seen.add(key)
            obj_owner[key] = owner
        for key in list(obj_inst):
            if key not in seen:
                pending.append((key, obj_inst.pop(key)))
    for side in (1, 2):
        if sides[side]["inst"] is not None:
            pending.append((side, sides[side]["inst"]))
    for key, inst in obj_inst.items():
        pending.append((key, inst))
    char = {s: character_of(sprites[s]) for s in (1, 2)}
    for key, inst in pending:
        if key in (1, 2):
            if not inst.rows:
                continue
            ch = char[key]
            has_hit = any(r[0] or r[8] >= 0 for r in inst.rows.values())
            if keep_action(ch, inst.act, has_hit):
                close_inst(inst, stats, ("move", ch or "?", inst.act), source)
        else:
            owner = obj_owner.get(key, "")
            ch = char.get(int(owner[1])) if owner[:1] == "P" else None
            if any(r[0] for r in inst.rows.values()):
                close_inst(inst, stats, ("object", ch or "?", inst.act), source)
    return {"stats": {k: freeze(v) for k, v in stats.items()}, "types": types, "chars": char, "rows": n_rows,
            "path": os.path.basename(path)}


_training = None


def training_logs():
    """Logs made by drive/train_sweep.py (the other logs are replays)."""
    global _training
    if _training is None:
        _training = set()
        try:
            st = json.load(open(os.path.join(LOGS, "sweep_state.json"), encoding="utf-8"))
            _training = {c["log"] for c in st.get("chars", {}).values()}
        except (OSError, ValueError):
            pass
    return _training


def freeze(st):
    return {"n": st["n"], "n_rest": st["n_rest"], "frames": {k: dict(v) for k, v in st["frames"].items()},
            "active_whiff": dict(st["active_whiff"]), "active_hit": dict(st["active_hit"]),
            "dx": dict(st["dx"]), "dx_rest": dict(st["dx_rest"]), "dy": dict(st["dy"]),
            "spawned_by": dict(st["spawned_by"]), "sources": dict(st["sources"])}


def merge(a, b):
    a["n"] += b["n"]
    a["n_rest"] += b["n_rest"]
    for f, cn in b["frames"].items():
        d = a["frames"].setdefault(f, {})
        for k, v in cn.items():
            d[k] = d.get(k, 0) + v
    for name in ("active_whiff", "active_hit", "spawned_by", "sources"):
        d = a[name]
        for k, v in b[name].items():
            d[k] = d.get(k, 0) + v
    for name in ("dx", "dx_rest", "dy"):
        d = a[name]
        for f, v in b[name].items():
            lst = d.setdefault(f, [])
            if len(lst) < MAX_DX_SAMPLES:
                lst.extend(v[:MAX_DX_SAMPLES - len(lst)])


def med(v):
    return int(statistics.median(v)) if v else None


def ranges(frames):
    out = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def summarize(key, st, push_centre):
    """DB entry of one move or object."""
    kind, char, act = key
    frames = []
    total_n = st["n"]
    for f in sorted(st["frames"]):
        cn = st["frames"][f]
        n = sum(cn.values())
        if n < max(1, 0.15 * total_n) and f > 60 and n < 2:
            continue
        top, c = max(cn.items(), key=lambda kv: kv[1])
        hit, hurt, other, push, a, ch, si, ti, thr = top
        rest = st["dx_rest"].get(f) or []
        anyx = st["dx"].get(f) or []
        frames.append({"f": f, "n": n, "share": round(c / n, 3), "hit": list(hit), "hurt": list(hurt), "other": list(other),
                       "push": list(push), "a": int(a), "ch": int(ch), "si": int(si), "ti": int(ti), "thr": thr,
                       "dx": med(rest) if len(rest) >= 1 else med(anyx), "dx_any": med(anyx), "dy": med(st["dy"].get(f) or [])})
    # active frames: the most common whole pattern of whiffs, else of connecting instances
    src = "whiff" if st["active_whiff"] else "connected"
    pool = st["active_whiff"] or st["active_hit"]
    active, first, cons = [], None, None
    if pool:
        firsts = collections.Counter()
        for t, n in pool.items():
            if t:
                firsts[t[0]] += n
        non_empty = {t: n for t, n in pool.items() if t}
        if non_empty:
            active = list(max(non_empty.items(), key=lambda kv: kv[1])[0])
            first = active[0]
            cons = round(firsts[first] / sum(firsts.values()), 3)
    by_f = {r["f"]: r for r in frames}
    reach = behind = reach_moved = behind_moved = reach_cur = None
    top = bot = None
    throw = False
    for f in active:
        r = by_f.get(f)
        if not r:
            continue
        hb = list(r["hit"])
        if r["thr"] >= 0 and not hb and r["push"]:      # a throw's range: pushbox front + range
            hb = [(0, 0, r["push"][2] + r["thr"], r["push"][3])]
            r["throw_box"] = list(hb[0])
            throw = True
        if not hb:
            continue
        fr = max(b[2] for b in hb) - push_centre
        bk = min(b[0] for b in hb) - push_centre
        reach = fr if reach is None else max(reach, fr)
        behind = -bk if behind is None else max(behind, -bk)
        dx = r["dx"] or 0
        reach_moved = fr + dx if reach_moved is None else max(reach_moved, fr + dx)
        behind_moved = -(bk + dx) if behind_moved is None else max(behind_moved, -(bk + dx))
        cur_c = (r["push"][0] + r["push"][2]) / 2 if r["push"] and r["push"][2] != r["push"][0] else push_centre
        v = max(b[2] for b in hb) - cur_c
        reach_cur = v if reach_cur is None else max(reach_cur, v)
        tp, bt = max(b[3] for b in hb), min(b[1] for b in hb)
        top = tp if top is None else max(top, tp)
        bot = bt if bot is None else min(bot, bt)
    box_share = None
    if active:
        sh = [by_f[f]["share"] for f in active if f in by_f]
        box_share = round(min(sh), 3) if sh else None
    mi = MV.move_info(char, act) if kind == "move" else None
    if kind == "object":
        sp = collections.Counter()
        for (oa, at), n in st["spawned_by"].items():
            sp[oa] += n
        spawned = sp.most_common(3)
        ats = [at for (oa, at), n in st["spawned_by"].items() if sp and oa == sp.most_common(1)[0][0] for _ in range(min(n, 50))]
        spawn_at = med(ats)
    else:
        spawned, spawn_at = None, None
    return {"kind": kind, "char": char, "act": act, "input": (mi[0] if mi else "") or "", "name": (mi[1] if mi else "") or "",
            "category": (mi[2] if mi else ""), "flags": sorted(mi[3]) if mi else [],
            "n": total_n, "n_rest": st["n_rest"], "n_whiff": sum(st["active_whiff"].values()), "n_conn": sum(st["active_hit"].values()),
            "sources": st["sources"], "first_active": first, "first_active_consistency": cons, "active_source": src if pool else None,
            "active_ranges": ranges(active), "active_frames": len(active), "box_consistency": box_share, "throw": throw,
            "reach": reach, "reach_cur_push": reach_cur, "behind": behind, "reach_moved": reach_moved, "behind_moved": behind_moved,
            "top": top, "bottom": bot, "dx_first_active": by_f.get(first, {}).get("dx") if first else None,
            "spawned_by": spawned, "spawn_at": spawn_at, "total_frames": max(by_f) if by_f else 0, "frames": frames}


# ---- Dustloop comparison ----------------------------------------------------------------------

def nk(s):
    return re.sub(r"[\s._\-]", "", str(s)).lower()


def load_dustloop():
    p = os.path.join(DUST, "hitboxes.json")
    d = json.load(open(p, encoding="utf-8"))
    by = collections.defaultdict(list)
    for e in d["entries"]:
        by[(e["character"], nk(e["input"]))].append(e)
    return by


def parse_active(raw):
    """Dustloop 'active' text -> total frames (sum of plain numbers, gaps in parentheses left out) or None."""
    if not raw:
        return None
    s = re.sub(r"\([^)]*\)", " ", raw)
    n = [int(x) for x in re.findall(r"\d+", s)]
    return sum(n) if n else None


def compare_dustloop(entries, shift):
    by = load_dustloop()
    rows = []
    for e in entries:
        if e["kind"] != "move" or not e["input"]:
            continue
        dn = DL_NAME.get(e["char"], e["char"])
        cands = by.get((dn, nk(e["input"])))
        if not cands:
            continue
        ours = e["first_active"]
        d = cands[0]
        reach_dl = [c["m"]["reach_norm"] for c in cands if c.get("m") and c["m"].get("reach_norm") is not None]
        reach_ours = None if e["reach"] is None else e["reach"] * PX_PER_UNIT
        best = min(reach_dl, key=lambda v: abs(v - reach_ours)) if reach_dl and reach_ours is not None else None
        best_cur = None
        if reach_dl and e["reach_cur_push"] is not None:
            rc = e["reach_cur_push"] * PX_PER_UNIT
            best_cur = min(reach_dl, key=lambda v: abs(v - rc)) - rc
        sh = None
        c = shift.get(dn) or {}
        r = c.get(e["input"]) or next((v for k, v in c.items() if nk(k) == nk(e["input"])), None)
        if r:
            sh = r
        rows.append({"char": e["char"], "input": e["input"], "act": e["act"], "n": e["n"], "n_whiff": e["n_whiff"],
                     "first_active": ours, "consistency": e["first_active_consistency"], "dl_startup": d.get("startup"), "dl_startup_raw": d.get("startup_raw"),
                     "startup_diff": None if ours is None or d.get("startup") is None else ours - d["startup"],
                     "active": e["active_frames"], "active_ranges": e["active_ranges"], "dl_active_raw": d.get("active_raw"), "dl_active": parse_active(d.get("active_raw")),
                     "active_diff": None if not e["active_frames"] or parse_active(d.get("active_raw")) is None else e["active_frames"] - parse_active(d.get("active_raw")),
                     "active_source": e["active_source"], "reach": e["reach"], "reach_px": None if reach_ours is None else round(reach_ours, 1),
                     "dl_reach_px": best, "reach_diff_px": None if best is None else round(reach_ours - best, 1),
                     "reach_diff_units": None if best is None else round((reach_ours - best) / PX_PER_UNIT),
                     "reach_cur_diff_px": None if best_cur is None else round(-best_cur, 1), "dl_images": len(reach_dl),
                     "shift_units": None if not sh else sh.get("shift_units"), "shift_startup": None if not sh else sh.get("startup"),
                     "shift_confidence": None if not sh else sh.get("confidence"), "dx_first_active": e["dx_first_active"],
                     "shift_diff_units": None if not sh or sh.get("shift_units") is None or e["dx_first_active"] is None else e["dx_first_active"] - sh["shift_units"]})
    return rows


# ---- build ------------------------------------------------------------------------------------

def build(args):
    files = sorted(glob.glob(os.path.join(LOGS, "inputs_*.csv")))
    print(len(files), "logs", flush=True)
    stats = {}
    types = collections.Counter()
    n_files = n_rows = 0
    logs_by_char = collections.Counter()
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(scan, files, chunksize=2):
            if not res["rows"]:
                continue
            n_files += 1
            n_rows += res["rows"]
            for s in (1, 2):
                if res["chars"][s]:
                    logs_by_char[res["chars"][s]] += 1
            for k, v in res["stats"].items():
                if k in stats:
                    merge(stats[k], v)
                else:
                    stats[k] = v
            types.update(res["types"])
    os.makedirs(os.path.join(OUT, "moves"), exist_ok=True)
    centres = {}
    for (kind, ch, act), st in stats.items():
        if kind == "move" and act == "CmnActStand":
            cn = collections.Counter()
            for f, d in st["frames"].items():
                for k, n in d.items():
                    cn[(k[3][0] + k[3][2]) / 2] += n
            centres[ch] = cn.most_common(1)[0][0] if cn else 0
    entries = []
    for key, st in stats.items():
        entries.append(summarize(key, st, centres.get(key[1], 0)))
    by_char = collections.defaultdict(list)
    for e in entries:
        by_char[e["char"]].append(e)
    index = []
    for ch, es in sorted(by_char.items()):
        es.sort(key=lambda e: (e["kind"], e["input"] or "~", e["act"]))
        with open(os.path.join(OUT, "moves", ch.replace("?", "") + ".json"), "w", encoding="utf-8") as f:
            json.dump({"char": ch, "push_centre": centres.get(ch, 0), "moves": es}, f)
        index += [{k: v for k, v in e.items() if k != "frames"} for e in es]
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"logs": n_files, "rows": n_rows, "logs_by_char": logs_by_char,
                   "box_types": {f"{t}{'' if real else ' (point)'}": n for (t, real), n in sorted(types.items())}, "moves": index}, f)
    cols = ["char", "kind", "act", "input", "name", "category", "n", "n_whiff", "n_conn", "n_rest", "first_active", "first_active_consistency",
            "active_ranges", "active_frames", "active_source", "box_consistency", "reach", "reach_cur_push", "behind", "reach_moved", "behind_moved",
            "dx_first_active", "top", "bottom", "total_frames", "spawn_at"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in index:
            w.writerow([e.get(c) for c in cols])
    # coverage per character: table moves seen, with hit boxes
    cov = []
    for ch in sorted(set(by_char) | set(MV.M)):
        es = {e["act"]: e for e in by_char.get(ch, []) if e["kind"] == "move"}
        table = set(MV.M.get(ch, {}))
        seen = table & set(es)
        hb = [a for a in seen if es[a]["first_active"] is not None]
        nm = [a for a, e in es.items() if a.startswith("NmlAtk") and e["first_active"] is not None]
        cov.append({"char": ch, "logs": logs_by_char.get(ch, 0), "table_moves": len(table), "seen": len(seen), "with_hitbox": len(hb),
                    "normals_with_hitbox": len(nm), "objects": sum(1 for e in by_char.get(ch, []) if e["kind"] == "object"),
                    "unseen": sorted(table - seen)})
    with open(os.path.join(OUT, "coverage.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0]))
        w.writeheader()
        w.writerows(cov)
    shift = json.load(open(os.path.join(DUST, "startup_shift.json"), encoding="utf-8"))
    cmp_rows = compare_dustloop(index, shift)
    if cmp_rows:
        with open(os.path.join(OUT, "compare_dustloop.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0]))
            w.writeheader()
            w.writerows(cmp_rows)
    json.dump(report(index, cov, cmp_rows, types, n_files, n_rows), open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(json.dumps(report(index, cov, cmp_rows, types, n_files, n_rows), indent=1)[:6000])


def report(index, cov, rows, types, n_files, n_rows):
    mv = [e for e in index if e["kind"] == "move" and e["first_active"] is not None]
    st = [r for r in rows if r["startup_diff"] is not None]
    ac = [r for r in rows if r["active_diff"] is not None]
    rc = [r for r in rows if r["reach_diff_px"] is not None]
    sf = [r for r in rows if r["shift_diff_units"] is not None]
    abs_med = lambda v: statistics.median(abs(x) for x in v) if v else None
    return {"logs": n_files, "rows": n_rows, "moves_with_hitbox": len(mv), "named_moves_with_hitbox": sum(1 for e in mv if e["input"]),
            "objects_with_hitbox": sum(1 for e in index if e["kind"] == "object"),
            "box_types": {f"{t}{'' if real else ' (point)'}": n for (t, real), n in sorted(types.items())},
            "startup": {"compared": len(st), "agree": sum(r["startup_diff"] == 0 for r in st), "within1": sum(abs(r["startup_diff"]) <= 1 for r in st)},
            "active": {"compared": len(ac), "agree": sum(r["active_diff"] == 0 for r in ac)},
            "reach": {"compared": len(rc), "median_abs_px": abs_med([r["reach_diff_px"] for r in rc]), "median_abs_units": abs_med([r["reach_diff_units"] for r in rc]),
                      "within_5px": sum(abs(r["reach_diff_px"]) <= 5 for r in rc), "median_abs_px_cur_push": abs_med([r["reach_cur_diff_px"] for r in rc if r["reach_cur_diff_px"] is not None])},
            "shift": {"compared": len(sf), "median_abs_units": abs_med([r["shift_diff_units"] for r in sf]), "within_5000": sum(abs(r["shift_diff_units"]) <= 5000 for r in sf)},
            "coverage": [{k: c[k] for k in ("char", "logs", "table_moves", "seen", "with_hitbox", "normals_with_hitbox", "objects")} for c in cov]}


# ---- verification: hits happen where boxes overlap ------------------------------------------

def overlaps(a, ax, asgn, b, bx, bsgn):
    """Boxes of two entities in world coordinates: x gap (<= 0 = touching in x) and whether they overlap."""
    a0, a1 = sorted((ax + asgn * a[0], ax + asgn * a[2]))
    b0, b1 = sorted((bx + bsgn * b[0], bx + bsgn * b[2]))
    return max(a0 - b1, b0 - a1)


def verify_file(path):
    """For each HP drop (hit-by player or object): did an active hit box overlap a hurt box that frame or just before?"""
    exp = Expander()
    res = collections.Counter()
    gaps = []
    win = collections.deque(maxlen=3)
    prev_hp = {1: None, 2: None}
    for ix, c in rows_of(path) or ():
        frame = {}
        for side in (1, 2):
            parsed = parse_record(c[ix[f"p{side}_boxes"]])
            if not parsed:
                continue
            st, s = parsed
            frame[side] = (st, parse_boxes(exp(f"P{side}", s)))
        objs = {}
        for item in filter(None, c[ix["obj_boxes"]].split("|")):
            key, _, rec = item.partition("~")
            p = parse_record(rec)
            if p:
                objs[key] = (p[0], parse_boxes(exp(key, p[1])))
        win.append((frame, objs))
        for d in (1, 2):
            try:
                hp = int(c[ix[f"p{d}_hp"]])
            except ValueError:
                continue
            drop = prev_hp[d] is not None and hp < prev_hp[d]
            prev_hp[d] = hp
            if not drop or c[ix[f"p{d}_act"]].startswith(("CmnActThrow", "ThrowExe", "Bunri")):
                continue
            by = c[ix[f"p{d}_hitby"]]
            if not by:
                res["no_source"] += 1
                continue
            a_side = 3 - d
            act = c[ix[f"p{a_side}_act"]]
            if by.startswith("P") and ("Throw" in act or "Throw" in c[ix[f"p{d}_act"]]):
                res["throw"] += 1
                continue
            best = None
            n_hit = 0
            for fr, ob in win:
                if d not in fr:
                    continue
                dst, dbx = fr[d]
                hurts = [b for b in dbx if b[0] == 0 and b[3] != b[1] and b[4] != b[2]]
                if by.startswith("P"):
                    src = fr.get(a_side)
                else:
                    src = ob.get(by)
                if not src:
                    continue
                ast, abx = src
                hits = [b for b in abx if b[0] in HIT_TYPES and b[3] != b[1] and b[4] != b[2]]
                n_hit += bool(hits)
                ax, ay, af = ast[:3]
                dx, dy, df = dst[:3]
                for h in hits:
                    for u in hurts:
                        gx = overlaps(h[1:], ax, 1 if af == 0 else -1, u[1:], dx, 1 if df == 0 else -1)
                        gy = max(ay + h[2] - (dy + u[4]), dy + u[2] - (ay + h[4]))
                        g = max(gx, gy)
                        best = g if best is None else min(best, g)
            kind = "player" if by.startswith("P") else "object"
            if not n_hit:
                res[kind + "_no_hitbox"] += 1
            elif best is None:
                res[kind + "_no_hurtbox"] += 1
            else:
                res[kind + ("_overlap" if best <= 0 else "_miss")] += 1
                if best > 0:
                    gaps.append(best)
    return res, gaps


def verify(args):
    files = sorted(glob.glob(os.path.join(LOGS, "inputs_*.csv")))
    tot = collections.Counter()
    gaps = []
    with Pool(args.workers) as pool:
        for r, g in pool.imap_unordered(verify_file, files, chunksize=2):
            tot.update(r)
            gaps += g
    gaps.sort()
    out = {"counts": dict(tot), "miss_gap_median": gaps[len(gaps) // 2] if gaps else None,
           "miss_gap_p90": gaps[int(len(gaps) * 0.9)] if gaps else None}
    print(json.dumps(out, indent=1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "verify_overlap.json"), "w"), indent=1)


# ---- export for compare.html ------------------------------------------------------------------

def export(args):
    dest = os.path.join(DUST, "gamedata")
    os.makedirs(dest, exist_ok=True)
    dust = load_dustloop()
    index = []
    for path in sorted(glob.glob(os.path.join(OUT, "moves", "*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        ch = d["char"]
        dn = DL_NAME.get(ch, ch)
        keep = []
        for e in d["moves"]:
            if e["first_active"] is None and not (e["act"] in BASELINE or e["kind"] == "object"):
                continue
            fr = [[r["f"], r["hit"], r["hurt"], r["other"], r["push"], r["a"], r["ch"], r["si"], r["ti"], r["dx"] or 0, r["dy"] or 0, r.get("throw_box")]
                  for r in e["frames"]]
            cands = dust.get((dn, nk(e["input"]))) if e["input"] else None
            name = cands[0]["name"] if cands else e["name"]
            keep.append({"id": e["act"], "k": e["kind"][0], "in": e["input"], "nm": name, "n": e["n"], "fa": e["first_active"],
                         "ar": e["active_ranges"], "af": e["active_frames"], "c": e["first_active_consistency"], "fr": fr, "dx": e["dx_first_active"]})
            index.append({"c": dn, "k": e["kind"][0], "a": e["act"], "in": e["input"], "nm": name, "n": e["n"], "fa": e["first_active"],
                          "af": e["active_frames"], "ar": e["active_ranges"], "r": e["reach"], "b": e["behind"], "rm": e["reach_moved"], "f": len(fr)})
        fname = re.sub(r"[^A-Za-z0-9]+", "_", dn)
        with open(os.path.join(dest, fname + ".js"), "w", encoding="utf-8") as f:
            f.write("window.GAME_DATA=window.GAME_DATA||{};window.GAME_DATA[" + json.dumps(dn) + "]=" + json.dumps({"moves": keep}, separators=(",", ":")) + ";\n")
    with open(os.path.join(dest, "index.js"), "w", encoding="utf-8") as f:
        f.write("window.GAME_INDEX=" + json.dumps({"px_per_unit": PX_PER_UNIT, "moves": index}, separators=(",", ":")) + ";\n")
    print(len(index), "moves/objects exported to", dest)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "export"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"build": build, "verify": verify, "export": export}[a.cmd](a)
