"""Check the move tables against the logged inputs.

For every special / super / command-normal action start in the logs, read the attacker's
inputs just before it (p*_rec, screen-relative, turned into numpad notation with p*_face)
and find the motion + button that started it: the latest button press up to 8 frames
before the action, and the longest motion in the 24 frames before that press. Per
(character, action) the most common reading is compared with the table's notation.

    .venv/Scripts/python.exe tools/ggst_logger/dataset/verify_inputs.py [--char Giovanna] [--all]

Prints one line per action: n, the top readings with their share, and the table's
notation; `!!` marks a table entry whose motion+button the inputs don't support (unless
the entry is flagged `dustloop`: checked on Dustloop's frame data), `??` an
action with no table entry. Writes the full table to $GGST_DATA/ggst_dataset/input_check.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import moves  # noqa: E402

DATA = Path(os.environ.get("GGST_DATA", r"G:\Projects\replanal-data"))
MANIFEST = DATA / "ggst_rec" / "manifest.jsonl"
OUT = DATA / "ggst_dataset" / "input_check.json"

U, D, L, R = 0x1, 0x2, 0x4, 0x8
BUTTONS = [(0x10, "P"), (0x20, "K"), (0x40, "S"), (0x80, "H"), (0x100, "D")]
PRESS_BEFORE = 8       # the action starts this many frames after the press at most (buffered)
MOTION_WINDOW = 24     # frames before the press searched for the motion
CHARGE = 30            # frames a charge direction must be held for [4]6 / [2]8
CHARGE_WINDOW = 90     # ...within this many frames before the release
# longest first: a 632146 also contains 214, a 41236 a 236. The 5-direction half circles
# are Goldlewis's Behemoth Typhoon.
MOTIONS = ["632146", "236236", "214214", "41236", "63214", "21478", "23698", "47896", "69874",
           "87412", "89632", "623", "421", "236", "214", "22"]


def numpad(bits: int, facing: int) -> str:
    """Screen-relative direction bits -> numpad digit for a player facing right (0) or left."""
    fwd, back = (R, L) if facing == 0 else (L, R)
    v = 1 if bits & D else (7 if bits & U else 4)
    h = 1 if bits & fwd else (-1 if bits & back else 0)
    return str(v + 1 + h)


def has_motion(dirs: list[str], m: str) -> bool:
    """`m` as an in-order subsequence of the direction changes (sloppy inputs pass)."""
    if m == "22":
        seq = "".join(d for i, d in enumerate(dirs) if i == 0 or d != dirs[i - 1])
        return re.search(r"[123][^123]+[123]", seq) is not None
    it = iter(d for i, d in enumerate(dirs) if i == 0 or d != dirs[i - 1])
    return all(any(c == d for d in it) for c in m)


def read_command(rec: list[int], face: list[int], t: int) -> list[str] | None:
    """The motion + button readings that could have started an action at frame t, the most
    likely first (None: no press found). All matching motions are kept: a jump-cancelled 236
    rolled into 6236 also reads as 623."""
    for b in range(t, max(0, t - PRESS_BEFORE) - 1, -1):
        pressed = [n for bit, n in BUTTONS if rec[b] & bit and not (b and rec[b - 1] & bit)]
        if pressed:
            break
    else:
        return None
    btn = "/".join(pressed) if len(pressed) > 1 else pressed[0]
    lo = max(0, b - MOTION_WINDOW)
    dirs = [numpad(rec[i], face[i]) for i in range(lo, b + 1)]
    out = [m + btn for m in MOTIONS if has_motion(dirs, m)]
    # charge: back (or down) held for CHARGE frames (in total: the game keeps a charge through
    # short gaps), then forward (up) by the press
    lo2 = max(0, b - CHARGE_WINDOW)
    held = [numpad(rec[i], face[i]) for i in range(lo2, b + 1)]
    for ch, rel, name in (("147", "369", "[4]6"), ("123", "789", "[2]8")):
        k = len(held) - 1
        while k >= 0 and held[k] in rel:      # the release, up to the press
            k -= 1
        if k < len(held) - 1 and sum(c in ch for c in held[:k + 1]) >= CHARGE:
            out.append(name + btn)
    out.append(dirs[-1].replace("5", "") + btn)
    return out


def load(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def norm_notation(n: str) -> set[str]:
    """Table notation -> the start commands it allows ('236K|214K' -> {236K, 214K})."""
    out = set()
    for alt in n.split("|"):
        alt = alt.strip().replace("j.", "")
        if "~" in alt or not alt:
            continue                     # follow-ups start from another action
        m = re.match(r"^(\[\d\]\d|\d*)([PKSHDX/]+)$", alt)
        if not m:
            continue
        motion, btns = m.groups()
        for b in btns.split("/"):
            out.add(motion + b)
    return out


def supports(read: str, allowed: set[str]) -> bool:
    """A reading backs a table command when the buttons agree (X = any) and the read motion
    contains the table's (41236S backs 236S)."""
    rm = re.match(r"^(\[\d\]\d|\d*)(.*)$", read)
    for a in allowed:
        am = re.match(r"^(\[\d\]\d|\d*)(.*)$", a)
        if am.group(2) in ("X", rm.group(2)) and am.group(1) in rm.group(1):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", help="only this character")
    ap.add_argument("--all", action="store_true", help="also list the actions whose table entry agrees")
    ap.add_argument("--min", type=int, default=5, help="skip actions seen fewer times (default 5)")
    args = ap.parse_args()

    entries = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    reads = collections.defaultdict(collections.Counter)
    cands = collections.defaultdict(list)       # every occurrence's candidate readings
    for e in entries:
        if not os.path.exists(e["csv"]):
            continue
        rows = load(e["csv"])
        if not rows or "p1_sprite" not in rows[0]:
            continue
        for s in (1, 2):
            acts = [r[f"p{s}_act"] for r in rows]
            char, _ = moves.resolve_character(set(acts), sprites=[r[f"p{s}_sprite"] for r in rows])
            if args.char and char != args.char:
                continue
            rec = [int(r[f"p{s}_rec"] or 0) for r in rows]
            face = [int(r[f"p{s}_face"] or 0) for r in rows]
            for t in range(1, len(rows)):
                a = acts[t]
                if a == acts[t - 1] or a.startswith(moves.COMMON_PREFIXES):
                    continue
                mi = moves.move_info(char, a)
                if mi and mi[2] not in (moves.SPECIAL, moves.SUPER, moves.CMD, moves.PROJ):
                    continue
                cmds = read_command(rec, face, t) or ["-"]
                reads[(char, a)][cmds[0]] += 1
                cands[(char, a)].append(cmds)

    report = {}
    for (char, a), c in sorted(reads.items()):
        n = sum(c.values())
        if n < args.min:
            continue
        mi = moves.move_info(char, a)
        table = mi[0] if mi else None
        allowed = norm_notation(table) if table else set()
        top = c.most_common(3)
        support = sum(any(supports(k, allowed) for k in cmds) for cmds in cands[(char, a)])
        state = c["-"] / n >= 0.7           # entered from another action, not by a press
        # a follow-up (a~b, 236HH) or a later state of the move (exe / cancel / follow-up
        # parts) starts from the earlier action, not from its own command
        follow = (bool(table) and ("~" in table or re.search(r"([PKSHD])\1$", table) is not None)
                  or re.search(r"(_?[Ee]xe|_Cancel|_Add)$", a) is not None)
        flag = ("??" if mi is None else
                "!!" if (allowed and support / n < 0.3 and not (state or follow)
                         and "dustloop" not in mi[3]) else "  ")
        report.setdefault(char, {})[a] = {"n": n, "top": top, "table": table, "support": round(support / n, 2)}
        if flag.strip() or args.all:
            tops = ", ".join(f"{k} {v / n:.0%}" for k, v in top)
            print(f"{flag} {char:12s} {a:32s} n={n:4d}  {tops:40s} table={table}")
    OUT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
