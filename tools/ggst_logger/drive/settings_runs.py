"""Training runs that need a character setting: open the pause menu, change named rows of
"Impostazioni specifiche personaggi", leave, reset (key 8), sample moves, and record which setting each
sample was made under in variants.jsonl (dataset/hitbox_db.py keeps those as separate rows).

Offline only. Every key is sent on a recognised screen and from that screen's list (SCREEN_KEYS);
values change only after the selected row has been read back (char_settings.py).

Usage: python settings_runs.py nagoriyuki          (the game on the training HUD or the settings screens)
"""
import json
import os
import sys
import time

import cv2

import char_settings as cs
import ggst
import nav
import train_sweep as ts

TRAIN_TITLE = (190, 150, 950, 215)      # "Impostazioni allenamenti"
SCREEN_KEYS = {"hud": {"enter", "8"}, "pause": {"w", "s", "u", "o"}, "train_settings": {"w", "s", "u", "o"}}
PAUSE_FIRST_ROW = (380, 232, 940, 260)
TRAIN_SPECIFIC_ROW = (380, 432, 940, 460)
VARIANTS = os.path.join(ts.LOGS, "variants.jsonl")


def key_allowed(screen, key):
    return key in SCREEN_KEYS.get(screen, ())


def screen_now():
    """Which screen of this path the game is on, or None."""
    if cs.on_screen():
        return "settings"
    if ts.ref_score("train_settings", TRAIN_TITLE) < 10:
        return "train_settings"
    if ts.ref_score("pause", ts.PAUSE_TITLE) < 10:
        return "pause"
    if ts.on_training():
        return "hud"
    return None


def send(screen, key):
    if not key_allowed(screen, key):
        raise SystemExit(f"key '{key}' is not allowed on '{screen}'")
    if screen_now() != screen:
        raise SystemExit(f"expected '{screen}', not there; no key sent")
    ggst.keys(f"{key}:70", 200)
    time.sleep(0.9)


def wait_screen(name, timeout=12):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if screen_now() == name:
            return
        time.sleep(0.5)
    raise SystemExit(f"'{name}' did not show; stopping")


def open_settings():
    """HUD -> pause -> training settings -> character settings."""
    s = screen_now()
    if s == "settings":
        return
    if s == "hud":
        send("hud", "enter")
        wait_screen("pause")
        s = "pause"
    if s == "pause":
        for _ in range(9):
            if ts.red_at(PAUSE_FIRST_ROW):
                break
            send("pause", "w")
        else:
            raise SystemExit("could not reach the pause menu's first row")
        send("pause", "u")
        wait_screen("train_settings")
    for _ in range(10):
        if ts.red_at(TRAIN_SPECIFIC_ROW):
            break
        send("train_settings", "s")
    else:
        raise SystemExit("could not reach the character settings row")
    send("train_settings", "u")
    wait_screen("settings")


def close_settings():
    """O back out to the HUD, one recognised screen at a time."""
    for _ in range(5):
        s = screen_now()
        if s == "hud":
            return
        if s == "settings":
            cs.press("o", after=0.9)
        elif s in ("train_settings", "pause"):
            send(s, "o")
        else:
            raise SystemExit("unrecognised screen while leaving the settings")
    raise SystemExit("could not get back to the training screen")


def apply(values):
    open_settings()
    for row, value in values:
        print("  set", row, "=", cs.set_value(row, value), flush=True)
    close_settings()
    ts.reset_position()
    time.sleep(1.5)


def note_variant(path, f0, f1, variant):
    with open(VARIANTS, "a") as f:
        f.write(json.dumps({"log": os.path.basename(path), "f0": f0, "f1": f1, "side": 1, "variant": variant}) + "\n")


def sample(log, notation, want, variant, out, reset=True):
    """One move after a fresh reset. The frames it took are recorded under the variant when there is one."""
    if reset:
        ts.reset_position()
    atts = ts.parse(notation)
    if not atts:
        out.write(json.dumps({"notation": notation, "skip": "not typable", "variant": variant}) + "\n")
        return []
    got = []
    for att in atts:
        r0 = log.last()
        f0 = int(r0["game_frame"])
        acts, tries = ts.attempt(log, att, want=want)
        r1 = log.last()
        if variant:
            note_variant(log.path, f0, int(r1["game_frame"]), variant)
        out.write(json.dumps({"notation": notation, "act": want, "acts": acts, "tries": tries, "variant": variant}) + "\n")
        out.flush()
        got += acts
    return got


NAGO_LEVELS = [
    ("Liv. 1", [("Blood Rage", "Normale"), ("Indicatore sangue", "Liv. 1 (fisso)")]),
    ("Liv. 2", [("Blood Rage", "Normale"), ("Indicatore sangue", "Liv. 2 (fisso)")]),
    ("Liv. 3", [("Blood Rage", "Normale"), ("Indicatore sangue", "Liv. 3 (fisso)")]),
    ("blood rage", [("Indicatore sangue", "Normale"), ("Blood Rage", "Continuo")]),
]
NAGO_NORMALS = ["5S", "2S", "5H", "2H", "6S", "6H", "j.S", "j.H", "j.2H"]


def nagoriyuki():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
    import moves as MV
    log = ts.Log()
    r = log.last()
    if not (r["p1_sprite"] or "").startswith("nag"):
        raise SystemExit(f"not Nagoriyuki: {r['p1_sprite']}")
    with open(os.path.join(ts.LOGS, "settings_Nagoriyuki.jsonl"), "a") as out:
        # specials do not change with the level: once, outside blood rage, reset after each
        apply(NAGO_LEVELS[0][1])
        for act, (notation, _n, _c, flags) in MV.M["Nagoriyuki"].items():
            if notation and "nonattack" not in flags:
                print(" ", notation, sample(log, notation, act, "", out), flush=True)
        for name, values in NAGO_LEVELS:
            print("level", name, flush=True)
            apply(values)
            cv2.imwrite(os.path.join(ts.LOGS, "ref", f"nago_{name.replace(' ', '_').replace('.', '')}.jpg"), ts.capture())
            for n in NAGO_NORMALS:
                print(" ", n, sample(log, n, None, name, out), flush=True)
            ts.reset_position()
            r0 = log.last()
            ts.near_phase(log, out)
            note_variant(log.path, int(r0["game_frame"]), int(log.last()["game_frame"]), name)
    # the user's settings back
    apply([("Indicatore sangue", "Normale"), ("Blood Rage", "Continuo")])
    print("done; Blood Rage back on Continuo", flush=True)


# ---- press-and-release sequences ----------------------------------------------------------------
# Ops: "236P+" motion and button, the button stays held; "^P" releases it; "6S" taps; "~300" waits;
# "j" jumps forward. Held buttons stay down through later ops.

def run_ops(log, ops, face):
    hwnd = ggst.find_window()
    if not hwnd:
        raise SystemExit("Strive window not found")
    held = set()

    def down(k):
        ggst._post(hwnd, ggst.SCAN[k], False)

    def up(k):
        ggst._post(hwnd, ggst.SCAN[k], True)

    try:
        for op in ops:
            if op.startswith("~"):
                time.sleep(int(op[1:]) / 1000)
            elif op.startswith("h"):                 # h3:600 holds a direction, d4 taps one
                d, _, ms = op[1:].partition(":")
                keys = ts.dir_keys(d, face)
                for k in keys:
                    down(k)
                time.sleep(int(ms) / 1000)
                for k in keys:
                    up(k)
            elif op.startswith("d"):
                keys = ts.dir_keys(op[1:], face)
                for k in keys:
                    down(k)
                time.sleep(0.06)
                for k in keys:
                    up(k)
            elif op == "j":
                keys = ["space", "d" if face == 0 else "a"]
                for k in keys:
                    down(k)
                time.sleep(0.07)
                for k in keys:
                    up(k)
                time.sleep(0.25)
            elif op.startswith("^"):
                k = ts.BTN[op[1]]
                up(k)
                held.discard(k)
            else:
                keep = op.endswith("+")
                motion, button = op.rstrip("+")[:-1] or "5", op.rstrip("+")[-1]
                last = set()
                for d in motion[:-1]:
                    now = set(ts.dir_keys(d, face))
                    for k in now - last:
                        down(k)
                    for k in last - now:
                        up(k)
                    last = now
                    time.sleep(0.04)
                now = set(ts.dir_keys(motion[-1], face)) | {ts.BTN[button]}
                for k in now - last - held:
                    down(k)
                for k in last - now:
                    up(k)
                time.sleep(0.06)
                for k in now:
                    if k == ts.BTN[button] and keep:
                        held.add(k)
                    elif k not in held:
                        up(k)
    finally:
        for k in held:
            up(k)


def seq(log, name, ops, out, variant=""):
    """One sequence after a reset; records the actions it produced."""
    ts.reset_position()
    time.sleep(0.5)
    ts.wait_idle(log, need=4, timeout=4)
    if not ts.on_training():
        raise SystemExit("not on the training screen; no key sent")
    r0 = log.last()
    f0, face = int(r0["game_frame"]), int(r0["p1_face"])
    run_ops(log, ops, face)
    time.sleep(0.5)
    ts.wait_idle(log, need=6, timeout=6)
    acts = []
    for x in log.rows(900):
        if int(x["game_frame"]) >= f0 and x["p1_act"] not in ts.IDLE and (not acts or acts[-1] != x["p1_act"]):
            acts.append(x["p1_act"])
    if variant:
        note_variant(log.path, f0, int(log.last()["game_frame"]), variant)
    out.write(json.dumps({"seq": name, "ops": ops, "acts": acts, "variant": variant}) + "\n")
    out.flush()
    print(" ", name, acts, flush=True)
    return acts


def char_order():
    """Character -> position on the character select, as the sweep found them."""
    st = json.load(open(ts.STATE))
    return {c: i for i, c in enumerate(st["chars"])}


def switch(char):
    log = ts.Log()
    r = log.last()
    code = (r["p1_sprite"] or "")[:3].upper()
    close_settings()
    if ts.MV.CHAR_CODES.get(code) == char or ts.MV.CHAR_CODES.get(code.lower()) == char:
        return ts.Log()
    ts.pause_to_charsel()
    ts.go_to_cell(ts.cell_list()[char_order()[char]])
    ts.confirm_into_training()
    log = ts.Log()
    code = (log.last()["p1_sprite"] or "")[:3].upper()
    got = ts.MV.CHAR_CODES.get(code) or ts.MV.CHAR_CODES.get(code.lower())
    if got != char:
        raise SystemExit(f"selected {got}, wanted {char}; stopping")
    return log


def outfile(char):
    return open(os.path.join(ts.LOGS, f"settings_{char.replace('?', '')}.jsonl"), "a")


def aba():
    log = switch("A.B.A")
    open_settings()
    print("  Jealous Rage options", cs.options("Jealous Rage"), flush=True)
    close_settings()
    apply([("Jealous Rage", "Normale")])
    with outfile("A.B.A") as out:
        done, want = ts.sweep_char("A.B.A", log, out)
    print("  plain moves:", sorted(want & done), flush=True)
    apply([("Jealous Rage", "Continuo")])
    print("done A.B.A; Jealous Rage back on Continuo", flush=True)


def zato():
    log = switch("Zato-1")
    open_settings()
    print("  Eddie options", cs.options("Indicatore vita Eddie"), flush=True)
    close_settings()
    apply([("Indicatore vita Eddie", "Infinito")])
    with outfile("Zato-1") as out:
        for summon in ("214H", "236K"):
            for b in "PKSH":
                seq(log, f"{summon} {b} hold, release", [summon, "~700", b + "+", "~450", "^" + b, "~1200"], out)
    apply([("Indicatore vita Eddie", "Normale")])
    print("done Zato-1; Eddie gauge back on Normale", flush=True)


def johnny():
    log = switch("Johnny")
    with outfile("Johnny") as out:
        for b in "PKS":
            seq(log, f"214{b} hold, release", [f"214{b}+", "~500", "^" + b, "~900"], out)
            seq(log, f"j.214{b} hold, release", ["j", f"214{b}+", "~300", "^" + b, "~900"], out)
        seq(log, "stance moves", ["214S+", "~300", "6S", "~400", "4S", "~400", "66S", "~400", "44S", "~400", "^S"], out)
        for m, name in (("6", "walk f"), ("4", "walk b")):
            seq(log, "stance " + name, ["214S+", "~300", m + "D", "~500", "^S"], out)
        seq(log, "stance D", ["214S+", "~300", "5D", "~500", "^S"], out)
    print("done Johnny", flush=True)


def jacko():
    log = switch("Jack-O")
    with outfile("Jack-O") as out:
        for name, ops in (("6S", ["6S"]), ("D", ["5D"]), ("214K", ["214K"]), ("236K", ["236K"])):
            seq(log, "held servant " + name, ["236P+", "~350"] + ops + ["~600", "^P", "~800"], out)
        seq(log, "held servant j.6S", ["236P+", "~300", "j", "6S", "~500", "^P", "~800"], out)
        seq(log, "held servant j.D", ["236P+", "~300", "j", "5D", "~500", "^P", "~800"], out)
    print("done Jack-O", flush=True)


def leo():
    log = switch("Leo")
    with outfile("Leo") as out:
        seq(log, "enter, S kept held, then 5P", ["214S+", "~400", "5P", "~700", "^S", "~500"], out)
        seq(log, "enter, S released, then 5P", ["214S+", "~100", "^S", "~500", "5P", "~700"], out)
        enter = ["214S+", "~100", "^S", "~500"]
        for b in "PKSHD":
            seq(log, f"bt.{b}", enter + ["5" + b, "~1000"], out)
        for m in ("214K", "214H", "632146S"):
            seq(log, "bt." + m, enter + [m, "~1500"], out)
        for m in ("66", "44"):
            seq(log, "bt." + m, enter + [m[0] + "D", "~400"], out)
        seq(log, "bt.22S", enter + ["22S", "~1000"], out)
    print("done Leo", flush=True)


def ky():
    log = switch("Ky")
    with outfile("Ky") as out:
        for m in ("236S", "236H"):
            for gap in (250, 450, 700):
                seq(log, f"{m} then 4P after {gap} ms", [m, f"~{gap}", "4P", "~1000"], out)
    open_settings()
    print("  Dragon Install options", cs.options("Dragon Install"), flush=True)
    close_settings()
    print("done Ky first part", flush=True)


# Menu order, as read from the row starting at Normale. "Peso 100t" is read differently each time.
FAUST_ITEMS = ["Normale", "Bomba", "Peso 100t", "Ciambella", "Banana", "Afro", "Meteora", "Mini Faust", "Tromba", "Martello"]


def faust(first_item=2):
    log = switch("Faust")
    with outfile("Faust") as out:
        opts = FAUST_ITEMS
        for i, item in enumerate(opts[first_item:], first_item):
            open_settings()
            raw = cs.step_to("What Could This Be?", opts, i)
            print(f"  item {i} {item}: OCR now reads '{raw}'", flush=True)
            if cs.closest(raw, opts) != i:          # never label a sample with the wrong item
                print(f"  item {i}: position not confirmed, skipped", flush=True)
                close_settings()
                continue
            close_settings()
            ts.reset_position()
            time.sleep(1.5)
            seq(log, f"236P, item {i} {item}", ["236P", "~1800"], out, variant=f"item {i} {item}")
    open_settings()
    print("  back to", cs.step_to("What Could This Be?", opts, 0), flush=True)
    close_settings()
    print("done Faust; What Could This Be? back on Normale", flush=True)


def faust2():
    log = switch("Faust")
    with outfile("Faust") as out:
        for pull, h in ((350, 150), (350, 500), (500, 250), (500, 700), (700, 400)):
            seq(log, f"41236K 4 H, gaps {pull} {h}", ["41236K", f"~{pull}", "d4", f"~{h}", "5H", "~1200"], out)
    print("done Faust NiceShot", flush=True)


def leo2():
    """Backturn: S stays held through Turbulenz; does the stance last after release?"""
    log = switch("Leo")
    enter = ["214S+", "~900", "^S", "~300"]
    with outfile("Leo") as out:
        seq(log, "enter, S held 900 ms, released, wait", enter + ["~1200"], out)
        for b in "PKSHD":
            seq(log, f"bt.{b}", enter + ["5" + b, "~1000"], out)
            seq(log, f"bt.{b}, S still held", ["214S+", "~900", "5" + b, "~1000", "^S", "~300"], out)
        for m in ("214K", "214H", "632146S"):
            seq(log, "bt." + m, enter + [m, "~1500"], out)
        for d in "64":
            seq(log, f"bt.{d}", enter + [f"h{d}:600"], out)
            seq(log, f"bt.{d}{d}", enter + [f"d{d}", "~50", f"d{d}", "~500"], out)
        seq(log, "bt.8", enter + ["h8:100", "~900"], out)
        seq(log, "bt.22", enter + ["d2", "~50", "d2", "~800"], out)
        seq(log, "bt.22S", enter + ["22S", "~1000"], out)
    print("done Leo backturn", flush=True)


def johnny2():
    log = switch("Johnny")
    with outfile("Johnny") as out:
        for d in "64":
            seq(log, f"stance {d}{d}", ["214S+", "~300", f"d{d}", "~50", f"d{d}", "~500", "^S", "~400"], out)
    print("done Johnny dashes", flush=True)


def kydi():
    log = switch("Ky")
    apply([("Dragon Install", "Continuo")])
    with outfile("Ky") as out:
        for act, (notation, _n, _c, flags) in ts.MV.M["Ky"].items():
            if notation and "nonattack" not in flags:
                print(" ", notation, sample(log, notation, None, "Dragon Install", out), flush=True)
    apply([("Dragon Install", "Normale")])
    print("done Ky Dragon Install; back on Normale", flush=True)


def testament():
    log = switch("Testament")
    with outfile("Testament") as out:
        for m in ("236S", "236H"):
            for gap in (1100, 1500, 2000):
                seq(log, f"{m} then 214K after {gap} ms", [m, f"~{gap}", "214K", "~1500"], out)
    print("done Testament", flush=True)


def chipp():
    log = switch("Chipp")
    with outfile("Chipp") as out:
        for b in "PKSD":
            seq(log, f"214H held, then {b}", ["214H+", "~500", "5" + b, "~900", "^H", "~700"], out)
        seq(log, "214H held, released", ["214H+", "~700", "^H", "~900"], out)
        seq(log, "236S 236S", ["236S", "~250", "236S", "~1100"], out)
        seq(log, "236S 236K", ["236S", "~250", "236K", "~1100"], out)
        for b in "PK":
            ops = ["h6:4500", f"236{b}+", "~800", "5S", "~800", "^" + b, "~800"]
            seq(log, f"at the far wall: 236{b} held, then S", ops, out)
    print("done Chipp", flush=True)


def asuka_list():
    """Read the spell list once, top to bottom, from the open SPELL screen. Comes back to where it started."""
    start = cs.spell_highlighted()
    names, steps = [start], 0
    for _ in range(90):
        cs.press_spell("s")
        steps += 1
        n = cs.spell_highlighted()
        if cs.norm(n) == cs.norm(start):
            break
        names.append(n)
    # rotate so the list starts at the first row
    first = next(i for i, n in enumerate(names) if cs.norm(n) == "normale")
    ordered = names[first:] + names[:first]
    json.dump(ordered, open(os.path.join(ts.LOGS, "asuka_spells.json"), "w"), indent=0)
    print("  spell list", len(ordered), ordered, flush=True)
    return ordered


def asuka(deadline_hhmm="19:40"):
    """Each spell in slot 1, cast with 236P after a reset; which objects it makes is the spell's identity."""
    log = switch("Asuka")
    names = json.load(open(os.path.join(ts.LOGS, "asuka_spells.json")))
    original = "Delayed Howling Metron"
    with outfile("Asuka") as out:
        cur = cs.closest(original, names)
        for i, name in enumerate(names):
            if time.strftime("%H:%M") >= deadline_hhmm:
                print(f"  deadline: stopped before spell {i} {name}", flush=True)
                break
            open_settings()
            cs.open_slot(1)
            # the cursor opens on the slot's current spell
            for _ in range(abs(i - cur)):
                cs.press_spell("s" if i > cur else "w", after=0.1)
            raw = cs.spell_highlighted()
            if cs.closest(raw, names) != i:
                print(f"  spell {i} {name}: cursor reads '{raw}', not confirmed, skipped", flush=True)
                cs.press_spell("o", after=1.0)
                close_settings()
                continue
            cs.press_spell("u", after=1.0)
            cur = i
            close_settings()
            ts.reset_position()
            time.sleep(1.0)
            r0 = log.last()
            f0, face = int(r0["game_frame"]), int(r0["p1_face"])
            run_ops(log, ["236P", "~1800"], face)
            ts.wait_idle(log, need=4, timeout=4)
            objs, acts = set(), []
            for x in log.rows(500):
                if int(x["game_frame"]) < f0:
                    continue
                if x["p1_act"] not in ts.IDLE and (not acts or acts[-1] != x["p1_act"]):
                    acts.append(x["p1_act"])
                for it in filter(None, x["objs"].split("|")):
                    v = it.split("/")
                    if len(v) >= 3 and "Magic" in v[2]:
                        objs.add(v[2])
            out.write(json.dumps({"spell_index": i, "spell": name, "ocr": raw, "slot": 1, "deck": "Test case 2", "acts": acts,
                                  "objects": sorted(objs), "log": os.path.basename(log.path), "f0": f0}) + "\n")
            out.flush()
            print(f"  spell {i} {name}: {acts} objects {sorted(objs)}", flush=True)
    # the user's spell back
    open_settings()
    cs.open_slot(1)
    target = cs.closest(original, names)
    for _ in range(abs(target - cur)):
        cs.press_spell("s" if target > cur else "w", after=0.1)
    raw = cs.spell_highlighted()
    if cs.closest(raw, names) != target:
        raise SystemExit(f"could not put slot 1 back: cursor reads '{raw}'")
    cs.press_spell("u", after=1.0)
    print("  slot 1 now reads", cs.highlighted(), flush=True)
    close_settings()
    print("done Asuka; slot 1 back on", original, flush=True)


# Moves the sweep never saw that the user says are not in the game, or are the same move under another name.
NOT_MOVES = {"ElysionDriverEx", "SunVoid_Air", "AirZanseiRouga", "Zweihander_Land_Add", "SoregaOrenonada_exe"}


def retest(char):
    """The moves of this character still unseen, with the fixed inputs; command grabs next to the dummy."""
    st = json.load(open(ts.STATE))
    missing = [a for a in st["chars"][char]["missing"] if a not in NOT_MOVES]
    log = switch(char)
    with outfile(char) as out:
        for act in missing:
            notation, _n, _c, flags = ts.MV.M[char][act]
            if not notation or "nonattack" in flags:
                continue
            exe = act.endswith(("Exe", "_exe")) or "grab" in flags
            text = notation.replace(" ", "~")
            atts = ts.parse(text)
            if not atts:
                out.write(json.dumps({"notation": notation, "act": act, "skip": "not typable"}) + "\n")
                continue
            for att in atts:
                ops = []
                if exe:
                    ops.append("h6:600")
                for k, (air, charge, motion, button) in enumerate(att):
                    if k:
                        ops.append("~400")
                    if air:
                        ops.append("j")
                    if charge:
                        ops.append(f"h{charge}:750")
                    ops.append(motion + button)
                ops.append("~1200")
                seq(log, f"{act} {notation}", ops, out)
    print("done retest", char, flush=True)


if __name__ == "__main__":
    ts.MV = __import__("moves") if "moves" in sys.modules else None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
    import moves
    ts.MV = moves
    runs = {"nagoriyuki": nagoriyuki, "aba": aba, "zato": zato, "johnny": johnny, "jacko": jacko, "leo": leo, "ky": ky,
            "faust": faust, "faust2": faust2, "leo2": leo2, "johnny2": johnny2, "kydi": kydi,
            "testament": testament, "chipp": chipp, "asuka": asuka}
    for name in sys.argv[1:]:
        ts.guard()
        try:
            if name.startswith("retest:"):
                retest(name[7:])
            else:
                runs[name]()
        except SystemExit as e:                     # one failed run must not freeze the queue unnoticed
            print(f"RUN FAILED {name}: {e}", flush=True)
            if not ggst.find_window() or screen_now() is None:
                print("STOPPED: no game or an unrecognised screen; no key sent", flush=True)
                raise
    print("QUEUE DONE", flush=True)
