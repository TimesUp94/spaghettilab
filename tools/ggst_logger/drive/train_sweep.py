"""Training-mode sweep for the hitbox database: every character performs every move we can
input, whiffing at long range, with the logger's box columns on (GGST_LOG_BOXES=1).

Offline only: Dojo -> training, never a network mode. Keys in battle are limited to BATTLE_KEYS,
sent only while the training HUD is recognised; menus are entered only through pause_to_charsel,
which checks each screen (references made by hand in $GGST_BOXLOGS/ref, see make_refs).

Usage (game at the character select or in training, or at the title with --from-title):
  python train_sweep.py --chars Sol,Ky [--from-title] [--state sweep_state.json]
Writes one sweep_<char>.jsonl per character (input tried, actions that came out) next to the logs.
"""
import argparse
import ctypes
import ctypes.wintypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

import ggst
import nav

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dataset"))
import moves as MV  # noqa: E402

DATA = os.environ.get("GGST_DATA", r"G:\Projects\replanal-data")
LOGS = os.environ.get("GGST_INPUT_LOG_DIR") or os.path.join(DATA, "ggst_boxlogs")
REFS = os.path.join(LOGS, "ref")
STATE = os.path.join(LOGS, "sweep_state.json")

BTN = {"P": "i", "K": "j", "S": "o", "H": "l", "D": "p"}
BATTLE_KEYS = {"space", "a", "s", "d"} | set(BTN.values())     # in battle up is Space; W only moves menus
HUD_EMBLEM = (880, 70, 1040, 180)   # the round clock's infinity emblem: the same on every stage
HUD_CORR = 0.9
RESET_KEY = "8"                     # "Reimposta posizioni" in this PC's key configuration
PAUSE_TITLE = (480, 150, 960, 220)
DIALOG_BOX = (480, 270, 1440, 800)
SIDE_BOX = None
CHARSEL_BOX = (0, 700, 1920, 1080)
MIN_DISK = 25 * 1024 ** 3
MIN_COMMIT_KB = 6 * 1024 ** 2
START_COMMIT_KB = 12 * 1024 ** 2
CLOSE_COMMIT_KB = 2 * 1024 ** 2
UE4SS_LOG = r"C:\Program Files (x86)\Steam\steamapps\common\GUILTY GEAR STRIVE\RED\Binaries\Win64\UE4SS.log"
NEAR_WALL_FAR = 900_000       # walk back when the opponent is closer than this
IDLE = {"CmnActStand", "CmnActCrouch", "CmnActStand2Crouch", "CmnActCrouch2Stand", "CmnActStandTurn",
        "CmnActCrouchTurn", "CmnActBWalk", "CmnActFWalk", "CmnActCrouchGuardEnd", "CmnActHighGuardEnd",
        "CmnActMidGuardEnd", "CmnActJumpLanding", "CmnActLandingStiff"}

# Character select: rows as shown on screen, left to right (cursor walks them in this order).
GRID = [12, 11, 11]


def capture(tries=6):
    """A screenshot; captures can time out while the GPU is busy, so retry. Stops if none comes."""
    for _ in range(tries):
        try:
            ggst.shot(nav.TMP)
            im = cv2.imread(nav.TMP)
            if im is not None:
                return im
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("no screenshot from the game; stopping, no key sent")


def session_problem():
    """Batches run only on the unlocked console session."""
    out = subprocess.run(["query", "session"], capture_output=True, text=True, errors="ignore").stdout
    me = [l for l in out.splitlines() if l.startswith(">")]
    if not me or not me[0][1:].lower().startswith("console"):
        return "not on the console session: " + (me[0] if me else "?")
    if "LogonUI.exe" in subprocess.run(["tasklist", "/FI", "IMAGENAME eq LogonUI.exe"], capture_output=True, text=True, errors="ignore").stdout:
        return "the session is locked"
    return None


def free_commit_kb():
    out = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 1 << 40


def wait_for_monitor():
    """The screen checks need the borderless 1920x1080 window on the monitor at WINDOW_POS."""
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    said = False
    while not ggst.on_a_monitor(*ggst.WINDOW_POS):
        if not said:
            print(f"no monitor at {ggst.WINDOW_POS}: waiting, no key sent", flush=True)
            said = True
        time.sleep(30)


def guard():
    """Stop on a locked or remote session, or low disk; wait while free commit is under 6 GB
    or the game's monitor is missing."""
    p = session_problem()
    if p:
        raise Stop(p)
    wait_for_monitor()
    if shutil.disk_usage(LOGS).free < MIN_DISK:
        raise Stop("data drive under 25 GB free")
    said = False
    while True:
        if free_commit_kb() >= MIN_COMMIT_KB:
            return
        if not said:
            print("free commit under 6 GB: pausing", flush=True)
            said = True
        hwnd = ggst.find_window()
        if hwnd and free_commit_kb() < CLOSE_COMMIT_KB:      # give the machine the game's 5 GB back
            print("free commit under 2 GB: closing Strive (WM_CLOSE)", flush=True)
            ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
        time.sleep(60)


class Stop(Exception):
    pass


def make_refs(name):
    """Keep the current screen as a reference (only ever run by hand on a checked screen)."""
    os.makedirs(REFS, exist_ok=True)
    ggst.shot(os.path.join(REFS, name + ".png"))


def ref_score(name, box=None):
    cur = capture()
    ref = cv2.imread(os.path.join(REFS, name + ".png"))
    if ref is None:
        return 255.0
    return float(np.mean(np.abs(nav.thumb(nav.half_res(cur), box) - nav.thumb(nav.half_res(ref), box))))


def wait_ref(name, box=None, timeout=30, thr=10.0):
    t0 = time.time()
    s = 255.0
    while time.time() - t0 < timeout:
        s = ref_score(name, box)
        if s < thr:
            return
        time.sleep(0.7)
    raise SystemExit(f"timed out waiting for '{name}' (last diff {s:.1f}); stopping, no key sent")


def red_at(box):
    im = capture()
    x0, y0, x1, y1 = box
    b, g, r = np.median(im[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    return r > 110 and g < 45 and b < 45


def on_training():
    """The round clock's emblem is there. Correlation, not difference: a bright stage shows through it
    (0.99 on every stage so far; the pause menu, dialogs and character select are 0.3 or less)."""
    cur = nav.half_res(capture())
    ref = cv2.resize(cv2.imread(os.path.join(HERE, "ref", "training.jpg")), (1920, 1080))
    return float(np.corrcoef(nav.thumb(cur, HUD_EMBLEM).ravel(), nav.thumb(ref, HUD_EMBLEM).ravel())[0, 1]) >= HUD_CORR


def latest_log():
    files = glob.glob(os.path.join(LOGS, "inputs_*.csv"))
    return max(files, key=os.path.getmtime) if files else None


class Log:
    """The last rows of the newest log."""
    def __init__(self):
        self.path = latest_log()
        with open(self.path, encoding="ascii", errors="ignore") as f:
            self.cols = f.readline().strip().split(",")

    def rows(self, n=120):
        path = latest_log()
        if path != self.path:
            self.__init__()
        size = os.path.getsize(self.path)
        with open(self.path, "rb") as f:
            f.seek(max(0, size - 4000 * n))
            lines = f.read().decode("ascii", "ignore").split("\n")[1:-1]
        out = []
        for line in lines[-n:]:
            c = line.split(",")
            if len(c) >= 20:
                out.append(dict(zip(self.cols, c)))
        return out

    def last(self):
        r = self.rows(3)
        return r[-1] if r else None


def wait_idle(log, need=6, timeout=6.0):
    """Until P1 has been idle for `need` frames. Returns the rows seen since the call."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = log.rows(max(need, 8))
        if len(rows) >= need and all(r["p1_act"] in IDLE for r in rows[-need:]):
            return True
        time.sleep(0.1)
    return False


# ---- inputs -----------------------------------------------------------------------------------

def dir_keys(d, face):
    fwd, back = ("d", "a") if face == 0 else ("a", "d")
    return {"1": ["s", back], "2": ["s"], "3": ["s", fwd], "4": [back], "5": [], "6": [fwd],
            "7": ["space", back], "8": ["space"], "9": ["space", fwd]}[d]


def tokens(motion, button, face, hold=34):
    """Motion digits + button -> ggst.keys tokens, the button with the last direction."""
    toks = []
    ds = [c for c in motion]
    for i, d in enumerate(ds[:-1]):
        k = dir_keys(d, face)
        if k:
            toks.append("+".join(k) + f":{hold}")
        if ds[i + 1] == d:                 # 22H, 214214H: a repeated direction needs neutral between
            toks.append("wait")
    last = dir_keys(ds[-1], face) if ds else []
    toks.append("+".join(last + ([BTN[button]] if button else [])) + f":{max(hold, 50)}")
    return toks


def parse(notation):
    """Notation -> list of attempts, each [(air, charge, motion, button), ...] (one step per '~').
    None for what cannot be typed: stances, states, spells, buttons by name."""
    s = notation.strip().replace(" ", "")
    if not s:
        return None
    air = s.startswith("j.")
    if air:
        s = s[2:]
    out = []
    # alternatives inside one step: P/K -> two attempts
    alts = [""]
    steps = s.split("~")
    parsed = []
    for st in steps:
        m = re.fullmatch(r"(\[[1-9]\])?([1-9]*)((?:[PKSHDX]/?)+)", st)
        if not m:
            return None
        charge, motion, buttons = m.groups()
        bs = [b for b in buttons.replace("/", "") if b]
        bs = ["S" if b == "X" else b for b in bs]
        parsed.append((charge[1] if charge else "", motion or "5", bs))
    # Cartesian only for the first step (the rest take their first button)
    for b in parsed[0][2]:
        att = [(air, parsed[0][0], parsed[0][1], b)] + [(False, p[0], p[1], p[2][0]) for p in parsed[1:]]
        out.append(att)
    return out or None


def normals():
    g = [f"{d}{b}" for d in "52" for b in "PKSHD"] + ["6P", "6K", "6S", "6H", "3K"]
    a = [f"j.{b}" for b in "PKSHD"] + ["j.2K", "j.2H"]
    return g + a


def targets(char):
    """(notation, action or None) to try: common normals, then the character's table."""
    t = [(n, None) for n in normals()]
    seen = {n for n, _ in t}
    for act, (notation, _name, _cat, flags) in MV.M.get(char, {}).items():
        if not notation or notation in seen or "nonattack" in flags and "~" in notation:
            continue
        seen.add(notation)
        t.append((notation, act))
    return t


def roll(toks):
    """Play tokens with no neutral between directions: the next keys go down before the last come up, like
    a stick rolled by hand. Key-by-key calls left 4-13 neutral frames and motions failed (Bridget 236K)."""
    hwnd = ggst.find_window()
    if not hwnd:
        raise SystemExit("Strive window not found")
    down = set()
    try:
        for t in toks:
            if t == "wait":                         # a repeated direction: 22H needs neutral between
                for k in down:
                    ggst._post(hwnd, ggst.SCAN[k], True)
                down = set()
                time.sleep(0.034)
                continue
            names, _, ms = t.partition(":")
            now = set(names.split("+"))
            for k in now - down:
                ggst._post(hwnd, ggst.SCAN[k], False)
            for k in down - now:
                ggst._post(hwnd, ggst.SCAN[k], True)
            down = now
            time.sleep(int(ms or 50) / 1000)
    finally:
        for k in down:
            ggst._post(hwnd, ggst.SCAN[k], True)


def press(step, face, hold=34):
    air, charge, motion, button = step
    toks = []
    if charge:
        k = dir_keys(charge, face)
        ggst.keys("+".join(k) + ":750", 0)
    toks = tokens(motion, button, face, hold)
    for t in toks:
        name = t.split(":")[0].split("+")
        if t != "wait" and not set(name) <= BATTLE_KEYS:
            raise SystemExit(f"key outside the battle set: {t}")
    roll(toks)


def reset_position():
    """Training reset (key 8), only on the recognised training HUD."""
    if not on_training():
        raise SystemExit("not on the training screen; no reset sent")
    ggst.keys(f"{RESET_KEY}:70", 200)
    time.sleep(1.0)


def walk_back(log):
    """Keep the opponent far enough that the moves whiff. Long walk only if needed."""
    r = log.last()
    if not r:
        return
    gap = abs(int(r["p1_x"]) - int(r["p2_x"]))
    if gap >= NEAR_WALL_FAR:
        return
    back = "a" if int(r["p1_face"]) == 0 else "d"
    ms = min(3000, int((NEAR_WALL_FAR + 150_000 - gap) / 18))    # about 18 units/ms walking
    ggst.keys(f"{back}:{ms}", 0)
    time.sleep(0.25)


def attempt(log, att, want=None, retries=2):
    """One input, again with longer key holds when the wanted action (if known) did not come out.
    Returns the P1 actions (other than idle ones) during the attempt."""
    acts = []
    for k in range(retries + 1):
        for _ in range(40):                 # a super's cinematic hides the HUD: wait, send nothing
            if on_training():
                break
            time.sleep(0.5)
        else:
            raise SystemExit("not on the training screen; stopping, no key sent")
        if not wait_idle(log):
            time.sleep(1.0)
        walk_back(log)
        wait_idle(log, need=4, timeout=2)
        r0 = log.last()
        if not r0:
            raise SystemExit("no log rows")
        f0, face = int(r0["game_frame"]), int(r0["p1_face"])
        if att[0][0]:
            jump = "space+" + ("d" if face == 0 else "a") + ":70"      # forward jump
            ggst.keys(jump, 0)
            time.sleep(0.25 if len(att[0][2]) <= 1 else 0.16)
        hold = (34, 50, 67)[min(k, 2)]
        press(att[0], face, hold)
        for step in att[1:]:
            time.sleep(0.35)
            press(step, face, hold)
        time.sleep(0.35)
        for _ in range(60):                 # up to about 6 s of the move
            if wait_idle(log, need=4, timeout=0.2):
                break
        rows = [x for x in log.rows(500) if int(x["game_frame"]) >= f0]
        acts = []
        for x in rows:
            if x["p1_act"] not in IDLE and (not acts or acts[-1] != x["p1_act"]) and not x["p1_act"].startswith("CmnActJump"):
                acts.append(x["p1_act"])
        if want in acts if want else any(a.startswith(("NmlAtk", "Semuke")) or not a.startswith("Cmn") for a in acts):
            return acts, k
    return acts, retries


def near_phase(log, out):
    """Close-range versions, next to the dummy: c.S and the throw."""
    if not on_training():
        return
    wait_idle(log)
    r = log.last()
    face = int(r["p1_face"])
    fwd = "d" if face == 0 else "a"
    for _ in range(60):
        r = log.last()
        if abs(int(r["p1_x"]) - int(r["p2_x"])) < 200_000:
            break
        ggst.keys(f"{fwd}:400", 0)
    for name, att in (("c.S", [(False, "", "5", "S")]), ("Throw", [(False, "", "6", "D")])):
        wait_idle(log, need=4, timeout=4)
        r0 = log.last()
        f0 = int(r0["game_frame"])
        if not on_training():
            return
        press(att[0], int(r0["p1_face"]), 50)
        time.sleep(1.5)
        wait_idle(log, need=4, timeout=4)
        acts = []
        for x in log.rows(400):
            if int(x["game_frame"]) >= f0 and x["p1_act"] not in IDLE and (not acts or acts[-1] != x["p1_act"]):
                acts.append(x["p1_act"])
        out.write(json.dumps({"notation": name, "act": None, "acts": acts, "tries": 0}) + "\n")
        out.flush()


def sweep_char(char, log, out):
    done, want = set(), set(MV.M.get(char, {}))
    for notation, act in targets(char):
        atts = parse(notation)
        if not atts:
            out.write(json.dumps({"notation": notation, "act": act, "skip": "not typable"}) + "\n")
            continue
        for att in atts:
            acts, tries = attempt(log, att, want=act)
            done |= set(acts)
            out.write(json.dumps({"notation": notation, "act": act, "acts": acts, "tries": tries}) + "\n")
            out.flush()
        guard()
    near_phase(log, out)
    return done, want


# ---- menus -------------------------------------------------------------------------------------

def pause_to_charsel():
    """Training -> pause menu -> character select, every screen checked."""
    if ref_score("pause", PAUSE_TITLE) >= 10:        # a stopped run can leave the pause menu open
        for _ in range(30):                          # a super's cinematic: look again
            if on_training():
                break
            time.sleep(1)
        else:
            raise SystemExit("not on the training screen")
        wait_idle(Log(), need=6, timeout=8)
        for _ in range(3):
            ggst.keys("enter:80", 300)
            time.sleep(1.5)
            if ref_score("pause", PAUSE_TITLE) < 10:
                break
    wait_ref("pause", PAUSE_TITLE, timeout=8)
    for _ in range(8):
        if red_at((380, 432, 940, 460)):            # the "Character selection" row
            break
        ggst.keys("s", 300)
        time.sleep(0.5)
    else:
        raise SystemExit("could not reach the character select row")
    ggst.keys("u", 300)
    time.sleep(1.5)
    wait_ref("dialog", DIALOG_BOX, timeout=8)
    ggst.keys("a", 300)
    time.sleep(0.8)
    if not red_at((510, 700, 940, 760)):            # "Yes" highlighted
        raise SystemExit("dialog: Yes not highlighted; stopping")
    ggst.keys("u", 300)
    time.sleep(5)
    nav.wait_for("side", timeout=60)
    ggst.keys("a", 300)
    time.sleep(1)
    nav.wait_for("side", timeout=10)
    ggst.keys("u", 300)
    time.sleep(4)
    nav.wait_for("charsel", box=CHARSEL_BOX, timeout=90)


def confirm_into_training():
    nav.wait_for("charsel", box=CHARSEL_BOX, timeout=20)
    for _ in range(8):
        if on_training():
            break
        ggst.keys("u", 300)
        time.sleep(3)
    t0 = time.time()
    while not on_training():
        if time.time() - t0 > 60:
            raise SystemExit("the training screen did not show; stopping")
        time.sleep(1)
    time.sleep(2)


# Portrait centres on the 1080p character select: row -> (y, first x, count); cells are 100 px apart.
CELLS = {0: (820, 410, 12), 1: (900, 474, 11), 2: (980, 440, 11)}


def cell_list():
    return [(r, c) for r in CELLS for c in range(CELLS[r][2])]


def cursor_cell():
    """(row, col) of the P1 cursor: the portrait with the red frame."""
    im = capture()
    best, where = -1.0, None
    for r, (y, x0, n) in CELLS.items():
        for c in range(n):
            cx = x0 + 100 * c
            ring = np.vstack([im[y - 40:y - 30, cx - 40:cx + 40].reshape(-1, 3), im[y + 30:y + 40, cx - 40:cx + 40].reshape(-1, 3),
                              im[y - 40:y + 40, cx - 40:cx - 30].reshape(-1, 3), im[y - 40:y + 40, cx + 30:cx + 40].reshape(-1, 3)]).astype(int)
            v = float(np.mean((ring[:, 2] > 150) & (ring[:, 1] < 60) & (ring[:, 0] < 70)))
            if v > best:
                best, where = v, (r, c)
    return where if best > 0.2 else None


def go_to_cell(target):
    """Move the cursor to a portrait, checking where it is after each key."""
    for _ in range(40):
        nav.wait_for("charsel", box=CHARSEL_BOX, timeout=20)
        cur = cursor_cell()
        if cur is None:
            raise SystemExit("cursor not found on the character select")
        if cur == target:
            return
        if cur[0] != target[0]:
            key = "s" if cur[0] < target[0] else "w"
        else:
            key = "d" if cur[1] < target[1] else "a"
        ggst.keys(key, 200)
        time.sleep(0.5)
    raise SystemExit(f"could not reach cell {target}")


def start_game():
    """Start Strive through Steam after a crash; never closes or kills anything. The game takes
    about 5 GB of commit, so it starts only with room for that above the 6 GB floor."""
    wait_for_monitor()
    said = False
    while free_commit_kb() < START_COMMIT_KB:
        if not said:
            print("waiting for 12 GB of free commit before starting Strive", flush=True)
            said = True
        time.sleep(60)
    try:
        os.remove(UE4SS_LOG)
    except OSError:
        pass
    ggst.settings_windowed()
    os.startfile("steam://rungameid/1384160")
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            if "[InputLogger] UpdateBattle" in open(UE4SS_LOG, encoding="utf-8", errors="ignore").read():
                time.sleep(25)                 # a fresh launch can still close and relaunch itself
                if not ggst.find_window():
                    time.sleep(45)
                # A launch in exclusive fullscreen changes the monitor's mode for a moment, so the
                # monitor can look missing; keep trying until the game area is right.
                rc = ctypes.wintypes.RECT()
                for _ in range(24):
                    ggst.ensure_windowed()
                    ggst.place_window()
                    ctypes.windll.user32.GetClientRect(ggst.find_window(), ctypes.byref(rc))
                    if (rc.right, rc.bottom) == (1920, 1080):
                        return
                    time.sleep(5)
                raise SystemExit(f"game area is {rc.right}x{rc.bottom}, not 1920x1080; no key sent")
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit("Strive did not start")


def to_main_checked():
    """Title -> main menu. Unlike nav.to_main, no key on a screen it does not recognise: wait, then stop."""
    t0 = time.time()
    while time.time() - t0 < 240:
        if nav.score("main", nav.RETE_TAB) < 14:
            print("  on 'main'", flush=True)
            return
        if nav.score("title") < 14:
            ggst.keys("u", 300)
            time.sleep(6)
            continue
        time.sleep(3)
    raise SystemExit("neither the title nor the main menu showed; no key sent")


def to_charsel_from_title():
    """Fresh game: title -> character select with nav's checked steps."""
    to_main_checked()
    for name, box, keys, pause in nav.STEPS[:-1]:
        nav.wait_for(name, box=box, timeout=90)
        ggst.keys(keys, 300)
        time.sleep(pause)
    nav.wait_for("charsel", box=CHARSEL_BOX, timeout=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-charsel", action="store_true", help="start on the character select")
    ap.add_argument("--in-training", action="store_true", help="already in training as this index's character")
    ap.add_argument("--max-chars", type=int, default=34)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="start the game if needed; from the title screen")
    args = ap.parse_args()
    st = {"i": 0, "chars": {}}
    if args.resume and os.path.exists(STATE):
        st = json.load(open(STATE))
    path = cell_list()
    if args.fresh:
        if not ggst.find_window():
            start_game()
        elif nav.score("title") >= 14 and nav.score("main", nav.RETE_TAB) >= 14:
            raise SystemExit("--fresh with the game on another screen; no key sent")
        to_charsel_from_title()
    elif not args.from_charsel and not args.in_training:
        pause_to_charsel()
    n = 0
    restarts = 0
    while st["i"] < len(path) and n < args.max_chars:
        guard()
        if not (args.in_training and n == 0):
            go_to_cell(path[st["i"]])
            confirm_into_training()
        log = Log()
        r = log.last()
        code = (r["p1_sprite"] or "???")[:3].upper()
        char = MV.CHAR_CODES.get(code) or MV.CHAR_CODES.get(code.lower()) or code
        print(f"[{st['i']}] {char} ({r['p1_sprite']}) in {os.path.basename(log.path)}", flush=True)
        back = "a" if int(r["p1_face"]) == 0 else "d"
        ggst.keys(f"{back}:4000", 0)                   # to the wall
        try:
            with open(os.path.join(LOGS, f"sweep_{char.replace('?', '')}.jsonl"), "a") as out:
                done, want = sweep_char(char, log, out)
        except SystemExit as e:
            if ggst.find_window() or restarts >= 5:
                raise
            restarts += 1                              # Strive crashed: start it again, redo this character
            print(f"Strive closed during {char} ({e}); restart {restarts}", flush=True)
            time.sleep(20)
            start_game()
            to_charsel_from_title()
            args.in_training = False
            continue
        got = sorted(want & done)
        print(f"    {len(got)} of {len(want)} table moves came out; log rows {r['game_frame']}+", flush=True)
        st["chars"][char] = {"log": os.path.basename(log.path), "moves": len(got), "table": len(want),
                             "missing": sorted(want - done)}
        st["i"] += 1
        with open(STATE, "w") as f:
            json.dump(st, f, indent=1)
        n += 1
        if st["i"] < len(path) and n < args.max_chars:
            pause_to_charsel()
    print("done", st["i"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Stop as e:
        raise SystemExit(f"stopped: {e}")
