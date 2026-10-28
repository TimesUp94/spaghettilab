"""Record other players' replays, character by character, from Replay -> Search.

For each value of the search's "Character 1" filter and each rank tier (ranked
matches only), it searches and records the first --per-rank results with
batch.record_one, saving character and rank range in the manifest, then moves on. It stops when the filter wraps back to "all characters". The game
must already be on the Search tab's results list (after a first search).

Usage: python search_batch.py [--per-rank 2] [--index 1000] [--resume]

Progress is saved to ggst_rec/search_state.json after every replay. If Strive
disappears (it crashed after ~5 h and ~9 h of replays) or gets stuck, the
game is restarted and the batch continues where it stopped.
"""
import argparse
import ctypes
import json
import os
import subprocess
import time

import cv2

import batch
import ggst
import nav

HERE = os.path.dirname(os.path.abspath(__file__))

SEARCH_HEADER = (90, 200, 560, 290)      # "Cerca ..." title: tells the search tab from the other replay tabs
SEARCH_VIEW = (480, 495, 1410, 545)      # "Visualizza replay" in the search entry menu
SEARCH_END = (480, 340, 1440, 500)       # end-of-replay menu for downloaded replays (5 items)
FILTER_TITLE = (480, 130, 1440, 220)
CONFIRM_BOX = (480, 270, 1440, 780)      # "Modificare la condizione del filtro?" with "Sì" highlighted     # "MENU FILTRO" header of the J filter menu
CHAR_VALUE = (960, 555, 1410, 595)       # "Character 1" value in the filter menu


def read_char_name():
    """OCR the "Character 1" value; returns e.g. 'Sol' or 'TUTTI'."""
    import easyocr  # slow import, only when needed
    global _reader
    try:
        _reader
    except NameError:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    ggst.shot(nav.TMP)
    x0, y0, x1, y1 = CHAR_VALUE
    crop = cv2.imread(nav.TMP)[y0:y1, x0:x1]
    texts = [t for _, t, c in _reader.readtext(crop) if c > 0.3]
    return " ".join(texts).strip()


# Rank options in the filter (index: name in the Italian UI), grouped into tiers.
TIERS = [("Iron", 1, 3), ("Bronze", 4, 6), ("Silver", 7, 9), ("Gold", 10, 12),
         ("Platinum", 13, 15), ("Diamond", 16, 18), ("Vanquisher", 19, 22)]
# Filter-menu rows by name: centre y of the label. The menu gained floor rows
# ("Piano") in 2026-10, so rows are found by the highlighted (red) label, not counted.
ROWS_Y = {"filter": 308, "rank_lo": 361, "rank_hi": 414, "floor_lo": 467, "floor_hi": 520,
          "char1": 573, "char2": 626, "reset": 838, "close": 891}
# The rank options as the Italian UI reads them; the list wraps around at both ends.
RANKS = ["Non classificato", "Ferro 1", "Ferro 2", "Ferro 3", "Bronzo 1", "Bronzo 2", "Bronzo 3",
         "Argento", "Argento 2", "Argento 3", "Oro", "Oro 2", "Oro 3", "Platino 1", "Platino 2",
         "Platino 3", "Diamante", "Diamante 2", "Diamante 3", "DOMINATORE", "Dominatore Ignis",
         "Dominatore Virtus", "Dominatore III Vindex"]


def rank_index(text):
    import difflib
    m = difflib.get_close_matches(text, RANKS, n=1, cutoff=0.5)
    if not m:
        raise SystemExit(f"unreadable rank '{text}'")
    return RANKS.index(m[0])


def set_rank(row, target):
    """Cursor on a rank row: step right until it shows RANKS[target]."""
    for _ in range(3):
        cur = rank_index(read_row(row))
        if cur == target:
            return RANKS[target]
        ggst.keys(" ".join(["d"] * ((target - cur) % len(RANKS))), 120)
        time.sleep(0.5)
    raise SystemExit(f"could not set rank row {row} to {RANKS[target]}")


def highlighted():
    """Name of the highlighted filter-menu row (red label), None for unnamed rows."""
    ggst.shot(nav.TMP)
    im = cv2.imread(nav.TMP)
    for name, y in ROWS_Y.items():
        b, g, r = im[y - 15:y + 15, 500:940].reshape(-1, 3).mean(0)
        if r > 120 and g < 40 and b < 40:
            return name
    return None


def go_to_row(target):
    """Move the filter-menu cursor to the named row (up or down by the rows' order)."""
    order = list(ROWS_Y)
    for _ in range(16):
        cur = highlighted()
        if cur == target:
            return
        ggst.keys("w" if cur and order.index(cur) > order.index(target) else "s", 250)
        time.sleep(0.5)
    raise SystemExit(f"could not reach the filter row '{target}'")


def read_row(row):
    global CHAR_VALUE
    keep = CHAR_VALUE
    y = ROWS_Y[row]
    CHAR_VALUE = (960, y - 20, 1410, y + 20)
    try:
        return read_char_name()
    finally:
        CHAR_VALUE = keep


def set_filters(step_char, lo, hi, target=None):
    """From the results list: open the filter menu (J); set ranked-match rank range
    [lo, hi] (option indices); optionally step 'Character 1' right; Close so the
    results refresh. Returns (character, rank_lo, rank_hi) as read on screen."""
    if nav.score("filter_menu", FILTER_TITLE) >= 10:      # not already open (resumed run)
        nav.wait_for("search_list", box=SEARCH_HEADER, timeout=30)
        for _ in range(3):
            ggst.keys("j", 300)
            time.sleep(2)
            if nav.score("filter_menu", FILTER_TITLE) < 10:
                break
    nav.wait_for("filter_menu", box=FILTER_TITLE, timeout=10, thr=10)
    go_to_row("filter")
    for _ in range(5):                                    # make it "ranked match"
        if "CLASSIF" in read_row("filter").upper():
            break
        ggst.keys("d", 300)
        time.sleep(0.6)
    else:
        raise SystemExit("could not set the search filter to 'ranked match'")
    # Highest to the top first so the lowest can move freely, then the real range.
    go_to_row("rank_hi")
    set_rank("rank_hi", len(RANKS) - 1)
    go_to_row("rank_lo")
    rank_lo = set_rank("rank_lo", lo)
    go_to_row("rank_hi")
    rank_hi = set_rank("rank_hi", hi)
    go_to_row("char1")
    if step_char:
        ggst.keys("d", 300)
        time.sleep(0.8)
    name = read_row("char1")
    if target:                                            # resume: step until it shows target
        target = FILTER_NAMES.get(target, target)
        seen = []
        for _ in range(45):
            if norm(name) == norm(target):
                break
            ggst.keys("d", 300)
            time.sleep(0.8)
            name = read_row("char1")
            seen.append(name)
        else:
            raise SystemExit(f"could not find character '{target}' in the filter (read: {seen})")
    go_to_row("close")
    ggst.keys("u", 300)
    time.sleep(1.5)
    if nav.score("filter_confirm", CONFIRM_BOX) < 10:      # "apply the changed filter?" -> Yes
        ggst.keys("u", 300)
    time.sleep(8)
    nav.wait_for("search_list", box=SEARCH_HEADER, timeout=60)
    return name, rank_lo, rank_hi


# The filter's spelling where it differs from the character's usual name.
FILTER_NAMES = {"Happy Chaos": "Chaos"}


def norm(name):
    """Name as compared: letters and digits only, OCR look-alikes folded ('Jack-0')."""
    s = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    return s.translate(str.maketrans("015", "ols"))


SEARCH_START = (490, 310, 1430, 420)     # first visit: "set search conditions" with "Start search" highlighted
STATE = os.environ.get("GGST_SEARCH_STATE") or os.path.join(batch.REC, "search_state.json")
# Scheduled restarts are off: killing Strive while it shows its save icon can damage
# save data, so the game is only restarted after it has crashed (it did after ~5 h and
# ~9 h of replays).
RESTART_EVERY_S = None


class Restart(Exception):
    pass


UE4SS_LOG = r"C:\Program Files (x86)\Steam\steamapps\common\GUILTY GEAR STRIVE\RED\Binaries\Win64\UE4SS.log"


def online(host="steamcommunity.com", port=443, timeout=5):
    import socket
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def wait_online():
    """Downloaded replays and Steam's cloud sync need the network: wait for it rather
    than restarting the game into a Steam 'cloud sync failed' prompt."""
    if online():
        return
    print("!! no internet connection; waiting for it to come back")
    while not online():
        time.sleep(30)
    print("   back online")
    time.sleep(30)


def restart_game():
    """Start Strive through Steam and wait until the logger mod has loaded. A game
    that is still running is first asked to close (WM_CLOSE) and only killed if it
    has not exited after a minute."""
    hwnd = ggst.find_window()
    if hwnd:
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
        for _ in range(60):
            if "GGST-Win64" not in subprocess.run(["tasklist"], capture_output=True, text=True).stdout:
                break
            time.sleep(1)
    for exe in ("GGST-Win64-Shipping.exe", "GGST.exe"):
        subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True)
    for _ in range(60):
        out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
        if "GGST" not in out:
            break
        time.sleep(1)
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
                time.sleep(5)
                ggst.ensure_windowed()
                ggst.place_window()
                return
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit("Strive did not start")


def go_to_search():
    """Fresh game at the title -> Replay room -> Search tab results (no-op if already there)."""
    if nav.score("search_list", SEARCH_HEADER) < 10 and nav.score("search_start", SEARCH_START) >= 12:
        return
    if nav.score("filter_menu", FILTER_TITLE) < 10:       # filter menu open: set_filters goes on
        return
    if nav.score("search_end", SEARCH_END) < 10:         # a replay's end menu: back to the list
        ggst.keys("u", 300)
        time.sleep(6)
        nav.wait_for("search_list", box=SEARCH_HEADER, timeout=60)
        return
    nav.to_main()
    ggst.keys("d d d d d d", 300)
    time.sleep(1.5)
    nav.wait_for("collection", box=batch.COLLECTION_TAB, timeout=20)
    ggst.keys("u", 300)
    time.sleep(8)
    for _ in range(10):
        # The conditions dialog shares the list's header, so check it first.
        if nav.score("search_start", SEARCH_START) < 12:
            ggst.keys("u", 300)                           # "Start search"
            time.sleep(8)
            continue
        if nav.score("search_list", SEARCH_HEADER) < 14:
            return
        ggst.keys("l", 300)
        time.sleep(4)
    nav.wait_for("search_list", box=SEARCH_HEADER, timeout=30)


def save_state(**st):
    with open(STATE, "w") as f:
        json.dump(st, f)


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run(per_rank, state):
    """Record from `state` on ({'char', 'tier', 'k', 'index', 'step'}). Raises Restart
    after RESTART_EVERY_S so the game is restarted before it crashes."""
    t_start = time.time()
    index, target = state["index"], state.get("char")
    tier0, k0, step = state.get("tier", 0), state.get("k", 0), state.get("step", False)
    while True:
        for t in range(tier0, len(TIERS)):
            tier, lo, hi = TIERS[t]
            # First tier of this run: find the saved character, stepping one on if
            # it was finished. Later tiers keep the filter as it is.
            first = t == tier0
            name, rank_lo, rank_hi = set_filters(step and first, lo, hi,
                                                 target=target if first and not step else None)
            step, target = False, name
            if not name or name.upper().startswith("TUTT"):
                print("filter wrapped back to all characters; done")
                save_state(index=index, char=name, tier=0, k=0, step=False, done=True)
                return
            print(f"== character: {name} | {tier}: {rank_lo} .. {rank_hi}")
            k_first = k0 if first else 0
            if k_first:
                ggst.keys(" ".join(["s"] * k_first), 200)
                time.sleep(1)
            for k in range(k_first, per_rank):
                try:
                    batch.record_one(index, list_ref="search_list", menu_ref="search_menu", menu_box=SEARCH_VIEW,
                                     extra={"source": "search", "char_filter": name, "tier": tier,
                                            "rank_lo": rank_lo, "rank_hi": rank_hi},
                                     end_ref="search_end", end_box=SEARCH_END)
                except SystemExit as e:
                    if not ggst.find_window():
                        raise
                    if not online():                      # the replay could not be downloaded
                        raise
                    print(f"   skipped ({e})")
                    break
                index += 1
                if k + 1 < per_rank:
                    nxt = dict(index=index, char=name, tier=t, k=k + 1, step=False)
                elif t + 1 < len(TIERS):
                    nxt = dict(index=index, char=name, tier=t + 1, k=0, step=False)
                else:
                    nxt = dict(index=index, char=name, tier=0, k=0, step=True)
                save_state(**nxt)
                nav.wait_for("search_end", box=SEARCH_END, timeout=10, thr=10)
                ggst.keys("u", 300)
                time.sleep(6)
                nav.wait_for("search_list", box=SEARCH_HEADER, timeout=60)
                if RESTART_EVERY_S and time.time() - t_start > RESTART_EVERY_S:
                    raise Restart("scheduled restart")
                ggst.keys("s", 300)
                time.sleep(1)
            ggst.keys(" ".join(["w"] * per_rank), 200)       # back to the top of the list
        tier0, k0, step, target = 0, 0, True, None


def run_chars(chars, per_rank, state):
    """Like run(), for a list of characters (filter spellings or usual names): every rank
    tier of each. State {'index', 'ci', 'tier', 'k'} = next replay to record."""
    index = state["index"]
    for ci in range(state.get("ci", 0), len(chars)):
        for t in range(state.get("tier", 0) if ci == state.get("ci", 0) else 0, len(TIERS)):
            tier, lo, hi = TIERS[t]
            name, rank_lo, rank_hi = set_filters(False, lo, hi, target=chars[ci])
            print(f"== character: {name} | {tier}: {rank_lo} .. {rank_hi}")
            k0 = state.get("k", 0) if (ci, t) == (state.get("ci", 0), state.get("tier", 0)) else 0
            if k0:
                ggst.keys(" ".join(["s"] * k0), 200)
                time.sleep(1)
            for k in range(k0, per_rank):
                try:
                    batch.record_one(index, list_ref="search_list", menu_ref="search_menu", menu_box=SEARCH_VIEW,
                                     extra={"source": "search", "char_filter": name, "tier": tier,
                                            "rank_lo": rank_lo, "rank_hi": rank_hi},
                                     end_ref="search_end", end_box=SEARCH_END)
                except SystemExit as e:
                    if not ggst.find_window() or not online():
                        raise
                    print(f"   skipped ({e})")
                    break
                index += 1
                nxt = (ci, t, k + 1) if k + 1 < per_rank else (ci, t + 1, 0) if t + 1 < len(TIERS) else (ci + 1, 0, 0)
                save_state(index=index, ci=nxt[0], tier=nxt[1], k=nxt[2], chars=chars)
                nav.wait_for("search_end", box=SEARCH_END, timeout=10, thr=10)
                ggst.keys("u", 300)
                time.sleep(6)
                nav.wait_for("search_list", box=SEARCH_HEADER, timeout=60)
                ggst.keys("s", 300)
                time.sleep(1)
            ggst.keys(" ".join(["w"] * per_rank), 200)       # back to the top of the list
    save_state(index=index, ci=len(chars), tier=0, k=0, chars=chars, done=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-rank", type=int, default=2, help="replays per character and rank tier")
    ap.add_argument("--index", type=int, default=1000, help="first manifest index (new runs)")
    ap.add_argument("--resume", action="store_true",
                    help="restart Strive and continue from search_state.json")
    ap.add_argument("--chars", help="comma-separated characters to record (default: all, stepping)")
    args = ap.parse_args()
    chars = args.chars.split(",") if args.chars else None
    state = load_state() if args.resume else None
    if state and state.get("done"):
        raise SystemExit("search_state.json says the batch is complete")
    if not state:
        state = dict(index=args.index, char=None, tier=0, k=0, step=False, ci=0)
    if args.resume:
        wait_online()
        if not ggst.find_window():
            restart_game()
    need_nav = args.resume
    restarts = 0
    while True:
        try:
            if need_nav:                                  # a server error here restarts too
                go_to_search()
            if chars:
                run_chars(chars, args.per_rank, state)
            else:
                run(args.per_rank, state)
            break
        except (Restart, SystemExit, Exception) as e:
            restarts += 1
            print(f"!! restarting Strive ({type(e).__name__}: {e}); restart {restarts}")
            if restarts > 20:
                raise
            subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe"], capture_output=True)
            wait_online()
            try:
                restart_game()
            except SystemExit as e:                       # e.g. a Steam prompt blocking the launch
                print(f"!! {e}; trying again in 5 minutes")
                time.sleep(300)
                continue
            need_nav = True
            state = load_state() or state
