"""Record other players' replays, character by character, from Replay -> Search.

For each value of the search's "Character 1" filter and each rank tier (ranked
matches only), it searches and records the first --per-rank results with
batch.record_one, saving character and rank range in the manifest, then moves on. It stops when the filter wraps back to "all characters". The game
must already be on the Search tab's results list (after a first search).

Usage: python search_batch.py [--per-rank 2] [--index 1000]
"""
import argparse
import os
import time

import cv2

import batch
import ggst
import nav

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
ROW_Y = {1: (290, 330), 2: (345, 385), 3: (398, 438), 6: (555, 595)}  # filter-menu rows
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


def read_row(row):
    global CHAR_VALUE
    keep = CHAR_VALUE
    CHAR_VALUE = (960, ROW_Y[row][0], 1410, ROW_Y[row][1])
    try:
        return read_char_name()
    finally:
        CHAR_VALUE = keep


def set_filters(step_char, lo, hi):
    """From the results list: open the filter menu (J); set ranked-match rank range
    [lo, hi] (option indices); optionally step 'Character 1' right; Close so the
    results refresh. Returns (character, rank_lo, rank_hi) as read on screen."""
    nav.wait_for("search_list", box=batch.LIST_HEADER, timeout=30)
    for _ in range(3):
        ggst.keys("j", 300)
        time.sleep(2)
        if nav.score("filter_menu", FILTER_TITLE) < 10:
            break
    nav.wait_for("filter_menu", box=FILTER_TITLE, timeout=10, thr=10)
    ggst.keys("s", 250)                                   # row 1: search filter
    time.sleep(0.4)
    if "CLASSIF" not in read_row(1).upper():
        raise SystemExit("search filter is not 'ranked match'")
    # Highest to the top first so the lowest can move freely, then the real range.
    ggst.keys("s s", 250)
    time.sleep(0.4)
    set_rank(3, len(RANKS) - 1)
    ggst.keys("w", 250)
    time.sleep(0.4)
    rank_lo = set_rank(2, lo)
    ggst.keys("s", 250)
    time.sleep(0.4)
    rank_hi = set_rank(3, hi)
    ggst.keys("s s s", 250)                               # row 6: character 1
    time.sleep(0.4)
    if step_char:
        ggst.keys("d", 300)
        time.sleep(0.8)
    name = read_row(6)
    ggst.keys("s s s s s s", 200)                         # row 12: close
    time.sleep(0.5)
    ggst.keys("u", 300)
    time.sleep(1.5)
    if nav.score("filter_confirm", CONFIRM_BOX) < 10:      # "apply the changed filter?" -> Yes
        ggst.keys("u", 300)
    time.sleep(8)
    nav.wait_for("search_list", box=batch.LIST_HEADER, timeout=60)
    return name, rank_lo, rank_hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-rank", type=int, default=2, help="replays per character and rank tier")
    ap.add_argument("--index", type=int, default=1000, help="first manifest index")
    args = ap.parse_args()
    index = args.index
    char_step = False   # the first character is whatever the filter shows now
    while True:
        for t, (tier, lo, hi) in enumerate(TIERS):
            name, rank_lo, rank_hi = set_filters(char_step and t == 0, lo, hi)
            char_step = False
            if not name or name.upper().startswith("TUTT"):
                print("filter wrapped back to all characters; done")
                raise SystemExit(0)
            print(f"== character: {name} | {tier}: {rank_lo} .. {rank_hi}")
            for k in range(args.per_rank):
                try:
                    batch.record_one(index, list_ref="search_list", menu_ref="search_menu", menu_box=SEARCH_VIEW,
                                     extra={"source": "search", "char_filter": name, "tier": tier,
                                            "rank_lo": rank_lo, "rank_hi": rank_hi},
                                     end_ref="search_end", end_box=SEARCH_END)
                except SystemExit as e:
                    print(f"   skipped ({e})")
                    break
                index += 1
                try:
                    nav.wait_for("search_end", box=SEARCH_END, timeout=10, thr=10)
                    ggst.keys("u", 300)
                    time.sleep(6)
                    nav.wait_for("search_list", box=batch.LIST_HEADER, timeout=60)
                except SystemExit:
                    ggst.shot(os.path.join(batch.REC, f"after_search_{index}.png"))
                    raise SystemExit(f"unexpected screen after search replay {index}")
                ggst.keys("s", 300)
                time.sleep(1)
            ggst.keys(" ".join(["w"] * args.per_rank), 200)  # back to the top of the list
        char_step = True
