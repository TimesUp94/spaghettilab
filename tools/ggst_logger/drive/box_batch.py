"""Play other players' replays from Replay -> Search with the logger's box columns on, without
recording video. Reuses search_batch's filter handling and batch's playing and screen checks; only
the log directory, the manifest and the way the game is restarted differ.

Usage: python box_batch.py --tiers Silver,Diamond,Vanquisher --per-rank 1 [--chars Sol,Ky] [--resume]
Needs GGST_LOG_BOXES=1 and GGST_INPUT_LOG_DIR=<boxlogs> in %LOCALAPPDATA%\\ggst_logger.env (the game
is started through Steam, so it does not inherit this process's environment).
Writes <boxlogs>/manifest.jsonl and box_batch_state.json. Never touches ggst_logs or ggst_rec.
"""
import argparse
import ctypes
import json
import os
import subprocess
import time

import batch
import ggst
import nav
import search_batch as sb
import train_sweep as ts

ALL = ["Sol", "Ky", "May", "Axl", "Chipp", "Potemkin", "Faust", "Millia", "Zato-1", "Ramlethal", "Leo", "Nagoriyuki",
       "Giovanna", "Anji", "I-No", "Goldlewis", "Jack-O", "Happy Chaos", "Baiken", "Testament", "Bridget", "Sin", "Bedman?",
       "Asuka", "Johnny", "Elphelt", "A.B.A", "Slayer", "Dizzy", "Venom", "Unika", "Lucy", "Jam", "Robo-Ky"]


def close_game():
    """WM_CLOSE only. If the game will not close, stop: it is not killed."""
    hwnd = ggst.find_window()
    if not hwnd:
        return
    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
    for _ in range(120):
        if "GGST-Win64" not in subprocess.run(["tasklist"], capture_output=True, text=True).stdout:
            return
        time.sleep(1)
    raise ts.Stop("Strive did not close after WM_CLOSE; left running")


def restart_game():
    close_game()
    time.sleep(10)
    ts.start_game()


def guard_every_replay(fn):
    def wrapped(*a, **k):
        ts.guard()
        return fn(*a, **k)
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="Silver,Diamond,Vanquisher")
    ap.add_argument("--per-rank", type=int, default=1)
    ap.add_argument("--chars", help="comma-separated (default: all 34)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    nav.to_main = ts.to_main_checked               # no key on an unrecognised screen, even after a restart
    batch.RECORD = False
    batch.LOGS = ts.LOGS
    batch.REC = ts.LOGS
    batch.MANIFEST = os.path.join(ts.LOGS, "manifest.jsonl")
    sb.STATE = os.path.join(ts.LOGS, "box_batch_state.json")
    sb.TIERS = [t for t in sb.TIERS if t[0] in a.tiers.split(",")]
    batch.record_one = guard_every_replay(batch.record_one)
    chars = a.chars.split(",") if a.chars else ALL
    state = sb.load_state() if a.resume else None
    if state and state.get("done"):
        raise SystemExit("box_batch_state.json says the batch is complete")
    if not state:
        index = 0
        if os.path.exists(batch.MANIFEST):
            with open(batch.MANIFEST) as f:
                index = 1 + max([json.loads(line)["index"] for line in f if line.strip()], default=-1)
        state = dict(index=index, ci=0, tier=0, k=0)
    sb.wait_online()
    if not ggst.find_window():
        ts.start_game()
    restarts = 0
    while True:
        try:
            sb.go_to_search()
            sb.run_chars(chars, a.per_rank, state)
            print("done", flush=True)
            return
        except ts.Stop:
            raise
        except (sb.Restart, SystemExit, Exception) as e:
            restarts += 1
            print(f"!! restarting Strive ({type(e).__name__}: {e}); restart {restarts}", flush=True)
            if restarts > 20:
                raise
            sb.wait_online()
            try:
                restart_game()
            except SystemExit as e2:                  # e.g. a Steam prompt blocking the launch
                print(f"!! {e2}; trying again in 5 minutes", flush=True)
                time.sleep(300)
                continue
            state = sb.load_state() or state


if __name__ == "__main__":
    try:
        main()
    except ts.Stop as e:
        raise SystemExit(f"stopped: {e}")
