"""Record Strive's saved replays one after another, with the logger mod running.

For each replay: open it from Replay -> Saved, record the Strive window with
FFmpeg while the mod writes its per-frame CSV, stop when the end-of-replay menu
shows (or the log says the match is long over), then move to the next entry.
Every screen is checked against a reference screenshot before a key is pressed
(the replay menu also offers "Delete").

Usage: python batch.py [--start N] [--count N]
Writes $GGST_DATA/ggst_rec/manifest.jsonl (one line per replay).
"""
import argparse
import glob
import json
import os
import subprocess
import time

import ggst
import nav

DATA = os.environ.get("GGST_DATA", r"G:\Projects\replanal-data")
LOGS = os.path.join(DATA, "ggst_logs")
REC = os.path.join(DATA, "ggst_rec")
MANIFEST = os.path.join(REC, "manifest.jsonl")

COLLECTION_TAB = (1230, 830, 1410, 1010)
LIST_HEADER = (90, 200, 1830, 290)        # "Replay salvato" title row + tab icons
VIEW_ITEM = (480, 440, 1410, 490)         # "Visualizza replay" highlighted
END_MENU = (480, 390, 1440, 680)          # end-of-replay menu, "Torna all'elenco" highlighted

# The end menu normally shows 7-27 s after a player's CmnActMatchWin (149 logged
# replays); replay 2002 sat under it for 7.5 min because the live screen check never
# matched (the recording shows the menu from 15 s on). Stop that long after the win.
MATCH_OVER_S = 60
# Longest real replay: 26090 frames (7.3 min, 4 rounds with time-outs) of 596 recorded
# (median 8314, p99 14188). Nothing ranked runs past 10 min.
MAX_RECORD_S = 600


def to_saved_list():
    nav.to_main()
    ggst.keys("d d d d d d", 300)
    time.sleep(1.5)
    nav.wait_for("collection", box=COLLECTION_TAB, timeout=20)
    ggst.keys("u", 300)
    time.sleep(8)
    # Tabs load slowly and swallow presses made while loading: step one tab at a
    # time until the saved-replays tab shows.
    for _ in range(6):
        if nav.score("saved_list", LIST_HEADER) < 14:
            print("  on 'saved_list'")
            return
        ggst.keys("l", 300)
        time.sleep(4)
    nav.wait_for("saved_list", box=LIST_HEADER, timeout=30)


def newest_csv():
    files = glob.glob(os.path.join(LOGS, "inputs_*.csv"))
    return max(files, key=os.path.getmtime) if files else None


class LogTail:
    """Follows the logger's CSV as it grows: has a player gone into the match-win
    action yet (the match is over, only the outro is left)?"""

    def __init__(self, path):
        self.f = open(path, newline="")
        self.buf = ""
        self.acts = None
        self.match_over = False

    def poll(self):
        if self.match_over:
            return self
        self.buf += self.f.read()
        *lines, self.buf = self.buf.split("\n")   # a half-written last line waits
        for line in lines:
            cols = line.rstrip("\r").split(",")
            if self.acts is None:
                self.acts = [cols.index("p1_act"), cols.index("p2_act")]
            elif any(len(cols) > i and cols[i] == "CmnActMatchWin" for i in self.acts):
                self.match_over = True
        return self


def record_one(index, list_ref="saved_list", menu_ref="replay_menu", menu_box=VIEW_ITEM, extra=None,
               end_ref="replay_end", end_box=END_MENU):
    """Open the highlighted entry of a replay list, play it and record it."""
    before = newest_csv()
    nav.wait_for(list_ref, box=LIST_HEADER, timeout=30)
    # A press while the list is still loading is swallowed: retry, but only while
    # the list itself is still showing.
    for _ in range(5):
        ggst.keys("u", 300)
        time.sleep(2)
        if nav.score(menu_ref, menu_box) < 10:
            break
        if nav.score(list_ref, LIST_HEADER) >= 14:
            raise SystemExit("unexpected screen after opening a replay entry")
    nav.wait_for(menu_ref, box=menu_box, timeout=10, thr=10)
    ggst.keys("u", 300)

    t0 = time.time()
    while newest_csv() == before:
        if time.time() - t0 > 90:
            raise SystemExit("replay did not start (no new log)")
        time.sleep(0.2)
    csv = newest_csv()
    video = os.path.join(REC, os.path.basename(csv)[:-4] + ".mp4")
    ff = subprocess.Popen(
        [ggst.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex",
         "gfxcapture=window_exe=GGST-Win64-Shipping:capture_cursor=0:max_framerate=60,"
         "hwdownload,format=bgra,scale=1920:1080,fps=60,format=nv12",
         "-t", str(MAX_RECORD_S + 60), "-c:v", "h264_mf", "-hw_encoding", "1", "-b:v", "30M", video],
        stdin=subprocess.PIPE)
    print(f"[{index}] recording {os.path.basename(video)}")

    # The game frame keeps counting under the end-of-replay menu, so the log never
    # stops by itself: watch the screen for that menu instead, and give up on it a
    # while after the log shows the match won, or after MAX_RECORD_S. Only the
    # recording stops here; the caller still waits for the menu before any key.
    t0 = time.time()
    log = LogTail(csv)
    won_at = None
    stop = "menu"
    said = False
    while True:
        s = nav.score(end_ref, end_box)
        if s < 10:
            break
        if s >= 255 and not said:
            print(f"[{index}]   screen capture failing")
            said = True
        if won_at is None and log.poll().match_over:
            won_at = time.time()
        if won_at is not None and time.time() - won_at > MATCH_OVER_S:
            stop = "match over"
            break
        if time.time() - t0 > MAX_RECORD_S:
            stop = "max duration"
            break
        time.sleep(2)
    ff.communicate(b"q", timeout=30)
    log.f.close()
    if stop != "menu":
        print(f"[{index}] stopped: {stop}, end menu not seen (last diff {s:.1f})")
    rows = sum(1 for _ in open(csv)) - 1
    print(f"[{index}] done: {rows} frames")
    with open(MANIFEST, "a") as f:
        f.write(json.dumps({"index": index, "csv": csv, "video": video, "frames": rows, "stop": stop,
                            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--here", action="store_true", help="already on the saved list, cursor on --start")
    args = ap.parse_args()
    if not args.here:
        to_saved_list()
    if args.start and not args.here:
        ggst.keys(" ".join(["s"] * args.start), 120)
        time.sleep(1)
    for i in range(args.start, args.start + args.count):
        record_one(i)
        try:
            nav.wait_for("replay_end", box=END_MENU, timeout=10, thr=10)
            ggst.keys("u", 300)  # back to the replay list
            time.sleep(6)
            nav.wait_for("saved_list", box=LIST_HEADER, timeout=60)
        except SystemExit:
            ggst.shot(os.path.join(REC, f"after_replay_{i}.png"))
            raise SystemExit(f"unexpected screen after replay {i}; see after_replay_{i}.png")
        ggst.keys("s", 300)
        time.sleep(1)
