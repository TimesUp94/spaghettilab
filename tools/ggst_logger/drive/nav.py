"""Navigate Strive's menus to training mode, checking every screen first.

Each step waits until the window matches a reference screenshot (ref/*.png,
compared on a small grey thumbnail, optionally a sub-region) and only then
presses keys. Any unexpected screen aborts, so a key never lands on the wrong
menu (e.g. ranked matchmaking).
"""
import os
import sys
import tempfile
import time

import cv2
import numpy as np

import ggst

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(tempfile.gettempdir(), "ggst_nav.png")


def thumb(img, box=None):
    if box:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54)).astype(np.float32)


def half_res(img):
    """The references are stored at 960x540; put the live capture through the same
    down- and up-scaling so both sides blur alike."""
    small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (1920, 1080))


def score(name, box=None):
    for _ in range(5):  # a capture occasionally comes back empty; retry
        try:
            ggst.shot(TMP)
        except Exception:
            cur = None
        else:
            cur = cv2.imread(TMP)
        if cur is not None:
            break
        time.sleep(0.5)
    else:
        return 255.0
    ref = cv2.resize(cv2.imread(os.path.join(HERE, "ref", f"{name}.jpg")), (1920, 1080))
    cur = half_res(cur)
    return float(np.mean(np.abs(thumb(cur, box) - thumb(ref, box))))


def wait_for(name, timeout=60, thr=14.0, box=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = score(name, box)
        if s < thr:
            print(f"  on '{name}' (diff {s:.1f})")
            return
        time.sleep(1)
    raise SystemExit(f"timed out waiting for '{name}' (last diff {s:.1f}); stopping")


# Regions that stay the same whatever the menu art behind them.
BOTTOM_TABS = (0, 830, 1920, 1080)
# The selected tab's highlight: tells RETE from DOJO apart (whole-bar diff is only ~12).
RETE_TAB = (60, 830, 330, 1010)
DOJO_TAB = (640, 830, 960, 1010)
HUD_TOP = (0, 0, 1920, 200)

STEPS = [
    ("main", RETE_TAB, "d d d", 1.5),
    ("dojo", DOJO_TAB, "u", 4),
    ("side", None, "a", 1),
    ("side", None, "u", 4),
    ("charsel", (0, 700, 1920, 1080), "u", 2),
]

def to_main():
    """Title screen -> main menu. The title loops into an attract video after a
    while; a key press there returns to the title, so retry until the menu shows."""
    t0 = time.time()
    while time.time() - t0 < 180:
        if score("main", RETE_TAB) < 14:
            print("  on 'main'")
            return
        if score("title") < 14:
            ggst.keys("u", 300)
            time.sleep(6)
            continue
        if score("server") < 14:  # "communicating with the server" dialog
            time.sleep(3)
            continue
        ggst.keys("enter", 300)  # attract video: skip back to the title
        time.sleep(3)
    raise SystemExit("could not reach the main menu")


if __name__ == "__main__" and sys.argv[1:] == ["main"]:
    to_main()
elif __name__ == "__main__":
    to_main()
    for name, box, keys, pause in STEPS:
        wait_for(name, box=box, timeout=90)
        ggst.keys(keys, 300)
        time.sleep(pause)
    # P1 colour, P2 character/colour, stage: confirm until the training HUD shows.
    for _ in range(8):
        if score("training", HUD_TOP) < 14:
            break
        ggst.keys("u")
        time.sleep(3)
    wait_for("training", box=HUD_TOP, timeout=60)
    print("in training mode")
