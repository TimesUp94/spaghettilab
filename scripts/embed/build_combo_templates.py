"""Build GGST combo-counter templates from real frames.

Two kinds of templates are written to config/templates/combo_templates.npz:

* ``w2`` .. ``w9`` -- the spelled-out word ("two" .. "nine") that GGST draws
  next to the hit digit for 2-9 hits. Crisp, opaque-ish text: the most
  reliable thing to read.
* ``g0`` .. ``g9`` -- glyphs of the huge two-digit counter shown from 10 hits
  on (no word is drawn then). Tight glyph box plus BIG_PAD margin. Tens and
  ones digits, P1 and P2 side all use the same font, so every instance of a
  digit feeds the same template (the P1 counter is drawn ~1.2x larger than
  the P2 one and both zoom on every hit; alignment is multi-scale).

Each template starts from a hand-measured seed box (SEEDS). Then every
instance in INSTANCES (all checked by eye) is aligned to the current template
by multi-scale NCC inside a search region, cropped, resized to the template
size, and the crops are combined: mean for words, per-pixel MEDIAN for big
glyphs (characters often cover part of a big digit; the median drops them).
Two alignment passes are run (2nd pass aligns to the combined template).

Usage:
    .venv/Scripts/python.exe scripts/embed/build_combo_templates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = {
    "s17": ROOT / "ggstsets" / "17_DIHLScarecrow_vs_QSalkazko.mp4",
    "g02": ROOT / "ggstsets" / "02_DeeSlyCoops_vs_QLeltdver.mp4",
    "g07": ROOT / "ggstsets" / "07_3DaveFennoy_vs_4EneroneJR.mp4",
    "g12": ROOT / "ggstsets" / "12_4Revill_vs_QEneroneJR.mp4",
    "g13": ROOT / "ggstsets" / "13_Dumadacchi_vs_HLScarecrow.mp4",
    "g14": ROOT / "ggstsets" / "14_Qumadacchi_vs_4IHLScarecrow.mp4",
    "g19": ROOT / "ggstsets" / "19_4Revill_vs_Salkazko.mp4",
    "l03": ROOT / "league" / "03_QSalkazko_vs_QEneroneJR.mp4",
    "l06": ROOT / "league" / "06_Jason_vs_QSalkazko.mp4",
    "l10": ROOT / "league" / "10_IKenun_vs_0Sallazko.mp4",
    "l13": ROOT / "league" / "13_Guessing_vs_Salkazko.mp4",
    "l18": ROOT / "league" / "18_Jason_vs_1Guessing.mp4",
}
VIDEO = VIDEOS["s17"]
OUT = ROOT / "config" / "templates" / "combo_templates.npz"
PREVIEW = ROOT / "output" / "embed_test" / "combo_counter" / "templates_preview.png"

WORD_H = 48  # stored word template height (px); width keeps the seed aspect
BIG_H = 96  # stored big-glyph template height (incl. padding)
BIG_PAD = 0.04  # big glyph seed box is padded by this fraction of its height
BIG_TOP_CUT = 0.16  # top of a big glyph often hides under the HUD: not matched
# Big instances aligning worse than this (glyph mostly covered, or zoomed past
# BIG_GLYPH_H) are left out of the median.
BIG_MIN_ALIGN = 0.55
# Instances with an eye-measured approximate box are searched only near it,
# and kept down to this score (the box was checked by eye).
BIG_MIN_ALIGN_GUIDED = 0.30

# label -> (frame, x1, y1, x2, y2) full-res seed box, measured on gridded
# crops (set 17 unless the frame is given as (video_key, frame)).
SEEDS = {
    "w2": (4640, 403, 286, 517, 353),
    "w3": (9000, 1478, 286, 1642, 348),
    "w4": (5196, 1328, 212, 1562, 333),
    "w5": (6970, 1317, 262, 1467, 343),
    "w6": (3100, 326, 300, 414, 376),
    "w7": (9520, 1342, 288, 1596, 376),
    "w8": (10418, 1230, 276, 1462, 362),
    "w9": (8287, 1212, 282, 1422, 368),
    # big glyphs: tight glyph box (tops can sit under the HUD bars)
    "g0": (7880, 1405, 110, 1840, 810),
    "g1": (1462, 215, 75, 435, 905),  # P1 tens digit
    "g2": (8304, 1420, 105, 1840, 810),
    "g3": (8319, 1395, 100, 1830, 810),
    "g4": (10456, 1355, 140, 1860, 845),
    "g5": (10500, 1365, 105, 1780, 815),
    "g7": (1444, 690, 70, 1180, 900),  # P1 ones digit
    "g8": (1462, 700, 75, 1170, 905),  # P1 ones digit
    "g6": (("l03", 21211), 640, 75, 1120, 905),  # P1 ones digit, top under the HUD
    "g9": (("l03", 31265), 645, 40, 1130, 845),
}
# (frame, side, label) -- every one looked at in a contact sheet
INSTANCES = [
    (4640, 1, "w2"), (472, 1, "w2"), (480, 1, "w2"), (660, 1, "w2"), (870, 1, "w2"),
    (6420, 1, "w2"), (7240, 1, "w2"), (9055, 2, "w2"), (9790, 2, "w2"),
    (9000, 2, "w3"), (9005, 2, "w3"), (885, 1, "w3"), (900, 1, "w3"), (7330, 1, "w3"), (11060, 1, "w3"),
    (5196, 2, "w4"), (5204, 2, "w4"), (7600, 2, "w4"), (9870, 2, "w4"), (9370, 2, "w4"),
    (5540, 1, "w4"), (7742, 2, "w4"),
    (6970, 2, "w5"), (6020, 2, "w5"), (9390, 2, "w5"), (10250, 2, "w5"), (9500, 2, "w5"), (1420, 1, "w5"),
    (3100, 1, "w6"), (3150, 1, "w6"), (3070, 1, "w6"), (8282, 2, "w6"), (10750, 1, "w6"),
    (9520, 2, "w7"), (10280, 2, "w7"), (11405, 2, "w7"),
    (9532, 2, "w8"), (10290, 2, "w8"), (10318, 2, "w8"), (11412, 2, "w8"), (10418, 2, "w8"),
    (8287, 2, "w9"), (10422, 2, "w9"),
]

# Big-glyph instances: (video_key, frame, region, label). region = which
# digit slot to search: p1t/p1o = P1 tens/ones, p2t/p2o = P2 tens/ones.
BIG_INSTANCES = [
    # "1": tens digit of every 10-19 counter, and ones digit of "11"
    ("s17", 1430, "p1t", "g1"), ("s17", 1444, "p1t", "g1"), ("s17", 1470, "p1t", "g1"),
    ("s17", 7880, "p2t", "g1"), ("s17", 9650, "p2t", "g1"), ("s17", 10456, "p2t", "g1"),
    ("s17", 10500, "p2t", "g1"), ("s17", 7959, "p2o", "g1"), ("s17", 7975, "p2o", "g1"),
    ("g02", 8241, "p1t", "g1"), ("g12", 10206, "p1t", "g1"), ("l13", 21051, "p1t", "g1"),
    ("g19", 4906, "p1t", "g1"), ("l18", 33246, "p2t", "g1"), ("g13", 1170, "p2t", "g1"),
    ("l10", 11598, "p2t", "g1"), ("l03", 31377, "p1t", "g1"),
    ("g02", 8304, "p1o", "g1"), ("g19", 4913, "p1o", "g1"), ("l13", 12857, "p1o", "g1"),
    ("g19", 9953, "p1o", "g1"),
    ("s17", 7880, "p2o", "g0"), ("s17", 10428, "p2o", "g0"), ("s17", 9650, "p2o", "g0"),
    ("g02", 8241, "p1o", "g0"), ("g02", 8252, "p1o", "g0"), ("g12", 10206, "p1o", "g0"),
    ("g12", 10221, "p1o", "g0"), ("l13", 21051, "p1o", "g0"), ("l13", 21069, "p1o", "g0"),
    ("g19", 9854, "p1o", "g0"),
    ("s17", 8300, "p2o", "g2"), ("s17", 8304, "p2o", "g2"), ("s17", 10436, "p2o", "g2"),
    ("g19", 4990, "p1o", "g2"), ("g19", 5004, "p1o", "g2"), ("g19", 5017, "p1o", "g2"),
    ("g13", 1170, "p2o", "g2"),
    ("g07", 18560, "p1t", "g2"), ("g07", 18565, "p1t", "g2"),  # tens digit of a 2x counter
    ("s17", 8316, "p2o", "g3"), ("s17", 8322, "p2o", "g3"), ("g14", 3199, "p1o", "g3"),
    ("l10", 11598, "p2o", "g3"), ("g13", 11468, "p1o", "g3"), ("l03", 20987, "p1o", "g3"),
    ("s17", 10448, "p2o", "g4"), ("s17", 10456, "p2o", "g4"), ("g14", 3206, "p1o", "g4"),
    ("g14", 3213, "p1o", "g4"), ("g13", 11519, "p1o", "g4"), ("g13", 11536, "p1o", "g4"),
    ("l18", 33246, "p2o", "g4"), ("l18", 33281, "p2o", "g4"), ("g07", 25649, "p1o", "g4"),
    ("s17", 10510, "p2o", "g5"), ("s17", 10531, "p2o", "g5"), ("g14", 3262, "p1o", "g5"),
    ("g14", 3280, "p1o", "g5"), ("l03", 31361, "p1o", "g5"), ("l03", 31393, "p1o", "g5"),
    ("l03", 21170, "p1o", "g5"),
    ("l03", 21211, "p1o", "g6"), ("l03", 21204, "p1o", "g6", (641, 72, 834)),
    ("l03", 21197, "p1o", "g6", (641, 72, 834)),
    ("s17", 1441, "p1o", "g7"),
    ("s17", 5565, "p1o", "g7"), ("s17", 588, "p1o", "g7"), ("s17", 5560, "p1o", "g7"),
    ("s17", 1444, "p1o", "g7"), ("l06", 11185, "p1o", "g7", (560, 90, 860)),
    ("l06", 11200, "p1o", "g7", (560, 90, 860)),
    ("s17", 1462, "p1o", "g8"), ("s17", 5585, "p1o", "g8"),
    ("s17", 609, "p1o", "g8"), ("s17", 1465, "p1o", "g8"), ("s17", 10936, "p1o", "g8"),
    ("s17", 10942, "p1o", "g8", (640, 70, 900)), ("s17", 10948, "p1o", "g8", (640, 70, 900)),
    ("s17", 2702, "p1o", "g8", (700, 95, 780)),
    ("l03", 31253, "p1o", "g9"), ("l03", 31259, "p1o", "g9"), ("l03", 31265, "p1o", "g9"),
    ("l03", 31270, "p1o", "g9"),
]

SEARCH = {1: (0, 120, 1000, 620), 2: (920, 120, 1920, 620)}
# full-res search regions per big digit slot (cover the zoom-in on each hit)
BIG_SEARCH = {
    "p1t": (40, 0, 640, 1000),
    "p1o": (500, 0, 1350, 1000),
    "p2t": (860, 0, 1400, 1000),
    "p2o": (1240, 0, 1920, 1000),
}
BIG_GLYPH_H = (600, 940)  # glyph height range (px) searched when aligning


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class Video:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))

    def frame(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {n}")
        if img.shape[:2] != (1080, 1920):
            img = cv2.resize(img, (1920, 1080))
        return img


def align(g_region: np.ndarray, tmpl: np.ndarray, scales, ds: int):
    """Best (score, x, y, w, h) of tmpl (full-res gray) inside g_region, at 1/ds."""
    small = cv2.resize(g_region, (g_region.shape[1] // ds, g_region.shape[0] // ds), interpolation=cv2.INTER_AREA)
    best = (-2.0, 0, 0, 0, 0)
    for s in scales:
        th, tw = int(tmpl.shape[0] * s / ds), int(tmpl.shape[1] * s / ds)
        if th < 6 or tw < 6 or th > small.shape[0] or tw > small.shape[1]:
            continue
        t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (float(mx), loc[0] * ds, loc[1] * ds, tw * ds, th * ds)
    return best


def main():
    vids = {k: Video(v) for k, v in VIDEOS.items()}
    frames: dict[tuple[str, int], np.ndarray] = {}

    def fr(n, key="s17"):
        if (key, n) not in frames:
            frames[(key, n)] = gray(vids[key].frame(n))
        return frames[(key, n)]

    seeds: dict[str, np.ndarray] = {}
    for label, (f, x1, y1, x2, y2) in SEEDS.items():
        key, f = f if isinstance(f, tuple) else ("s17", f)
        if label.startswith("g"):
            pad = int(round(BIG_PAD * (y2 - y1)))
            x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        img = fr(f, key)
        crop = img[max(0, y1) : y2, max(0, x1) : x2].astype(np.float32)
        if y1 < 0 or x1 < 0:
            crop = cv2.copyMakeBorder(crop, max(0, -y1), 0, max(0, -x1), 0, cv2.BORDER_REPLICATE)
        H = WORD_H if label.startswith("w") else BIG_H
        W = int(round(crop.shape[1] * H / crop.shape[0]))
        seeds[label] = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)

    jobs = [("s17", f, side, label) for f, side, label in INSTANCES]
    jobs += [(k, f, slot, label, *rest) for k, f, slot, label, *rest in BIG_INSTANCES]
    current = {k: v.copy() for k, v in seeds.items()}
    log = []
    for it in range(2):
        acc: dict[str, list[np.ndarray]] = {}
        log = []
        for key, f, where, label, *approx in jobs:
            g = fr(f, key)
            is_big = label.startswith("g")
            x1, y1, x2, y2 = BIG_SEARCH[where] if is_big else SEARCH[where]
            if approx:  # eye-measured (left, top, height) of a heavily covered glyph
                al, at, ah = approx[0]
                x1, y1 = max(0, al - 80), max(0, at - 80)
                x2, y2 = min(1920, al + int(0.75 * ah) + 80), min(1080, at + ah + 80)
            region = g[y1:y2, x1:x2]
            tmpl = current[label].astype(np.uint8)
            c0 = 0
            if is_big:
                lo, hi = (approx[0][2] * 0.9, approx[0][2] * 1.1) if approx else BIG_GLYPH_H
                # align on the glyph minus its top BIG_TOP_CUT (hidden by the HUD)
                c0 = int(round(tmpl.shape[0] * (BIG_PAD + BIG_TOP_CUT) / (1 + 2 * BIG_PAD)))
                scales = [h * (1 + 2 * BIG_PAD) / tmpl.shape[0] for h in np.geomspace(lo, hi, 24)]
                ds = 4
            else:
                scales = [s for s in np.geomspace(0.9, 2.6, 22)]
                ds = 2
            score, x, y, w, h = align(region, tmpl[c0:], scales, ds)
            if c0:
                sc = h / (tmpl.shape[0] - c0)
                y, h = int(round(y - c0 * sc)), int(round(tmpl.shape[0] * sc))
            top = 200  # room above the frame for boxes reaching past y=0
            regp = cv2.copyMakeBorder(region, top, 0, 0, 0, cv2.BORDER_REPLICATE)
            crop = regp[y + top : y + top + h, x : x + w].astype(np.float32)
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            crop = cv2.resize(crop, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_AREA)
            pad = BIG_PAD * h / (1 + 2 * BIG_PAD) if is_big else 0
            gx, gy, gh = x + x1 + pad, y + y1 + pad, h - 2 * pad
            skip = is_big and score < (BIG_MIN_ALIGN_GUIDED if approx else BIG_MIN_ALIGN)
            log.append((label, key, f, where, round(score, 3), (int(gx), int(gy), int(w - 2 * pad), int(gh)), "SKIP" if skip else ""))
            if not skip:
                acc.setdefault(label, []).append(crop)
        current = {
            k: (np.median(v, axis=0) if k.startswith("g") and len(v) >= 3 else np.mean(v, axis=0)) for k, v in acc.items()
        }
    for row in log:
        print(*row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(OUT), **{k: v.astype(np.float32) for k, v in current.items()})
    # preview
    tiles = []
    for k in sorted(current):
        t = np.clip(current[k], 0, 255).astype(np.uint8)
        t = cv2.resize(t, (int(t.shape[1] * 96 / t.shape[0]), 96))
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        cv2.putText(t, k, (2, 14), 0, 0.45, (0, 255, 255), 1)
        tiles.append(np.pad(t, ((0, 0), (0, 6), (0, 0))))
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW), np.hstack(tiles))
    print(f"saved {len(current)} templates -> {OUT}")
    print(f"preview -> {PREVIEW}")


if __name__ == "__main__":
    sys.exit(main())
