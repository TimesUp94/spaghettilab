"""Both characters from the game's own HUD: the portrait in the top corner and the name on the health bar,
per side and over time. Neither changes with the player's colour (only a mod would change them), so this
is the authority, with video_players' visual vote (which looks at the fighters) as the last fallback.

Only game-native HUD is read, never a stream overlay: the boxes below are the portrait's inner disc (without
the "Player 1" label across its bottom, or the ring/ornaments, which depend on the HUD style) and the start of
the name (the bar behind it changes with HP, and the frame around it with the style).

Matching: normalised correlation against per-character, per-side references, with a small shift search,
on frames reduced to about 720p detail, so that 720p30 copies read like 1080p60 recordings. The name has two
references, on the empty bar (bright text) and on the full bar (faint text), both high-passed.

A video may hold several games with other characters (a tournament set), so the read is a timeline:
segments() cuts it where a side's portrait stays changed, and resolve() picks each segment's characters.

    .venv/Scripts/python.exe tools/ggst_logger/model/hud_chars.py collect               # crops of the logged replays -> <data>/hud_chars/bands
    .venv/Scripts/python.exe tools/ggst_logger/model/hud_chars.py build [--test]        # references (--test: without the test fold)
    .venv/Scripts/python.exe tools/ggst_logger/model/hud_chars.py eval                  # test fold, 1080p60 and 720p30, against the log and the visual vote
    .venv/Scripts/python.exe tools/ggst_logger/model/hud_chars.py eval --sets <dir>     # videos with no labels: every source per stretch of each video
    .venv/Scripts/python.exe tools/ggst_logger/model/hud_chars.py read <video> [--parquet p] [--refs f]

-> <GGST_MODEL_DIR>/hud_chars_refs.npz (hud_chars_refs_test.npz), results_hud_chars.json, results_hud_chars_sets.json
The references are crops of game art: they live with the weights, not in the repository. Without the file the
chain falls back to the visual vote and says so (source "visual").
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

W, H = 1920, 1080
# The crop stored per sampled frame (x0, y0, x1, y1 for P1; P2 is mirrored in x): room for the shift search.
PORT_BOX = (5, 50, 165, 190)
NAME_BOX = (140, 102, 420, 152)
# The portrait's disc. P1 is at x 82 everywhere. P2 depends on the game's HUD: the recordings (ornate HUD) show the art as
# on P1, centred at x 1845; the streams (ink HUD) show it mirrored, at x 1838. So one art per character, tried both
# ways round, with a shift search wide enough for both places.
PORT_CENTRE = {0: 82, 1: 1841}               # middle of the search, frame px
PORT_Y = 118
PORT_R = 48
PORT_YMAX = 26                               # below the centre, frame px: the "Player 1" label starts there
PORT_SCALE = 0.5
PORT_SIZE = 50                               # reference window, px at PORT_SCALE (a square around the disc)
PORT_SHIFT = 4                               # search either way, px at PORT_SCALE
# The start of the name within the crop: names are left aligned on P1, right aligned on P2.
NAME_CROP = {0: (14, 14, 264, 38), 1: (16, 14, 266, 38)}     # x0, y0, x1, y1 (text rows only: the bar's edges match anything; P2's end alike, "... VALENTINE", so wide)
NAME_SCALE = 2 / 3
NAME_SHIFT = 3
NAME_SIGMA = 4.0                             # high pass, at NAME_SCALE

# A frame counts for a character when its score passes these and clears the runner-up (tune: `eval --tune`).
PORT_MIN, PORT_MARGIN = 0.75, 0.10
NAME_MIN, NAME_MARGIN = 0.70, 0.08
# A side's reading of a video or segment is confident with at least this many frames, this share of them agreeing.
MIN_FRAMES = 5
MIN_SHARE = 0.8
# A segment lasts at least this many counted frames (fewer: a misread); a game is far longer.
MIN_RUN = 4
MERGE_CUTS_S = 20.0                          # the two sides' changes closer than this are one change
SAMPLE_EVERY_S = 2.0
MAX_SAMPLES = 600
SOURCES = ("hud_portrait", "hud_name", "visual")


def refs_path(test: bool = False) -> Path:
    from common import OUT
    return OUT / ("hud_chars_refs_test.npz" if test else "hud_chars_refs.npz")


def cut_box(img, box, side: int):
    """side 0 / 1: the box, mirrored in x for P2 (the pixels are not flipped)."""
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1] if side == 0 else img[y0:y1, W - x1:W - x0]


def port_mask() -> np.ndarray:
    """Boolean mask of the art in the reference window: the disc, without the label across its bottom."""
    yy, xx = np.mgrid[0:PORT_SIZE, 0:PORT_SIZE]
    c = (PORT_SIZE - 1) / 2
    return ((xx - c) ** 2 + (yy - c) ** 2 <= (PORT_R * PORT_SCALE) ** 2) & ((yy - c) / PORT_SCALE <= PORT_YMAX)


def port_small(crop, side: int) -> np.ndarray:
    """The search area of the portrait crop at PORT_SCALE: PORT_SIZE + 2 * PORT_SHIFT px square, float32 BGR."""
    import cv2
    half = int((PORT_SIZE / 2 + PORT_SHIFT) / PORT_SCALE)
    x0 = PORT_BOX[0] if side == 0 else W - PORT_BOX[2]
    cx, cy = PORT_CENTRE[side] - x0, PORT_Y - PORT_BOX[1]
    c = crop[cy - half:cy + half, cx - half:cx + half]
    n = PORT_SIZE + 2 * PORT_SHIFT
    return cv2.resize(c, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)


def name_small(crop, side: int):
    """The name's start, high-passed grey, at about 720p detail (with its shift margin)."""
    import cv2
    x0, y0, x1, y1 = NAME_CROP[side]
    m = int(round(NAME_SHIFT / NAME_SCALE))              # the margin for the shift search
    h, w = crop.shape[:2]
    c = crop[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=NAME_SCALE, fy=NAME_SCALE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), NAME_SIGMA)


def _unit(v, axis=-1):
    v = v - v.mean(axis=axis, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-6)


def shifted(small: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Every window of (th, tw) in small -> [S, th, tw, ...]."""
    ny, nx = small.shape[0] - th + 1, small.shape[1] - tw + 1
    return np.stack([small[y:y + th, x:x + tw] for y in range(ny) for x in range(nx)])


class Refs:
    """The references of one npz: chars, port [C, 2, D] (unit vectors of the masked art, as drawn and mirrored),
    name [2, C, 2, D] (per side and bar state)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.chars = [str(c) for c in z["chars"]]
        self.port = z["port"]
        self.name = z["name"]
        self.name_shape = tuple(int(v) for v in z["name_shape"])
        self.mask = port_mask()

    def score_port(self, crop, side: int) -> np.ndarray:
        """[C] best correlation of the portrait over the shifts and both orientations."""
        win = shifted(port_small(crop, side), PORT_SIZE, PORT_SIZE)    # [S, h, w, 3]
        x = _unit(win[:, self.mask].reshape(len(win), -1))             # [S, D]
        return (self.port @ x.T).max(-1).max(1)                        # [C, 2, S] -> [C]

    def score_name(self, crop, side: int) -> np.ndarray:
        """[C] best correlation of the name over the small shifts and both bar states."""
        h, w = self.name_shape
        x = _unit(shifted(name_small(crop, side), h, w).reshape(-1, h * w))
        return (self.name[side] @ x.T).max(-1).max(1)                  # [C, 2, S] -> [C]


def build(stems: list[str], truth: dict, bands_dir: Path, out: Path):
    """References from the collected crops of the given replays: the median per character (portraits: P1's, whose place never
    changes), per side and bar state for the names."""
    port, names, seen = {}, {}, collections.Counter()
    n = PORT_SIZE
    for s in stems:
        f = bands_dir / f"{s}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        port.setdefault(truth[s, 1], []).extend(port_small(p, 0)[PORT_SHIFT:PORT_SHIFT + n, PORT_SHIFT:PORT_SHIFT + n] for p in z["port"][:, 0])
        for k in (0, 1):
            c = truth[s, k + 1]
            for im in z["name"][:, k]:
                x0, y0, x1, y1 = NAME_CROP[k]
                light = im[y0:y1, x0:x1].mean() > 150          # the bar is lit behind the name
                names.setdefault((c, k, int(light)), []).append(name_small(im, k))
                seen[c, k, int(light)] += 1
    chars = sorted(port)
    m = NAME_SHIFT
    sample = next(iter(names.values()))[0]
    nh, nw = sample.shape[0] - 2 * m, sample.shape[1] - 2 * m
    mask = port_mask()
    P = np.zeros((len(chars), 2, int(mask.sum()) * 3), np.float32)
    N = np.zeros((2, len(chars), 2, nh * nw), np.float32)
    for i, c in enumerate(chars):
        med = np.median(np.stack(port[c]), 0)
        P[i, 0] = _unit(med[mask].reshape(-1))
        P[i, 1] = _unit(med[:, ::-1][mask].reshape(-1))
        for k in (0, 1):
            dark = np.median(np.stack(names[c, k, 0]), 0)[m:m + nh, m:m + nw]
            lit = names.get((c, k, 1))
            N[k, i, 0] = _unit(dark.reshape(-1))
            N[k, i, 1] = _unit((np.median(np.stack(lit), 0)[m:m + nh, m:m + nw] if lit else dark).reshape(-1))
    np.savez_compressed(out, chars=np.array(chars), port=P, name=N, name_shape=np.array([nh, nw]))
    print(f"{out}: {len(chars)} characters, {len(stems)} replays; lit-bar name crops per char min "
          f"{min(seen[c, k, 1] for c in chars for k in (0, 1))}, P1 portraits min {min(len(v) for v in port.values())}")


# ---- reading frames ---------------------------------------------------------------------------------------

def sample_times(parquet=None, duration_s: float = 0.0, every: float = SAMPLE_EVERY_S, cap: int = MAX_SAMPLES) -> np.ndarray:
    """The seconds to look at: gameplay (the pipeline's round timer reads) when the parquet is given, about one
    frame every `every` seconds, at most `cap`, evenly over the video."""
    if parquet is not None and Path(parquet).exists():
        import pyarrow.parquet as pq
        names = pq.read_schema(parquet).names
        tcol = "timer_raw" if "timer_raw" in names else "timer_value"
        c = pq.read_table(parquet, columns=["timestamp_ms", tcol]).to_pydict()
        t = np.array([ms / 1000.0 for ms, v in zip(c["timestamp_ms"], c[tcol]) if v is not None])
        if len(t):
            n = min(cap, max(1, int(len(t) / (30.0 * every))))
            return t[np.linspace(0, len(t) - 1, n).round().astype(int)]
    if duration_s <= 0:
        return np.zeros(0)
    n = min(cap, max(1, int(duration_s / every)))
    return (np.arange(n) + 0.5) * duration_s / n


def read_crops(video, times):
    """(time s, portrait crops [2], name crops [2]) of the frames at the given seconds, scaled to 1920x1080."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            if f.shape[:2] != (H, W):
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
            yield float(t), [cut_box(f, PORT_BOX, 0), cut_box(f, PORT_BOX, 1)], [cut_box(f, NAME_BOX, 0), cut_box(f, NAME_BOX, 1)]
    finally:
        cap.release()


def read_scores(video, refs: Refs, times) -> dict:
    """Scores of every sampled frame: t [N], port [N, 2, C], name [N, 2, C]."""
    ts, ps, ns = [], [], []
    for t, p, n in read_crops(video, times):
        ts.append(t)
        ps.append([refs.score_port(p[k], k) for k in (0, 1)])
        ns.append([refs.score_name(n[k], k) for k in (0, 1)])
    C = len(refs.chars)
    return dict(t=np.array(ts, np.float64),
                port=np.array(ps, np.float32).reshape(-1, 2, C), name=np.array(ns, np.float32).reshape(-1, 2, C))


# ---- from scores to characters (pure: no video, no refs file) ------------------------------------------------

def frame_wins(s: np.ndarray, min_score: float, margin: float) -> np.ndarray:
    """s [N, C]: per frame the winning character's index, or -1 where it doesn't clear the score and the margin."""
    if not len(s):
        return np.zeros(0, int)
    top = np.argsort(-s, axis=1)[:, :2]
    a = s[np.arange(len(s)), top[:, 0]]
    b = s[np.arange(len(s)), top[:, 1]] if s.shape[1] > 1 else np.full(len(s), -1.0)
    return np.where((a >= min_score) & (a - b >= margin), top[:, 0], -1)


def vote(w: np.ndarray) -> tuple[int, float, int]:
    """w: frame wins (-1 = none). (winner or -1, share of the counted frames, number of counted frames)."""
    w = np.asarray(w)
    w = w[w >= 0]
    if not len(w):
        return -1, 0.0, 0
    u, n = np.unique(w, return_counts=True)
    return int(u[n.argmax()]), float(n.max() / len(w)), int(len(w))


def confident(share: float, n: int) -> bool:
    return n >= MIN_FRAMES and share >= MIN_SHARE


def timeline(wp: np.ndarray, min_run: int = MIN_RUN) -> list[tuple[int, int, int]]:
    """Runs (first, last, char) of one side over the counted frames of wp (-1 = not counted): a character holds until
    another one has shown min_run counted frames in a row, so single misreads never open a run. Indices are of wp."""
    idx = np.nonzero(np.asarray(wp) >= 0)[0]
    runs: list[list[int]] = []
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and wp[idx[j + 1]] == wp[idx[i]]:
            j += 1
        if j - i + 1 >= min_run:
            if runs and runs[-1][2] == int(wp[idx[i]]):
                runs[-1][1] = int(idx[j])
            else:
                runs.append([int(idx[i]), int(idx[j]), int(wp[idx[i]])])
        i = j + 1
    return [tuple(r) for r in runs]


def segments(t: np.ndarray, wp: np.ndarray) -> list[tuple[float, float]]:
    """t [N] seconds, wp [N, 2] portrait wins per side -> time spans [(start, end)] of constant characters, joined at
    the midpoint of the gap between two runs. One span over everything when nothing changes."""
    if not len(t):
        return [(0.0, float("inf"))]
    cuts = []
    for k in (0, 1):
        r = timeline(wp[:, k])
        cuts += [(float(t[a[1]]) + float(t[b[0]])) / 2 for a, b in zip(r, r[1:])]
    merged: list[list[float]] = []
    for c in sorted(cuts):          # both sides change at the same break, each with its own last and first readable frame
        if merged and c - merged[-1][-1] < MERGE_CUTS_S:
            merged[-1].append(c)
        else:
            merged.append([c])
    edges = [0.0] + [round(sum(m) / len(m), 2) for m in merged] + [float("inf")]
    return list(zip(edges, edges[1:]))


def side_reading(wp: np.ndarray, wn: np.ndarray, visual: str | None, chars: list[str]) -> dict:
    """One side over one segment: portrait first, then name, then the visual vote; the HUD only when confident. A portrait
    and a name that are both confident yet disagree go to whichever has more counted frames, flagged in `conflict`."""
    cp, sp, np_ = vote(wp)
    cn, sn, nn = vote(wn)
    okp, okn = confident(sp, np_), confident(sn, nn)
    out = dict(portrait=chars[cp] if cp >= 0 else None, portrait_share=round(sp, 3), portrait_frames=np_,
               name=chars[cn] if cn >= 0 else None, name_share=round(sn, 3), name_frames=nn, visual=visual)
    conflict = okp and okn and cp != cn
    if okp and (not conflict or np_ >= nn):
        out.update(char=chars[cp], source="hud_portrait", confidence=round(sp, 3))
    elif okn:
        out.update(char=chars[cn], source="hud_name", confidence=round(sn, 3))
    else:
        out.update(char=visual, source="visual", confidence=0.0)
    out["conflict"] = bool(conflict)
    return out


def resolve(scores: dict, chars: list[str], visual=None) -> list[dict]:
    """The scores of read_scores -> [{start_s, end_s, frames, P1: reading, P2: reading}], one per constant pair of characters.
    visual(start_s, end_s) -> (P1 name, P2 name) gives the fallback for a span (None: no fallback)."""
    t = scores["t"]
    wp = np.stack([frame_wins(scores["port"][:, k], PORT_MIN, PORT_MARGIN) for k in (0, 1)], 1) if len(t) else np.zeros((0, 2), int)
    wn = np.stack([frame_wins(scores["name"][:, k], NAME_MIN, NAME_MARGIN) for k in (0, 1)], 1) if len(t) else np.zeros((0, 2), int)
    out = []
    for a, b in segments(t, wp):
        m = (t >= a) & (t < b) if len(t) else np.zeros(0, bool)
        vis = visual(a, b) if visual else (None, None)
        seg = dict(start_s=round(a, 2), end_s=None if b == float("inf") else round(b, 2), frames=int(m.sum()))
        for k in (0, 1):
            seg[f"P{k + 1}"] = side_reading(wp[m, k], wn[m, k], vis[k], chars)
        out.append(seg)
    return out


def main_characters(segs: list[dict]) -> dict:
    """The video's characters when one name per side is wanted: per side the one seen over the most sampled frames."""
    out = {}
    for side in ("P1", "P2"):
        n = collections.Counter()
        for s in segs:
            n[s[side]["char"]] += max(1, s["frames"])
        out[side] = n.most_common(1)[0][0] if n else None
    return out


def segment_index(segs: list[dict], t: float) -> int:
    for k, s in enumerate(segs):
        if t >= s["start_s"] and (s["end_s"] is None or t < s["end_s"]):
            return k
    return len(segs) - 1


def segment_at(segs: list[dict], t: float) -> dict:
    return segs[segment_index(segs, t)]


def visual_reading(char: str | None) -> dict:
    """A side read by the visual vote only (no references, or the HUD unreadable)."""
    return dict(portrait=None, portrait_share=0.0, portrait_frames=0, name=None, name_share=0.0, name_frames=0, visual=char,
                char=char, source="visual", confidence=0.0, conflict=False)


def characters_fields(segs: list[dict]) -> dict:
    """The result fields: characters (one per side, as before), characters_source, characters_confidence, and
    character_segments, the whole timeline with what each source read."""
    main = main_characters(segs)
    src, conf = {}, {}
    for side in ("P1", "P2"):
        here = [g[side] for g in segs if g[side]["char"] == main[side]]
        best = max(here, key=lambda r: r["confidence"])
        src[side], conf[side] = best["source"], best["confidence"]
    return dict(characters=main, characters_source=src, characters_confidence=conf, character_segments=segs)


# ---- CLI ---------------------------------------------------------------------------------------------------

def read_video(video, refs: Refs, parquet=None, every: float = SAMPLE_EVERY_S) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return read_scores(video, refs, sample_times(parquet, n / fps if fps else 0.0, every))


def load_truth():
    """{(replay, 1|2): character} from the logged openers."""
    import pyarrow.parquet as pq
    from common import DATA
    rows = pq.read_table(DATA / "openers.parquet", columns=["replay", "attacker", "attacker_char", "defender_char", "video"]).to_pylist()
    tru, vid = {}, {}
    for r in rows:
        a = int(r["attacker"][1])
        tru[r["replay"], a] = r["attacker_char"]
        tru[r["replay"], 3 - a] = r["defender_char"]
        vid[r["replay"]] = r["video"]
    return tru, vid


def collect():
    """Six frames of every logged replay, shortly after an opener (HUD up, both bar states) -> <data>/hud_chars/bands/<stem>.npz."""
    import random
    from multiprocessing import Pool
    import pyarrow.parquet as pq
    from common import DATA, low_priority
    low_priority()
    bands = DATA / "hud_chars" / "bands"
    bands.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(DATA / "openers.parquet", columns=["replay", "video", "video_frame"]).to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        if r["video_frame"] is not None:
            by[r["replay"]].append(r)
    rng = random.Random(0)
    tasks = []
    for s, rr in sorted(by.items()):
        rr = sorted(rr, key=lambda r: r["video_frame"])
        tasks.append((s, rr[0]["video"], sorted(r["video_frame"] + rng.randint(10, 120) for r in rr[::max(1, len(rr) // 6)][:6]), str(bands)))
    with Pool(3) as p:
        for k, _ in enumerate(p.imap_unordered(_collect_one, tasks)):
            if k % 100 == 0:
                print(k, len(tasks), flush=True)


def _collect_one(a):
    import cv2
    s, video, frames, bands = a
    o = Path(bands) / f"{s}.npz"
    if o.exists():
        return s
    cv2.setNumThreads(1)
    c = cv2.VideoCapture(video)
    A, B, F = [], [], []
    for f in frames:
        c.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = c.read()
        if ok:
            A.append([cut_box(im, PORT_BOX, 0), cut_box(im, PORT_BOX, 1)])
            B.append([cut_box(im, NAME_BOX, 0), cut_box(im, NAME_BOX, 1)])
            F.append(f)
    c.release()
    np.savez_compressed(o, port=np.array(A, np.uint8), name=np.array(B, np.uint8), frames=np.array(F))
    return s


# ---- evaluation --------------------------------------------------------------------------------------------

def _eval_one(a):
    """Scores of one replay at one quality -> a cache file (a rerun skips it). clip: read only a 720p30 clip of that many seconds."""
    stem, video, parquet, refs_file, cache, clip = a
    import cv2
    from video_crops import FFMPEG
    cv2.setNumThreads(1)
    f = Path(cache) / f"{stem}.npz"
    if f.exists():
        return stem
    refs = _refs_cached(refs_file)
    if clip:
        t = sample_times(parquet)
        mid = float(t[len(t) // 2]) if len(t) else 30.0
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "clip.mp4"
            subprocess.run([FFMPEG, "-hide_banner", "-v", "error", "-ss", str(max(0.0, mid - clip / 2)), "-t", str(clip), "-i", str(video),
                            "-vf", "scale=1280:720,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-b:v", "3M", "-maxrate", "3M",
                            "-bufsize", "6M", "-an", str(tmp)], check=True)
            sc = read_scores(tmp, refs, np.arange(1.0, clip, SAMPLE_EVERY_S))
    else:
        sc = read_scores(video, refs, sample_times(parquet))
    np.savez_compressed(f, **sc)
    return stem


_refs = {}


def _refs_cached(path):
    if path not in _refs:
        _refs[path] = Refs(path)
    return _refs[path]


def _score_set(tasks, jobs=3):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(jobs) as p:
        for k, _ in enumerate(p.imap_unordered(_eval_one, tasks)):
            if k % 40 == 0:
                print(f"  {k}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)


def summarise(stems, cache: Path, truth, chars, visual=None) -> dict:
    """Per-side accuracy of each source alone, and of the chain's rule, over the replays with a cache file."""
    n = 0
    right = collections.Counter()
    answered = collections.Counter()
    errs = collections.defaultdict(list)
    cuts = []
    frames = []
    for s in stems:
        f = cache / f"{s}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f))
        segs = resolve(z, chars, visual=(lambda a, b, s=s: visual[s]) if visual else None)
        if len(segs) > 1:
            cuts.append([s, len(segs)])
        frames.append(len(z["t"]))
        wp = [frame_wins(z["port"][:, k], PORT_MIN, PORT_MARGIN) for k in (0, 1)]
        wn = [frame_wins(z["name"][:, k], NAME_MIN, NAME_MARGIN) for k in (0, 1)]
        for k in (0, 1):
            n += 1
            tru = truth[s, k + 1]
            cp, sp, np_ = vote(wp[k])
            cn, sn, nn = vote(wn[k])
            for src, c, ok in (("hud_portrait", cp, confident(sp, np_)), ("hud_name", cn, confident(sn, nn))):
                answered[src] += ok
                right[src] += ok and chars[c] == tru
                if ok and chars[c] != tru:
                    errs[src].append([s, k + 1, tru, chars[c]])
            sg = max(segs, key=lambda g: g["frames"])[f"P{k + 1}"]
            answered["chain"] += sg["char"] is not None
            right["chain"] += sg["char"] == tru
            answered["chain_" + sg["source"]] += 1
            if sg["char"] != tru:
                errs["chain"].append([s, k + 1, tru, sg["char"], sg["source"]])
            if visual:
                right["visual"] += visual[s][k] == tru
                if visual[s][k] != tru:
                    errs["visual"].append([s, k + 1, tru, visual[s][k]])
    res = dict(sides=n, frames_median=float(np.median(frames)) if frames else 0, replays_cut=cuts)
    for src in ("hud_portrait", "hud_name", "chain", "visual"):
        if src != "visual" or visual:
            res[src] = dict(answered=answered[src] if src != "visual" else n, right=right[src],
                            acc_of_all=round(right[src] / max(1, n), 4), errors=errs[src])
    res["chain_by_source"] = {k[6:]: v for k, v in answered.items() if k.startswith("chain_")}
    return res


def visual_votes(root: Path, stems, chars) -> dict:
    out = {}
    for s in stems:
        f = root / s / "index.json"
        if f.exists():
            lp = json.loads(f.read_text()).get("chars_logp")
            if lp is not None:
                lp = np.array(lp)
                out[s] = (chars[int(lp[0].argmax())], chars[int(lp[1].argmax())])
    return out


def evaluate(a):
    """The test fold with references built without it: 1080p60 originals and 720p30 3 Mbit/s copies (degrade_eval's for the
    old replays; a clip encoded the same way for the new ones, which have none) -> results_hud_chars.json."""
    import torch
    from common import DATA, OUT, low_priority, replay_fold_map, save_json, test_subsets
    low_priority()
    subs = test_subsets(replay_fold_map())
    tru, vid = load_truth()
    refs_file = str(refs_path(True))
    chars = Refs(refs_file).chars
    root = DATA / "hud_chars" / "eval"
    c1080, c720 = root / "1080", root / "720"
    c1080.mkdir(parents=True, exist_ok=True)
    c720.mkdir(parents=True, exist_ok=True)
    deg = DATA / "degraded"
    t1080 = [(s, vid[s], str(DATA / "runs" / s / "frames" / f"{s}_frames.parquet"), refs_file, str(c1080), 0) for s in subs["all"]]
    t720 = [(s, str(deg / f"{s}.mp4"), str(deg / "runs" / s / "frames" / f"{s}_frames.parquet"), refs_file, str(c720), 0) for s in subs["old"]]
    t720 += [(s, vid[s], str(DATA / "runs" / s / "frames" / f"{s}_frames.parquet"), refs_file, str(c720), 60) for s in subs["new"]]
    print("1080p60 ...", flush=True)
    _score_set(t1080, a.jobs)
    print("720p30 ...", flush=True)
    _score_set(t720, a.jobs)
    pchars = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    assert list(pchars) == chars
    vis1080 = visual_votes(DATA / "vcrops_test", subs["all"], chars)
    vis720 = visual_votes(deg / "vcrops", subs["old"], chars)
    res = {}
    for name, stems, cache, vis in (("1080p60 all", subs["all"], c1080, vis1080), ("1080p60 old", subs["old"], c1080, vis1080),
                                    ("1080p60 new", subs["new"], c1080, vis1080), ("720p30 all", subs["all"], c720, None),
                                    ("720p30 old", subs["old"], c720, vis720), ("720p30 new (60 s clip)", subs["new"], c720, None)):
        res[name] = summarise(stems, cache, tru, chars, vis)
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "errors"}) for k, v in res[name].items()}, default=float), flush=True)
        for src in ("hud_portrait", "hud_name", "chain", "visual"):
            if src in res[name] and res[name][src]["errors"]:
                print("   ", src, res[name][src]["errors"][:12], flush=True)
    save_json(res, OUT / "results_hud_chars.json")


def eval_sets(a):
    """Videos without labels (--sets <dir>, tournament sets with a stream overlay): every source per stretch of the video and
    each stretch's visual vote on the same frames (the final net) -> results_hud_chars_sets.json. --frames <dir>: one frame per stretch, to look at."""
    import cv2
    import torch
    from common import OUT, low_priority, save_json
    low_priority()
    refs = Refs(a.refs or refs_path())
    net = str(OUT / "video_players_final.pt")
    names = torch.load(net, map_location="cpu")["chars"]
    assert list(names) == refs.chars
    res = {}
    for video in sorted(Path(a.sets).glob("*.mp4")):
        parquet = a.analysis / video.stem / "frames" / f"{video.stem}_frames.parquet" if a.analysis else None
        t0 = time.time()
        sc = read_video(video, refs, parquet)
        batch, parts = [], []
        for _t, im in _frames_at(video, sc["t"]):
            batch.append(im)
            if len(batch) == 32:
                parts.append(_visual_batch(batch, net))
                batch = []
        if batch:
            parts.append(_visual_batch(batch, net))
        V = np.concatenate(parts) if parts else np.zeros((0, 2, len(names)))

        def visual(x, y, t=sc["t"], V=V):
            m = (t >= x) & (t < y)
            if not m.any():
                return None, None
            v = V[m].sum(0)
            return names[int(v[0].argmax())], names[int(v[1].argmax())]
        segs = resolve(sc, refs.chars, visual)
        for g in segs:
            m = (sc["t"] >= g["start_s"]) & (sc["t"] < (g["end_s"] if g["end_s"] is not None else float("inf")))
            for k, side in enumerate(("P1", "P2")):
                c = g[side]["char"]
                if c in names and m.any():
                    # how the visual net reads this stretch frame by frame, against the HUD's answer
                    w = V[m, k].argmax(-1)
                    g[side]["visual_frame_acc"] = round(float((w == names.index(c)).mean()), 3)
                    top = collections.Counter(names[i] for i in w).most_common(3)
                    g[side]["visual_top"] = [[n, c_] for n, c_ in top]
            if a.frames and m.any():
                Path(a.frames).mkdir(parents=True, exist_ok=True)
                ts = sc["t"][m][int(m.sum()) // 2]
                cap = cv2.VideoCapture(str(video))
                cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
                ok, f = cap.read()
                cap.release()
                if ok:
                    f = cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(Path(a.frames) / f"{video.stem}_{int(g['start_s'])}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        res[video.name] = segs
        print(video.name, f"{len(sc['t'])} frames {time.time() - t0:.0f}s", flush=True)
        for g in segs:
            print(f"  {g['start_s']:7.1f} .. {g['end_s']} frames {g['frames']}: " + " | ".join(
                f"{g[x]['char']} [{g[x]['source']} {g[x]['confidence']}] p={g[x]['portrait']}({g[x]['portrait_frames']}) n={g[x]['name']}({g[x]['name_frames']}) "
                f"v={g[x]['visual']} vacc={g[x].get('visual_frame_acc')} {g[x].get('visual_top')}" for x in ("P1", "P2")), flush=True)
    save_json(res, OUT / "results_hud_chars_sets.json")


def _frames_at(video, times):
    import cv2
    cap = cv2.VideoCapture(str(video))
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if ok:
            yield t, cv2.resize(f, (256, 144), interpolation=cv2.INTER_AREA)
    cap.release()


def _visual_batch(smalls, net):
    import video_crops as vc
    return vc.players(smalls, net)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["collect", "build", "eval", "read"])
    ap.add_argument("video", nargs="?", type=Path)
    ap.add_argument("--test", action="store_true", help="build: without the test fold")
    ap.add_argument("--parquet", type=Path)
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sets", type=Path, help="eval: videos without labels, one per file")
    ap.add_argument("--analysis", type=Path, help="eval --sets: folder of analyze_replay outputs, <stem>/frames/<stem>_frames.parquet")
    ap.add_argument("--frames", type=Path, help="eval --sets: write one frame per stretch here")
    a = ap.parse_args()
    if a.mode == "collect":
        return collect()
    if a.mode == "eval":
        return eval_sets(a) if a.sets else evaluate(a)
    if a.mode == "build":
        from common import DATA, TEST_FOLD, replay_fold_map
        fold = replay_fold_map()
        tru, _ = load_truth()
        stems = sorted(s for s in fold if not (a.test and fold[s] == TEST_FOLD))
        return build(stems, tru, DATA / "hud_chars" / "bands", refs_path(a.test))
    refs = Refs(a.refs or refs_path())
    t0 = time.time()
    sc = read_video(a.video, refs, a.parquet)
    print(json.dumps(resolve(sc, refs.chars), indent=1))
    print(f"{len(sc['t'])} frames, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
