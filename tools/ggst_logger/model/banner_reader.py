"""Counter / punish of each detected opener from the game's own banner, read on the video: the
yellow COUNTER / PUNISH word (and REVERSAL, the red COUNTER) in the attacker's HUD box, matched
with the dataset build's templates (dataset/build.py banner_word_scores), plus the two looks of a
large counter hit, whose red word is rarely read: a cinematic letterbox (black bars over the HUD)
or a huge see-through white COUNTER! across the middle of the screen (banner_large_counter.png:
the mean white mask of 19 training frames). Every 2nd frame from -60 to
+60 around the detected start is scored on both sides, on frames decoded at 960x540 from any
video; a random-forest classifier on those series (and the nearest detections: a trade is no
counter hit, an earlier hit's banner may still be up) gives counter / punish / none.

    .venv/Scripts/python.exe tools/ggst_logger/model/banner_reader.py extract --set rec --jobs 5
    .venv/Scripts/python.exe tools/ggst_logger/model/banner_reader.py extract --set rec --base banners_rec_v1   # new detections: reuse the
        rows of an earlier extraction for detections at the same frame and side (a row depends on nothing else)
    .venv/Scripts/python.exe tools/ggst_logger/model/banner_reader.py extract --set degraded --jobs 5
    .venv/Scripts/python.exe tools/ggst_logger/model/banner_reader.py train      # folds 1-4 -> test fold 0
    .venv/Scripts/python.exe tools/ggst_logger/model/banner_reader.py final      # every replay

-> <data>/ggst_dataset/banners_<set>/<stem>.npz (scores per detection),
   model/banner_reader_{test,final}.pkl, results_banner.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "dataset"))
from common import DATA, OUT, TEST_FOLD, label_hit_kind, low_priority, replay_fold_map, report, save_json, test_subsets  # noqa: E402

OFFS = tuple(range(-60, 61, 2))
W, H = 960, 540                  # frames at half the 1920x1080 the boxes and templates are set for
CLASSES = ("counter", "none", "punish")
WORDS = ("counter", "punish", "reversal", "red_counter")
BIG_ROWS = (230, 466)            # the large CH's white COUNTER! (half scale)
NF = 2 * (len(WORDS) + 1) + 2 + 2
_T: dict = {}


def templates():
    import build
    import cv2
    if not _T:
        _T.update({w: [cv2.resize(t, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for t in ts]
                   for w, ts in build.banner_templates().items()})
    return _T


def half(box):
    return tuple(v // 2 for v in box)


def word_scores(hsv, side):
    """build.banner_word_scores at half scale: the yellow words, and the red COUNTER."""
    import build
    import cv2
    x1, y1, x2, y2 = half(build.KIND_BOX[side])
    hue, sat, val = (hsv[y1:y2, x1:x2, k] for k in range(3))
    yel = ((hue >= 15) & (hue <= 35) & (sat >= 110) & (val >= 160)).astype(np.float32)
    red = (((hue <= 8) | (hue >= 172)) & (sat >= 150) & (val >= 120)).astype(np.float32)
    out = []
    for w in WORDS:
        m, t = (red, templates()["counter"]) if w == "red_counter" else (yel, templates()[w])
        out.append(0.0 if m.mean() < 0.02 else max(float(cv2.matchTemplate(m, x, cv2.TM_CCOEFF_NORMED).max()) for x in t))
    x1, y1, x2, y2 = half(build.BANNER_BOX[side])
    hue, sat, val = (hsv[y1:y2, x1:x2, k] for k in range(3))
    return out + [float(((hue >= 15) & (hue <= 35) & (sat >= 110) & (val >= 160)).mean())]


def frames_at(video, wanted):
    """Yield (60 fps frame index, 960x540 BGR frame) for the sorted indices (the decoder selects
    the spans around them: a frame list overflows the command line)."""
    import video_crops as vc
    fps, _, _ = vc.video_format(video)
    pre = "fps=60," if abs(fps - 60) > 0.5 else ""
    ranges, a = [], None
    for f in wanted:
        if a is None or f > b + OFFS[1] - OFFS[0]:
            if a is not None:
                ranges.append((a, b))
            a = f
        b = f
    ranges.append((a, b))
    keep = set(wanted)
    sel = "+".join(f"between(n\\,{x}\\,{y})" for x, y in ranges)
    cmd = [vc.FFMPEG, "-hide_banner", "-v", "error", "-threads", "4", "-i", str(video), "-vf",
           f"{pre}select='{sel}',scale={W}:{H}:flags=area", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    n = W * H * 3
    for f in (f for x, y in ranges for f in range(x, y + 1)):
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        if f in keep:
            yield f, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    p.stdout.close()
    p.wait()


def big_template():
    """The large CH's white COUNTER! (mean white mask of training frames, at a quarter of W x H)."""
    import cv2
    if "big" not in _T:
        t = cv2.imread(str(Path(__file__).with_name("banner_large_counter.png")), cv2.IMREAD_GRAYSCALE)
        _T["big"] = cv2.resize(t.astype(np.float32) / 255, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)[:, 4:-4]
    return _T["big"]


def frame_feats(img):
    """Word scores and yellow share of both banner boxes, the top rows' darkness (letterbox), and
    the white share of the middle band and its match with the large CH's word."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = word_scores(hsv, 1) + word_scores(hsv, 2)
    top = img[1:9].mean(-1)
    out += [float(top.mean()) / 255, float((top < 20).mean())]
    band = hsv[BIG_ROWS[0]:BIG_ROWS[1]]
    white = ((band[..., 1] < 60) & (band[..., 2] > 190)).astype(np.float32)
    w4 = cv2.resize(white, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    out += [float(white.mean()), float(cv2.matchTemplate(w4, big_template(), cv2.TM_CCOEFF_NORMED).max())]
    return np.array(out, np.float32)


def scan(video, frames):
    """[len(frames), len(OFFS), NF] features around each detected frame (NaN where unread)."""
    import cv2
    cv2.setNumThreads(1)
    wanted = sorted({f + o for f in frames for o in OFFS if f + o >= 0})
    feats = {f: frame_feats(img) for f, img in frames_at(video, wanted)}
    nan = np.full(NF, np.nan, np.float32)
    return np.stack([np.stack([feats.get(f + o, nan) for o in OFFS]) for f in frames]) if frames else \
        np.zeros((0, len(OFFS), NF), np.float32)


def _scan_one(args):
    stem, video, dets, path, base = args
    low_priority()
    if Path(path).exists():
        return stem, -1
    old = {}
    if base and (Path(base) / Path(path).name).exists():
        z = np.load(Path(base) / Path(path).name)
        old = {(int(f), int(s)): z["x"][k] for k, (f, s) in enumerate(zip(z["frame"], z["side"]))}
    todo = [d["frame"] for d in dets if (d["frame"], d["side"]) not in old]
    fresh = dict(zip(todo, scan(video, todo))) if todo else {}
    x = np.stack([old[(d["frame"], d["side"])] if (d["frame"], d["side"]) in old else fresh[d["frame"]] for d in dets]) \
        if dets else np.zeros((0, len(OFFS), NF), np.float32)
    tmp = Path(path).with_suffix(".part.npz")
    np.savez_compressed(tmp, x=x, frame=np.array([d["frame"] for d in dets]), side=np.array([d["side"] for d in dets]))
    tmp.replace(path)
    return stem, len(todo)


def sources(which):
    """(detections per stem, video per stem, output folder) of a set: the recordings or the 720p30 copies."""
    import pyarrow.parquet as pq
    if which == "degraded":
        dets = json.loads((DATA / "degraded" / "detections.json").read_text())
        video = {s: str(DATA / "degraded" / f"{s}.mp4") for s in dets}
    else:
        dets = json.loads((OUT / "detections.json").read_text())
        video = {r["replay"]: r["video"] for r in pq.read_table(DATA / "openers.parquet", columns=["replay", "video"]).to_pylist()}
    return dets, video, DATA / f"banners_{which}"


def extract(which, jobs, only=None, base=None):
    dets, video, root = sources(which)
    root.mkdir(parents=True, exist_ok=True)
    fold = replay_fold_map()
    # the test fold first, so it can be measured before the training replays are done
    stems = sorted((s for s in dets if dets[s] and (not only or s in only)), key=lambda s: (fold.get(s) != TEST_FOLD, s))
    tasks = [(s, video[s], dets[s], str(root / f"{s}.npz"), str(DATA / base) if base else None) for s in stems]
    t0 = time.time()
    with Pool(jobs) as p:
        for k, (s, n) in enumerate(p.imap_unordered(_scan_one, tasks)):
            print(f"{k + 1}/{len(tasks)} {s} {n} ({time.time() - t0:.0f}s)", flush=True)


EDGES = (-60, -40, -20, -8, 0, 8, 16, 24, 32, 40, 50, 61)     # windows (frames from the detection) features take the max over


def features(x, frames, sides):
    """Per detection, the max over each window of: the attacker box's word scores, yellow share and
    COUNTER minus PUNISH, the defender box's, the letterbox and large CH word; the nearest other-side detection
    (a trade) and the previous same-side one (its banner may still be up), in frames (capped)."""
    na = len(WORDS) + 1
    att = np.where(sides[:, None, None] == 1, x[:, :, :na], x[:, :, na:2 * na])
    dfd = np.where(sides[:, None, None] == 1, x[:, :, na:2 * na], x[:, :, :na])
    ser = np.concatenate([att, att[:, :, :1] - att[:, :, 1:2], dfd, x[:, :, 2 * na:]], 2)
    ser = np.nan_to_num(ser, nan=-1.0)
    offs = np.array(OFFS)
    win = [ser[:, (offs >= lo) & (offs < hi)].max(1) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    near = np.full((len(frames), 2), 199.0, np.float32)
    for i in range(len(frames)):
        o = np.abs(frames[sides != sides[i]] - frames[i])
        p = frames[i] - frames[(sides == sides[i]) & (frames < frames[i])]
        near[i] = (min(199.0, float(o.min())) if len(o) else 199.0, min(199.0, float(p.min())) if len(p) else 199.0)
    return np.concatenate(win + [near], 1)


def load_set(which, stems, by_key, dets):
    """Features, labels (None: no logged hit kind / not an opener) and detection keys."""
    root = DATA / f"banners_{which}"
    X, y, keys = [], [], []
    for s in stems:
        f = root / f"{s}.npz"
        if not f.exists() or not dets.get(s):
            continue
        z = np.load(f)
        X.append(features(z["x"], z["frame"], z["side"]))
        for d in dets[s]:
            r = by_key.get(f"{s}/{d['row']}") if d["row"] else None
            y.append(label_hit_kind(r) if r is not None and not r["is_burst"] else None)
            keys.append((s, d["frame"], d["side"]))
    return (np.concatenate(X) if X else np.zeros((0, 1))), y, keys


def fit(X, y):
    from sklearn.ensemble import RandomForestClassifier
    m = [i for i, l in enumerate(y) if l is not None]
    # (HistGradientBoosting hangs in its OpenMP threads on this machine)
    clf = RandomForestClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=4, random_state=0)
    clf.fit(X[m], np.array([CLASSES.index(y[i]) for i in m]))
    return clf


def proba(clf, X):
    """Class probabilities in CLASSES order."""
    P = np.zeros((len(X), len(CLASSES)), np.float32)
    if len(X):
        P[:, clf.classes_] = clf.predict_proba(X)
    return P


def load(tag="final"):
    with open(OUT / f"banner_reader_{tag}.pkl", "rb") as f:
        return pickle.load(f)["clf"]


def hit_kind(video, dets, tag="final"):
    """Counter / none / punish probabilities (CLASSES order) for every detection of one video."""
    fr, sd = np.array([d["frame"] for d in dets]), np.array([d["side"] for d in dets])
    return proba(load(tag), features(scan(video, list(fr)), fr, sd))


def probs_for(items, which, tag="test"):
    """Probabilities for video_moves items (stem, frame, side) from the extracted banners_<which>."""
    clf, P = load(tag), np.full((len(items), len(CLASSES)), np.nan, np.float32)
    at = collections.defaultdict(list)
    for i, it in enumerate(items):
        at[it["stem"]].append(i)
    for s, ii in at.items():
        f = DATA / f"banners_{which}" / f"{s}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        pos = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(z["frame"], z["side"]))}
        Pz = proba(clf, features(z["x"], z["frame"], z["side"]))
        for i in ii:
            k = pos.get((items[i]["frame"], items[i]["side"]))
            if k is not None:
                P[i] = Pz[k]
    return P


def evaluate(clf, which, stems, by_key, dets):
    X, y, keys = load_set(which, stems, by_key, dets)
    P = proba(clf, X)
    m = [i for i, l in enumerate(y) if l is not None]
    res = report([CLASSES.index(y[i]) for i in m], P[m].argmax(1), list(CLASSES), P[m])
    res["replays"] = len({k[0] for k in keys})
    return res, P, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "train", "final"])
    ap.add_argument("--set", default="rec", choices=["rec", "degraded"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--base", help="extract: banners dir under the data root to reuse rows from")
    a = ap.parse_args()
    low_priority()
    if a.mode == "extract":
        return extract(a.set, a.jobs, a.only, a.base)
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    fold = replay_fold_map()
    dets, _, _ = sources("rec")
    stems = sorted(fold)
    train = stems if a.mode == "final" else [s for s in stems if fold[s] != TEST_FOLD]
    X, y, _ = load_set("rec", train, by_key, dets)
    print(len(X), "training detections,", sum(l is not None for l in y), "with a hit kind", flush=True)
    clf = fit(X, y)
    tag = "final" if a.mode == "final" else "test"
    with open(OUT / f"banner_reader_{tag}.pkl", "wb") as f:
        pickle.dump({"clf": clf, "classes": CLASSES, "offs": OFFS}, f)
    if a.mode == "train":
        res = {}
        for sub, test in test_subsets(fold).items():
            for which in ("rec", "degraded"):
                d = dets if which == "rec" else sources("degraded")[0]
                if not any(d.get(s) for s in test):
                    continue
                r, _, _ = evaluate(clf, which, test, by_key, d)
                res[which if sub == "all" else f"{which}_{sub}"] = r
                print(which, sub, {k: r[k] for k in ("n", "accuracy", "balanced_accuracy", "replays")}, r["top_confusions"][:4], flush=True)
        save_json(res, OUT / "results_banner.json")


if __name__ == "__main__":
    main()
