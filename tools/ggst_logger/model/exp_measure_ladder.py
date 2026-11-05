"""Measurement pass, step 3: the opener chain on stream-like copies at lower bitrates and at 480p, on the first N old test
replays (sorted by name), with degrade_eval's own stages. The 3 Mbit/s rung is the existing 720p30 copies, scored on the same N.
Uses libx264. Everything goes under measure/ladder/<tag>/ and is deleted by `clean <tag>`; results stay in model/.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_ladder.py base --limit 30         # the existing copies
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_ladder.py run 720p_1500 --limit 30
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_ladder.py clean 720p_1500
-> model/results_exp_measure_ladder_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, low_priority, save_json  # noqa: E402
from exp_measure_common import MEASURE, opener_counts, paired_bootstrap, ratio  # noqa: E402
import degrade_eval as de  # noqa: E402

RUNGS = {  # tag: (w, h, bitrate, maxrate, bufsize): the same filters and x264 settings as degrade_eval.encode
    "720p_1500": (1280, 720, "1500k", "1750k", "3M"),
    "720p_750": (1280, 720, "750k", "875k", "1500k"),
    "480p_1500": (854, 480, "1500k", "1750k", "3M"),
    "480p_750": (854, 480, "750k", "875k", "1500k"),
}


def encode(stems, video, spec):
    import video_crops
    w, h, b, m, buf = spec
    de.DEG.mkdir(parents=True, exist_ok=True)

    def one(s):
        out = de.DEG / f"{s}.mp4"
        if out.exists():
            return
        tmp = de.DEG / f"{s}.part.mp4"
        subprocess.run([video_crops.FFMPEG, "-hide_banner", "-v", "error", "-y", "-i", video[s],
                        "-vf", f"fps=30,scale={w}:{h}:flags=bicubic", "-c:v", "libx264", "-preset", "veryfast", "-threads", "4",
                        "-b:v", b, "-maxrate", m, "-bufsize", buf, "-an", str(tmp)], check=True)
        tmp.rename(out)

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(one, stems))
    print("encoded MB:", round(sum((de.DEG / f"{s}.mp4").stat().st_size for s in stems) / 2 ** 20), flush=True)


def score(tag, stems, rows):
    """Full ensemble and --light, without the banner reader (the chain metric doesn't use it)."""
    import pyarrow.parquet as pq
    import torch
    import video_moves as vm
    allrows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in allrows}
    net, ck = vm.load(OUT / "video_moves_test.pt")
    root = de.DEG / "vcrops"
    items = vm.items_for(root, by_key, stems)
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    vch = vm.video_chars(root, stems, chars_list)
    out = dict(tag=tag, replays=len(stems), detections=len(items))
    counts = {}
    for name, n, sh in (("full", net, vm.TTA), ("light", net[0] if isinstance(net, list) else net, [0])):
        P = vm.predict(n, items, ck["voc"], vch, shifts=sh)
        r = vm.end_to_end(items, P, ck["voc"], vch, allrows, set(stems))
        out[name] = {k: r[k] for k in ("openers", "detected", "detected_and_kept", "precision_after_none", "video_char_right", "move_acc", "chain_move_right_of_all_openers", "detector_precision")}
        out[name]["category"] = r["category"]["accuracy"]
        counts[name] = opener_counts(items, P, ck["voc"], allrows, stems)
    out["counts"] = counts
    out["chars_right"] = sum(all(vch[s][k] == next((x["char"] for x in allrows if x["replay"] == s and x["attacker"] == f"P{k + 1}"), vch[s][k]) for k in (0, 1)) for s in stems if s in vch)
    return out


def compare(tag, res):
    """Against the base rung on the same replays."""
    f = OUT / "results_exp_measure_ladder_base.json"
    if tag == "base" or not f.exists():
        return
    base = json.loads(f.read_text())
    for name in ("full", "light"):
        a, b = base["counts"][name], res["counts"][name]
        res[f"vs_base_{name}"] = dict(chain=paired_bootstrap(a, b, "chain", "n"), move=paired_bootstrap(a, b, "move", "det"), category=paired_bootstrap(a, b, "cat", "det"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("base", "run", "clean"))
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    low_priority()
    if a.cmd == "clean":
        shutil.rmtree(MEASURE / "ladder" / a.tag, ignore_errors=True)
        return
    rows = de.test_rows("old")
    video = {r["replay"]: r["video"] for r in rows}
    stems = sorted(video)[:a.limit]
    print(len(stems), "replays", flush=True)
    t00 = time.time()
    if a.cmd == "base":
        tag = "base"
        de.DEG = DATA / "degraded"
        # the same detection code path must give the saved detections, or the comparison isn't like for like
        saved = json.loads((de.DEG / "detections.json").read_text())
        import video_openers as vo
        by = {}
        for r in rows:
            by.setdefault(r["replay"], []).append(r)
        net = vo.load(OUT / "video_openers_test.pt")
        same = total = 0
        for s in stems:
            X, frames = vo.parquet_inputs(de.DEG / "runs" / s / "frames" / f"{s}_frames.parquet")
            d = vo.peaks(vo.predict(net, X, vo.replay_tiny(s, de.DEG) if net.tiny is not None else None), vo.THR)
            now = {(int(frames[i] * 2), sd) for i, sd, _ in d}
            was = {(x["frame"], x["side"]) for x in saved[s]}
            same += len(now & was)
            total += len(now | was)
        print(f"detections now against the saved ones: {same} of {total} the same", flush=True)
    else:
        tag = a.tag
        de.DEG = MEASURE / "ladder" / tag
        for name, fn in (("encode", lambda: encode(stems, video, RUNGS[tag])), ("analyze", lambda: de.analyze(stems, a.jobs)),
                         ("tiny", lambda: de.tiny(stems, a.jobs)), ("detect", lambda: de.detect(stems, rows)), ("crops", lambda: de.crops(stems)),
                         ("scene", lambda: de.scene(stems, a.jobs))):
            t0 = time.time()
            fn()
            print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
    res = score(tag, stems, rows)
    compare(tag, res)
    res["seconds"] = round(time.time() - t00)
    if a.cmd == "run":
        res["disk_gb"] = round(sum(f.stat().st_size for f in de.DEG.rglob("*") if f.is_file()) / 2 ** 30, 2)
    save_json(res, OUT / f"results_exp_measure_ladder_{tag}.json")
    print(json.dumps({k: v for k, v in res.items() if k != "counts"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
