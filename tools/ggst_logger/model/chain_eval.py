"""The whole chain on the test fold with another opener detector: its detections are cut into crops
and whole frames (only those the chain's crops don't already have, by frame and side), then
video_moves_test.pt runs end to end on them, as in video_moves.py train.

    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --name tiny              # detections_exp_tiny.json (video_openers.py train --name tiny)
    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --name tiny --degraded   # video_openers_exp_tiny.pt on the 720p30 copies

-> <data>/ggst_dataset/vcrops_exp_<name>/ (or degraded/vcrops_exp_<name>/), model/results_exp_chain_<name>[_degraded].json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, TEST_FOLD, low_priority, replay_fold_map, save_json  # noqa: E402


def degraded_dets(name, stems, rows):
    """Detections of video_openers_exp_<name>.pt on the 720p30 copies (60 fps frame numbers)."""
    import video_openers as vo
    deg = DATA / "degraded"
    net = vo.load(OUT / f"video_openers_exp_{name}.pt")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["replay"]].append(r)
    out, n_tru, n_hit, n_det = {}, 0, 0, 0
    for s in stems:
        X, frames = vo.parquet_inputs(deg / "runs" / s / "frames" / f"{s}_frames.parquet")
        im = vo.replay_tiny(s, deg) if net.tiny is not None else None
        f60 = frames * 2
        tru = vo.truth(by, s, f60)
        out[s], h = vo.matched_dets(vo.predict(net, X, im), f60, tru)
        n_tru, n_hit, n_det = n_tru + len(tru), n_hit + h, n_det + len(out[s])
    print(f"detector on degraded video: recall {n_hit / n_tru:.3f}, precision {n_hit / max(1, n_det):.3f}")
    return out, dict(recall=n_hit / n_tru, precision=n_hit / max(1, n_det))


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build(args):
    """out/<stem> (crops) and its vscene twin from base's tiles where a detection is the same, cut where new."""
    stem, dets, base, out, video, players = args
    import video_crops as vc
    import video_moves as vm
    import video_scene as vs
    cdir, sdir = out / stem, vm.scene_root(out) / stem
    if (cdir / "index.json").exists():
        return stem, 0
    old = json.loads((base / stem / "index.json").read_text())
    have = {(d["frame"], d["side"]): d for d in old["dets"]}
    cdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    new = [d for d in dets if (d["frame"], d["side"]) not in have]
    cut = {}
    if new:
        tmp = out / "_tmp" / stem
        shutil.rmtree(tmp, ignore_errors=True)
        vc.process((stem, video, new, str(tmp), players))
        stmp = out / "_tmp_scene" / stem
        shutil.rmtree(stmp, ignore_errors=True)
        vs.process((stem, video, str(tmp), str(stmp)))
        for d in json.loads((tmp / "index.json").read_text())["dets"]:
            shutil.move(tmp / d["file"], cdir / d["file"])
            if (stmp / d["file"]).exists():
                shutil.move(stmp / d["file"], sdir / d["file"])
            cut[(d["frame"], d["side"])] = d
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(stmp, ignore_errors=True)
    entries = []
    for d in dets:
        k = (d["frame"], d["side"])
        if k in have:
            o = have[k]
            link(base / stem / o["file"], cdir / o["file"])
            sc = vm.scene_root(base) / stem / o["file"]
            if sc.exists():
                link(sc, sdir / o["file"])
            entries.append(dict(d, file=o["file"], pos=o["pos"]))
        elif k in cut:
            entries.append(dict(d, file=cut[k]["file"], pos=cut[k]["pos"]))
    idx = dict(old, dets=entries)
    (cdir / "index.json").write_text(json.dumps(idx))
    return stem, len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="detector experiment (video_openers_exp_<name>.pt / detections_exp_<name>.json)")
    ap.add_argument("--degraded", action="store_true", help="the 720p30 copies of the test replays")
    ap.add_argument("--moves", default="video_moves_test.pt")
    ap.add_argument("--base", help="crops dir to reuse tiles from (default: the chain's, vcrops_test or degraded/vcrops)")
    ap.add_argument("--jobs", type=int, default=3, help="processes cutting new crops (each loads the player net)")
    a = ap.parse_args()
    low_priority()
    import pyarrow.parquet as pq
    import torch
    import video_moves as vm
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    test = sorted(s for s in fold if fold[s] == TEST_FOLD)
    video = {r["replay"]: r["video"] for r in rows}
    root = DATA / "degraded" if a.degraded else DATA
    base = root / (a.base or ("vcrops" if a.degraded else "vcrops_test"))
    out = root / f"vcrops_exp_{a.name}"
    det_stats = None
    if a.degraded:
        dets, det_stats = degraded_dets(a.name, test, [r for r in rows if r["replay"] in set(test)])
        video = {s: str(root / f"{s}.mp4") for s in test}
    else:
        dets = json.loads((OUT / f"detections_exp_{a.name}.json").read_text())
    players = str(OUT / "video_players_test.pt")
    n_new = 0
    tasks = [(s, dets.get(s, []), base, out, video[s], players) for s in test]
    with Pool(a.jobs) as p:
        for k, (s, n) in enumerate(p.imap_unordered(build, tasks)):
            n_new += n
            if k % 20 == 0:
                print(f"  {k + 1}/{len(test)} replays, {n_new} new detections cut", flush=True)
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    net, ck = vm.load(OUT / a.moves)
    items = vm.items_for(out, by_key, test)
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    vch = vm.video_chars(out, test, chars_list)
    P = vm.predict(net, items, ck["voc"], vch)
    res = vm.end_to_end(items, P, ck["voc"], vch, rows, set(test))
    res["new_detections_cut"] = n_new
    if det_stats:
        res["detector"] = det_stats
    save_json(res, OUT / f"results_exp_chain_{a.name}{'_degraded' if a.degraded else ''}.json")
    brief = {k: v for k, v in res.items() if k not in ("category", "hit_kind", "move_acc_by_char")}
    brief["category_acc"] = res["category"]["accuracy"]
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
