"""The whole video-only chain on stream-like copies of the test replays: 720p, 30 fps, 3 Mbit/s
H.264 (a typical Twitch / YouTube VOD), against the game log.

Stages (each skips what is done):
    encode   test-fold recordings -> <data>/ggst_dataset/degraded/<stem>.mp4
    analyze  the app's CV pipeline (scripts/analyze_replay.py) -> degraded/runs/<stem>/
    detect   video_openers_test.pt on those HUD readings, matched to the log -> degraded/detections.json
    crops    video_crops.py on the degraded videos -> degraded/vcrops/<stem>/
    scene    video_scene.py (whole frames, for a move net trained with --scene) -> degraded/vscene/<stem>/
    banners  banner_reader.py on the degraded videos -> banners_degraded/<stem>.npz
    eval     video_moves_test.pt end to end (counter / punish from banner_reader_test.pkl) -> model/results_degraded.json

    .venv/Scripts/python.exe tools/ggst_logger/model/degrade_eval.py encode analyze detect crops scene banners eval
    .venv/Scripts/python.exe tools/ggst_logger/model/degrade_eval.py eval --moves video_moves_exp_x.pt   # -> results_exp_degraded_x.json
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, TEST_FOLD, low_priority, replay_fold_map, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
DEG = DATA / "degraded"


def test_rows():
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    return [r for r in rows if fold.get(r["replay"]) == TEST_FOLD]


def encode(stems, video):
    import video_crops
    DEG.mkdir(parents=True, exist_ok=True)

    def one(s):
        out = DEG / f"{s}.mp4"
        if out.exists():
            return
        tmp = DEG / f"{s}.part.mp4"
        subprocess.run([video_crops.FFMPEG, "-hide_banner", "-v", "error", "-y", "-i", video[s],
                        "-vf", "fps=30,scale=1280:720:flags=bicubic", "-c:v", "libx264", "-preset", "veryfast", "-threads", "4",
                        "-b:v", "3M", "-maxrate", "3.5M", "-bufsize", "6M", "-an", str(tmp)], check=True)
        tmp.rename(out)

    with ThreadPoolExecutor(3) as ex:
        for k, _ in enumerate(ex.map(one, stems)):
            if k % 20 == 0:
                print(f"  encoded {k + 1}/{len(stems)}", flush=True)


def analyze(stems, jobs):
    def one(s):
        rd = DEG / "runs" / s
        if (rd / "frames" / f"{s}_frames.parquet").exists():
            return s, 0.0
        rd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(rd / "pipeline.log", "w") as log:
            subprocess.run([str(PY), str(ROOT / "scripts" / "analyze_replay.py"), str(DEG / f"{s}.mp4"),
                            "--config", str(ROOT / "config" / "default.yaml"), "--output", str(rd)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), creationflags=0x4000)
        return s, time.time() - t0

    with ThreadPoolExecutor(jobs) as ex:
        for k, (s, dt) in enumerate(ex.map(one, stems)):
            print(f"  analysed {k + 1}/{len(stems)} {s} ({dt:.0f}s)", flush=True)


def detect(stems, rows):
    """Detections in 60 fps frame numbers (the original recordings' timeline), matched to the log."""
    import torch
    import video_openers as vo
    by = collections.defaultdict(list)
    for r in rows:
        by[r["replay"]].append(r)
    net = vo.make_net().cuda()
    net.load_state_dict(torch.load(OUT / "video_openers_test.pt"))
    out, n_tru, n_hit, n_det = {}, 0, 0, 0
    for s in stems:
        X, frames = vo.parquet_inputs(DEG / "runs" / s / "frames" / f"{s}_frames.parquet")
        f60 = frames * 2                                   # 30 fps -> 60 fps frame numbers
        tru = vo.truth(by, s, f60)
        dets = vo.peaks(vo.predict(net, X), 0.3)
        hit, _ = vo.evaluate(dets, tru, f60)
        row_of = {(d[0], d[1]): tru[j][2] for j, d in hit.items()}
        out[s] = [dict(frame=int(f60[i]), side=sd, score=round(sc, 4),
                       row=(f"{row_of[(i, sd)]['log_row']}_{row_of[(i, sd)]['attacker']}" if (i, sd) in row_of else None))
                  for i, sd, sc in dets]
        n_tru, n_hit, n_det = n_tru + len(tru), n_hit + len(hit), n_det + len(dets)
    (DEG / "detections.json").write_text(json.dumps(out))
    print(f"detector on degraded video: recall {n_hit / n_tru:.3f}, precision {n_hit / max(1, n_det):.3f}")


def crops(stems):
    import video_crops as vc
    dets = json.loads((DEG / "detections.json").read_text())
    netfile = str(OUT / "video_players_test.pt")
    for k, s in enumerate(stems):
        stem, n = vc.process((s, str(DEG / f"{s}.mp4"), dets[s], str(DEG / "vcrops" / s), netfile))
        print(f"  crops {k + 1}/{len(stems)} {s} {n}", flush=True)


def scene(stems, jobs):
    import video_scene as vs
    from multiprocessing import Pool
    tasks = [(s, str(DEG / f"{s}.mp4"), str(DEG / "vcrops" / s), str(DEG / "vscene" / s)) for s in stems]
    with Pool(jobs) as p:
        for k, (s, n) in enumerate(p.imap_unordered(vs.process, tasks)):
            print(f"  scene {k + 1}/{len(stems)} {s} {n}", flush=True)


def evaluate(stems, rows, moves="video_moves_test.pt"):
    import pyarrow.parquet as pq
    import torch
    import video_moves as vm
    allrows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in allrows}
    net, ck = vm.load(OUT / moves)
    root = DEG / "vcrops"
    items = vm.items_for(root, by_key, stems)
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    vch = vm.video_chars(root, stems, chars_list)
    P = vm.predict(net, items, ck["voc"], vch)
    if (OUT / "banner_reader_test.pkl").exists():
        import banner_reader as br
        assert tuple(ck["voc"]["hit_kind"]) == br.CLASSES
        Pb = br.probs_for(items, "degraded")
        P["hit_kind"] = np.where(np.isnan(Pb).any(1)[:, None], P["hit_kind"], Pb)
    res = vm.end_to_end(items, P, ck["voc"], vch, allrows, set(stems))
    exp = moves.startswith("video_moves_exp_")
    save_json(res, OUT / (f"results_exp_degraded_{moves[16:-3]}.json" if exp else "results_degraded.json"))
    brief = {k: v for k, v in res.items() if k not in ("category", "hit_kind", "move_acc_by_char")}
    brief["category_acc"] = res["category"]["accuracy"]
    brief["hit_kind_acc"] = res["hit_kind"]["accuracy"]
    print(json.dumps(brief, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["encode", "analyze", "detect", "crops", "scene", "banners", "eval"])
    ap.add_argument("--moves", default="video_moves_test.pt", help="move net weights for eval")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N test replays only")
    a = ap.parse_args()
    low_priority()
    rows = test_rows()
    video = {r["replay"]: r["video"] for r in rows}
    stems = sorted(video)[:a.limit or None]
    print(len(stems), "test replays")
    for st in a.stages:
        t0 = time.time()
        {"encode": lambda: encode(stems, video), "analyze": lambda: analyze(stems, a.jobs),
         "detect": lambda: detect(stems, rows), "crops": lambda: crops(stems), "scene": lambda: scene(stems, a.jobs),
         "banners": lambda: __import__("banner_reader").extract("degraded", a.jobs, stems),
         "eval": lambda: evaluate(stems, rows, a.moves)}[st]()
        print(f"[{st}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
