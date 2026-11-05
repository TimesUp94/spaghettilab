"""Measurement pass, step 5: where the CPU server's time goes. recognize.run_chain --light on one video, device CPU, torch and the
pipeline limited as server/opener_service limits them (OMP / MKL / OpenBLAS threads, VM_WORKERS=0, process tree pinned to
OPENERS_THREADS logical CPUs), each stage timed: wall clock and the CPU seconds of the whole process tree (sampled once a
second; a child's last sample counts). The same stage wrappers run for every video, so a clip and a long set split into
fixed cost and cost per minute / per opener.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_timing.py prep                   # the test videos, at 720p30 3 Mbit/s like the app's uploads
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_timing.py run <name>             # clip90 | set4m | set12m
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_timing.py fit                    # the fit, from the results
-> model/results_exp_measure_timing_<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, save_json  # noqa: E402
from exp_measure_common import MEASURE  # noqa: E402

SETS = Path(__file__).resolve().parents[3] / "data" / "sets"
WORK = Path(os.environ.get("GGST_TIMING_DIR") or MEASURE / "timing")    # a local SSD: G: is a busy disk, which is not what is being measured
VIDEOS = {  # name: (source, start s, duration s or None)
    "clip90": ("11_WFinal_GuessingGame_vs_Jason.mp4", 120, 90),
    "set4m": ("08_UF_Revill_vs_EneroNeJR.mp4", 0, None),
    "set12m": ("11_WFinal_GuessingGame_vs_Jason.mp4", 0, None),
}
FFMPEG = r"G:\Projects\replanal-data\ggst_mod\dl\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"


def prep():
    WORK.mkdir(parents=True, exist_ok=True)
    for name, (src, start, dur) in VIDEOS.items():
        out = WORK / f"{name}.mp4"
        if out.exists():
            continue
        cmd = [FFMPEG, "-hide_banner", "-v", "error", "-y", "-ss", str(start), "-i", str(SETS / src)]
        if dur:
            cmd += ["-t", str(dur)]
        cmd += ["-vf", "fps=30,scale=1280:720:flags=bicubic", "-c:v", "libx264", "-preset", "veryfast", "-threads", "4", "-b:v", "3M",
                "-maxrate", "3.5M", "-bufsize", "6M", "-an", str(WORK / f"{name}.part.mp4")]
        subprocess.run(cmd, check=True)
        (WORK / f"{name}.part.mp4").rename(out)
        print(out, flush=True)


class Tree:
    """CPU seconds of this process and every descendant, sampled."""

    def __init__(self):
        import psutil
        self.psutil, self.me = psutil, psutil.Process()
        self.last = {}
        self.stop_ = False
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()

    def sample(self):
        try:
            ps = [self.me] + self.me.children(recursive=True)
        except self.psutil.Error:
            return
        for p in ps:
            try:
                t = p.cpu_times()
                self.last[(p.pid, p.create_time())] = t.user + t.system
            except self.psutil.Error:
                pass

    def loop(self):
        while not self.stop_:
            self.sample()
            time.sleep(1)

    def total(self):
        self.sample()
        return sum(self.last.values())


class Stages:
    def __init__(self):
        self.tree = Tree()
        self.rec = {}
        self.depth = 0

    def wrap(self, name, fn):
        def f(*a, **k):
            outer = self.depth == 0
            self.depth += 1
            w0, c0 = time.time(), self.tree.total()
            try:
                return fn(*a, **k)
            finally:
                self.depth -= 1
                d = self.rec.setdefault(name, dict(wall=0.0, cpu=0.0, calls=0))
                d["wall"] += time.time() - w0
                d["cpu"] += self.tree.total() - c0
                d["calls"] += 1
                if outer:
                    print(f"  {name}: {time.time() - w0:.1f}s", flush=True)
        return f


def run(name, threads, cpus):
    video = WORK / f"{name}.mp4"
    n = str(threads)
    os.environ.update(GGST_DEVICE="cpu", OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, VM_WORKERS="0", PYTHONUNBUFFERED="1")
    import psutil
    psutil.Process().cpu_affinity(cpus)
    wall0 = time.time()
    st = Stages()
    import torch
    torch.set_num_threads(threads)
    import banner_reader as br
    import cv2
    cv2.setNumThreads(threads)
    import hud_chars as hc  # noqa: F401
    import recognize as rc
    import video_crops as vc
    import video_moves as vm
    import video_scene as vs
    import_s = time.time() - wall0
    rc.analysis = st.wrap("hud_pipeline", rc.analysis)
    rc.detect = st.wrap("detector", rc.detect)
    rc.read_characters = st.wrap("hud_characters", rc.read_characters)
    vc.process = st.wrap("crops", vc.process)
    vs.process = st.wrap("scene", vs.process)
    vm.load = st.wrap("move_load", vm.load)
    vm.predict = st.wrap("move_nets", vm.predict)
    br.hit_kind = st.wrap("banners", br.hit_kind)
    out = WORK / f"out_{name}"
    shutil.rmtree(out, ignore_errors=True)
    t0, c0 = time.time(), st.tree.total()
    try:
        doc, _ = rc.run_chain(video, out, None, light=True)
    except BaseException as e:                     # keep what was measured
        save_json(dict(name=name, failed=repr(e)[:300], wall_s=round(time.time() - t0, 1), stages=st.rec), OUT / f"results_exp_measure_timing_{name}_failed.json")
        raise
    wall, cpu = time.time() - t0, st.tree.total() - c0
    fps, w, h = vc.video_format(video)
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    idx = json.loads((out / "crops" / "index.json").read_text())
    res = dict(work=str(WORK), name=name, video_s=round(dur, 1), size=[w, h], fps=fps, threads=threads, cpus=cpus, import_s=round(import_s, 1), wall_s=round(wall, 1), cpu_s=round(cpu, 1),
               wall_per_video_s=round(wall / dur, 2), detections=len(idx["dets"]), openers=len(doc["openers"]),
               stages={k: {x: round(y, 1) for x, y in v.items()} for k, v in st.rec.items()})
    res["unaccounted_wall_s"] = round(wall - sum(v["wall"] for v in st.rec.values()), 1)
    print(json.dumps(res, indent=1), flush=True)
    save_json(res, OUT / f"results_exp_measure_timing_{name}.json")
    shutil.rmtree(out, ignore_errors=True)


def fit():
    import numpy as np
    rs = [json.loads((OUT / f"results_exp_measure_timing_{n}.json").read_text()) for n in VIDEOS if (OUT / f"results_exp_measure_timing_{n}.json").exists()]
    rs.sort(key=lambda r: r["video_s"])
    names = sorted({k for r in rs for k in r["stages"]})
    out = dict(videos=[{k: r[k] for k in ("name", "video_s", "detections", "openers", "wall_s", "cpu_s", "wall_per_video_s")} for r in rs], stages={})
    for k in names + ["total"]:
        for m in ("wall", "cpu"):
            y = [r[f"{m}_s"] if k == "total" else r["stages"].get(k, {}).get(m, 0.0) for r in rs]
            x = [r["video_s"] for r in rs]
            b, a = np.polyfit(x, y, 1)
            out["stages"].setdefault(k, {})[m] = dict(values=[round(v, 1) for v in y], fixed_s=round(float(a), 1), per_video_s=round(float(b), 3))
    t = out["stages"]["total"]
    for m in ("wall", "cpu"):
        t[m]["at_15min_s"] = round(t[m]["fixed_s"] + 900 * t[m]["per_video_s"])
    print(json.dumps(out, indent=1))
    save_json(out, OUT / "results_exp_measure_timing_fit.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "run", "fit"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cpus", default="16,18", help="logical CPUs the tree is pinned to")
    a = ap.parse_args()
    if a.cmd == "prep":
        return prep()
    if a.cmd == "fit":
        return fit()
    run(a.name, a.threads, [int(x) for x in a.cpus.split(",")])


if __name__ == "__main__":
    main()
