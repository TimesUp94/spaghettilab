"""Combo openers of any Strive video, from the video alone: when each combo starts, who hits, both
characters and the opening move (category, button/stance, exact move, counter/punish).

Chain (final weights, trained on every logged replay):
1. the app's CV pipeline (scripts/analyze_replay.py) reads the HUD, unless --analysis has it;
2. video_openers: the frames combos start at, and the attacker's side (from the HUD readings and,
   when the weights were trained with it, the whole frame at 64x36: video_tiny);
3. video_players + video_crops: both players on each frame around them, the attacker crop, and
   both characters by vote over every frame seen;
4. video_moves: the opener's classes ('none' = not an opener after all, dropped), with the whole
   frame around each start too (video_scene) when the weights were trained with it;
5. banner_reader: counter / punish from the game's banner (in place of video_moves' guess).

    .venv/Scripts/python.exe tools/ggst_logger/model/recognize.py <video> [--out <dir>] [--analysis <dir>]

-> <out>/openers.json (and the crops under <out>/crops, <out>/scene)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, low_priority  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def analysis(video: Path, out: Path, given: Path | None) -> Path:
    """The pipeline's frames parquet for the video (run the pipeline if needed)."""
    if given is not None:
        found = list(given.glob("frames/*_frames.parquet"))
        if found:
            return found[0]
    rd = out / "analysis"
    found = list(rd.glob("frames/*_frames.parquet"))
    if found:
        return found[0]
    rd.mkdir(parents=True, exist_ok=True)
    print("running the CV pipeline ...", flush=True)
    with open(rd / "pipeline.log", "w") as log:
        subprocess.run([str(PY), str(ROOT / "scripts" / "analyze_replay.py"), str(video), "--config",
                        str(ROOT / "config" / "default.yaml"), "--output", str(rd)],
                       stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), check=True)
    return next(rd.glob("frames/*_frames.parquet"))


def detect(video: Path, parquet: Path, fps: float):
    import video_openers as vo
    import video_tiny as vt
    X, frames = vo.parquet_inputs(parquet)
    net = vo.load(OUT / "video_openers_final.pt")
    im = vt.tiny_frames(video, frames) if net.tiny is not None else None
    f60 = np.round(frames * 60.0 / fps).astype(int)
    return [dict(frame=int(f60[i]), side=s, score=round(sc, 4), row=None)
            for i, s, sc in vo.peaks(vo.predict(net, X, im), vo.THR)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analysis", type=Path, help="an existing analyze_replay output folder")
    a = ap.parse_args()
    low_priority()
    import torch
    import banner_reader as br
    import video_crops as vc
    import video_moves as vm
    import video_scene as vs
    out = a.out or a.video.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fps, w, h = vc.video_format(a.video)
    pq_path = analysis(a.video, out, a.analysis)
    dets = detect(a.video, pq_path, fps)
    print(f"{len(dets)} combo starts found ({time.time() - t0:.0f}s)", flush=True)
    cdir = out / "crops"
    if (cdir / "index.json").exists():
        (cdir / "index.json").unlink()
    vc.process((a.video.stem, str(a.video), dets, str(cdir), str(OUT / "video_players_final.pt")))
    idx = json.loads((cdir / "index.json").read_text())
    players_ck = torch.load(OUT / "video_players_final.pt", map_location="cpu")
    lp = np.array(idx["chars_logp"])
    chars = (players_ck["chars"][int(lp[0].argmax())], players_ck["chars"][int(lp[1].argmax())])
    net, ck = vm.load(OUT / "video_moves_final.pt")
    sdir = out / "scene"
    if ck.get("scene"):
        (sdir / "done").unlink(missing_ok=True)
        vs.process((a.video.stem, str(a.video), str(cdir), str(sdir)))
    items = [dict(stem="v", file=str(cdir / d["file"]), frame=d["frame"], side=d["side"], score=d["score"], row=None,
                  scene=str(sdir / d["file"])) for d in idx["dets"]]
    P = vm.predict(net, items, ck["voc"], {"v": chars})
    voc = ck["voc"]
    hk = {(d["frame"], d["side"]): p for d, p in zip(dets, br.hit_kind(a.video, dets))}
    print(f"banners read ({time.time() - t0:.0f}s)", flush=True)
    res = []
    for i, it in enumerate(items):
        cat = voc["category"][int(P["category"][i].argmax())]
        if cat == "none":
            continue
        top = np.argsort(-P["move"][i])[:3]
        res.append(dict(
            time_s=round(it["frame"] / 60.0, 3), video_frame=int(round(it["frame"] * fps / 60.0)),
            attacker=f"P{it['side']}", attacker_char=chars[it["side"] - 1], defender_char=chars[2 - it["side"]],
            detector_score=it["score"],
            category=cat, category_p=round(float(P["category"][i].max()), 3),
            move=voc["move"][top[0]].split(":", 1)[1], move_p=round(float(P["move"][i][top[0]]), 3),
            move_top3=[(voc["move"][k].split(":", 1)[1], round(float(P["move"][i][k]), 3)) for k in top],
            normal=voc["normal"][int(P["normal"][i].argmax())] if cat == "normal" else None,
            hit_kind=br.CLASSES[int(hk[(it["frame"], it["side"])].argmax())],
            hit_kind_p=round(float(hk[(it["frame"], it["side"])].max()), 3),
        ))
    doc = dict(video=str(a.video), fps=fps, size=[w, h], characters={"P1": chars[0], "P2": chars[1]}, openers=res)
    (out / "openers.json").write_text(json.dumps(doc, indent=1))
    print(f"{len(res)} openers -> {out / 'openers.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
