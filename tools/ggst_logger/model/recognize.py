"""Combo openers of any Strive video, from the video alone: when each combo starts, who hits, both
characters and the opening move (category, button/stance, exact move, counter/punish).

Chain (final weights, trained on every logged replay):
1. the app's CV pipeline (scripts/analyze_replay.py) reads the HUD, unless --analysis has it;
2. video_openers: the frames combos start at, and the attacker's side (from the HUD readings and,
   when the weights were trained with it, the whole frame at 64x36: video_tiny);
3. hud_chars: both characters from the game's own HUD (portrait, then the name on the bar), per stretch of the video
   (a tournament set can change characters between games); video_players + video_crops: both players
   on each frame around the starts, the attacker crop, and the characters' visual vote as the fallback;
4. video_moves: the opener's classes ('none' = not an opener after all, dropped), with the whole
   frame around each start too (video_scene) when the weights were trained with it;
5. banner_reader: counter / punish from the game's banner (in place of video_moves' guess).

    .venv/Scripts/python.exe tools/ggst_logger/model/recognize.py <video> [--out <dir>] [--analysis <dir>]

Runs on the GPU when there is one, else on the CPU (GGST_DEVICE / --device to choose); weights from
GGST_MODEL_DIR (default <GGST_DATA>/ggst_dataset/model). --light uses only the move ensemble's
first net and no frame-shift averaging: about a third of the CPU time, a point or two of accuracy.

-> <out>/openers.json (and the crops under <out>/crops, <out>/scene)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, device, low_priority  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PY = Path(sys.executable)


def analysis(video: Path, out: Path, given: Path | None) -> Path:
    """The pipeline's frames parquet for the video (run the pipeline if needed)."""
    if given is not None:
        found = list(given.glob("frames/*_frames.parquet"))
        # a shared output folder holds every replay's parquet: take this video's own, never just the first
        mine = [f for f in found if f.name.startswith(video.stem)]
        if mine or len(found) == 1:
            return (mine or found)[0]
        if found:
            raise SystemExit(f"{given} has {len(found)} frames parquets and none named after {video.stem}")
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


def detect(video: Path, parquet: Path, fps: float) -> tuple[list[dict], list[dict]]:
    """(combo starts, the detections dropped because the attacker's combo counter shows they continue a combo)."""
    import pyarrow.parquet as pq
    import video_openers as vo
    import video_tiny as vt
    X, frames = vo.parquet_inputs(parquet)
    net = vo.load(OUT / "video_openers_final.pt")
    im = vt.tiny_frames(video, frames) if net.tiny is not None else None
    f60 = np.round(frames * 60.0 / fps).astype(int)
    hud = pq.read_table(parquet, columns=["combo_side", "combo_count"]).to_pydict()
    shown = {s: vo.counter_readings(hud["combo_side"], hud["combo_count"], s) for s in (1, 2)}
    kept, dropped = [], []
    for i, s, sc in vo.peaks(vo.predict(net, X, im), vo.THR):
        (dropped if vo.continues_combo(shown[s], i) else kept).append(dict(frame=int(f60[i]), side=s, score=round(sc, 4), row=None))
    return kept, dropped


def read_characters(video: Path, parquet: Path) -> tuple[list | None, dict | None]:
    """The HUD's scores over the video's gameplay and the references' character names; (None, None) without references."""
    import hud_chars as hc
    path = hc.refs_path()
    if not path.exists():
        print(f"no {path.name}: characters from the visual vote", flush=True)
        return None, None
    refs = hc.Refs(path)
    return refs.chars, hc.read_video(video, refs, parquet)


def visual_span(idx: dict, names: list[str]):
    """The visual vote over a span of the video: visual(start_s, end_s) -> (P1, P2). The detections' own votes (frame numbers at 60 fps);
    the whole video's when the span has none."""
    votes = [(d["frame"] / 60.0, np.array(d["chlp"])) for d in idx["dets"] if d.get("chlp") is not None]

    def visual(a: float, b: float):
        lp = [v for t, v in votes if a <= t < b]
        s = np.sum(lp, 0) if lp else np.array(idx["chars_logp"])
        return names[int(s[0].argmax())], names[int(s[1].argmax())]
    return visual


def run_chain(video: Path, out: Path, given: Path | None = None, light: bool = False, stage=None, check_hud=None) -> tuple[dict, Path]:
    """The whole opener chain on one video -> (the openers.json document, the HUD parquet path); writes
    <out>/openers.json. stage(name) is called as each part starts: hud, detect, moves. check_hud(parquet) runs
    right after the HUD pass and may raise, before the models spend any CPU."""
    import torch
    import banner_reader as br
    import hud_chars as hc
    import video_crops as vc
    import video_moves as vm
    import video_scene as vs
    stage = stage or (lambda name: None)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fps, w, h = vc.video_format(video)
    stage("hud")
    pq_path = analysis(video, out, given)
    if check_hud:
        check_hud(pq_path)
    stage("detect")
    dets, dropped = detect(video, pq_path, fps)
    print(f"{len(dets)} combo starts found, {len(dropped)} dropped inside a combo ({time.time() - t0:.0f}s)", flush=True)
    stage("moves")
    ref_chars, hud_scores = read_characters(video, pq_path)
    hud_segs = hc.resolve(hud_scores, ref_chars) if hud_scores is not None else None
    print(f"HUD read ({time.time() - t0:.0f}s)", flush=True)

    def known_at(frame: int):
        """Both characters at a frame, when the HUD is sure of them."""
        if not hud_segs:
            return None
        g = hc.segment_at(hud_segs, frame / 60.0)
        return (g["P1"]["char"], g["P2"]["char"]) if g["P1"]["char"] and g["P2"]["char"] else None
    cdir = out / "crops"
    if (cdir / "index.json").exists():
        (cdir / "index.json").unlink()
    vc.process((video.stem, str(video), dets, str(cdir), str(OUT / "video_players_final.pt"), known_at))
    idx = json.loads((cdir / "index.json").read_text())
    players_ck = torch.load(OUT / "video_players_final.pt", map_location="cpu")
    visual = visual_span(idx, players_ck["chars"])
    if hud_scores is not None:
        segs = hc.resolve(hud_scores, ref_chars, visual)
    else:
        v1, v2 = visual(0, float("inf"))
        segs = [dict(start_s=0.0, end_s=None, frames=0, P1=hc.visual_reading(v1), P2=hc.visual_reading(v2))]
    pairs = {f"v{k}": (g["P1"]["char"], g["P2"]["char"]) for k, g in enumerate(segs)}
    print("characters:", "; ".join(f"{g['start_s']:.0f}s {g['P1']['char']} ({g['P1']['source']}) vs {g['P2']['char']} ({g['P2']['source']})"
                                   for g in segs), flush=True)
    net, ck = vm.load(OUT / "video_moves_final.pt")
    shifts = vm.TTA
    if light:
        net, shifts = (net[0] if isinstance(net, list) else net), [0]
    sdir = out / "scene"
    if ck.get("scene"):
        (sdir / "done").unlink(missing_ok=True)
        vs.process((video.stem, str(video), str(cdir), str(sdir)))
    # the crops stay; each detection's moves are restricted to the characters of its stretch of the video
    items = [dict(stem=f"v{hc.segment_index(segs, d['frame'] / 60.0)}", file=str(cdir / d["file"]), frame=d["frame"], side=d["side"],
                  score=d["score"], row=None, scene=str(sdir / d["file"])) for d in idx["dets"]]
    P = vm.predict(net, items, ck["voc"], pairs, shifts=shifts)
    voc = ck["voc"]
    hk = {(d["frame"], d["side"]): p for d, p in zip(dets, br.hit_kind(video, dets))}
    print(f"banners read ({time.time() - t0:.0f}s)", flush=True)
    res = []
    for i, it in enumerate(items):
        cat = voc["category"][int(P["category"][i].argmax())]
        if cat == "none":
            continue
        top = np.argsort(-P["move"][i])[:3]
        chars = pairs[it["stem"]]
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
    doc = dict(video=str(video), fps=fps, size=[w, h], **hc.characters_fields(segs),
               dropped_in_combo=[dict(time_s=round(d["frame"] / 60.0, 3), attacker=f"P{d['side']}", detector_score=d["score"]) for d in dropped],
               device=device(), light=light, seconds=round(time.time() - t0, 1), openers=res)
    (out / "openers.json").write_text(json.dumps(doc, indent=1))
    print(f"{len(res)} openers -> {out / 'openers.json'} ({time.time() - t0:.0f}s)", flush=True)
    return doc, pq_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analysis", type=Path, help="an existing analyze_replay output folder")
    ap.add_argument("--device", choices=["cuda", "cpu"], help="default: the GPU when there is one")
    ap.add_argument("--light", action="store_true", help="one move net, no frame-shift averaging (CPU servers)")
    ap.add_argument("--threads", type=int, help="CPU threads for the networks (default: torch's)")
    a = ap.parse_args()
    if a.device:
        os.environ["GGST_DEVICE"] = a.device
    low_priority()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    run_chain(a.video, a.out or a.video.with_suffix(""), a.analysis, a.light)


if __name__ == "__main__":
    main()
