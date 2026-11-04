"""Sample lab_result.json for the UIs, from a test-fold replay's saved outputs (no GPU, no video decode):
lang_testdata.py's openers.json, transcript_preds.json and frames.parquet (ggst_dataset/lang/<stem>/), the
state tower's saved predictions (search/exp_se_clip_sup_pred/<stem>.npz), through lab_analyze's own
mapping and assembly code.

The saved transcript predictions have no outcome probability. With --outcome-p (CPU only, a few seconds),
the ens5grid head reruns on the replay's saved V-JEPA 2 features (transcript/feats*, read only) and its
p_out is joined to the saved events by (side, start frame); a missing match stays null.

    set CUDA_VISIBLE_DEVICES=
    .venv/Scripts/python.exe tools/ggst_logger/model/lab_fixture.py inputs_20261022_181805 --outcome-p \
        --video G:/Projects/replanal-data/ggst_rec/inputs_20261022_181805.mp4 \
        --out tools/ggst_logger/model/fixtures/lab_result_sample.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import lab_analyze as la
import lang_summary as ls
from common import DATA


def build(stem: str, video: Path, outcome_p: bool = False) -> dict:
    d = DATA / "lang" / stem
    op = json.loads((d / "openers.json").read_text(encoding="utf-8"))
    preds = json.loads((d / "transcript_preds.json").read_text(encoding="utf-8"))
    if outcome_p:
        import torch
        torch.set_num_threads(2)
        fresh = la.transcript_predict(DATA / "transcript" / "feats", DATA / "transcript" / "feats_grid", stem, "cpu")
        po = {(p["side"], p["start_frame"]): p["p_out"] for p in fresh}
        same = sum((p["side"], p["start_frame"]) in po for p in preds)
        print(f"outcome p: {same} of {len(preds)} saved events matched by the CPU rerun ({len(fresh)} events)")
        for p in preds:
            p["p_out"] = po.get((p["side"], p["start_frame"]))
    states = la.smooth_states(np.load(DATA / "search" / f"exp_se_{la.STATE_NAME}_pred" / f"{stem}.npz")["cls_p"])
    hud = ls.Hud.load(d / "frames.parquet")
    duration, fps = la.video_info(video) if video.exists() else (hud.t[-1], float(op.get("fps") or 60.0))
    t0 = time.time()
    doc = la.assemble(str(video), duration, fps, op, preds, states, hud, match_id=stem)
    doc["timings_s"] = dict(facts=round(time.time() - t0, 2))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, nargs="+", required=True)
    ap.add_argument("--outcome-p", action="store_true", help="rerun the transcript head on the CPU for outcome_p")
    a = ap.parse_args()
    doc = build(a.stem, a.video, a.outcome_p)
    for o in a.out:
        o.parent.mkdir(parents=True, exist_ok=True)
        la.write_json(doc, o)
        print(f"-> {o} ({o.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
