"""Inputs for the language layer on the test fold, from predictions that already exist (CPU only, no
network, no GPU): ROADMAP point 5.

For each test replay it writes what the deployed chain would have produced from the video alone:
- <out>/<stem>/openers.json in recognize.py's schema, rebuilt from the test-fold chain outputs that were
  saved when the chain was measured: the detector's starts (vcrops_test/<stem>/index.json), the v2 move
  ensemble's test probabilities (model/exp_bb_v2_P_video_moves_test.npz, frame-shift averaged), the
  characters voted by video_players_test.pt and banner_reader_test.pkl's counter / punish on banners_rec.
  The 'row' link of each detection (to the log) is not read: the documents carry nothing from the log.
- <out>/<stem>/transcript_preds.json: that replay's rows of transcript/exp_tr_ens5grid_test_preds.parquet.
- <out>/<stem>/frames.parquet: a copy of the CV pipeline's HUD readings (runs/<stem>/frames/...).

    .venv/Scripts/python.exe tools/ggst_logger/model/lang_testdata.py --n 4 [--stems a b ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # never touch the GPU
os.environ["GGST_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT  # noqa: E402

CROPS = DATA / "vcrops_test"
TR = DATA / "transcript"
LANG = DATA / "lang"


def chain_docs(stems: list[str]) -> dict[str, dict]:
    """recognize.py-style openers.json documents for test replays, from the saved test outputs."""
    import torch
    import banner_reader as br
    voc = torch.load(OUT / "video_moves_test.pt", map_location="cpu", weights_only=False)["voc"]
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu", weights_only=False)["chars"]
    z = np.load(OUT / "exp_bb_v2_P_video_moves_test.npz")
    pos = {str(Path(f)): i for i, f in enumerate(z["files"])}
    out = {}
    for s in stems:
        idx = json.loads((CROPS / s / "index.json").read_text())
        lp = np.array(idx["chars_logp"])
        chars = (chars_list[int(lp[0].argmax())], chars_list[int(lp[1].argmax())])
        items = [dict(stem=s, file=str(CROPS / s / d["file"]), frame=d["frame"], side=d["side"], score=d["score"])
                 for d in idx["dets"]]
        Pb = br.probs_for(items, "rec", "test")
        res = []
        for it, pb in zip(items, Pb):
            k = pos.get(str(Path(it["file"])))
            if k is None:
                continue
            pc, pm = z["category"][k], z["move"][k]
            cat = voc["category"][int(pc.argmax())]
            if cat == "none":
                continue
            top = np.argsort(-pm)[:3]
            hk = pb if not np.isnan(pb).any() else z["hit_kind"][k]
            res.append(dict(
                time_s=round(it["frame"] / 60.0, 3), video_frame=int(it["frame"]),
                attacker=f"P{it['side']}", attacker_char=chars[it["side"] - 1], defender_char=chars[2 - it["side"]],
                detector_score=it["score"], category=cat, category_p=round(float(pc.max()), 3),
                move=voc["move"][top[0]].split(":", 1)[1], move_p=round(float(pm[top[0]]), 3),
                move_top3=[(voc["move"][j].split(":", 1)[1], round(float(pm[j]), 3)) for j in top],
                normal=voc["normal"][int(z["normal"][k].argmax())] if cat == "normal" else None,
                hit_kind=br.CLASSES[int(hk.argmax())], hit_kind_p=round(float(hk.max()), 3)))
        res.sort(key=lambda r: r["time_s"])
        out[s] = dict(video=str(Path(r"G:\Projects\replanal-data\ggst_rec") / f"{s}.mp4"), fps=60.0, size=[1920, 1080],
                      characters={"P1": chars[0], "P2": chars[1]}, device="cpu", light=False,
                      source="test-fold chain outputs (lang_testdata.py), not a recognize.py run", openers=res)
    return out


def transcript_rows(stems: list[str]) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    t = pq.read_table(TR / "exp_tr_ens5grid_test_preds.parquet")
    t = t.filter(pc.is_in(t["replay"], value_set=__import__("pyarrow").array(stems)))
    out = {s: [] for s in stems}
    for r in t.to_pylist():
        out[r["replay"]].append(r)
    for s in out:
        out[s].sort(key=lambda r: r["start_frame"])
    return out


def pick(n: int) -> list[str]:
    """n test replays, half old half new, spread by name (deterministic)."""
    sp = json.loads((TR / "transcript_splits.json").read_text())
    folds = json.loads((OUT / "folds.json").read_text())
    test = sorted(s for s, v in sp.items() if v == "test" and (CROPS / s / "index.json").exists()
                  and (DATA / "runs" / s / "frames" / f"{s}_frames.parquet").exists())
    old = [s for s in test if s in folds]
    new = [s for s in test if s not in folds]
    k = (n + 1) // 2
    take = lambda xs, m: [xs[int(i)] for i in np.linspace(0, len(xs) - 1, m)] if m and xs else []
    return take(old, k) + take(new, n - k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--stems", nargs="*")
    ap.add_argument("--out", type=Path, default=LANG)
    a = ap.parse_args()
    stems = a.stems or pick(a.n)
    sp = json.loads((TR / "transcript_splits.json").read_text())
    bad = [s for s in stems if sp.get(s) != "test"]
    if bad:
        raise SystemExit(f"not test-fold replays: {bad}")
    docs, tr = chain_docs(stems), transcript_rows(stems)
    for s in stems:
        d = a.out / s
        d.mkdir(parents=True, exist_ok=True)
        (d / "openers.json").write_text(json.dumps(docs[s], indent=1))
        (d / "transcript_preds.json").write_text(json.dumps(tr[s], indent=0))
        shutil.copyfile(DATA / "runs" / s / "frames" / f"{s}_frames.parquet", d / "frames.parquet")
        print(s, len(docs[s]["openers"]), "openers,", len(tr[s]), "transcript events", docs[s]["characters"])


if __name__ == "__main__":
    main()
