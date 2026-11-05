"""Measurement pass, step 4: fixed 50/50 average of the fine-tuned V-JEPA 2 head with v2, from saved test probabilities
(exp_bb_ensemble.py wrote exp_bb_P_vjepa2_large_ft.npz), against v2 alone: chain and move, paired bootstrap over replays.

    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_ensemble.py [--tag vjepa2_large_ft]
-> model/results_exp_measure_ensemble_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbone_feats as bf  # noqa: E402
from common import DATA, OUT, replay_fold_map, save_json, test_subsets  # noqa: E402
from exp_measure_common import opener_counts, paired_bootstrap, ratio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vjepa2_large_ft")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    import torch
    voc = torch.load(OUT / "video_moves_test.pt", map_location="cpu")["voc"]
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    subs = test_subsets(fold)
    titems, _ = bf.test_setup(rows, fold)
    files = [it["file"] for it in titems]
    zb, zv = np.load(OUT / f"exp_bb_P_{a.tag}.npz"), np.load(OUT / "exp_bb_v2_P_video_moves_test.npz")
    assert list(zb["files"]) == files and list(zv["files"]) == files
    Pv = {h: zv[h] for h in ("category", "move")}
    Pb = {h: zb[h] for h in ("category", "move")}
    Pe = {h: 0.5 * Pb[h] + 0.5 * Pv[h] for h in Pv}
    cv, cb, ce = (opener_counts(titems, P, voc, rows, subs["all"]) for P in (Pv, Pb, Pe))
    out = dict(tag=a.tag, w=0.5)
    for sub in ("old", "new", "all"):
        st = subs[sub]
        out[sub] = dict(head_chain=round(ratio(cb, "chain", "n", st), 4), head_move=round(ratio(cb, "move", "det", st), 4),
                        chain=paired_bootstrap(cv, ce, "chain", "n", st), move=paired_bootstrap(cv, ce, "move", "det", st),
                        category=paired_bootstrap(cv, ce, "cat", "det", st))
    print(json.dumps(out, indent=1))
    save_json(out, OUT / f"results_exp_measure_ensemble_{a.tag}.json")


if __name__ == "__main__":
    main()
