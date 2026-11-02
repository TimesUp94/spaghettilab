"""Average a frozen-backbone head's test probabilities (backbone_feats train, model/video_moves_exp_bb_<tag>.pt)
with the v2 ensemble's (model/exp_bb_v2_P_video_moves_test.npz, from backbone_feats baseline), equal weights
(fixed in advance, not tuned on the test fold), scored with backbone_feats.score (the v2 eval's own path).

    .venv/Scripts/python.exe tools/ggst_logger/model/exp_bb_ensemble.py --tag dinov2_base
-> model/results_exp_ensv2_<tag>.json, model/exp_bb_P_<tag>.npz (the head's test probabilities)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbone_feats as bf  # noqa: E402
from common import DATA, OUT, replay_fold_map, save_json  # noqa: E402
import video_moves as vm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--w", type=float, default=0.5, help="weight of the backbone head (v2 gets 1 - w)")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    import torch
    ck = torch.load(OUT / f"video_moves_exp_bb_{a.tag}.pt", map_location="cpu")
    voc, name, kind = ck["voc"], ck["backbone"], ck["kind"]
    v2voc = torch.load(OUT / "video_moves_test.pt", map_location="cpu")["voc"]
    assert voc == v2voc
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    titems, vch = bf.test_setup(rows, fold)
    files = [it["file"] for it in titems]
    pcache = OUT / f"exp_bb_P_{a.tag}.npz"
    if pcache.exists():
        z = np.load(pcache)
        assert list(z["files"]) == files
        Pb = {h: z[h] for h in vm.HEADS}
    else:
        Tc, Ts = bf.load_feats(name, titems)
        TF = Tc if ck["crop_only"] else np.concatenate([Tc, Ts], -1)
        del Tc, Ts
        TM = bf.masks_for(voc, [vch[it["stem"]][it["side"] - 1] for it in titems])
        net = bf.make_head(TF.shape[-1], voc).to(bf.device())
        net.load_state_dict(ck["state"])
        Pb = bf.predict_head(net, TF, TM, kind)
        np.savez(pcache, files=np.array(files), **Pb)
    z = np.load(OUT / "exp_bb_v2_P_video_moves_test.npz")
    assert list(z["files"]) == files
    Pv = {h: z[h] for h in vm.HEADS}
    out = {}
    for label, P in (("head", Pb), ("ensemble", {h: a.w * Pb[h] + (1 - a.w) * Pv[h] for h in vm.HEADS})):
        res, ex = bf.score(titems, P, voc, vch, rows, fold)
        out[label] = bf.summary(res, ex)
        print(label, json.dumps({k: (v["chain"], v["move"], v["category"]) for k, v in out[label].items()}), flush=True)
    save_json(dict(tag=a.tag, w_head=a.w, summary=out), OUT / f"results_exp_ensv2_{a.tag}.json")


if __name__ == "__main__":
    main()
