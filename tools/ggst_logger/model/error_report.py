"""Where the video-only chain loses openers on the test fold: every logged opener (bursts apart) is
missed by the detector, found but dropped as 'none', found and kept with the wrong move, or right,
broken down by category, character, move, rare bucket, counter/punish, training count of the move,
and for the misses why (no HP drop logged, HUD hidden, no signal, low score, off timing, wrong side,
merged with a nearby opener).

Uses the chain's detections (detections.json, video_crops vcrops_test) and video_moves_test.pt; the
miss reasons use video_openers_test.pt's probabilities (the same split as the crossfit fold 0).

    .venv/Scripts/python.exe tools/ggst_logger/model/error_report.py
    .venv/Scripts/python.exe tools/ggst_logger/model/error_report.py --moves video_moves_exp_frac25.pt --out results_exp_errors_frac25.json

-> <data>/ggst_dataset/model/results_errors.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, TEST_FOLD, label_category, label_hit_kind, low_priority, replay_fold_map, save_json  # noqa: E402

STATUS = ("missed", "dropped_none", "wrong_move", "right")


def key(r):
    return f"{r['log_row']}_{r['attacker']}"


def move_preds(weights, root, by_key, test):
    import torch
    import video_moves as vm
    net, ck = vm.load(OUT / weights)
    voc = ck["voc"]
    items = vm.items_for(root, by_key, test)
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    vch = vm.video_chars(root, test, chars_list)
    return items, vm.predict(net, items, voc, vch), voc, vch


def detector_probs(test):
    import video_openers as vo
    net = vo.load(OUT / "video_openers_test.pt")
    out = {}
    for s in test:
        X, frames = vo.replay_inputs(s)
        out[s] = (vo.predict(net, X, vo.replay_tiny(s) if net.tiny is not None else None), frames, X)
    return out


def miss_reason(r, s, det_list, probs, rows_here):
    """Why a logged opener has no matched detection."""
    import video_openers as vo
    v = r["first_damage_video_frame"]
    if v is None:
        return "no_hp_drop_logged", None
    p, frames, X = probs[s]
    i = int(np.argmin(np.abs(frames - v)))
    side = int(r["attacker"][1])
    dfd = 3 - side
    lo, hi = max(0, i - vo.MATCH), i + vo.MATCH + 1
    pmax = float(p[lo:hi, side].max())
    # HUD hidden (super flash, cinematic): defender HP unread or no timer around the hit
    ok = X[:, (dfd - 1) * vo.NF + 1]
    tim = X[:, 2 * vo.NF]
    hidden = bool((ok[i:i + 4] < 0.5).any() or (tim[i:i + 4] < 0.5).any())
    near = [d for d in det_list if abs(d["frame"] - v) <= vo.MATCH * vo.STEP]
    if any(d["row"] == key(r) for d in det_list):
        why = "crop_missing"    # detected, but no crop in vcrops
    elif any(d["side"] == side and d["row"] for d in near):
        why = "merged"          # the detection there matched another opener
    elif any(d["side"] != side for d in near):
        why = "wrong_side"
    elif any(d["side"] == side and abs(d["frame"] - v) <= 16 * vo.STEP for d in det_list):
        why = "off_timing"
    elif pmax >= 0.1:
        why = "low_score"
    else:
        why = "no_signal"
    others = [o for o in rows_here if o is not r and o["first_damage_video_frame"] is not None
              and abs(o["first_damage_video_frame"] - v) <= 2 * vo.MATCH * vo.STEP]
    return why, dict(hud_hidden=hidden, pmax=round(pmax, 3), opener_nearby=bool(others))


def share(c, n):
    return {k: dict(n=v, share=round(v / n, 4)) for k, v in c.most_common()}


def table(rows, status, keyf, min_n=1):
    t = collections.defaultdict(collections.Counter)
    for r in rows:
        t[keyf(r)][status[key_full(r)]] += 1
    out = {}
    for k, c in sorted(t.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        if n < min_n:
            continue
        out[str(k)] = dict(n=n, **{st: round(c[st] / n, 3) for st in STATUS}, lost=n - c["right"])
    return out


def key_full(r):
    return f"{r['replay']}/{key(r)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default="video_moves_test.pt")
    ap.add_argument("--out", default="results_errors.json")
    a = ap.parse_args()
    os.environ.setdefault("VM_WORKERS", "4")
    low_priority()
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by_key = {key_full(r): r for r in rows}
    fold = replay_fold_map()
    test = sorted(s for s in fold if fold[s] == TEST_FOLD)
    tset = set(test)
    train_rows = [r for r in rows if r["replay"] in fold and fold[r["replay"]] != TEST_FOLD]
    root = DATA / "vcrops_test"
    items, P, voc, vch = move_preds(a.moves, root, by_key, test)
    cat_none = voc["category"].index("none")
    dets = json.loads((OUT / "detections.json").read_text())
    probs = detector_probs(test)

    tru = [r for r in rows if r["replay"] in tset and not r["is_burst"]]
    n = len(tru)
    by_replay = collections.defaultdict(list)
    for r in tru:
        by_replay[r["replay"]].append(r)
    item_of = {key_full(it["row"]): i for i, it in enumerate(items) if it["row"] is not None}
    train_moves = collections.Counter(f"{r['char']}:{r['move']}" for r in train_rows if not r["is_burst"])

    status, info = {}, {}
    for r in tru:
        k = key_full(r)
        lab = f"{r['char']}:{r['move']}"
        rare = lab not in voc["move"]
        if rare:
            lab = f"{r['char']}:(rare)"
        d = dict(label=lab, rare=rare, train_count=train_moves[f"{r['char']}:{r['move']}"])
        i = item_of.get(k)
        if i is None:
            st = "missed"
            d["why"], extra = miss_reason(r, r["replay"], dets.get(r["replay"], []), probs, by_replay[r["replay"]])
            d.update(extra or {})
        else:
            pm = voc["move"][int(P["move"][i].argmax())]
            pc = P["category"][i].copy()
            d["pred_move"] = pm
            d["video_char_right"] = vch[r["replay"]][int(r["attacker"][1]) - 1] == r["char"]
            pcn = pc.copy()
            pcn[cat_none] = 0
            d["pred_cat"] = voc["category"][int(pcn.argmax())]
            if pc.argmax() == cat_none:
                st = "dropped_none"
                d["p_none"] = float(pc[cat_none])
            else:
                st = "right" if pm == lab else "wrong_move"
        status[k], info[k] = st, d

    res = dict(openers=n, moves_weights=a.moves, status=share(collections.Counter(status.values()), n))
    cat = lambda r: label_category(r)
    res["by_category"] = table(tru, status, cat)
    res["by_raw_category"] = table(tru, status, lambda r: r["category"])
    res["by_char"] = table(tru, status, lambda r: r["char"])
    res["by_hit_kind"] = table(tru, status, lambda r: label_hit_kind(r) or "unknown")
    res["by_rare"] = table(tru, status, lambda r: "rare bucket" if info[key_full(r)]["rare"] else "own class")

    def bucket(r):
        c = info[key_full(r)]["train_count"]
        for lo, name in ((200, "200+"), (100, "100-199"), (50, "50-99"), (20, "20-49"), (5, "5-19")):
            if c >= lo:
                return name
        return "<5 (rare)"
    res["by_move_train_count"] = table(tru, status, bucket)
    mv = table(tru, status, lambda r: f"{r['char']}:{r['move']} ({label_category(r)})")
    res["top_lost_moves"] = dict(sorted(mv.items(), key=lambda kv: -kv[1]["lost"])[:40])
    res["worst_moves_min20"] = dict(sorted(((k, v) for k, v in mv.items() if v["n"] >= 20),
                                          key=lambda kv: kv[1]["right"])[:25])

    miss = [r for r in tru if status[key_full(r)] == "missed"]
    res["missed"] = dict(
        n=len(miss),
        why=share(collections.Counter(info[key_full(r)]["why"] for r in miss), len(miss)),
        by_category=share(collections.Counter(cat(r) for r in miss), len(miss)),
        hud_hidden=share(collections.Counter(str(info[key_full(r)].get("hud_hidden")) for r in miss), len(miss)),
        miss_rate_hud_hidden_vs_not={},
        why_by_category={c: dict(collections.Counter(info[key_full(r)]["why"] for r in miss if cat(r) == c))
                         for c in sorted({cat(r) for r in miss})},
        opener_nearby=sum(bool(info[key_full(r)].get("opener_nearby")) for r in miss),
    )
    # how often the HUD is hidden at found openers too, for the miss rate split
    hid = collections.Counter()
    for r in tru:
        if r["first_damage_video_frame"] is None:
            continue
        p, frames, X = probs[r["replay"]]
        i = int(np.argmin(np.abs(frames - r["first_damage_video_frame"])))
        dfd = 2 if r["attacker"] == "P1" else 1
        import video_openers as vo
        h = bool((X[i:i + 4, (dfd - 1) * vo.NF + 1] < 0.5).any() or (X[i:i + 4, 2 * vo.NF] < 0.5).any())
        hid[(h, cat(r), status[key_full(r)] == "missed")] += 1
    for h in (True, False):
        tot = sum(v for (hh, _, _), v in hid.items() if hh == h)
        m = sum(v for (hh, _, mm), v in hid.items() if hh == h and mm)
        res["missed"]["miss_rate_hud_hidden_vs_not"]["hidden" if h else "visible"] = dict(openers=tot, missed=m, rate=round(m / max(1, tot), 3))
    res["missed"]["super_hud"] = {("hidden" if h else "visible"): dict(openers=sum(v for (hh, c, _), v in hid.items() if hh == h and c == "super"),
                                                                       missed=sum(v for (hh, c, mm), v in hid.items() if hh == h and c == "super" and mm))
                                  for h in (True, False)}

    drop = [r for r in tru if status[key_full(r)] == "dropped_none"]
    res["dropped_none"] = dict(n=len(drop), by_category=share(collections.Counter(cat(r) for r in drop), max(1, len(drop))),
                               by_char=share(collections.Counter(r["char"] for r in drop), max(1, len(drop))))
    wrong = [r for r in tru if status[key_full(r)] == "wrong_move"]
    conf = collections.Counter((info[key_full(r)]["label"], info[key_full(r)]["pred_move"]) for r in wrong)
    res["wrong_move"] = dict(
        n=len(wrong),
        video_char_wrong=sum(not info[key_full(r)]["video_char_right"] for r in wrong),
        cross_category=sum(info[key_full(r)]["pred_cat"] != cat(r) for r in wrong),
        true_rare=sum(info[key_full(r)]["rare"] for r in wrong),
        pred_rare=sum(info[key_full(r)]["pred_move"].endswith("(rare)") for r in wrong),
        by_category=share(collections.Counter(cat(r) for r in wrong), max(1, len(wrong))),
        by_char=share(collections.Counter(r["char"] for r in wrong), max(1, len(wrong))),
        top_confusions=[[c, t, p] for (t, p), c in conf.most_common(40)],
    )
    right = [r for r in tru if status[key_full(r)] == "right"]
    res["right_via_rare_bucket"] = sum(info[key_full(r)]["rare"] for r in right)
    res["chain_move_right_of_all_openers"] = len(right) / n
    save_json(res, OUT / a.out)
    print(json.dumps({k: res[k] for k in ("openers", "status", "chain_move_right_of_all_openers", "right_via_rare_bucket")}, indent=1))
    print("missed:", json.dumps({k: res["missed"][k] for k in ("why", "by_category", "miss_rate_hud_hidden_vs_not", "super_hud")}))
    print("wrong:", {k: res["wrong_move"][k] for k in ("n", "video_char_wrong", "cross_category", "true_rare", "pred_rare")})
    print("by category:", json.dumps(res["by_category"]))


if __name__ == "__main__":
    main()
