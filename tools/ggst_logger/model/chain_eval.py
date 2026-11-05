"""The whole chain on the test fold with another opener detector: its detections are cut into crops
and whole frames (only those the chain's crops don't already have, by frame and side), then
video_moves_test.pt runs end to end on them, as in video_moves.py train.

    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --name tiny              # detections_exp_tiny.json (video_openers.py train --name tiny)
    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --name tiny --degraded   # video_openers_exp_tiny.pt on the 720p30 copies
    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --rebuild --base vcrops_v1   # the chain's own vcrops_test /
        vscene_test for every replay at detections.json (new crossfit detections), reusing vcrops_v1's tiles

    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --mid-combo   # the chain's cached test probabilities, with and without the rule
        that drops detections inside a combo (video_openers.continues_combo) and with the HUD's characters -> model/results_mid_combo.json

    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --trades cut      # crops and whole frames of the trades' second detections
    .venv/Scripts/python.exe tools/ggst_logger/model/chain_eval.py --trades score    # before / after with paired bootstraps -> model/results_trades.json

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
    bidx = base / stem / "index.json"
    old = json.loads(bidx.read_text()) if bidx.exists() else None
    have = {(d["frame"], d["side"]): d for d in old["dets"]} if old else {}
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
        if old is not None:
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
    if old is None:                       # nothing to reuse: the new cut's index (its character vote)
        old = json.loads((tmp / "index.json").read_text()) if new else {"win": vc.WIN, "chars_logp": None, "frames_seen": 0}
        shutil.rmtree(tmp, ignore_errors=True)
    idx = dict(old, dets=entries)
    (cdir / "index.json").write_text(json.dumps(idx))
    return stem, len(new)


def trade_dets(stem, dets, rows_here, root, fps):
    """A replay's trades' second detections as recognize.py finds them, each with its logged opener (row) or null."""
    import pyarrow.parquet as pq
    import video_openers as vo
    t = pq.read_table(root / "runs" / stem / "frames" / f"{stem}_frames.parquet", columns=["frame_number", "combo_side", "combo_count"] + vo.BAR_COLUMNS).to_pydict()
    fr = np.asarray(t["frame_number"]) * (60 // fps)
    shown = {s: vo.counter_readings(t["combo_side"], t["combo_count"], s) for s in (1, 2)}
    hp, pend = vo.hud_bars(t)
    step = lambda f: min(int(np.searchsorted(fr, f)), len(fr) - 1)
    all_ = [(step(d["frame"]), d["side"], d["score"]) for d in dets]
    kept = [x for x in all_ if not vo.continues_combo(shown[x[1]], x[0])]
    taken = {d["row"] for d in dets if d["row"]}
    tru = [r for r in rows_here if not r["is_burst"] and r["first_damage_video_frame"] is not None]
    out = []
    for i, s, sc in vo.trade_partners(kept, all_, hp, pend, shown):
        f = int(fr[i])
        cand = [r for r in tru if int(r["attacker"][1]) == s and abs(r["first_damage_video_frame"] - f) <= vo.MATCH * vo.STEP
                and f"{r['log_row']}_{r['attacker']}" not in taken]
        row = None
        if cand:
            r = min(cand, key=lambda r: abs(r["first_damage_video_frame"] - f))
            row = f"{r['log_row']}_{r['attacker']}"
            taken.add(row)
        out.append(dict(frame=f, side=s, score=sc, row=row))
    return out


def _cut_trades(args):
    import video_crops as vc
    import video_moves as vm
    import video_scene as vs
    stem, video, dets, out, players = args
    vc.process((stem, video, dets, str(out / stem), players))
    vs.process((stem, video, str(out / stem), str(vm.scene_root(out) / stem)))
    return stem, len(dets)


def trades_sets():
    """(name, replays, HUD root, detections, video per replay, fps of the HUD's readings, HUD-character cache, the chain's crops, the trades' crops)."""
    import pyarrow.parquet as pq
    from common import test_subsets
    rows = pq.read_table(DATA / "openers.parquet", columns=["replay", "video"]).to_pylist()
    video = {r["replay"]: r["video"] for r in rows}
    subs = test_subsets(replay_fold_map())
    deg = DATA / "degraded"
    return [("normal", subs["all"], DATA, json.loads((OUT / "detections.json").read_text()), video, 60, DATA / "hud_chars" / "eval" / "1080", DATA / "vcrops_test", DATA / "vcrops_trade"),
            ("720p30", subs["old"], deg, json.loads((deg / "detections.json").read_text()), {s: str(deg / f"{s}.mp4") for s in subs["old"]}, 30,
             DATA / "hud_chars" / "eval" / "720", deg / "vcrops", deg / "vcrops_trade")]


def trades_cut(jobs):
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["replay"]].append(r)
    players = str(OUT / "video_players_test.pt")
    for name, stems, root, dets, video, fps, _, _, out in trades_sets():
        tasks = []
        for s in stems:
            added = trade_dets(s, dets.get(s, []), by[s], root, fps)
            if added and not (out / s / "index.json").exists():
                tasks.append((s, video[s], added, out, players))
        print(f"{name}: {len(tasks)} replays to cut, {sum(len(t[2]) for t in tasks)} detections", flush=True)
        with Pool(jobs) as p:
            for k, (s, n) in enumerate(p.imap_unordered(_cut_trades, tasks)):
                if k % 10 == 0:
                    print(f"  {k + 1}/{len(tasks)} {s} {n}", flush=True)


def per_replay(items, P, voc, tru_by_stem):
    """{stem: [openers, right, kept logged, false kept]} of the whole chain, counted as video_moves.end_to_end does."""
    cat_none = voc["category"].index("none")
    pm, pc = P["move"].argmax(1), P["category"].argmax(1)
    out = {s: [len(t), 0, 0, 0] for s, t in tru_by_stem.items()}
    for i, it in enumerate(items):
        kept = pc[i] != cat_none
        c = out[it["stem"]]
        r = it["row"]
        if r is None:
            c[3] += bool(kept)
            continue
        if r["is_burst"]:
            continue
        lab = f"{r['char']}:{r['move']}"
        lab = lab if lab in voc["move"] else f"{r['char']}:(rare)"
        c[2] += bool(kept)
        c[1] += bool(kept and lab == voc["move"][pm[i]])
    return out


def paired(a, b, stems, n=4000, seed=0):
    """Chain and precision, before (a) and after (b), with a paired bootstrap over replays."""
    A, B = np.array([a[s] for s in stems], float), np.array([b[s] for s in stems], float)
    chain = lambda X: X[..., 1].sum(-1) / X[..., 0].sum(-1)
    prec = lambda X: X[..., 2].sum(-1) / (X[..., 2] + X[..., 3]).sum(-1)
    idx = np.random.default_rng(seed).integers(0, len(stems), (n, len(stems)))
    res = dict(replays=len(stems), openers=int(A[:, 0].sum()))
    for name, f in (("chain", chain), ("precision", prec)):
        d = f(B[idx]) - f(A[idx])
        res[name] = dict(before=round(float(f(A)), 4), after=round(float(f(B)), 4), delta=round(float(f(B) - f(A)), 4),
                         ci95=[round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])], share_above_zero=round(float((d > 0).mean()), 4))
    return res


def trades_score():
    """The chain with the mid-combo rule and the HUD's characters, without and with the trades' second detections."""
    import pyarrow.parquet as pq
    import torch
    import hud_chars as hc
    import video_moves as vm
    from common import test_subsets
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["replay"]].append(r)
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    subs = test_subsets(replay_fold_map())
    net, ck = vm.load(OUT / "video_moves_test.pt")
    voc = ck["voc"]
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    refs = hc.Refs(hc.refs_path(True))
    res = {}
    for name, stems, root, dets, _, fps, hud, crops, out in trades_sets():
        items = vm.items_for(crops, by_key, stems)
        vis = vm.video_chars(crops, stems, chars_list)
        chars = dict(vis)
        for s in stems:
            f = hud / f"{s}.npz"
            if f.exists():
                g = max(hc.resolve(dict(np.load(f)), refs.chars, lambda a, b, s=s: vis[s]), key=lambda g: g["frames"])
                chars[s] = (g["P1"]["char"], g["P2"]["char"])
        cache = OUT / f"exp_trades_P_{name}.npz"
        if cache.exists():
            z = np.load(cache)
            P = {h: z[h] for h in z.files}
        elif name == "normal":      # the chain's cached probabilities; only the replays whose characters change run again, as in mid_combo_eval
            z = np.load(OUT / "exp_bb_v2_P_video_moves_test.npz")
            pos = {str(f): i for i, f in enumerate(z["files"])}
            order = np.array([pos[it["file"]] if it["file"] in pos else pos[Path(it["file"]).parent.name + "/" + Path(it["file"]).name] for it in items])
            P = {h: z[h][order] for h in vm.HEADS}
            changed = {s for s in chars if chars[s] != vis.get(s)}
            ix = [i for i, it in enumerate(items) if it["stem"] in changed]
            if ix:
                Pn = vm.predict(net, [items[i] for i in ix], voc, chars)
                for h in P:
                    P[h][ix] = Pn[h]
            np.savez_compressed(cache, **P)
        else:
            P = vm.predict(net, items, voc, chars)
            np.savez_compressed(cache, **P)
        at = collections.defaultdict(list)
        for i, it in enumerate(items):
            at[it["stem"]].append(i)
        drop = np.zeros(len(items), bool)
        for s, ix in at.items():
            drop[ix] = rule_flags(s, [(items[i]["frame"], items[i]["side"]) for i in ix], root, fps)
        keep = np.nonzero(~drop)[0]
        base_items = [items[i] for i in keep]
        Pb = {h: P[h][keep] for h in P}
        new = vm.items_for(out, by_key, stems)
        Pn = vm.predict(net, new, voc, chars)
        tru = {s: [r for r in by[s] if not r["is_burst"]] for s in stems}
        before = per_replay(base_items, Pb, voc, tru)
        after = per_replay(base_items + new, {h: np.concatenate([Pb[h], Pn[h]]) for h in Pb}, voc, tru)
        kept_new = Pn["category"].argmax(1) != voc["category"].index("none")
        n_log = sum(it["row"] is not None for it in new)
        r = dict(added=len(new), added_logged=n_log, added_false=len(new) - n_log, added_precision=round(n_log / max(1, len(new)), 4),
                 added_kept=int(kept_new.sum()), added_kept_logged=int(sum(k and it["row"] is not None for k, it in zip(kept_new, new))),
                 added_right=int(sum(after[s][1] - before[s][1] for s in stems)),
                 added_by_side={str(k): sum(it["side"] == k for it in new) for k in (1, 2)})
        for sub, ss in subs.items():
            ss = [s for s in ss if s in set(stems)]
            if ss:
                r[sub] = paired(before, after, ss)
        res[name] = r
        print(name, json.dumps(r), flush=True)
    save_json(res, OUT / "results_trades.json")


DEDUP_FRAMES = 20


def rule_flags(stem, frames_sides, root=DATA, fps=60):
    """For a replay's (60 fps frame, side) detections: does video_openers.continues_combo drop them? Uses its pipeline parquet under root,
    whose frame numbers are the video's own (fps)."""
    import pyarrow.parquet as pq
    import video_openers as vo
    t = pq.read_table(root / "runs" / stem / "frames" / f"{stem}_frames.parquet", columns=["frame_number", "combo_side", "combo_count"]).to_pydict()
    fr = np.asarray(t["frame_number"])
    shown = {s: vo.counter_readings(t["combo_side"], t["combo_count"], s) for s in (1, 2)}
    return [vo.continues_combo(shown[s], min(int(np.searchsorted(fr, f * fps // 60)), len(fr) - 1)) for f, s in frames_sides]


def mid_combo_eval():
    """The whole chain on the test fold from cached probabilities: no retraining and no crops recut. Baseline; with the rule; with the HUD's characters too
    (the replays whose visual vote was wrong are predicted again). Also the rule alone on the training folds' crossfit detections and the 720p30 copies."""
    import pyarrow.parquet as pq
    import torch
    import hud_chars as hc
    import video_moves as vm
    from common import test_subsets
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    subs = test_subsets(fold)
    by_key = {f"{r['replay']}/{r['log_row']}_{r['attacker']}": r for r in rows}
    root = DATA / "vcrops_test"
    items = vm.items_for(root, by_key, subs["all"])
    net, ck = vm.load(OUT / "video_moves_test.pt")
    voc = ck["voc"]
    z = np.load(OUT / "exp_bb_v2_P_video_moves_test.npz")
    pos = {str(f): i for i, f in enumerate(z["files"])}
    order = np.array([pos[it["file"]] if it["file"] in pos else pos[Path(it["file"]).parent.name + "/" + Path(it["file"]).name] for it in items])
    P = {h: z[h][order] for h in ("category", "normal", "hit_kind", "move")}
    chars_list = torch.load(OUT / "video_players_test.pt", map_location="cpu")["chars"]
    vch = vm.video_chars(root, subs["all"], chars_list)
    # the HUD's characters on the same replays
    refs = hc.Refs(hc.refs_path(True))
    hch = dict(vch)
    for s in subs["all"]:
        f = DATA / "hud_chars" / "eval" / "1080" / f"{s}.npz"
        if f.exists():
            segs = hc.resolve(dict(np.load(f)), refs.chars, lambda a, b, s=s: vch[s])
            g = max(segs, key=lambda g: g["frames"])
            hch[s] = (g["P1"]["char"], g["P2"]["char"])
    changed = sorted(s for s in hch if hch[s] != vch.get(s))
    print("replays whose characters change:", [(s, vch[s], hch[s]) for s in changed])
    Phud = {h: v.copy() for h, v in P.items()}
    if changed:
        idx = [i for i, it in enumerate(items) if it["stem"] in set(changed)]
        Pn = vm.predict(net, [items[i] for i in idx], voc, hch)
        for h in Phud:
            Phud[h][idx] = Pn[h]
    # the rule
    drop = np.zeros(len(items), bool)
    per = {}
    for i, it in enumerate(items):
        per.setdefault(it["stem"], []).append(i)
    for s, ix in per.items():
        drop[ix] = rule_flags(s, [(items[i]["frame"], items[i]["side"]) for i in ix])
    is_false = np.array([it["row"] is None for it in items])
    zspin = np.array([it["row"] is not None and it["row"]["defender_act"] == "CmnActZSpin" for it in items])
    cat_none = voc["category"].index("none")
    kept_head = P["category"].argmax(1) != cat_none
    res = dict(test_drop=dict(dropped=int(drop.sum()), false_detections=int((drop & is_false).sum()), logged_openers=int((drop & ~is_false).sum()),
                              of_which_zspin_label=int((drop & zspin).sum()), false_kept_by_none_head_before=int((is_false & kept_head).sum()),
                              false_kept_removed=int((drop & is_false & kept_head).sum())))
    res["test_dropped_logged"] = [[items[i]["stem"], items[i]["frame"], items[i]["side"], items[i]["row"]["log_row"], items[i]["row"]["defender_act"], items[i]["row"]["move"]]
                                  for i in np.nonzero(drop & ~is_false)[0]]
    keep = np.nonzero(~drop)[0]
    # a further candidate: of two detections of one side under DEDUP_FRAMES apart, only the stronger
    dedup = np.zeros(len(items), bool)
    for s, ix in per.items():
        for side in (1, 2):
            mine = sorted((i for i in ix if items[i]["side"] == side), key=lambda i: items[i]["frame"])
            for a_, b_ in zip(mine, mine[1:]):
                if items[b_]["frame"] - items[a_]["frame"] <= DEDUP_FRAMES:
                    dedup[a_ if items[a_]["score"] < items[b_]["score"] else b_] = True
    res["test_dedup"] = dict(dropped=int(dedup.sum()), false_detections=int((dedup & is_false).sum()), logged_openers=int((dedup & ~is_false).sum()),
                             false_kept_removed=int((dedup & is_false & kept_head).sum()))
    keep2 = np.nonzero(~drop & ~dedup)[0]
    rows_all = {}
    for name, PP, ch, ii in (("baseline", P, vch, np.arange(len(items))), ("rule", P, vch, keep), ("hud_chars", Phud, hch, np.arange(len(items))),
                             ("rule+hud_chars", Phud, hch, keep), ("rule+dedup+hud_chars", Phud, hch, keep2)):
        rows_all[name] = {}
        for sub, stems in subs.items():
            st = set(stems)
            jj = [i for i in ii if items[i]["stem"] in st]
            r = vm.end_to_end([items[i] for i in jj], {h: v[jj] for h, v in PP.items()}, voc, ch, rows, st)
            rows_all[name][sub] = {k: r[k] for k in ("openers", "detected", "detected_and_kept", "false_detections", "false_detections_kept", "precision_after_none",
                                                     "video_char_right", "move_acc", "chain_move_right_of_all_openers")}
            # the same with the log's mid-combo labels out of the truth: a hit on a spinning defender (CmnActZSpin, not in
            # build.py's damage states) is logged as an opener, though the game's counter goes on (see README)
            tr = [x for x in rows if x["defender_act"] != "CmnActZSpin"]
            it2 = [dict(items[i], row=None) if items[i]["row"] is not None and items[i]["row"]["defender_act"] == "CmnActZSpin" else items[i] for i in jj]
            r2 = vm.end_to_end(it2, {h: v[jj] for h, v in PP.items()}, voc, ch, tr, st)
            rows_all[name][sub]["clean_truth"] = {k: r2[k] for k in ("openers", "false_detections_kept", "precision_after_none", "chain_move_right_of_all_openers")}
    res["chain"] = rows_all
    # the training folds: the crossfit detections
    dets = json.loads((OUT / "detections.json").read_text())
    tally = collections.Counter()
    for s, dd in dets.items():
        if s not in fold or fold[s] == TEST_FOLD or not (DATA / "runs" / s / "frames" / f"{s}_frames.parquet").exists():
            continue
        fl = rule_flags(s, [(d["frame"], d["side"]) for d in dd])
        for d, x in zip(dd, fl):
            r = by_key.get(f"{s}/{d['row']}") if d["row"] else None
            c = "false" if r is None else ("zspin" if r["defender_act"] == "CmnActZSpin" else "opener")
            tally[c] += 1
            tally[c + "_dropped"] += bool(x)
    res["train_folds"] = dict(tally)
    # the 720p30 copies (their own parquets, their own detections)
    ddet = json.loads((DATA / "degraded" / "detections.json").read_text())
    tally = collections.Counter()
    for s, dd in ddet.items():
        fl = rule_flags(s, [(d["frame"], d["side"]) for d in dd], DATA / "degraded", fps=30)
        for d, x in zip(dd, fl):
            r = by_key.get(f"{s}/{d['row']}") if d["row"] else None
            c = "false" if r is None else ("zspin" if r["defender_act"] == "CmnActZSpin" else "opener")
            tally[c] += 1
            tally[c + "_dropped"] += bool(x)
    res["degraded_old"] = dict(tally)
    save_json(res, OUT / "results_mid_combo.json")
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid-combo", action="store_true")
    ap.add_argument("--trades", choices=["cut", "score"], help="the trades' second detections (video_openers.trade_partners)")
    ap.add_argument("--name", help="detector experiment (video_openers_exp_<name>.pt / detections_exp_<name>.json)")
    ap.add_argument("--rebuild", action="store_true",
                    help="cut the chain's vcrops_test / vscene_test for every replay at detections.json (from --base's tiles where the same)")
    ap.add_argument("--degraded", action="store_true", help="the 720p30 copies of the test replays")
    ap.add_argument("--moves", default="video_moves_test.pt")
    ap.add_argument("--base", help="crops dir to reuse tiles from (default: the chain's, vcrops_test or degraded/vcrops)")
    ap.add_argument("--reverse", action="store_true", help="--rebuild: last replay first, skipping any already started")
    ap.add_argument("--jobs", type=int, default=3, help="processes cutting new crops (each loads the player net)")
    a = ap.parse_args()
    low_priority()
    if a.mid_combo:
        return mid_combo_eval()
    if a.trades:
        return trades_cut(a.jobs) if a.trades == "cut" else trades_score()
    import pyarrow.parquet as pq
    import torch
    import video_moves as vm
    rows = pq.read_table(DATA / "openers.parquet").to_pylist()
    fold = replay_fold_map()
    test = sorted(s for s in fold if fold[s] == TEST_FOLD)
    video = {r["replay"]: r["video"] for r in rows}
    if a.rebuild:
        dets = json.loads((OUT / "detections.json").read_text())
        tasks = [(s, dets[s], DATA / a.base, DATA / "vcrops_test", video[s], str(OUT / "video_players_test.pt"))
                 for s in sorted(dets, reverse=a.reverse) if s in video]
        if a.reverse:     # a second runner from the other end: only replays nobody started
            tasks = [t for t in tasks if not (DATA / "vcrops_test" / t[0]).exists()]
        n_new = 0
        with Pool(a.jobs) as p:
            for k, (s, n) in enumerate(p.imap_unordered(build, tasks)):
                n_new += n
                print(f"  {k + 1}/{len(tasks)} {s}: {n} new detections cut ({n_new} in all)", flush=True)
        return
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
