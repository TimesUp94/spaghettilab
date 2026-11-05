"""Measurement pass, step 2: the match transcript (ens5grid, unchanged) on the 720p30 3 Mbit/s copies of the old test
replays (degraded/<stem>.mp4), scored with transcript_eval against the same replays at 1080p60.

Features are those of transcript_feats + transcript_feats_grid, but one combined pass: the grid tokens of the player squares
come from the same encoder output as their mean tokens, so the second pass isn't needed. On these copies:
- HUD: their own pipeline parquet (degraded/runs). Its frame_number counts 30 fps frames, so row j is hud row j on the
  30 Hz timeline (transcript_feats.hud's fr // 2 assumes 60 fps frame numbers).
- Player positions and the character vote come from the same player net on the copy; the decode goes through fps=60 like any
  non-60 fps video, then every 4th frame.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_transcript.py check                 # combined pass = the two passes, on 1080p
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_transcript.py extract [--max N]     # resumable, random order
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_transcript.py score                 # whichever replays are done
    .venv/Scripts/python.exe tools/ggst_logger/model/exp_measure_transcript.py score --hud-half      # the HUD timeline lab_analyze.transcript_features gives a 30 fps video
-> transcript/degraded/{feats,feats_grid}/, transcript/degraded/ens5grid_preds[_hudhalf].parquet, model/results_exp_measure_transcript_720p30[_hudhalf].json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, low_priority, replay_fold_map, save_json, test_subsets  # noqa: E402

DEG = DATA / "degraded"
TD = DATA / "transcript" / "degraded"
PCA = DATA / "transcript" / "feats_grid" / "pca.npz"


def old_tasks(seed=0):
    stems = test_subsets(replay_fold_map())["old"]
    order = np.random.default_rng(seed).permutation(sorted(stems))
    return [(str(s), str(DEG / f"{s}.mp4")) for s in order]


def combined_encoder(tf, tfg, pca, players_net):
    import torch

    class Combined(tfg.GridEncoder):
        """tf.Encoder.chunk, with the quadrant tokens of the two player views taken from the same encoder output."""

        def vjepa(self, views):
            n = len(views[0])
            pad = (-n) % tf.CLIP
            clips = []
            for v in views:
                v = (v - self.mean) / self.std
                if pad:
                    v = torch.cat([v, v[-1:].expand(pad, -1, -1, -1)])
                clips.append(v.view(-1, tf.CLIP, 3, tf.TILE, tf.TILE))
            x = torch.cat(clips).to(self.dt)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                h = self.m(pixel_values_videos=x, skip_predictor=True).last_hidden_state
            P = tf.TILE // 16
            g = h.view(len(x), tf.CLIP // 2, tfg.G, P // tfg.G, tfg.G, P // tfg.G, tf.DIM).float().mean((3, 5))
            g = g.reshape(len(views), -1, tfg.G * tfg.G, tf.DIM)[1:]
            g = (g - self.pca[0]) @ self.pca[1].T
            self.last_grid = [z[:(n + 1) // 2].half().cpu().numpy() for z in g]
            m = h.view(len(x), tf.CLIP // 2, -1, tf.DIM).float().mean(2).reshape(len(views), -1, tf.DIM)
            return [z[:(n + 1) // 2].half().cpu().numpy() for z in m]

        def chunk(self, frames, st):
            super().chunk(frames, st)
            if len(frames):
                st.setdefault("g1", []).append(self.last_grid[0])
                st.setdefault("g2", []).append(self.last_grid[1])
    return Combined(players_net, pca)


def redirect(tf, tfg, root, hud30=True):
    """The modules' dataset paths -> root; writes both files per replay."""
    tf.TR, tf.FEATS, tfg.GRID = root, root / "feats", root / "feats_grid"
    tf.FEATS.mkdir(parents=True, exist_ok=True)
    tfg.GRID.mkdir(parents=True, exist_ok=True)
    write0 = tf.write

    def write(stem, st, enc):
        r = write0(stem, st, enc)
        tmp = tfg.GRID / f"{stem}.tmp.npz"
        np.savez(tmp, p1g=np.concatenate(st["g1"]), p2g=np.concatenate(st["g2"]))
        os.replace(tmp, tfg.GRID / f"{stem}.npz")
        return r[0], r[1], r[2] + (tfg.GRID / f"{stem}.npz").stat().st_size
    tf.write = write
    if hud30:
        import video_openers as vo

        def hud(stem, n_frames):
            X, fr = vo.parquet_inputs(DEG / "runs" / stem / "frames" / f"{stem}_frames.parquet")
            return hud_rows(X, np.asarray(fr), n_frames)
        tf.hud = hud


def hud_rows(X, idx, n_frames):
    """The 30 Hz HUD rows from parquet rows at index idx on that timeline, forward-filled."""
    n30 = (n_frames + 1) // 2
    out = np.zeros((n30, X.shape[1]), np.float32)
    ok = np.zeros(n30, bool)
    keep = (idx >= 0) & (idx < n30)
    out[idx[keep]] = X[keep]
    ok[idx[keep]] = True
    last = np.maximum.accumulate(np.where(ok, np.arange(n30), -1))
    out = np.where((last >= 0)[:, None], out[np.maximum(last, 0)], 0)
    return out.astype(np.float16), ok


def cmd_extract(a):
    low_priority()
    import transcript_feats as tf
    import transcript_feats_grid as tfg
    redirect(tf, tfg, TD)
    pca = dict(np.load(PCA))
    players = str(OUT / "video_players_test.pt")
    tf.Encoder = lambda net: combined_encoder(tf, tfg, pca, net)
    tasks = [t for t in old_tasks() if not (tf.FEATS / f"{t[0]}.npz").exists()]
    if a.max:
        tasks = tasks[:a.max]
    print(f"{len(tasks)} replays to extract", flush=True)
    r = tf.run(tasks, players, workers=a.workers, threads=a.threads)
    r.pop("stats")
    print(json.dumps(r), flush=True)


def cmd_check(a):
    """Combined pass against the dataset's two passes, on the first 256 sampled frames of two test replays at 1080p60."""
    import shutil
    import transcript_feats as tf
    import transcript_feats_grid as tfg
    orig_feats, orig_grid = tf.FEATS, tfg.GRID
    tmp = DATA / "measure" / "combined_check"
    vid = dict(tf.replays())
    redirect(tf, tfg, tmp, hud30=False)
    pca = dict(np.load(PCA))
    tf.Encoder = lambda net: combined_encoder(tf, tfg, pca, net)
    stems = test_subsets(replay_fold_map())["old"][:2]
    tf.run([(s, vid[s]) for s in stems], str(OUT / "video_players_test.pt"), workers=0, threads=4, limit=256)
    for s in stems:
        a1, b1 = np.load(tf.FEATS / f"{s}.npz"), np.load(orig_feats / f"{s}.npz")
        a2, b2 = np.load(tfg.GRID / f"{s}.npz"), np.load(orig_grid / f"{s}.npz")
        for name, x, y in (("scene", a1, b1), ("p1", a1, b1), ("p2", a1, b1), ("p1g", a2, b2), ("p2g", a2, b2)):
            u, v = x[name].astype(np.float32), y[name][:len(x[name])].astype(np.float32)
            cos = (u.reshape(len(u), -1) * v.reshape(len(v), -1)).sum(1) / (np.linalg.norm(u.reshape(len(u), -1), axis=1) * np.linalg.norm(v.reshape(len(v), -1), axis=1))
            print(s, name, u.shape, f"cosine min {cos[:-2].min():.4f} mean {cos[:-2].mean():.4f}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


def per_replay(truth, preds, level, k, vocab, te):
    """replay -> {tp, pred, true} at a level."""
    out = {}
    tb, pb = {}, {}
    for r in truth:
        tb.setdefault(r["replay"], []).append(r)
    for r in preds:
        pb.setdefault(r["replay"], []).append(r)
    for s in tb:
        d = te.score_level(tb[s], pb.get(s, []), level, k, vocab)
        out[s] = dict(tp=d["tp"], pred=d["n_pred"], true=d["n_true"])
    return out


def boot_f1(a, b, stems, n=2000, seed=0):
    """F1 difference b - a, paired over replays."""
    A = np.array([[a[s]["tp"], a[s]["pred"], a[s]["true"]] for s in stems], float)
    B = np.array([[b[s]["tp"], b[s]["pred"], b[s]["true"]] for s in stems], float)
    f1 = lambda m: 2 * m[..., 0] / np.maximum(m[..., 1] + m[..., 2], 1)
    idx = np.random.default_rng(seed).integers(0, len(stems), (n, len(stems)))
    d = f1(B[idx].sum(1)) - f1(A[idx].sum(1))
    return dict(a=round(float(f1(A.sum(0))), 4), b=round(float(f1(B.sum(0))), 4), diff=round(float(f1(B.sum(0)) - f1(A.sum(0))), 4),
                lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4))


def cmd_score(a):
    low_priority()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch  # noqa: F401
    import transcript_eval as te
    import video_openers as vo
    import video_transcript as vt
    feats = TD / "feats"
    stems = sorted(s for s, _ in old_tasks() if (feats / f"{s}.npz").exists() and (TD / "feats_grid" / f"{s}.npz").exists())
    print(len(stems), "replays with features", flush=True)
    vt.FEATS, vt.GRIDF = feats, TD / "feats_grid"
    if a.hud_half:
        # lab_analyze.transcript_features on a 30 fps video: idx = frame_number // 2 where frame_number counts 30 fps frames
        real_np = np

        class Z(dict):
            pass

        class Shim:
            def __getattr__(self, k):
                return getattr(real_np, k)

            def load(self, path, *x, **y):
                z = real_np.load(path, *x, **y)
                p = Path(path)
                if p.parent != feats:
                    return z
                d = Z({k: z[k] for k in z.files})
                X, fr = vo.parquet_inputs(DEG / "runs" / p.stem / "frames" / f"{p.stem}_frames.parquet")
                d["hud"], d["hud_ok"] = hud_rows(X, real_np.asarray(fr) // 2, int(z["n_frames"]))
                return d
        vt.np = Shim()
    rows, sp, vocab, voc_idx, chars, ev_by = vt.setup()
    net, ck = vt.load_net("ens5grid")
    assert ck["vocab"] == vocab
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    t0 = time.time()
    st = vt.Store(stems, {}, voc_idx, ck["chars"], norm, grid=True)
    cands = vt.candidates(net, st, vocab, ck["chars"], 0.05)
    preds = vt.decode(cands, ck["thr"], ck["merge"])
    print(f"head: {len(preds)} events in {time.time() - t0:.0f}s (thr {ck['thr']}, merge {ck['merge']})", flush=True)
    tag = "_hudhalf" if a.hud_half else ""
    cols = ["replay", "side", "start_frame", "score", "category", "char_move", "char", "normal", "outcome", "p_cat", "p_move"]
    pq.write_table(pa.Table.from_pylist([{c: p[c] for c in cols} for p in preds]), TD / f"ens5grid_preds{tag}.parquet")
    ss = set(stems)
    truth = [r for r in rows if r["replay"] in ss]
    ref = [r for r in pq.read_table(DATA / "transcript" / "exp_tr_ens5grid_test_preds.parquet").to_pylist() if r["replay"] in ss]
    res = dict(replays=len(stems), of=len(old_tasks()), events=len(truth), thr=ck["thr"], merge=ck["merge"], levels={})
    for k in te.TOLS:
        res["levels"][f"k{k}"] = {}
        for lv in te.LEVELS:
            pa_, pb_ = per_replay(truth, ref, lv, k, vocab, te), per_replay(truth, preds, lv, k, vocab, te)
            res["levels"][f"k{k}"][lv] = boot_f1(pa_, pb_, stems)
    d = te.score_level(truth, preds, "category", 4, vocab)
    d0 = te.score_level(truth, ref, "category", 4, vocab)
    res["category_k4"] = dict(precision=[round(d0["micro"]["precision"], 4), round(d["micro"]["precision"], 4)], recall=[round(d0["micro"]["recall"], 4), round(d["micro"]["recall"], 4)])
    res["openers_recovered_k4"] = [te.openers_recovered(truth, x, 4, vocab)["frac"] for x in (ref, preds)]
    # found (any class) by the true outcome, k = 4
    by = {}
    for o in ("hit", "block", "whiff", "interrupted", "counter"):
        t = [r for r in truth if te.outcome_label(r) == o]
        found = []
        for x in (ref, preds):
            tb, pb, got = {}, {}, 0
            for r in t:
                tb.setdefault((r["replay"], r["side"]), []).append(r["start_frame"])
            for r in x:
                pb.setdefault((r["replay"], r["side"]), []).append((r["start_frame"], r.get("score", 1.0)))
            # a true event of this outcome is found if any prediction of its side is within k (each prediction once, among all events of the side)
            for key, fr in tb.items():
                got += len(te.match(fr, pb.get(key, []), 4))
            found.append(round(got / max(1, len(t)), 4))
        by[o] = dict(n=len(t), found_1080p=found[0], found_720p=found[1])
    res["found_by_outcome_k4"] = by
    # the character vote on the copies
    res["vote_sides_right"] = sum(ck["chars"][st.vote[r, s]] == next((x["char"] for x in ev_by[n] if x["side"] == f"P{s + 1}"), None)
                                  for r, n in enumerate(st.stems) for s in (0, 1))
    res["vote_sides"] = 2 * len(st.stems)
    print(json.dumps(res, indent=1), flush=True)
    save_json(res, OUT / f"results_exp_measure_transcript_720p30{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "extract", "score"))
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hud-half", action="store_true")
    a = ap.parse_args()
    {"check": cmd_check, "extract": cmd_extract, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
