"""Label-free EmbeddingGemma-2 probes on set 17 (Faust vs Anji Mito).

Probes (all outputs under output/embed_test/probes/):
  A  scene zero-shot at 1 fps vs the timer-based gameplay mask from output/exp17/analysis.db
  B  character zero-shot on gameplay frames
  C  biggest HP-drop frames vs Dustloop images/texts (contact sheets)
  D  audio 1 s windows (0.5 s hop) zero-shot + alignment with HP drops / supers
  E  per-frame embedding throughput at 140/280 tokens (full-res vs pre-downscaled)

Usage:
  .venv/Scripts/python.exe scripts/embed/probe_set17.py [--probes ABCDE]
Embeddings are cached in probes/cache_*.npz; delete them to recompute.
Reads output/exp17/analysis.db read-only; never touches output/analysis.db.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_embed as ge  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "ggstsets" / "17_DIHLScarecrow_vs_QSalkazko.mp4"
EXP_DB = ROOT / "output" / "exp17" / "analysis.db"
DUST_NPZ = ROOT / "output" / "embed_test" / "dustloop_embeddings.npz"
DUST_MANIFEST = ROOT / "data" / "dustloop" / "manifest.json"
OUT = ROOT / "output" / "embed_test" / "probes"
FPS = 30.0
BUDGET = 280

SCENES_V1 = {  # first attempt: long, descriptive prompts (SearchQuery). Collapsed to "gameplay".
    "gameplay": ["a fighting game match in progress with two health bars at the top",
                 "two characters fighting in Guilty Gear Strive gameplay"],
    "character_select": ["a fighting game character select screen"],
    "versus_splash": ["a versus splash screen showing two fighters before the match"],
    "menu_lobby": ["a game menu or online lobby screen"],
    "transition": ["a replay or loading transition screen"],
    "super_flash": ["a cinematic super move flash with a close-up of a character"],
    "wallbreak": ["a wall break cutscene in a fighting game, the stage wall shattering"],
    "overlay_only": ["a stream overlay with commentator webcams and no gameplay"],
    "round_announce": ["a round announcement with large text like Let's Rock, Slash or Perfect"],
}
SCENES = {  # v2: prompts describing what is visibly on screen; best of the variants tried (no prefix)
    "gameplay": ["a fighting game screen with health bars and a round timer"],
    "intro_cutscene": ["an anime cutscene with sky"],
    "character_select": ["a character select screen"],
    "versus_splash": ["a versus splash screen showing two fighters before the match"],
    "menu_lobby": ["a game menu or online lobby screen"],
    "loading": ["a loading screen with a dialog box"],
    "blank": ["a blank black screen", "a blank white screen"],
    "super_flash": ["a dramatic close-up of a character face without health bars"],
    "wallbreak": ["a wall break cutscene in a fighting game, the stage wall shattering"],
    "overlay_only": ["a stream overlay with commentator webcams and no gameplay"],
    "round_announce": ["large text DUEL or SLASH on screen"],
    "victory": ["a character posing after winning"],
}
SCENE_PREFIX = None  # measured: None beat SearchQuery for these prompts (agreement 0.776 vs 0.753)
GAMEPLAY_FAMILY = {"gameplay", "super_flash", "wallbreak", "round_announce"}  # in-round states

CHARS = ["Anji Mito", "Faust", "Sol Badguy", "Ky Kiske", "May", "Potemkin", "Ramlethal Valentine",
         "Nagoriyuki", "Zato-1", "Millia Rage", "Giovanna", "Leo Whitefang"]

AUDIO_LABELS = ["a character shouting a special move name", "crowd cheering",
                "a commentator talking", "menu music", "a hit impact sound",
                "fighting game background music", "silence"]


# --------------------------------------------------------------------------- data
def timer_mask():
    """Per-frame booleans: timer_value present, plus per-frame HP (normalized pct)."""
    con = sqlite3.connect(f"file:{EXP_DB.as_posix()}?mode=ro", uri=True)
    r = con.execute("select frame_number, timer_value, p1_health_pct, p2_health_pct "
                    "from frame_data order by frame_number").fetchall()
    ev = con.execute("select timestamp_ms, frame_start, target_side, damage_pct from damage_events "
                     "order by timestamp_ms").fetchall()
    con.close()
    fn = np.array([x[0] for x in r])
    has_t = np.array([x[1] is not None for x in r])
    return fn, has_t, ev


def read_frames_at(frame_idx, scale=None):
    """Sequential decode, keeping frames whose index is in frame_idx (sorted). RGB uint8."""
    want = sorted(set(int(i) for i in frame_idx))
    cap = cv2.VideoCapture(str(VIDEO))
    out, i, k = {}, 0, 0
    while k < len(want):
        if i < want[k]:
            if not cap.grab():
                break
            i += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        if scale:
            bgr = cv2.resize(bgr, scale, interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
        k += 1
    cap.release()
    return [out[j] for j in want if j in out], [j for j in want if j in out]


def softmax(x, t=0.01):
    z = (x - x.max(-1, keepdims=True)) / t
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def class_scores(img_emb, txt_emb_by_class):
    names = list(txt_emb_by_class)
    S = np.stack([ge.cosine_sim(img_emb, txt_emb_by_class[c]).max(1) for c in names], 1)
    return names, S


def get_model(cache={}):
    if "m" not in cache:
        t0 = time.time()
        cache["m"] = ge.load_model(("text", "vision", "audio"))
        print(f"model loaded in {time.time() - t0:.1f}s")
    return cache["m"]


# --------------------------------------------------------------------------- 1 fps frames
def frames_1fps():
    """Embeddings for one frame per second (frame at t+0.5s), cached."""
    cache = OUT / "cache_1fps.npz"
    thumbs = OUT / "frames_1fps"
    if cache.exists():
        z = np.load(cache)
        return z["frame_idx"], z["emb"]
    thumbs.mkdir(parents=True, exist_ok=True)
    idx = np.arange(15, 11801, 30)
    frames, got = read_frames_at(idx)
    for f, j in zip(frames, got):
        cv2.imwrite(str(thumbs / f"f{j:05d}_{j / FPS:06.1f}s.jpg"),
                    cv2.cvtColor(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    m = get_model()
    t0 = time.time()
    emb = ge.embed_images(m, frames, token_budget=BUDGET)
    print(f"1fps: {len(frames)} frames embedded in {time.time() - t0:.1f}s")
    np.savez(cache, frame_idx=np.array(got), emb=emb)
    return np.array(got), emb


# --------------------------------------------------------------------------- A
def knn_vs_timer(fidx, emb, gt, k=7, exclude_s=5):
    """Leave-one-out cosine kNN using the (noisy) timer mask as labels; neighbours within
    +-exclude_s seconds of the query are excluded so temporal near-duplicates don't leak."""
    S = ge.cosine_sim(emb, emb)
    t = fidx / FPS
    pred = np.zeros(len(fidx), bool)
    for i in range(len(fidx)):
        s = S[i].copy()
        s[np.abs(t - t[i]) <= exclude_s] = -9
        nn = np.argsort(-s)[:k]
        pred[i] = gt[nn].mean() >= 0.5
    return pred


def probe_a(fidx, emb, has_t, report):
    m = get_model()
    tfrac0 = np.array([has_t[max(0, j - 15):j + 15].mean() for j in fidx])
    gt0 = tfrac0 >= 0.5
    v1 = {c: ge.embed_texts(m, p, prompt_name="SearchQuery") for c, p in SCENES_V1.items()}
    n1, S1 = class_scores(emb, v1)
    p1 = np.array(n1)[S1.argmax(1)]
    report["A_v1_prompts"] = {
        "prefix": "SearchQuery", "pred_counts": _counts(p1.tolist()),
        "agreement_gameplay_only": float(((p1 == "gameplay") == gt0).mean()),
        "note": "degenerate: predicts gameplay for nearly every frame incl. intro cutscene, black/white, loading"}
    kp = knn_vs_timer(fidx, emb, gt0)
    report["A_knn_timer_labels"] = {
        "k": 7, "exclude_s": 5, "agreement": float((kp == gt0).mean()),
        "tp": int((kp & gt0).sum()), "fp": int((kp & ~gt0).sum()),
        "fn": int((~kp & gt0).sum()), "tn": int((~kp & ~gt0).sum()),
        "disagree_t": [round(float(j / FPS), 1) for j in fidx[kp != gt0]]}
    tx = {c: ge.embed_texts(m, p, prompt_name=SCENE_PREFIX) for c, p in SCENES.items()}
    names, S = class_scores(emb, tx)
    pred = np.array(names)[S.argmax(1)]
    srt = np.sort(S, 1)
    margin = srt[:, -1] - srt[:, -2]
    # timer mask: fraction of frames with a timer in the 1 s window around the sample
    tfrac = np.array([has_t[max(0, j - 15):j + 15].mean() for j in fidx])
    gt = tfrac >= 0.5
    pg = pred == "gameplay"
    pf = np.isin(pred, list(GAMEPLAY_FAMILY))
    conf = {c: {"timer_on": int(((pred == c) & gt).sum()), "timer_off": int(((pred == c) & ~gt).sum())}
            for c in names}
    res = {
        "n_frames": int(len(fidx)), "timer_on_frames": int(gt.sum()),
        "agreement_gameplay_only": float((pg == gt).mean()),
        "agreement_gameplay_family": float((pf == gt).mean()),
        "binary_gameplay_family": {
            "tp": int((pf & gt).sum()), "fp": int((pf & ~gt).sum()),
            "fn": int((~pf & gt).sum()), "tn": int((~pf & ~gt).sum())},
        "confusion_pred_vs_timer": conf,
        "score_range": [float(S.min()), float(S.max())],
        "median_top1_margin": float(np.median(margin)),
    }
    rows = []
    for j, p, mg, tf, s in zip(fidx, pred, margin, tfrac, S):
        rows.append({"frame": int(j), "t": round(j / FPS, 2), "pred": str(p), "margin": round(float(mg), 4),
                     "timer_frac": round(float(tf), 2),
                     "scores": {n: round(float(v), 4) for n, v in zip(names, s)}})
    dis = [r for r in rows if (r["pred"] in GAMEPLAY_FAMILY) != (r["timer_frac"] >= 0.5)]
    res["n_disagreements_family"] = len(dis)
    (OUT / "A_scene_per_second.json").write_text(json.dumps(rows, indent=1))
    # timeline image: row1 timer mask, row2 predicted class colour
    pal = {n: tuple(int(c) for c in cv2.applyColorMap(np.uint8([[i * 255 // max(1, len(names) - 1)]]), cv2.COLORMAP_TURBO)[0, 0])
           for i, n in enumerate(names)}
    W = len(fidx) * 3
    img = np.full((140 + 18 * len(names), W + 170, 3), 255, np.uint8)
    for i, (g, p) in enumerate(zip(gt, pred)):
        img[5:35, i * 3:i * 3 + 3] = (0, 160, 0) if g else (200, 200, 200)
        img[45:75, i * 3:i * 3 + 3] = pal[p]
    cv2.putText(img, "timer mask", (W + 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "zero-shot", (W + 5, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for k in range(0, len(fidx), 30):
        cv2.putText(img, f"{k}s", (k * 3, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    for i, n in enumerate(names):
        y = 110 + 18 * i
        cv2.rectangle(img, (5, y), (20, y + 12), pal[n], -1)
        cv2.putText(img, f"{n} ({int((pred == n).sum())})", (25, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(OUT / "A_timeline.png"), img)
    report["A"] = res
    return names, S, pred, gt, rows


# --------------------------------------------------------------------------- B
def probe_b(fidx, emb, gt, report):
    m = get_model()
    variants = {
        "plain": [f"{c} from Guilty Gear Strive" for c in CHARS],
        "fighting": [f"{c} fighting in Guilty Gear Strive" for c in CHARS],
    }
    out = {}
    ge_emb = emb[gt]
    for vn, prompts in variants.items():
        T = ge.embed_texts(m, prompts)
        S = ge.cosine_sim(ge_emb, T)
        top1 = S.argmax(1)
        top2 = np.argsort(-S, 1)[:, :2]
        both = np.mean([set(r) == {0, 1} for r in top2])
        out[vn] = {
            "top1_counts": {CHARS[i]: int((top1 == i).sum()) for i in range(len(CHARS)) if (top1 == i).any()},
            "top2_is_anji_and_faust": float(both),
            "mean_score": {c: round(float(S[:, i].mean()), 4) for i, c in enumerate(CHARS)},
            "mean_rank_anji": float((np.argsort(-S, 1) == 0).argmax(1).mean() + 1),
            "mean_rank_faust": float((np.argsort(-S, 1) == 1).argmax(1).mean() + 1),
        }
    out["n_gameplay_frames"] = int(gt.sum())
    report["B"] = out


# --------------------------------------------------------------------------- C
def probe_c(has_t, events, report):
    m = get_model()
    z = np.load(DUST_NPZ, allow_pickle=True)
    man = json.loads(DUST_MANIFEST.read_text())
    paths = {r["id"]: ROOT / r["local_path"] for r in man["records"]}
    # dedupe events within 1 s, gameplay only (timer present in >=50% of +-1 s), biggest first
    ev = sorted(events, key=lambda e: -e[3])
    picked = []
    for ts, f0, side, dmg in ev:
        f0 = int(f0)
        if has_t[max(0, f0 - 30):f0 + 30].mean() < 0.5:
            continue
        if any(abs(f0 - p[1]) < 30 for p in picked):
            continue
        picked.append((ts, f0, side, dmg))
        if len(picked) == 30:
            break
    # use the frame ~0.2 s before the bar drop is registered (hit is happening)
    fr_idx = [max(0, p[1] - 6) for p in picked]
    frames, got = read_frames_at(fr_idx)
    by = dict(zip(got, frames))
    frames = [by[i] for i in fr_idx]
    E = ge.embed_images(m, frames, token_budget=BUDGET)
    Si = ge.cosine_sim(E, z["emb_280"])
    St = ge.cosine_sim(E, z["emb_text"])
    sheets = OUT / "C_contact"
    sheets.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (p, f) in enumerate(zip(picked, frames)):
        oi = np.argsort(-Si[k])[:5]
        ot = np.argsort(-St[k])[:5]
        rows.append({"t": round(p[1] / FPS, 2), "target_side": int(p[2]), "damage_pct": round(float(p[3]), 3),
                     "top_img": [(str(z["ids"][i]), round(float(Si[k, i]), 4)) for i in oi],
                     "top_txt": [(str(z["move_keys"][i]), round(float(St[k, i]), 4)) for i in ot],
                     "img_top1_minus_top5": round(float(Si[k, oi[0]] - Si[k, oi[4]]), 4),
                     "img_top1_minus_median": round(float(Si[k, oi[0]] - np.median(Si[k])), 4),
                     "img_top1_char": str(z["character"][oi[0]])})
        # contact sheet: frame (640x360) + 5 thumbs (240x170) with captions
        H = 360 + 230
        sheet = np.full((H, 1280, 3), 255, np.uint8)
        sheet[:360, :640] = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        txt = [f"t={p[1] / FPS:.1f}s  P{p[2]} -{p[3] * 100:.0f}%", "top Dustloop texts:"] + \
              [f"  {z['move_keys'][i]}  {St[k, i]:.3f}" for i in ot]
        for li, s in enumerate(txt):
            cv2.putText(sheet, s, (655, 30 + 26 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        for r, i in enumerate(oi):
            im = cv2.imread(str(paths[str(z["ids"][i])]), cv2.IMREAD_UNCHANGED)
            if im.shape[2] == 4:
                a = im[:, :, 3:4] / 255.0
                im = (im[:, :, :3] * a + 128 * (1 - a)).astype(np.uint8)
            h, w = im.shape[:2]
            sc = min(250 / w, 190 / h)
            im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            x0, y0 = 5 + r * 255, 365
            sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.putText(sheet, f"{z['ids'][i]} {Si[k, i]:.3f}", (x0, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.imwrite(str(sheets / f"C{k:02d}_{p[1] / FPS:06.1f}s.png"), sheet)
    (OUT / "C_hp_drop_neighbours.json").write_text(json.dumps(rows, indent=1))
    top1 = Si.max(1)
    report["C"] = {
        "n": len(rows),
        "img_top1_mean": float(top1.mean()), "img_top1_range": [float(top1.min()), float(top1.max())],
        "img_all_mean": float(Si.mean()), "img_all_std": float(Si.std()),
        "img_top1_minus_top5_mean": float(np.mean([r["img_top1_minus_top5"] for r in rows])),
        "img_top1_minus_median_mean": float(np.mean([r["img_top1_minus_median"] for r in rows])),
        "img_top1_char_counts": {c: sum(r["img_top1_char"] == c for r in rows) for c in ("Anji Mito", "Faust")},
        "img_top1_id_counts": _counts([r["top_img"][0][0] for r in rows]),
        "txt_top1_id_counts": _counts([r["top_txt"][0][0] for r in rows]),
        "txt_top1_mean": float(St.max(1).mean()),
        "txt_top1_minus_top5_mean": float(np.mean(np.sort(St, 1)[:, -1] - np.sort(St, 1)[:, -5])),
    }


def _counts(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- D
def probe_d(has_t, events, report):
    m = get_model()
    cache = OUT / "cache_audio.npz"
    audio = ge.extract_audio(VIDEO)
    sr = ge.SAMPLE_RATE
    starts = np.arange(0, len(audio) - sr + 1, sr // 2)
    if cache.exists():
        E = np.load(cache)["emb"]
    else:
        wins = [audio[s:s + sr] for s in starts]
        t0 = time.time()
        E = ge.embed_audio(m, wins, batch_size=64)
        el = time.time() - t0
        print(f"audio: {len(wins)} windows in {el:.1f}s")
        report.setdefault("D", {})["embed_time_s"] = el
        np.savez(cache, emb=E, starts=starts)
    tc = (starts + sr / 2) / sr  # window centre, s
    rms = np.array([np.sqrt(np.mean(audio[s:s + sr] ** 2)) for s in starts])
    man = json.loads(DUST_MANIFEST.read_text())
    supers = sorted({f"{mv['character']} {mv['name']}" for mv in man["moves"] if mv.get("category") == "super"})
    labels = AUDIO_LABELS + supers + [f"{c} shouting {mv}" for c, mv in
                                      [("Anji Mito", "Issei Ougi: Sai"), ("Faust", "Bone-crushing Excitement")]]
    T = ge.embed_texts(m, labels)
    S = ge.cosine_sim(E, T)
    # HP drop per window centre (sum of damage events within +-0.5 s)
    ev_t = np.array([e[1] / FPS for e in events])
    ev_d = np.array([e[3] for e in events])
    hp = np.array([ev_d[np.abs(ev_t - c) <= 0.5].sum() for c in tc])
    gm = np.array([has_t[int(c * FPS)] if int(c * FPS) < len(has_t) else False for c in tc])
    per = {}
    for i, lab in enumerate(labels):
        s = S[:, i]
        z_ = (s - s.mean()) / (s.std() + 1e-9)
        peaks = np.argsort(-s)[:10]
        per[lab] = {
            "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
            "max": round(float(s.max()), 4),
            "argmax_frac": round(float((S.argmax(1) == i).mean()), 3),
            "corr_hp_drop": round(float(np.corrcoef(s, hp)[0, 1]), 3),
            "corr_rms": round(float(np.corrcoef(s, rms)[0, 1]), 3),
            "mean_gameplay_minus_non": round(float(s[gm].mean() - s[~gm].mean()), 4),
            "top10_peaks_s": [round(float(tc[p]), 1) for p in sorted(peaks)],
            "top10_peaks_near_hp_drop": int(sum(hp[max(0, p - 2):p + 3].sum() > 0.1 for p in peaks)),
        }
    base_rate = float(np.mean([hp[max(0, p - 2):p + 3].sum() > 0.1 for p in range(len(tc))]))
    # silent-window comparison for loudness
    db = 20 * np.log10(rms + 1e-9)
    res = report.setdefault("D", {})
    res.update({
        "n_windows": int(len(starts)), "audio_s": round(len(audio) / sr, 2),
        "rms_dbfs": {"p10": round(float(np.percentile(db, 10)), 1), "median": round(float(np.median(db)), 1),
                     "p90": round(float(np.percentile(db, 90)), 1),
                     "gameplay_median": round(float(np.median(db[gm])), 1),
                     "nongameplay_median": round(float(np.median(db[~gm])), 1)},
        "base_rate_window_near_hp_drop": round(base_rate, 3),
        "labels": per,
        "window_embedding_self_sim_mean": float(ge.cosine_sim(E[::4], E[::4]).mean()),
    })
    np.savez(OUT / "D_audio_scores.npz", t=tc, scores=S, labels=np.array(labels), rms=rms, hp=hp, gameplay=gm)
    # plot: rms, hp drops, z-scored selected label curves
    sel = [0, 1, 2, 4] + [len(AUDIO_LABELS) + k for k in range(len(supers))][:2]
    Wd = len(tc) * 2
    Hh = 70
    img = np.full((Hh * (len(sel) + 2) + 20, Wd + 330, 3), 255, np.uint8)

    def strip(row, v, col, name):
        y0 = row * Hh + 5
        vv = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
        for k in range(len(v)):
            if not gm[k]:
                img[y0:y0 + Hh - 10, 2 * k:2 * k + 2] = (235, 235, 235)
            h = int(vv[k] * (Hh - 12))
            img[y0 + Hh - 10 - h:y0 + Hh - 10, 2 * k:2 * k + 2] = col
        cv2.putText(img, name[:42], (Wd + 5, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    strip(0, rms, (80, 80, 80), "RMS (grey bg = no timer)")
    strip(1, hp, (0, 0, 200), "HP drop (sum dmg, +-0.5s)")
    for r, i in enumerate(sel):
        strip(r + 2, S[:, i], (200, 100, 0), labels[i])
    for k in range(0, len(tc), 60):
        cv2.putText(img, f"{int(tc[k])}s", (2 * k, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(str(OUT / "D_audio_timeline.png"), img)


# --------------------------------------------------------------------------- E
def probe_e(report):
    import torch
    m = get_model()
    n = 240
    start = 4500  # 150 s, gameplay
    res = {}
    for scale, sname in ((None, "1920x1080"), ((960, 540), "960x540")):
        t0 = time.time()
        frames, _ = read_frames_at(range(start, start + n), scale=scale)
        dec = time.time() - t0
        for b in (140, 280):
            ge.embed_images(m, frames[:16], token_budget=b)  # warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            ge.embed_images(m, frames, token_budget=b, batch_size=32)
            torch.cuda.synchronize()
            el = time.time() - t0
            fps = n / el
            res[f"{sname}_b{b}"] = {
                "embed_fps": round(fps, 1), "decode_fps": round(n / dec, 1),
                "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "whole_video_11801_frames_min": round((11801 / fps + 11801 / (n / dec)) / 60, 1),
                "x_realtime_at_30fps": round(fps / 30, 2),
            }
            print(sname, b, res[f"{sname}_b{b}"])
    # embedding agreement full-res vs downscaled
    full, _ = read_frames_at(range(start, start + 32, 4))
    small, _ = read_frames_at(range(start, start + 32, 4), scale=(960, 540))
    for b in (140, 280):
        a = ge.embed_images(m, full, token_budget=b)
        c = ge.embed_images(m, small, token_budget=b)
        res[f"cos_full_vs_960_b{b}"] = round(float(np.mean(np.sum(a * c, 1))), 4)
    report["E"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="ABCDE")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rep_path = OUT / "report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    fn, has_t, events = timer_mask()
    if set(args.probes) & set("AB"):
        fidx, emb = frames_1fps()
        tfrac = np.array([has_t[max(0, j - 15):j + 15].mean() for j in fidx])
        gt = tfrac >= 0.5
    if "A" in args.probes:
        probe_a(fidx, emb, has_t, report)
    if "B" in args.probes:
        probe_b(fidx, emb, gt, report)
    if "C" in args.probes:
        probe_c(has_t, events, report)
    if "D" in args.probes:
        probe_d(has_t, events, report)
    if "E" in args.probes:
        probe_e(report)
    rep_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "D"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
