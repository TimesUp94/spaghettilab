"""Spaghetti Lab analysis of one Strive match video -> <out>/lab_result.json (schema spaghetti-lab-result/1,
described in LAB_RESULT.md next to this file). This is what the opener service and the apps run.

1. the opener chain (recognize.run_chain: HUD pipeline, combo starts, characters, opening moves,
   counter / punish), always; it also writes <out>/openers.json as before;
2. the match transcript (ROADMAP point 2), only with --transcript on, or auto and a CUDA GPU: frozen
   V-JEPA 2 features of this one video (transcript_feats + transcript_feats_grid, ~77 s of GPU per match),
   then the ens5grid head (video_transcript), plus the state tower's per-tubelet state classes
   (state_embed clip_sup heads). A transcript failure is not fatal: the result says transcript false;
3. facts and prompts in Italian and English (lang_summary; pure Python), from the openers, the
   transcript when there is one, and the HUD parquet the pipeline wrote.

    .venv/Scripts/python.exe tools/ggst_logger/model/lab_analyze.py <video> --out <dir> [--device cuda|cpu]
        [--light] [--transcript auto|on|off] [--analysis <analyze_replay output dir>] [--threads N] [--title T]

Progress lines for the service: `stage: hud|detect|moves|transcript|facts|done`.
Weights from GGST_MODEL_DIR (default <GGST_DATA>/ggst_dataset/model): the chain's *_final files, and for the
transcript video_transcript_exp_ens5grid.pt, video_players_test.pt, state_embed_exp_se_clip_sup.pt and the grid
PCA (transcript_grid_pca.npz there, else <data>/transcript/feats_grid/pca.npz).

The mapping and assembly functions (map_transcript, link_openers, smooth_states, assemble) need no torch and
no models; lab_fixture.py builds the UI sample with them from saved outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab_combos  # noqa: E402
import lang_summary as ls  # noqa: E402

SCHEMA = "spaghetti-lab-result/1"
FPS60 = 60.0
STEP = 8                                  # 60 fps frames per transcript tubelet (7.5 Hz)
STATE_HZ = FPS60 / STEP
CLS = ["neutral", "attacking", "blockstun", "hitstun", "knockdown", "airborne"]
TRANSCRIPT_NAME = "ens5grid"
STATE_NAME = "clip_sup"
PLAYERS_NET = "video_players_test.pt"
LANGS = ("it", "en")
STAGES = ("hud", "detect", "moves", "transcript", "facts", "done")


# ---- mapping (pure) ----------------------------------------------------------------------------

def opener_frame(o: dict) -> int:
    return int(round(o["time_s"] * FPS60))


def link_openers(events: list[dict], openers: list[dict]) -> list[dict]:
    """Marks the event each opener starts at (search_index.build's rule): among the same side's events with
    -4 <= opener frame - event frame <= 40, the nearest at or before the opener (else the first after it);
    one event per opener. Sets opener / hit_kind in place; returns events."""
    for o in sorted(openers, key=lambda o: o["time_s"]):
        f, side = opener_frame(o), o["attacker"]
        cand = [e for e in events if e["side"] == side and -4 <= f - e["frame"] <= 40 and not e["opener"]]
        if not cand:
            continue
        e = min(cand, key=lambda e: ((f - e["frame"]) if f >= e["frame"] else 99, e["frame"]))
        e["opener"], e["hit_kind"] = True, o.get("hit_kind")
    return events


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def map_transcript(preds: list[dict], openers: list[dict]) -> list[dict]:
    """video_transcript prediction rows (side, start_frame, category, char_move, char, normal, outcome, p_move,
    p_out?) -> lab_result transcript events, sorted by frame, openers linked."""
    ev = []
    for p in sorted(preds, key=lambda p: (p["start_frame"], p["side"])):
        cm = p.get("char_move")
        move = cm.split(":", 1)[1] if cm else "(rare)"
        char = p.get("char") or (cm.split(":", 1)[0] if cm else None)
        f = int(p["start_frame"])
        ev.append(dict(t=round(f / FPS60, 3), frame=f, side=p["side"], char=char, category=p["category"], move=move,
                       move_p=_r(p.get("p_move")), normal=p.get("normal") if p["category"] == "normal" else None,
                       outcome=p["outcome"], outcome_p=_r(p.get("p_out")), opener=False, hit_kind=None))
    return link_openers(ev, openers)


def smooth_states(cls_p) -> dict:
    """cls_p [2, n, 6] class probabilities per side and tubelet -> {"hz", "P1", "P2"} class names, after the
    3-tap mean search_index.state_streams applies."""
    c = np.asarray(cls_p, np.float32)
    cs = c.copy()
    if c.shape[1] >= 3:
        cs[:, 1:-1] = (c[:, :-2] + c[:, 1:-1] + c[:, 2:]) / 3
    a = cs.argmax(-1)
    return dict(hz=STATE_HZ, P1=[CLS[int(k)] for k in a[0]], P2=[CLS[int(k)] for k in a[1]])


# ---- search presets: the reference for the UIs (LAB_RESULT.md "Search presets"; tests check them
# against search_index.query) ----------------------------------------------------------------------

def opener_events(openers: list[dict]) -> list[dict]:
    """The openers seen as events, for the presets when there is no transcript."""
    return [dict(t=o["time_s"], frame=opener_frame(o), side=o["attacker"], char=o["attacker_char"], category=o["category"],
                 move=o["move"], move_p=o.get("move_p"), normal=o.get("normal"),
                 outcome="counter" if o.get("hit_kind") == "counter" else "hit",
                 outcome_p=o.get("hit_kind_p") if o.get("hit_kind") == "counter" else None, opener=True,
                 hit_kind=o.get("hit_kind")) for o in sorted(openers, key=lambda o: o["time_s"])]


def search_events(doc: dict) -> list[dict]:
    return doc["transcript"]["events"] if doc.get("transcript") else opener_events(doc["openers"])


def outcome_matches(e: dict, want: str) -> bool:
    if want in (None, "", "any"):
        return True
    if want == "connected":
        return e["outcome"] in ("hit", "counter")
    return e["outcome"] == want


def _own(events, side):
    return sorted((e for e in events if e["side"] == side), key=lambda e: e["frame"])


def preset_move_outcome(events, side=None, move=None, outcome="any", openers_only=False):
    return sorted((e for e in events if (not side or side == "any" or e["side"] == side) and (not move or e["move"] == move)
                   and outcome_matches(e, outcome) and (not openers_only or e["opener"])), key=lambda e: e["frame"])


def preset_counter_by(events, victim, move=None):
    return sorted((e for e in events if e["side"] != victim and e["outcome"] == "counter" and (not move or e["move"] == move)),
                  key=lambda e: e["frame"])


def preset_after_block(events, side, within=40):
    """[(blocked event p, next own event r)]."""
    ev = _own(events, side)
    return [(p, r) for p, r in zip(ev, ev[1:]) if p["outcome"] == "block" and r["frame"] - p["frame"] <= within]


def wakeups(cls: list[str]) -> list[int]:
    """60 fps frames at which a side gets up: the end of a >= 2-tubelet knockdown run not followed by hitstun."""
    out, i, n = [], 0, len(cls)
    while i < n:
        if cls[i] == "knockdown":
            j = i
            while j < n and cls[j] == "knockdown":
                j += 1
            if j - i >= 2 and j < n and cls[j] != "hitstun":
                out.append(STEP * j)
            i = j
        else:
            i += 1
    return out


def preset_after_kd(events, states, side):
    """[(event, frames since the wake-up)]: the side's first own event after each wake-up."""
    wakes, ev, w, res = wakeups(states[side]), _own(events, side), 0, []
    for k, r in enumerate(ev):
        p = ev[k - 1] if k else None
        while w < len(wakes) and wakes[w] < r["frame"] - 60:
            w += 1
        if w < len(wakes) and wakes[w] - 8 <= r["frame"] <= wakes[w] + 60 and (p is None or p["frame"] < wakes[w] - 8):
            res.append((r, r["frame"] - wakes[w]))
            w += 1
    return res


def prompt_text(p: dict) -> str:
    """The same text lang_summary's NoneProvider writes to prompt_<lang>.txt."""
    return f"[system]\n{p['system']}\n\n[user]\n{p['user']}"


def assemble(video: str, duration_s: float, fps: float, openers_doc: dict, preds: list[dict] | None = None,
             states: dict | None = None, hud: "ls.Hud | None" = None, timings: dict | None = None,
             match_id: str | None = None) -> dict:
    """The lab_result document. preds: transcript prediction rows or None (transcript not run). match_id: the
    title shown in the facts and prompts (e.g. the original file name); None = none, only the characters.
    `video` keeps only the file name: no directory (a server's job folder) ends up in the result."""
    t0 = time.time()
    facts = ls.match_facts(openers_doc, preds, hud, match_id=clean_title(match_id))
    facts["md"] = {lang: ls.render_md(facts, lang) for lang in LANGS}
    prompts = {lang: prompt_text(ls.build_prompt(facts, lang)) for lang in LANGS}
    timings = dict(timings or {})
    timings.setdefault("facts", round(time.time() - t0, 2))
    return dict(schema=SCHEMA, video=Path(str(video)).name, duration_s=round(float(duration_s), 2), fps=round(float(fps), 3),
                characters=dict(openers_doc["characters"]),
                capabilities=dict(openers=True, transcript=preds is not None, facts=True),
                openers=openers_doc["openers"],
                transcript=dict(events=map_transcript(preds, openers_doc["openers"])) if preds is not None else None,
                states=states if preds is not None else None,
                facts=facts, prompts=prompts, timings_s=timings)


# A match shows the round timer: of 1,103 real replays the lowest had 21.7 s of readings. A test pattern
# has none, and its health bars still read 71% of the frames, so the bars can't tell.
MIN_TIMER_S = 5.0
NOT_A_MATCH_EXIT = 3


def timer_seconds(parquet) -> float:
    """Seconds of round-timer readings in the pipeline's frames parquet (about 30 samples a second)."""
    import pyarrow.parquet as pq
    c = pq.read_table(parquet, columns=["timer_value", "timestamp_ms"]).to_pydict()
    t, v = c["timestamp_ms"], c["timer_value"]
    return sum(v_ is not None for v_ in v) / (len(t) / max((t[-1] - t[0]) / 1000.0, 1e-3)) if len(t) > 1 else 0.0


def require_match(parquet):
    """Stop before the models when the video isn't a Strive match: the job fails, and the analysis is given back."""
    s = timer_seconds(parquet)
    if s < MIN_TIMER_S:
        print(f"not a match: {s:.1f} s of round timer readings (needs {MIN_TIMER_S:.0f})", flush=True)
        raise SystemExit(NOT_A_MATCH_EXIT)


def ai_summary(doc: dict, name: str, langs=LANGS, prov=None) -> dict | None:
    """Summary + coaching notes written by an LLM from the facts (lang_summary prompts), one call per
    language, each answer checked against the facts (lang_summary.check_answer). A failed answer gets one
    repair call with the problems listed; if that fails too it is withheld: text None, the problems kept.
    name "none" -> None (nothing leaves the machine); no openers -> None (nothing to write about)."""
    if name in (None, "", "none") or not doc.get("openers"):
        return None
    os.environ.setdefault("GGST_LANG_ALLOW_NETWORK", "1")     # choosing a provider is the opt-in
    prov = prov or ls.provider(name, enabled=True)
    out = dict(provider=name, model=getattr(prov, "model", None), langs={})
    for lang in langs:
        try:
            prompt = ls.build_prompt(doc["facts"], lang)
            text = prov.complete(prompt)
            chk = ls.check_answer(text or "", doc["facts"])
            usage = [getattr(prov, "last_usage", None)]
            first = list(chk.get("errors", []))
            if text and first:
                text = prov.complete(ls.repair_prompt(prompt, text, first))
                chk = ls.check_answer(text or "", doc["facts"])
                usage.append(getattr(prov, "last_usage", None))
            ok = bool(text) and not chk.get("errors")
            print(f"ai summary {lang}: {'grounded' if ok else 'withheld'}, calls {len(usage)}, "
                  f"first draft problems {sorted({e['kind'] for e in first})}", flush=True)
            out["langs"][lang] = dict(text=text if ok else None, grounded=ok, errors=chk.get("errors", []),
                                      warnings=chk.get("warnings", []), repaired=len(usage) > 1, usage=usage[-1],
                                      usage_all=[u for u in usage if u])
        except Exception as e:                     # the rest of the result still comes out
            out["langs"][lang] = dict(text=None, grounded=False, errors=[f"{type(e).__name__}: {e}"[:300]], warnings=[])
    return out


def clean_title(title: str | None) -> str | None:
    """A match title for the facts: the file name without directory or extension, printable, at most 80 chars."""
    if not title:
        return None
    t = Path(str(title).replace("\\", "/")).name
    t = Path(t).stem if Path(t).suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi") else t
    t = "".join(c for c in t if c.isprintable() and c not in "[]()`<>").strip()[:80]
    return t or None


def write_json(doc: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ---- transcript (torch, GPU) -------------------------------------------------------------------

def grid_pca() -> Path:
    from common import DATA, OUT
    for p in (OUT / "transcript_grid_pca.npz", DATA / "transcript" / "feats_grid" / "pca.npz"):
        if p.exists():
            return p
    raise FileNotFoundError("the transcript grid PCA (transcript_grid_pca.npz) is missing")


def transcript_ready() -> bool:
    """The transcript's weights are all there."""
    from common import OUT
    try:
        grid_pca()
    except FileNotFoundError:
        return False
    return all((OUT / f).exists() for f in (f"video_transcript_exp_{TRANSCRIPT_NAME}.pt", PLAYERS_NET,
                                            f"state_embed_exp_se_{STATE_NAME}.pt"))


def hud_rows(X, frames, fps: float, n_frames: int):
    """The HUD inputs on the transcript's grid: row j is 60 fps frame 2j. The parquet's frame numbers are the video's
    own, so a 30 fps video's frame f is row f (not f // 2, which squeezed the match into its first half)."""
    n30 = (n_frames + 1) // 2
    out = np.zeros((n30, X.shape[1]), np.float32)
    ok = np.zeros(n30, bool)
    idx = np.round(np.asarray(frames) * FPS60 / fps).astype(int) // 2
    keep = (idx >= 0) & (idx < n30)
    out[idx[keep]] = X[keep]
    ok[idx[keep]] = True
    last = np.maximum.accumulate(np.where(ok, np.arange(n30), -1))
    out = np.where((last >= 0)[:, None], out[np.maximum(last, 0)], 0)
    return out.astype(np.float16), ok


def transcript_features(video: Path, parquet: Path, work: Path, stem: str, workers: int = 1, limit: int = 0):
    """transcript_feats + transcript_feats_grid for one video, into work/feats and work/feats_grid (the modules'
    dataset paths are redirected there; the HUD comes from the given parquet). limit: sampled frames (smoke tests)."""
    import gc
    import torch
    import transcript_feats as tf
    import transcript_feats_grid as tfg
    import video_openers as vo
    from common import OUT
    pca = dict(np.load(grid_pca()))
    tf.TR, tf.FEATS, tfg.GRID = work, work / "feats", work / "feats_grid"

    fps = video_info(video)[1]

    def hud(_stem, n_frames):           # tf.hud, on this video's parquet
        X, fr = vo.parquet_inputs(parquet)
        return hud_rows(X, fr, fps, n_frames)
    tf.hud = hud
    task = [(stem, str(video))]
    r = tf.run(task, str(OUT / PLAYERS_NET), workers=workers, threads=4, limit=limit)
    if not (tf.FEATS / f"{stem}.npz").exists():
        raise RuntimeError(f"transcript features failed: {r.get('stats')}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    enc = tfg.GridEncoder(str(OUT / PLAYERS_NET), pca)
    tfg.run(task, enc, workers=workers, limit=limit)
    del enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not (tfg.GRID / f"{stem}.npz").exists():
        raise RuntimeError("transcript grid features failed")


def transcript_predict(feats: Path, grid: Path, stem: str, dev: str) -> list[dict]:
    """ens5grid predictions (video_transcript rows, with p_out) of one replay whose features are under
    feats / grid (<stem>.npz each). Reads only."""
    import video_transcript as vt
    vt.DEV, vt.FEATS, vt.GRIDF = dev, feats, grid
    net, ck = vt.load_net(TRANSCRIPT_NAME)
    norm = tuple(np.array(x, np.float32) for x in ck["norm"])
    st = vt.Store([stem], {}, {m: i for i, m in enumerate(ck["vocab"])}, ck["chars"], norm, grid=bool(vt.ck_grid(ck)))
    preds = vt.decode(vt.candidates(net, st, ck["vocab"], ck["chars"], vt.thr_min(ck["thr"])), ck["thr"], ck["merge"])
    return sorted(preds, key=lambda p: (p["start_frame"], p["side"]))


def state_probs(feats: Path, grid: Path, stem: str, dev: str, bs: int = 1024):
    """The state tower's class probabilities [2, n, 6] per tubelet (state_embed predict, for one replay)."""
    import torch
    import torch.nn.functional as F
    import state_embed as se
    import video_transcript as vt
    vt.FEATS, vt.GRIDF = feats, grid
    from common import OUT
    ck = torch.load(OUT / f"state_embed_exp_se_{STATE_NAME}.pt", map_location=dev, weights_only=False)
    c = ck["cfg"]
    net, _ = se.build_nets(c["n_aux"], c["n_state"], d=c["d"], emb=c["emb"], W=c["W"], layers=c["layers"], heads=c["heads"])
    net.load_state_dict(ck["video"])
    net = net.to(dev).eval()
    V, A, _ = vt.load_replay(stem, None, grid=True)
    mu, sd = (np.array(x, np.float32) for x in ck["norm"])
    dt = torch.float16 if dev == "cuda" else torch.float32
    Vt = torch.from_numpy(V).to(dev)
    At = torch.from_numpy(((A[0] - mu) / sd).astype(np.float32)).to(dev, dt)
    n, W = len(V), c["W"]
    rel = np.arange(W) - W // 2
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for i in range(0, n, bs):
            idx = np.arange(i, min(n, i + bs))
            g = torch.as_tensor(np.clip(idx[:, None] + rel[None], 0, n - 1), device=dev)
            z = F.normalize(net(Vt[g], At[g]).float(), dim=-1)
            out.append(torch.softmax(net.h_cls(z).view(len(z), 2, len(CLS)).float(), -1).cpu().numpy())
    return np.concatenate(out).transpose(1, 0, 2)


def run_transcript(video: Path, parquet: Path, work: Path, dev: str, workers: int = 1, limit: int = 0) -> tuple[list[dict], dict]:
    stem = "match"
    work.mkdir(parents=True, exist_ok=True)
    transcript_features(video, parquet, work, stem, workers, limit)
    preds = transcript_predict(work / "feats", work / "feats_grid", stem, dev)
    states = smooth_states(state_probs(work / "feats", work / "feats_grid", stem, dev))
    return preds, states


# ---- CLI ---------------------------------------------------------------------------------------

def video_info(video: Path) -> tuple[float, float]:
    """(duration s, fps)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    finally:
        cap.release()
    return (n / fps if fps else 0.0), fps


class Stages:
    """Prints `stage: <name>` and keeps the seconds spent in each."""

    def __init__(self):
        self.t0 = self.t = time.time()
        self.cur, self.times = None, {}

    def __call__(self, name: str):
        now = time.time()
        if self.cur:
            self.times[self.cur] = round(self.times.get(self.cur, 0) + now - self.t, 1)
        self.cur, self.t = (None if name == "done" else name), now
        if name == "done":
            self.times["total"] = round(now - self.t0, 1)
        print(f"stage: {name}", flush=True)


def want_transcript(mode: str) -> bool:
    if mode == "off":
        return False
    from common import device
    if mode == "on":
        return True
    return device() == "cuda" and transcript_ready()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analysis", type=Path, help="an existing analyze_replay output folder")
    ap.add_argument("--device", choices=["cuda", "cpu"], help="default: the GPU when there is one")
    ap.add_argument("--light", action="store_true", help="opener chain: one move net, no frame-shift averaging")
    ap.add_argument("--threads", type=int, help="CPU threads for the networks (default: torch's)")
    ap.add_argument("--transcript", choices=["auto", "on", "off"], default="auto",
                    help="auto: only with a CUDA GPU and the transcript weights present")
    ap.add_argument("--title", help="match title for the facts and prompts (e.g. the original file name); "
                    "default: none, the heading shows only the characters")
    ap.add_argument("--workers", type=int, default=1, help="transcript: video decode processes (one video: 0 or 1)")
    ap.add_argument("--llm", default=os.environ.get("GGST_LANG_PROVIDER", "none"), choices=sorted(ls.PROVIDERS),
                    help="write a summary + coaching notes with an LLM from the facts (claude: needs ANTHROPIC_API_KEY); "
                         "default GGST_LANG_PROVIDER or none")
    a = ap.parse_args()
    if a.device:
        os.environ["GGST_DEVICE"] = a.device
    from common import device, low_priority
    import recognize as rec
    low_priority()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    out = a.out or a.video.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    st = Stages()
    duration, fps = video_info(a.video)
    openers_doc, pq_path = rec.run_chain(a.video, out, a.analysis, a.light, stage=st, check_hud=require_match)
    lab_combos.add_combo_hits(openers_doc["openers"], pq_path)
    preds = states = None
    if want_transcript(a.transcript):
        st("transcript")
        if device() != "cuda":
            print("warning: the transcript on the CPU is very slow (V-JEPA 2 ViT-L)", flush=True)
        try:
            preds, states = run_transcript(a.video, pq_path, out / "transcript", device(), a.workers)
            print(f"{len(preds)} transcript events", flush=True)
        except Exception as e:          # openers and facts still come out
            print(f"transcript failed: {type(e).__name__}: {e}", flush=True)
            preds = states = None
    st("facts")
    hud = ls.Hud.load(pq_path) if pq_path and Path(pq_path).exists() else None
    timings = dict(st.times)
    t_f = time.time()
    doc = assemble(str(a.video), duration, fps, openers_doc, preds, states, hud, timings, match_id=a.title)
    doc["timings_s"]["facts"] = round(time.time() - t_f, 1)
    if a.llm != "none":
        st("summary")
        t_s = time.time()
        doc["ai_summary"] = ai_summary(doc, a.llm)
        doc["capabilities"]["ai_summary"] = bool(doc["ai_summary"] and any(v["grounded"] for v in doc["ai_summary"]["langs"].values()))
        if doc["ai_summary"] is None:
            doc.pop("ai_summary")
        doc["timings_s"]["summary"] = round(time.time() - t_s, 1)
    doc["timings_s"]["total"] = round(time.time() - st.t0, 1)
    write_json(doc, out / "lab_result.json")
    st("done")
    print(f"lab result -> {out / 'lab_result.json'} (transcript {'yes' if preds is not None else 'no'}, "
          f"{len(doc['openers'])} openers)", flush=True)


if __name__ == "__main__":
    main()
