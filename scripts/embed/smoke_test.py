"""Smoke test for EmbeddingGemma-2 on a GGST set video.

  .venv/Scripts/python.exe scripts/embed/smoke_test.py [--video ggstsets/17_DIHLScarecrow_vs_QSalkazko.mp4]

Embeds 3 frames, a 5 s audio clip, a short frame clip as video, and a few text labels;
prints shapes, label similarities, peak VRAM and throughput at vision budgets 70/280/1120
and audio seconds/second. Saves the sampled frames + json results under output/embed_test/smoke/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_embed as ge  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "ggstsets/17_DIHLScarecrow_vs_QSalkazko.mp4"))
    ap.add_argument("--times", type=float, nargs="+", default=[5.0, 150.0, 370.0])
    ap.add_argument("--out", default=str(ROOT / "output/embed_test/smoke"))
    ap.add_argument("--bench-n", type=int, default=64, help="images per throughput run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"video": args.video, "revision": ge.MODEL_REVISION, "torch": torch.__version__}

    import cv2
    cap = cv2.VideoCapture(args.video)
    res["video_fps"] = cap.get(cv2.CAP_PROP_FPS)
    res["video_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res["video_size"] = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cap.release()
    print("video", res["video_fps"], "fps", res["video_frames"], "frames", res["video_size"])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = ge.load_model(("text", "vision", "audio"))
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res["vram_after_load_mb"] = round(torch.cuda.memory_allocated() / 2**20)
    print(f"loaded in {res['load_s']}s, {res['vram_after_load_mb']} MB")

    frames = ge.extract_frames(args.video, args.times)
    for t, f in zip(args.times, frames):
        cv2.imwrite(str(out / f"frame_{int(t):05d}s.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    labels = [
        "a fighting game match",
        "a character select menu",
        "a fighting game loading screen",
        "two players sitting at a tournament table",
        "an anime cutscene of an eagle flying in the sky",
        "a victory screen after a fight",
    ]
    te = ge.embed_texts(model, labels)
    ie = ge.embed_images(model, frames, token_budget=280)
    print("text", te.shape, "image", ie.shape, "norms", np.round(np.linalg.norm(ie, axis=1), 4))
    sims = ge.cosine_sim(ie, te)
    res["frame_label_sims"] = {f"{t}s": dict(zip(labels, map(float, np.round(r, 4))))
                               for t, r in zip(args.times, sims)}
    for t, r in zip(args.times, sims):
        print(f"  {t:7.1f}s best={labels[int(r.argmax())]!r}  " + " ".join(f"{v:.3f}" for v in r))

    # video: 8 frames at 2 fps starting at the second time
    vt = [args.times[1] + i * 0.5 for i in range(8)]
    ve = ge.embed_video_frames(model, ge.extract_frames(args.video, vt), tokens_per_frame=140)
    res["video_shape"] = list(ve.shape)
    res["video_label_sims"] = dict(zip(labels, map(float, np.round(ge.cosine_sim(ve, te)[0], 4))))
    print("video", ve.shape, res["video_label_sims"])

    # audio: 5 s clip
    audio = ge.extract_audio(args.video, args.times[1], 5.0, out_wav=out / "clip5s.wav")
    alabels = ["fighting game sound effects and music", "people talking", "crowd cheering", "silence"]
    ae = ge.embed_audio(model, audio)
    res["audio_samples"] = int(audio.shape[0])
    res["audio_shape"] = list(ae.shape)
    res["audio_label_sims"] = dict(zip(alabels, map(float, np.round(ge.cosine_sim(ae, ge.embed_texts(model, alabels))[0], 4))))
    print("audio", audio.shape, ae.shape, res["audio_label_sims"])

    # throughput: images/sec per budget
    bench_times = list(np.linspace(5, 385, args.bench_n))
    bench = ge.extract_frames(args.video, bench_times)
    res["images_per_s"] = {}
    for budget in (70, 280, 1120):
        ge.embed_images(model, bench[:4], token_budget=budget)  # warmup
        torch.cuda.reset_peak_memory_stats()
        _sync(); t0 = time.perf_counter()
        ge.embed_images(model, bench, token_budget=budget, batch_size=16)
        _sync(); dt = time.perf_counter() - t0
        res["images_per_s"][budget] = {"ips": round(len(bench) / dt, 2),
                                       "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
        print(f"budget {budget}: {len(bench)/dt:.1f} img/s, peak {res['images_per_s'][budget]['peak_mb']} MB")

    # throughput: audio seconds / second (8 x 30 s clips)
    clips = [ge.extract_audio(args.video, 30 + 30 * i, 30.0) for i in range(8)]
    ge.embed_audio(model, clips[:1])
    torch.cuda.reset_peak_memory_stats()
    _sync(); t0 = time.perf_counter()
    ge.embed_audio(model, clips, batch_size=8)
    _sync(); dt = time.perf_counter() - t0
    tot = sum(len(c) for c in clips) / ge.SAMPLE_RATE
    res["audio_s_per_s"] = round(tot / dt, 1)
    res["audio_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    print(f"audio: {tot/dt:.1f} audio-s/s, peak {res['audio_peak_mb']} MB")

    (out / "smoke_results.json").write_text(json.dumps(res, indent=2))
    print("wrote", out / "smoke_results.json")


if __name__ == "__main__":
    main()
