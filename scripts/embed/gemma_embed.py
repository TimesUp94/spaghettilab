"""EmbeddingGemma-2 helpers: load the model locally (CUDA, bf16) and embed text / images /
video frames / audio into the shared 768-d space.

Built on the model card's documented API (sentence-transformers >= 6.1, transformers >= 5.19):
  * SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": bf16},
    config_kwargs={"vision_config": None, "audio_config": None})  -> selective encoders
  * model.encode(..., processing_kwargs={"image": {"max_soft_tokens": N}}) -> vision token budget
    (N must be one of 70, 140, 280, 560, 1120; images default 280, video frames default 140)
  * text prefixes apply to text only, never to media; for text->media use SearchQuery (see embed_texts)
  * audio = mono float32 @ 16 kHz, 25 tokens/s; video default 1 fps sampling
  * never use float16 (NaNs); bf16 or fp32 only

CLI:
  python scripts/embed/gemma_embed.py text "a fighting game match" "a menu"
  python scripts/embed/gemma_embed.py frames VIDEO 60 120 --text "a fighting game match"
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MODEL_ID = "google/embeddinggemma-2"
# Pinned revision the pipeline was validated with (HF main @ 2026-10-06).
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
SAMPLE_RATE = 16000
VISION_BUDGETS = (70, 140, 280, 560, 1120)
EMBED_DIM = 768
MAX_AUDIO_S = 320  # 25 tok/s in an 8192-token context


# --------------------------------------------------------------------------- loading
def load_model(modalities: Iterable[str] = ("text", "vision", "audio"), dtype=None,
               device: str = "cuda", revision: str | None = MODEL_REVISION):
    """Load EmbeddingGemma-2 as a SentenceTransformer with only the requested encoders.

    `modalities` is a subset of {"text","vision","audio"} (text is always present).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    mods = set(modalities)
    unknown = mods - {"text", "vision", "audio"}
    if unknown:
        raise ValueError(f"unknown modalities {unknown}")
    if dtype is None:
        dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32
    if dtype == torch.float16:
        raise ValueError("EmbeddingGemma-2 does not support float16 (NaN/degraded output); use bf16/fp32")
    config_kwargs = {}
    if "vision" not in mods:
        config_kwargs["vision_config"] = None
    if "audio" not in mods:
        config_kwargs["audio_config"] = None
    model = SentenceTransformer(
        MODEL_ID, revision=revision, device=device,
        model_kwargs={"torch_dtype": dtype}, config_kwargs=config_kwargs,
    )
    model.eval()
    return model


# --------------------------------------------------------------------------- utils
ALPHA_BG = (128, 128, 128)  # neutral grey used to flatten transparent images


def _flatten(im, bg=ALPHA_BG):
    """RGB copy of a PIL image. Images with transparency are alpha-composited onto a flat `bg`:
    a plain convert("RGB") exposes whatever junk RGB sits under transparent pixels (Dustloop
    sprites mix black, white and leftover colour there), which leaks into embeddings."""
    from PIL import Image
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, bg + (255,))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return im.convert("RGB")


def _to_pil(img):
    from PIL import Image
    if isinstance(img, Image.Image):
        return _flatten(img)
    if isinstance(img, (str, Path)):
        return _flatten(Image.open(img))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return Image.fromarray(arr.astype(np.uint8))  # assumes RGB order (extract_frames returns RGB)


def _check_budget(n: int):
    if n not in VISION_BUDGETS:
        raise ValueError(f"vision token budget must be one of {VISION_BUDGETS}, got {n}")


def normalize(x: np.ndarray, dim: int | None = None) -> np.ndarray:
    """L2-normalize rows; optionally Matryoshka-truncate to `dim` (128/256/512/768) first."""
    x = np.asarray(x, dtype=np.float32)
    if dim is not None:
        x = x[..., :dim]
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cosine_sim(a, b) -> np.ndarray:
    """Cosine similarity matrix between rows of a (N,D) and b (M,D) -> (N,M)."""
    a = normalize(np.atleast_2d(a))
    b = normalize(np.atleast_2d(b))
    return a @ b.T


def top_k(query, corpus, k: int = 5):
    """Return (indices, scores) of the k corpus rows most similar to a single query vector."""
    s = cosine_sim(query, corpus)[0]
    idx = np.argsort(-s)[:k]
    return idx, s[idx]


def _encode(model, inputs, batch_size: int, processing_kwargs=None, prompt=None, prompt_name=None):
    kw = dict(batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
              show_progress_bar=False)
    if processing_kwargs:
        kw["processing_kwargs"] = processing_kwargs
    if prompt is not None:
        kw["prompt"] = prompt
    if prompt_name is not None:
        kw["prompt_name"] = prompt_name
    return np.asarray(model.encode(inputs, **kw), dtype=np.float32)


# --------------------------------------------------------------------------- embedders
def embed_texts(model, texts: Sequence[str], prompt_name: str | None = "SearchQuery",
                batch_size: int = 64) -> np.ndarray:
    """Embed texts. Default prompt is 'task: search result | query: ' (text query vs media docs).

    Measured on set 17: for text->image/video/audio scoring use 'SearchQuery' or None.
    'Classification' (symmetric prefix) mis-ranked clear gameplay frames as "loading screen";
    use it only when comparing text with text. Media never gets a prefix.
    Cross-modal cosine scores are compressed (~0.55-0.75, gaps of 0.02-0.1): compare relatively
    (argmax / softmax over labels, or kNN against labelled frame prototypes), not by fixed threshold."""
    return _encode(model, list(texts), batch_size, prompt_name=prompt_name)


def embed_images(model, images: Sequence, token_budget: int = 280, batch_size: int = 16) -> np.ndarray:
    """Embed images (PIL / RGB uint8 arrays / paths), one 768-d vector each. No text prefix."""
    _check_budget(token_budget)
    pil = [_to_pil(im) for im in images]
    return _encode(model, pil, batch_size, {"image": {"max_soft_tokens": token_budget}})


def embed_video_frames(model, frames: Sequence, tokens_per_frame: int = 140,
                       batch_size: int = 4) -> np.ndarray:
    """Embed ONE clip made of the given frames (RGB arrays / PIL) as a single vector.

    `frames` may also be a list of clips (list of lists) -> one vector per clip.
    Frames are used as given (no resampling); pick your own times with extract_frames.
    Budget: tokens_per_frame * n_frames must stay well under 8192.
    """
    _check_budget(tokens_per_frame)
    clips = frames if (len(frames) and isinstance(frames[0], (list, tuple))) else [frames]
    inputs = []
    for clip in clips:
        arr = np.stack([np.asarray(_to_pil(f)) for f in clip])  # (T,H,W,3) uint8
        inputs.append({"video": arr})
    pk = {"video": {"max_soft_tokens": tokens_per_frame, "do_sample_frames": False}}
    return _encode(model, inputs, batch_size, pk)


def embed_audio(model, audio, batch_size: int = 8) -> np.ndarray:
    """Embed audio: a mono float32 16 kHz array, a path to an audio file, or a list of those.
    Returns (N,768). Clips longer than 30 s are NOT truncated (see max_length below)."""
    items = audio if isinstance(audio, (list, tuple)) else [audio]
    arrs = []
    for a in items:
        if isinstance(a, (str, Path)):
            import soundfile as sf
            data, sr = sf.read(str(a), dtype="float32", always_2d=True)
            data = data.mean(1)
            if sr != SAMPLE_RATE:
                raise ValueError(f"{a}: sample rate {sr}, expected {SAMPLE_RATE} (use extract_audio)")
            a = data
        arrs.append({"audio": {"array": np.asarray(a, dtype=np.float32), "sampling_rate": SAMPLE_RATE}})
    # The Gemma4 feature extractor silently truncates at max_length=480000 samples (30 s / 750
    # tokens) unless told otherwise; raise it to the longest clip (8192-token context ~= 327 s).
    longest = max(len(x["audio"]["array"]) for x in arrs)
    if longest > MAX_AUDIO_S * SAMPLE_RATE:
        raise ValueError(f"audio clip {longest / SAMPLE_RATE:.1f}s exceeds {MAX_AUDIO_S}s context limit")
    pk = {"audio": {"max_length": max(longest, 480_000)}}
    return _encode(model, arrs, batch_size, pk)


# --------------------------------------------------------------------------- media IO
def extract_frames(video: str | os.PathLike, times_s: Sequence[float]) -> list[np.ndarray]:
    """Grab RGB uint8 frames at the given timestamps (seconds) with cv2. Missing -> skipped
    with a ValueError listing them."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, missing = [], []
    for t in times_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, bgr = cap.read()
        if not ok:
            missing.append(t)
            continue
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if missing:
        raise ValueError(f"could not read frames at {missing}s")
    return out


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video: str | os.PathLike, start_s: float = 0.0, dur_s: float | None = None,
                  out_wav: str | os.PathLike | None = None) -> np.ndarray:
    """Decode [start_s, start_s+dur_s) of the video's audio as mono float32 @ 16 kHz using the
    imageio-ffmpeg bundled binary. Optionally also writes a wav."""
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{start_s:.3f}"]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.3f}"]
    cmd += ["-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.decode(errors='replace')[:500]}")
    audio = np.frombuffer(res.stdout, dtype=np.float32).copy()
    if out_wav is not None:
        import soundfile as sf
        sf.write(str(out_wav), audio, SAMPLE_RATE)
    return audio


# --------------------------------------------------------------------------- CLI
def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text")
    t.add_argument("texts", nargs="+")
    f = sub.add_parser("frames", help="embed frames at times and score against --text labels")
    f.add_argument("video")
    f.add_argument("times", nargs="+", type=float)
    f.add_argument("--text", action="append", default=[])
    f.add_argument("--budget", type=int, default=280)
    a = sub.add_parser("audio", help="embed an audio clip and score against --text labels")
    a.add_argument("video")
    a.add_argument("start", type=float)
    a.add_argument("dur", type=float)
    a.add_argument("--text", action="append", default=[])
    args = ap.parse_args()

    if args.cmd == "text":
        m = load_model(("text",))
        e = embed_texts(m, args.texts)
        print(e.shape)
        print(np.round(cosine_sim(e, e), 3))
    elif args.cmd == "frames":
        m = load_model(("text", "vision"))
        e = embed_images(m, extract_frames(args.video, args.times), token_budget=args.budget)
        print("frames", e.shape)
        if args.text:
            s = cosine_sim(e, embed_texts(m, args.text))
            for t_, row in zip(args.times, s):
                print(f"{t_:8.1f}s", "  ".join(f"{lab!r}={v:.3f}" for lab, v in zip(args.text, row)))
    elif args.cmd == "audio":
        m = load_model(("text", "audio"))
        e = embed_audio(m, extract_audio(args.video, args.start, args.dur))
        print("audio", e.shape)
        if args.text:
            s = cosine_sim(e, embed_texts(m, args.text))[0]
            for lab, v in zip(args.text, s):
                print(f"{v:.3f}  {lab}")


if __name__ == "__main__":
    _main()
