from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

# Rate the analysis runs at. Downstream code (health cleaning, combo grouping,
# the Rust round detector) counts in samples and assumes ~30 samples/s.
TARGET_SAMPLE_FPS = 30.0
DEFAULT_FPS = 30.0


def resolve_fps(video_fps: float | None, fallback_fps: float | None = None) -> float:
    """The fps used for timestamps: the container's fps, else the fallback.

    OpenCV reports 0 (or NaN) when the container has no rate; then
    ``fallback_fps`` (the config value) is used, then 30.
    """
    if video_fps is not None and video_fps == video_fps and video_fps > 0:
        return float(video_fps)
    if fallback_fps is not None and fallback_fps > 0:
        return float(fallback_fps)
    return DEFAULT_FPS


def resolve_sample_every(fps: float, sample_every: int | None = None) -> int:
    """Frames to step per sample. Explicit values win; ``None`` (auto) keeps
    the effective sample rate at ~``TARGET_SAMPLE_FPS`` (60 fps -> 2, 30 -> 1)."""
    if sample_every is not None:
        return max(1, int(sample_every))
    return max(1, int(round(fps / TARGET_SAMPLE_FPS)))


def _open(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    return cap


def iter_frames(
    video_path: str,
    fps: float | None = None,
    sample_every: int | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_number, timestamp_ms, frame_bgr) from a video file.

    ``frame_number`` is the index of the frame in the source video (so on a
    60 fps video sampled every 2nd frame it goes 0, 2, 4, ...) and
    ``timestamp_ms = frame_number / video_fps * 1000`` is real video time.
    ``fps`` is only a fallback for containers that report no frame rate.
    ``max_frames`` stops after that many source frames (for quick tests).
    """
    cap = _open(video_path)
    real_fps = resolve_fps(cap.get(cv2.CAP_PROP_FPS), fps)
    step = resolve_sample_every(real_fps, sample_every)

    frame_number = 0
    try:
        while max_frames is None or frame_number < max_frames:
            if frame_number % step:
                # Skipped frame: grab() decodes but avoids the BGR copy.
                if not cap.grab():
                    break
                frame_number += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if h != 1080 or w != 1920:
                frame = cv2.resize(frame, (1920, 1080))
            timestamp_ms = (frame_number / real_fps) * 1000.0
            yield frame_number, timestamp_ms, frame
            frame_number += 1
    finally:
        cap.release()


def get_video_info(video_path: str, fallback_fps: float | None = None, sample_every: int | None = None) -> dict:
    """Return basic video metadata, plus the fps and sampling step the
    pipeline will use (``effective_fps``, ``sample_every``)."""
    cap = _open(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    eff = resolve_fps(info["fps"], fallback_fps)
    info["effective_fps"] = eff
    info["sample_every"] = resolve_sample_every(eff, sample_every)
    info["duration_ms"] = (info["frame_count"] / eff) * 1000.0 if info["frame_count"] > 0 else 0
    return info
