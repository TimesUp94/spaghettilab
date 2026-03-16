from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def iter_frames(
    video_path: str,
    fps: int = 30,
    sample_every: int = 1,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_number, timestamp_ms, frame_bgr) from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_number = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % sample_every == 0:
                h, w = frame.shape[:2]
                if h != 1080 or w != 1920:
                    frame = cv2.resize(frame, (1920, 1080))
                timestamp_ms = (frame_number / fps) * 1000.0
                yield frame_number, timestamp_ms, frame
            frame_number += 1
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """Return basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_ms"] = (info["frame_count"] / info["fps"]) * 1000.0 if info["fps"] > 0 else 0
    return info
