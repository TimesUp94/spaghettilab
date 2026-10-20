"""Timestamps come from the video's own fps; sampling keeps ~30 samples/s."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from replanal.config import PipelineConfig
from replanal.video import get_video_info, iter_frames, resolve_fps, resolve_sample_every

ROOT = Path(__file__).resolve().parent.parent


def _write_video(path: Path, fps: float, n: int) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 36))
    assert w.isOpened()
    for i in range(n):
        # Encode the frame index in the brightness so we can check which frame came back.
        w.write(np.full((36, 64, 3), (i * 4) % 256, np.uint8))
    w.release()
    return path


@pytest.fixture(scope="module")
def vids(tmp_path_factory):
    d = tmp_path_factory.mktemp("vids")
    return {
        60: _write_video(d / "v60.avi", 60.0, 30),
        30: _write_video(d / "v30.avi", 30.0, 15),
    }


def test_resolve_fps_falls_back_when_container_reports_none():
    assert resolve_fps(60.0, 30) == 60.0
    assert resolve_fps(0.0, 25) == 25.0
    assert resolve_fps(float("nan"), 25) == 25.0
    assert resolve_fps(0.0, None) == 30.0


def test_auto_sampling_targets_30_per_second():
    assert resolve_sample_every(60.0) == 2
    assert resolve_sample_every(59.94) == 2
    assert resolve_sample_every(30.0) == 1
    assert resolve_sample_every(29.97) == 1
    assert resolve_sample_every(120.0) == 4
    assert resolve_sample_every(24.0) == 1
    assert resolve_sample_every(60.0, 1) == 1  # explicit value wins
    assert resolve_sample_every(30.0, 3) == 3


def test_60fps_video_gets_real_timestamps_and_half_the_frames(vids):
    out = list(iter_frames(str(vids[60]), fps=30))  # config fps 30 must not be used
    nums = [n for n, _, _ in out]
    ts = [t for _, t, _ in out]
    assert nums == list(range(0, 30, 2))
    assert ts == pytest.approx([n / 60 * 1000 for n in nums])
    assert ts[-1] == pytest.approx(28 / 60 * 1000)  # ~467 ms, not ~933 ms
    # Skipped frames are really skipped (frame 2k is returned, not frame k).
    vals = [int(f[18, 32, 0]) for _, _, f in out]
    assert vals == pytest.approx([(n * 4) % 256 for n in nums], abs=3)
    assert out[0][2].shape == (1080, 1920, 3)


def test_30fps_video_unchanged(vids):
    out = list(iter_frames(str(vids[30]), fps=30))
    assert [n for n, _, _ in out] == list(range(15))
    assert [t for _, t, _ in out] == [(n / 30) * 1000.0 for n in range(15)]


def test_explicit_sample_every_and_max_frames(vids):
    out = list(iter_frames(str(vids[60]), sample_every=1, max_frames=10))
    assert [n for n, _, _ in out] == list(range(10))
    assert out[-1][1] == pytest.approx(9 / 60 * 1000)


def test_video_info_reports_effective_rate(vids):
    info = get_video_info(str(vids[60]), 30, None)
    assert info["effective_fps"] == 60.0
    assert info["sample_every"] == 2
    assert info["duration_ms"] == pytest.approx(500.0)


def test_config_sample_every_auto(tmp_path):
    cfg = PipelineConfig.from_yaml(ROOT / "config" / "default.yaml")
    assert cfg.sample_every_n_frames is None
    assert cfg.fps == 30
    for raw, want in (("auto", None), ("null", None), ("2", 2)):
        p = tmp_path / f"c_{raw}.yaml"
        p.write_text(f"sample_every_n_frames: {raw}\n")
        assert PipelineConfig.from_yaml(p).sample_every_n_frames == want
    p = tmp_path / "missing.yaml"
    p.write_text("fps: 30\n")
    assert PipelineConfig.from_yaml(p).sample_every_n_frames is None
    p.write_text("sample_every_n_frames: 0\n")
    with pytest.raises(ValueError):
        PipelineConfig.from_yaml(p)
