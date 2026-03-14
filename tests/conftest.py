from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def data_dir():
    return ROOT / "data"


@pytest.fixture
def id_frames_dir(data_dir):
    return data_dir / "id_frames"


@pytest.fixture
def sample_ggs_frame(id_frames_dir):
    """A GGS frame (Dec 16 Goldlewis vs Potemkin)."""
    path = id_frames_dir / "Replay_2025-12-16_22-55-27.png"
    frame = cv2.imread(str(path))
    assert frame is not None, f"Cannot load sample frame: {path}"
    return frame


@pytest.fixture
def sample_ggs_frame_2(id_frames_dir):
    """A GGS frame (Mar 10 Anji vs Baiken)."""
    path = id_frames_dir / "Replay_2026-03-10_22-03-03.png"
    frame = cv2.imread(str(path))
    assert frame is not None, f"Cannot load sample frame: {path}"
    return frame


@pytest.fixture
def default_config():
    from replanal.config import PipelineConfig
    return PipelineConfig.from_yaml(ROOT / "config" / "default.yaml")
