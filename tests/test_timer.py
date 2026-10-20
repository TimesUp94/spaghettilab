"""Round-timer reader tests.

tests/data/timer/labels.json holds ~840 hand-labelled timer crops from our
overlay VODs (data/sets, ggstsets) and three 60 fps non-overlay YouTube
replays; rois.npz holds the grey timer_digits ROI of every entry.  Labels:
"NN" clear value, "NNt" digit change / partly covered (NN or NN+1 is fine),
"none" no timer, "unreadable" timer present but not legible (not scored).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from replanal.config import PipelineConfig
from replanal.extractors.timer import TimerExtractor
from replanal.models import FrameContext, FrameData

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data" / "timer"


def _build_module():
    spec = importlib.util.spec_from_file_location(
        "build_timer_templates", ROOT / "scripts" / "build_timer_templates.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build():
    return _build_module()


@pytest.fixture(scope="module")
def samples(build):
    return build.load(DATA / "labels.json", DATA / "rois.npz")


@pytest.fixture(scope="module")
def extractor():
    return TimerExtractor()


@pytest.fixture(scope="module")
def config():
    return PipelineConfig.from_yaml(ROOT / "config" / "default.yaml")


def _read_crop(extractor, config, path):
    """Paste a 130x85 crop (frame x 895, y 100) on a black 1080p frame and run
    the extractor through the normal ROI path."""
    crop = cv2.imread(str(path))
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[100:185, 895:1025] = crop
    rois = {n: config.rois[n].crop(frame) for n in extractor.required_rois}
    data = FrameData(frame_number=0, timestamp_ms=0.0)
    extractor.extract(FrameContext("test", 0, 0.0, frame, rois), data)
    return data.timer_value


def _score(extractor, samples):
    clean = clean_ok = wrong = scored = 0
    for e, roi in samples:
        label = e["label"]
        if label == "unreadable":
            continue
        scored += 1
        r = extractor.read(roi)
        if label == "none":
            wrong += r is not None
        elif label.endswith("t"):
            v = int(label[:-1])
            wrong += r is not None and r not in (v, v + 1)
        else:
            v = int(label)
            clean += 1
            clean_ok += r == v
            wrong += r is not None and r != v
    return clean_ok / clean, wrong / scored


# The old reader returned 90, 82, 85, 80 (ones 6-9 -> 0-2, tens 3/6 -> 8)
# and None for the squashed 2 of a fresh digit change.
@pytest.mark.parametrize("name,expected", [
    ("crop_0062_98.png", 98),
    ("crop_0189_67.png", 67),
    ("crop_0158_65.png", 65),
    ("crop_0805_37.png", 37),
    ("crop_0118_92.png", 92),
    ("crop_0015_none.png", None),
    ("crop_0242_none.png", None),  # "LV2596" lobby text, not the timer
])
def test_reads_timer_crops(extractor, config, name, expected):
    assert _read_crop(extractor, config, DATA / name) == expected


def test_labelled_set_accuracy(extractor, samples):
    exact, wrong = _score(extractor, samples)
    assert exact >= 0.98
    assert wrong <= 0.01


def test_partly_covered_digits_are_not_misread(extractor, samples):
    # 0672: "57" with an orange blob over the 5 (read 37 before the quadrant
    # check); 0780: "79" with a white bar over the 7 (read 19).
    truth = {"0672": 57, "0780": 79}
    for e, roi in samples:
        if e["id"] in truth:
            assert extractor.read(roi) in (None, truth[e["id"]])


def test_flat_and_bright_rois_read_none(extractor):
    assert extractor.read(np.zeros((55, 80, 3), np.uint8)) is None
    assert extractor.read(np.full((55, 80, 3), 255, np.uint8)) is None


def test_templates_generalise_to_held_out_videos(build, samples, tmp_path):
    """Templates built only from the training videos read the held-out
    videos (incl. a non-overlay 60 fps replay) without wrong values."""
    train = [s for s in samples if s[0]["video"] not in build.HOLDOUT]
    test = [s for s in samples if s[0]["video"] in build.HOLDOUT]
    assert len(test) > 150
    glyphs, cells, _ = build.train(train)
    path = tmp_path / "glyphs.npz"
    np.savez_compressed(path, glyphs=glyphs.astype(np.float16),
                        cells=np.array([cells["tens"], cells["ones"]], np.int16))
    exact, wrong = _score(TimerExtractor(path), test)
    assert exact >= 0.97
    assert wrong <= 0.01


def test_labels_cover_every_digit():
    labels = json.loads((DATA / "labels.json").read_text())
    values = [int(e["label"].rstrip("t")) for e in labels
              if e["label"] not in ("none", "unreadable")]
    assert {v % 10 for v in values} == set(range(10))
    assert {v // 10 for v in values} >= {2, 3, 4, 5, 6, 7, 8, 9}
