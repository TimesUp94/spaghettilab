"""Round-counter (GGST heart) reader tests.

Labeled set: ``tests/data/hearts``. Each ``crops/*.png`` is a 190x50 BGR crop
of a 1920x1080 frame: columns 0-94 = frame x 775-869 (P1 hearts), columns
95-189 = frame x 1050-1144 (P2 hearts), rows = frame y 60-109. ``labels.csv``
gives per side the number of LOST hearts (``x`` = not visible / not
determinable), labeled by eye from enlarged crops, plus a by-video
train/test split. Frames come from our stream VODs (data/sets, ggstsets) and
three non-overlay 60 fps YouTube replays: round starts, mid-round, KOs and
match ends, white flashes, super cinematics, bright stages, lobby screens.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from replanal.extractors.round_counter import RoundCounterExtractor, read_lost_hearts
from replanal.models import FrameContext, FrameData

HEARTS_DIR = Path(__file__).resolve().parent / "data" / "hearts"


def _rows():
    with open(HEARTS_DIR / "labels.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _frame(name: str) -> np.ndarray:
    crop = cv2.imread(str(HEARTS_DIR / "crops" / name))
    assert crop is not None, name
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[60:110, 775:870] = crop[:, :95]
    frame[60:110, 1050:1145] = crop[:, 95:]
    return frame


def _extract(name: str) -> FrameData:
    data = FrameData(frame_number=0, timestamp_ms=0.0)
    ctx = FrameContext(video_path="", frame_number=0, timestamp_ms=0.0, frame_bgr=_frame(name), rois={})
    RoundCounterExtractor().extract(ctx, data)
    return data


def _score(rows):
    n = correct = wrong = 0
    for r in rows:
        frame = _frame(r["file"])
        for side in ("p1", "p2"):
            label = r[f"{side}_lost"]
            truth = None if label == "x" else int(label)
            pred = read_lost_hearts(frame, side)
            n += 1
            correct += pred == truth
            wrong += pred is not None and pred != truth
    return n, correct / n, wrong / n


def test_labeled_set_is_large_and_covers_both_sources():
    rows = _rows()
    assert len(rows) >= 300
    assert {r["source"] for r in rows} >= {"sets", "yt"}
    assert {r["split"] for r in rows} == {"train", "test"}


def test_heldout_accuracy():
    n, correct, wrong = _score([r for r in _rows() if r["split"] == "test"])
    assert correct >= 0.98, f"held-out correct {correct:.2%} on {n} side readings"
    assert wrong <= 0.01, f"held-out wrong non-None {wrong:.2%}"


def test_full_set_accuracy():
    n, correct, wrong = _score(_rows())
    assert correct >= 0.97, f"correct {correct:.2%} on {n} side readings"
    assert wrong <= 0.005, f"wrong non-None {wrong:.2%}"


def test_grey_heart_is_a_round_for_the_opponent():
    # P2 outer heart is grey (P1 won a round). Grey measures ~140 brightness
    # and ~0 saturation on the heart itself; the old 20x20 box mixed in the
    # dark background and read it as red.
    data = _extract("s01_00556.73.png")
    assert (data.p1_rounds_won, data.p2_rounds_won) == (1, 0)


def test_grey_heart_on_non_overlay_bright_stage():
    data = _extract("KWXz7_00283.35.png")  # YouTube replay, sky stage, 1-1
    assert (data.p1_rounds_won, data.p2_rounds_won) == (1, 1)


def test_grey_heart_under_red_tint():
    # Red screen tint makes the lost heart pink; compared against the red
    # inner heart it is still a lost heart.
    assert read_lost_hearts(_frame("s05_00683.37.png"), "p2") == 1
    assert read_lost_hearts(_frame("k02_00420.85.png"), "p1") == 1


def test_cracking_inner_heart_still_counts_as_red():
    # Final round of a 1-1 match: P2's inner heart cracks and jumps up during
    # the KO; it is not lost yet.
    assert read_lost_hearts(_frame("k06_00314.75.png"), "p2") == 1


@pytest.mark.parametrize("name", [
    "s03_00120.77.png",  # all-white flash (old reader: 2/2)
    "s06_00008.40.png",  # red/orange screen fill
    "s02_00244.20.png",  # lobby: dark red stripes where the hearts would be
])
def test_no_hud_gives_no_reading(name):
    frame = _frame(name)
    assert read_lost_hearts(frame, "p1") is None
    assert read_lost_hearts(frame, "p2") is None
    data = _extract(name)
    assert data.p1_rounds_won is None and data.p2_rounds_won is None


@pytest.mark.parametrize("name, side", [
    ("s06_00277.80.png", "p2"),  # white block over the outer heart
    ("m09_01784.80.png", "p2"),  # light object over the outer heart
])
def test_occluded_outer_heart_is_not_counted(name, side):
    assert read_lost_hearts(_frame(name), side) is None


def test_one_side_hidden_writes_nothing():
    # P1 hearts visible (0 lost), P2 outer heart occluded -> no per-frame value
    data = _extract("s06_00277.80.png")
    assert data.p1_rounds_won is None and data.p2_rounds_won is None
