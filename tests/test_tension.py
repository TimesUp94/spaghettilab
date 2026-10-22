"""Tension gauge: absolute fill against the fixed bar width, both sources."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from replanal.extractors.tension import TensionExtractor, finalize_tension
from replanal.models import FrameContext, FrameData

STRIPS = Path(__file__).resolve().parent / "data" / "tension"


def _frame(name: str) -> np.ndarray:
    """Test strips are rows 995-1040 of a 1920x1080 frame."""
    img = cv2.imread(str(STRIPS / name))
    if img is None:
        pytest.skip(f"missing test strip {name}")
    full = np.zeros((1080, 1920, 3), np.uint8)
    full[995:995 + img.shape[0]] = img
    return full


# (strip, P1, P2) checked by eye on contact sheets with the measured fill end
# drawn (output/fix2/track3). tappa = Tappa #4 1080p30; newvod = the 720p60
# stream (webcam over the P1 anchor, chat over the P2 bar).
@pytest.mark.parametrize(
    "name,p1,p2",
    [
        ("tappa02_t78.9_y995.jpg", 0.32, 0.68),  # green P1, yellow P2 past the gear
        ("tappa02_t88.3_y995.jpg", 0.66, 0.20),
        ("tappa02_t213.3_y995.jpg", 0.70, 0.81),
        ("tappa02_t252.9_y995.jpg", 0.89, 0.47),
        ("tappa02_t160.6_y995.jpg", 0.0, 0.0),  # empty bars, labels only
        ("tappa09_t486.7_y995.jpg", 1.0, 0.10),  # full bar: white shine + end gear
        ("newvod03_t99.1_y995.jpg", 0.35, 0.29),
        ("newvod03_t129.0_y995.jpg", 0.72, 0.72),  # pale yellow fill (H 28-31, S 27-33)
        ("newvod03_t174.9_y995.jpg", 0.74, 0.60),
        ("newvod07_t154.9_y995.jpg", 0.47, 0.87),
        ("newvod07_t360.6_y995.jpg", 0.35, 0.92),
    ],
)
def test_reads_absolute_tension(name, p1, p2):
    r1, r2 = TensionExtractor().read(_frame(name))
    assert r1.pct == pytest.approx(p1, abs=0.03)
    assert r2.pct == pytest.approx(p2, abs=0.03)


def _extract(frame, timer):
    d = FrameData(0, 0.0)
    d.timer_value = timer
    TensionExtractor().extract(FrameContext("t", 0, 0.0, frame, {}), d)
    return d


def test_unknown_without_hud():
    """An empty bar means 0 only while the timer (HUD) is on screen."""
    empty = _frame("tappa02_t160.6_y995.jpg")
    assert _extract(empty, None).p1_tension_pct is None
    d = _extract(empty, 50)
    assert d.p1_tension_pct == 0.0 and d.p2_tension_pct == 0.0
    # both bars filled: believed even if the timer was misread
    d = _extract(_frame("tappa02_t78.9_y995.jpg"), None)
    assert d.p1_tension_pct == pytest.approx(0.32, abs=0.03)


def test_white_flash_is_not_tension():
    frame = np.full((1080, 1920, 3), 255, np.uint8)
    r1, r2 = TensionExtractor().read(frame)
    assert r1.pct == 0.0 and r2.pct == 0.0


def _frames(vals1, vals2):
    out = []
    for i, (a, b) in enumerate(zip(vals1, vals2)):
        f = FrameData(i, i * 33.3)
        f.p1_tension_pct, f.p2_tension_pct = a, b
        out.append(f)
    return out


def test_overlay_ceiling_drops_capped_readings():
    rng = np.random.default_rng(0)
    # P1: the bar is hidden beyond 0.836 by a logo: readings pile up there and
    # never pass it. P2: free bar, reaches full.
    p1 = list(np.clip(rng.uniform(0, 1.2, 400), 0, 0.836))
    p2 = list(rng.uniform(0, 1.0, 400))
    frames = _frames(p1, p2)
    ceil = finalize_tension(frames)
    assert ceil["p1_tension_pct"] == pytest.approx(0.836, abs=0.002)
    assert ceil["p2_tension_pct"] is None
    assert all(f.p1_tension_pct is None or f.p1_tension_pct < 0.83 for f in frames)
    assert sum(f.p2_tension_pct is not None for f in frames) == 400


def test_overlay_ceiling_found_however_rarely_it_is_reached():
    # New VOD sets 05-07: P1 reaches the logo cap in one stretch of 111+
    # readings out of thousands (under 3 %), and never passes it.
    rng = np.random.default_rng(1)
    p1 = list(rng.uniform(0, 0.7, 4000)) + [0.836] * 111 + list(rng.uniform(0, 0.7, 2000))
    p2 = list(rng.uniform(0, 1.0, len(p1)))
    frames = _frames(p1, p2)
    ceil = finalize_tension(frames)
    assert ceil["p1_tension_pct"] == pytest.approx(0.836, abs=0.002)
    assert all(f.p1_tension_pct is None or f.p1_tension_pct < 0.8 for f in frames)
    # Re-running (scripts/reclean_replay.py) starts from the raw readings.
    again = finalize_tension(frames)
    assert again == ceil
    assert frames[4050].p1_tension_raw == pytest.approx(0.836)


def test_highest_tension_held_briefly_is_not_a_ceiling():
    # Tappa #4 05_UF P2: never above 0.70 in the file, held there for 1.7 s.
    rng = np.random.default_rng(2)
    p2 = list(rng.uniform(0, 0.65, 4000)) + [0.697] * 52 + list(rng.uniform(0, 0.65, 2000))
    p1 = list(rng.uniform(0, 1.0, len(p2)))
    frames = _frames(p1, p2)
    ceil = finalize_tension(frames)
    assert ceil["p2_tension_pct"] is None
    assert frames[4020].p2_tension_pct == pytest.approx(0.697)


def test_median_smooths_single_frame_spikes_and_keeps_gaps():
    vals = [0.3] * 5 + [0.9] + [0.3] * 5 + [None] * 3 + [0.4] * 5
    frames = _frames(vals, vals)
    finalize_tension(frames)
    got = [f.p1_tension_pct for f in frames]
    assert got[5] == pytest.approx(0.3)
    assert got[11:14] == [None, None, None]
