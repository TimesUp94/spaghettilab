from pathlib import Path

import cv2
import numpy as np
import pytest

from replanal.extractors.combo import (
    ComboCounterExtractor,
    _confirmed_max,
    derive_combos,
)
from replanal.models import FrameContext, FrameData, HealthReading, Side

FRAMES = Path(__file__).resolve().parent / "data" / "combo"


def _load(path):
    """Test frames are JPG crops of a 1080p frame: rows 0-1000 from column x
    (the ``_x<x>`` file name suffix); pasted back on a black 1920x1080 frame."""
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"missing test frame {path}")
    x0 = int(path.stem.rsplit("_x", 1)[1])
    full = np.zeros((1080, 1920, 3), np.uint8)
    full[: img.shape[0], x0 : x0 + img.shape[1]] = img
    return full


def _read(extractor, config, path, frame_number=0):
    img = _load(path)
    rois = {n: config.rois[n].crop(img) for n in extractor.required_rois}
    ctx = FrameContext("test", frame_number, 0.0, img, rois)
    data = FrameData(frame_number=frame_number, timestamp_ms=0.0)
    extractor.extract(ctx, data)
    return data


# Frames from ggstsets/17_DIHLScarecrow_vs_QSalkazko.mp4 (set17) and
# league/03_QSalkazko_vs_QEneroneJR.mp4, league/18_Jason_vs_1Guessing.mp4,
# all checked by eye.
@pytest.mark.parametrize(
    "name,side,count",
    [
        ("set17_f04640_x0.jpg", Side.P1, 2),  # "2 two", Faust (P1) combo, left side
        ("set17_f09000_x860.jpg", Side.P2, 3),  # "3 three", Anji (P2) combo, right side
        ("set17_f08300_x860.jpg", Side.P2, 12),  # huge "12", tens "1" covered by Faust
        ("set17_f10500_x860.jpg", Side.P2, 15),  # huge "15"
        ("set17_f07959_x860.jpg", Side.P2, 11),  # huge "11"
        ("set17_f01462_x0.jpg", Side.P1, 18),  # P1 "18", partly behind Faust
        ("set17_f05565_x0.jpg", Side.P1, 17),  # P1 "17"
        ("set17_f10900_x0.jpg", Side.P1, 10),  # P1 "17", 7 covered: tens only -> 10 (lower bound)
        ("league03_f31265_x0.jpg", Side.P1, 19),  # P1 "19"
        ("league18_f33281_x860.jpg", Side.P2, 14),  # P2 "14"
        ("set17_f10330_x0.jpg", None, None),  # no counter on screen
        ("set17_f01284_x0.jpg", None, None),  # "LET'S ROCK" round-start banner
        # "LET'S ROCK" (data/sets/04_UF f1306): with the big ROI stopping at
        # y 935 its strokes read as 1 + 1 (sum ~1.35) unless "11" pairs need
        # BIG_PAIR_MIN_SUM_11
        ("set04_f01306_x0.jpg", None, None),
    ],
)
def test_reads_counter_from_real_frames(default_config, name, side, count):
    ext = ComboCounterExtractor(stride=1)
    if not ext.enabled:
        pytest.skip("combo templates not built")
    data = _read(ext, default_config, FRAMES / name)
    assert data.combo_side == side
    assert data.combo_count == count


def test_stride_reads_every_frame_while_a_counter_is_shown(default_config):
    ext = ComboCounterExtractor(stride=2)
    if not ext.enabled:
        pytest.skip("combo templates not built")
    calls = []
    real = ext.read_side
    ext.read_side = lambda rois, side: (calls.append(side), real(rois, side))[1]
    empty, three = FRAMES / "set17_f10330_x0.jpg", FRAMES / "set17_f09000_x860.jpg"
    _read(ext, default_config, empty, frame_number=100)
    _read(ext, default_config, empty, frame_number=101)     # idle: skipped
    assert len(calls) == 2
    # The counter shows up on a skipped frame: the next read finds it and
    # reads the skipped frame too.
    d3 = _read(ext, default_config, three, frame_number=103)
    d4 = _read(ext, default_config, three, frame_number=104)
    assert len(calls) == 6
    assert (d3.combo_side, d3.combo_count) == (Side.P2, 3)
    assert (d4.combo_side, d4.combo_count) == (Side.P2, 3)
    # While a counter was seen recently every frame is read (odd ones too).
    d5 = _read(ext, default_config, empty, frame_number=105)
    assert len(calls) == 8 and d5.combo_side is None


def test_stride_output_does_not_depend_on_frame_parity(default_config):
    """The same frames read as a cut starting one frame later (every frame
    number shifted by one) give the same per-frame readings."""
    if not ComboCounterExtractor(stride=2).enabled:
        pytest.skip("combo templates not built")
    empty, three, twelve = (FRAMES / n for n in ("set17_f10330_x0.jpg", "set17_f09000_x860.jpg",
                                                 "set17_f08300_x860.jpg"))
    seq = [empty] * 4 + [three] * 3 + [twelve] * 3 + [empty] * 25 + [twelve] * 2 + [empty] * 4
    out = []
    for shift in (0, 1):
        ext = ComboCounterExtractor(stride=2)
        data = [_read(ext, default_config, path, frame_number=k + shift) for k, path in enumerate(seq)]
        out.append([(d.combo_side, d.combo_count) for d in data])
    assert out[0] == out[1]
    assert out[0][4] == (Side.P2, 3) and out[0][7] == (Side.P2, 12) and out[0][35] == (Side.P2, 12)


def test_confirmed_max_drops_one_frame_misreads():
    assert _confirmed_max([(0, 13), (1, 14), (2, 13), (3, 13), (4, 13), (5, 14)]) == 13
    assert _confirmed_max([(0, 2), (1, 4), (2, 4), (3, 4), (4, 6), (5, 6), (6, 6)]) == 6


def _frame(i, side=None, count=None, p1=None, p2=None):
    f = FrameData(frame_number=i, timestamp_ms=i * 1000 / 30)
    f.combo_side, f.combo_count = side, count
    if p1 is not None:
        f.p1_health = HealthReading(Side.P1, p1, 0, 0)
    if p2 is not None:
        f.p2_health = HealthReading(Side.P2, p2, 0, 0)
    return f


def test_derive_combos_groups_and_measures_damage():
    frames = []
    # P1 attacks P2: 2,2,2,3,3,3,(gap 5),5,5,5 ; P2 HP 0.8 -> 0.5
    seq = [2, 2, 2, 3, 3, 3, None, None, None, None, None, 5, 5, 5]
    for i in range(10):
        frames.append(_frame(len(frames), p1=1.0, p2=0.8))
    for c in seq:
        hp2 = 0.8 - 0.3 * (len(frames) - 10) / len(seq)
        frames.append(_frame(len(frames), Side.P1 if c else None, c, p1=1.0, p2=hp2))
    for i in range(10):
        frames.append(_frame(len(frames), p1=1.0, p2=0.5))
    # then a separate P2 combo on P1
    for c in [2, 2, 2, 4, 4, 4]:
        frames.append(_frame(len(frames), Side.P2, c, p1=0.9, p2=0.5))
    for i in range(10):
        frames.append(_frame(len(frames), p1=0.75, p2=0.5))

    combos = derive_combos(frames)
    assert len(combos) == 2
    a, b = combos
    assert a.attacker == Side.P1 and a.max_hits == 5
    assert a.frame_start == 10 and a.frame_end == 23
    assert a.hp_before == pytest.approx(0.8)
    assert a.damage == pytest.approx(0.3, abs=0.02)
    assert b.attacker == Side.P2 and b.max_hits == 4
    assert b.hp_before == pytest.approx(1.0)  # last reading before the counter appeared
    assert b.damage == pytest.approx(0.25, abs=1e-6)


def test_derive_combos_splits_on_restart_and_drops_blips():
    frames = [_frame(i) for i in range(5)]
    # 2..8 then immediately a new combo at 2 on the same side
    for c in [2, 2, 2, 5, 5, 5, 8, 8, 8, 2, 2, 2, 3, 3, 3]:
        frames.append(_frame(len(frames), Side.P2, c))
    # single-frame blip on P1 must be ignored
    frames.append(_frame(len(frames), Side.P1, 9))
    frames.extend(_frame(len(frames) + i) for i in range(5))
    combos = derive_combos(frames)
    assert [(c.attacker, c.max_hits) for c in combos] == [(Side.P2, 8), (Side.P2, 3)]
    assert all(c.damage is None for c in combos)  # no health data


def test_derive_combos_merges_through_misreads_and_long_holds():
    frames = [_frame(i) for i in range(5)]
    # 2 -> 3, "three" misread as "nine" for 3 frames, gap, 4 held for 2 s,
    # gap, 5: one combo of 5 hits (set 17 combos at 28.9 s and 92 s)
    seq = [2] * 6 + [3] * 8 + [9] * 3 + [None] * 15 + [4] * 60 + [None] * 14 + [5] * 4
    for c in seq:
        frames.append(_frame(len(frames), Side.P1 if c else None, c))
    frames.extend(_frame(len(frames)) for _ in range(5))
    combos = derive_combos(frames)
    assert [(c.attacker, c.max_hits) for c in combos] == [(Side.P1, 5)]


def test_derive_combos_keeps_separate_combos_separate():
    frames = [_frame(i) for i in range(5)]
    # 2,3 lingering for 1.3 s, then a fresh 2,3,4 by the same side
    seq = [2] * 4 + [3] * 40 + [None] * 12 + [2] * 4 + [3] * 4 + [4] * 6
    for c in seq:
        frames.append(_frame(len(frames), Side.P2 if c else None, c))
    frames.extend(_frame(len(frames)) for _ in range(5))
    combos = derive_combos(frames)
    assert [(c.attacker, c.max_hits) for c in combos] == [(Side.P2, 3), (Side.P2, 4)]


def test_derive_combos_resumes_big_count_after_cinematic():
    frames = [_frame(i) for i in range(5)]
    # 6,7,8, a super cinematic hides the counter for 3 s, then "10"; then a
    # covered-ones "10" (lower bound) after the real 11 must not split it
    seq = [6] * 4 + [7] * 4 + [8] * 4 + [None] * 90 + [10] * 6 + [11] * 4 + [None] * 15 + [10] * 4
    for c in seq:
        frames.append(_frame(len(frames), Side.P2 if c else None, c))
    frames.extend(_frame(len(frames)) for _ in range(5))
    combos = derive_combos(frames)
    assert [(c.attacker, c.max_hits) for c in combos] == [(Side.P2, 11)]


def test_stride_counts_samples_not_source_frames(default_config):
    """A 60 fps video read every 2nd frame has only even frame numbers; the
    stride must still skip every other *sample*, not read all of them."""
    ext = ComboCounterExtractor(stride=2)
    if not ext.enabled:
        pytest.skip("combo templates not built")
    calls = []
    real = ext.read_side
    ext.read_side = lambda rois, side: (calls.append(side), real(rois, side))[1]
    empty, three = FRAMES / "set17_f10330_x0.jpg", FRAMES / "set17_f09000_x860.jpg"
    _read(ext, default_config, empty, frame_number=200)
    _read(ext, default_config, empty, frame_number=202)     # next sample: skipped
    assert len(calls) == 2
    _read(ext, default_config, empty, frame_number=204)
    assert len(calls) == 4
    # A counter on sample 208 also reads the skipped sample 206 before it.
    d6 = _read(ext, default_config, three, frame_number=206)
    d8 = _read(ext, default_config, three, frame_number=208)
    assert len(calls) == 8
    assert (d6.combo_side, d6.combo_count) == (Side.P2, 3)
    assert (d8.combo_side, d8.combo_count) == (Side.P2, 3)
