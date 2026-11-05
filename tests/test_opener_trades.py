"""Trades in the opener chain: a start where the marked attacker's own bar falls too gives both players an opener. No models, no videos."""
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "tools" / "ggst_logger" / "model"
sys.path.insert(0, str(MODEL))
import video_openers as vo  # noqa: E402

I = 20      # the detection's step


def bar(before: float, after: float, red_before: float = 0.0, red_after: float = 0.0, n: int = 50):
    """Readings of one bar: `before` up to the hit at step I, `after` from there on, and its red pending segment likewise."""
    hp, red = np.full(n, before), np.full(n, red_before)
    hp[I:], red[I:] = after, red_after
    return hp, red


@pytest.mark.parametrize("hp,red,why", [
    ((0.8, 0.72), (0.0, 0.08), "the bar loses what the red segment gains"),
    ((1.0, 0.99), (0.0, 0.0105), "a light hit"),
    ((0.5, 0.42), (0.02, 0.10), "the red segment of an earlier hit is still up"),
])
def test_a_hit_on_the_marked_attacker(hp, red, why):
    assert vo.own_bar_falls(*bar(hp[0], hp[1], red[0], red[1]), I), why


@pytest.mark.parametrize("hp,red,why", [
    ((0.8, 0.8), (0.0, 0.0), "the attacker's bar doesn't move: a plain opener"),
    ((0.8, 0.8), (0.08, 0.08), "the red segment of an earlier hit, unchanged"),
    ((0.8, 0.3), (0.0, 0.05), "a misread: far more bar gone than red gained"),
    ((0.8, 0.7), (0.0, 0.0), "no red: the bar fell without a hit, a misread"),
    ((0.5, 0.9), (0.0, 0.1), "the bar goes up"),
    ((0.8, 0.795), (0.0, 0.005), "under the threshold"),
])
def test_not_a_hit_on_the_marked_attacker(hp, red, why):
    assert not vo.own_bar_falls(*bar(hp[0], hp[1], red[0], red[1]), I), why


def test_no_readings_is_no_trade():
    hp, red = np.full(50, np.nan), np.full(50, np.nan)
    assert not vo.own_bar_falls(hp, red, I)
    assert not vo.own_bar_falls(*bar(0.8, 0.72, 0.0, 0.08), 0)     # nothing before the first step


def _sides():
    hp1, red1 = bar(0.8, 0.72, 0.0, 0.08)
    hp2, red2 = bar(0.6, 0.53, 0.0, 0.07)
    return {1: hp1, 2: hp2}, {1: red1, 2: red2}


def test_the_other_player_gets_an_opener():
    hp, red = _sides()
    assert vo.trade_partners([(I, 1, 0.99)], [(I, 1, 0.99)], hp, red) == [(I, 2, 0.99)]


def test_the_side_of_the_falling_bar_decides():
    hp, red = _sides()
    hp[2], red[2] = np.full(50, 0.6), np.zeros(50)                 # P2 marked, its own bar still: only P1 was hit
    assert vo.trade_partners([(I, 2, 0.9)], [(I, 2, 0.9)], hp, red) == []


@pytest.mark.parametrize("step,kept", [(I + vo.MATCH, False), (I + vo.MATCH + 1, True)])
def test_no_second_opener_where_the_net_marks_that_side(step, kept):
    hp, red = _sides()
    out = vo.trade_partners([(I, 1, 0.99)], [(I, 1, 0.99), (step, 2, 0.4)], hp, red)
    assert bool(out) == kept


def test_no_second_opener_inside_that_sides_combo():
    hp, red = _sides()
    shown = {1: np.zeros(50, int), 2: np.array([4] * 20 + [5] * 30)}   # P2's counter goes on rising through the hit
    assert vo.trade_partners([(I, 1, 0.99)], [(I, 1, 0.99)], hp, red, shown) == []


def test_bars_from_a_parquets_columns():
    t = {"p1_health_raw": [0.5, None], "p2_health_raw": [1.0, 0.9], "p1_pending_pct": [None, 0.1], "p2_pending_pct": [0.0, 0.0]}
    hp, red = vo.hud_bars(t)
    assert hp[1][0] == 0.5 and np.isnan(hp[1][1]) and np.isnan(red[1][0]) and red[2].tolist() == [0.0, 0.0]
