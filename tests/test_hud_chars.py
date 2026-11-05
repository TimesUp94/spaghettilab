"""The HUD character reader's decisions and the mid-combo rule: no models, no videos, no references file."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "tools" / "ggst_logger" / "model"
sys.path.insert(0, str(MODEL))
import hud_chars as hc  # noqa: E402
import lab_analyze as la  # noqa: E402
import lang_summary as ls  # noqa: E402
import video_openers as vo  # noqa: E402

CHARS = ["Happy Chaos", "May", "Slayer", "Unika"]
UNIKA, SLAYER, MAY, CHAOS = 3, 2, 1, 0


def scores(rows):
    """rows of (t, P1 char, P2 char) -> a read_scores document with those winning clearly; None = an unreadable frame."""
    n = len(rows)
    port = np.full((n, 2, len(CHARS)), 0.5, np.float32)
    name = np.full((n, 2, len(CHARS)), 0.3, np.float32)
    for i, (_, a, b) in enumerate(rows):
        for k, c in enumerate((a, b)):
            if c is not None:
                port[i, k, c] = 0.97
                name[i, k, c] = 0.9
    return dict(t=np.array([r[0] for r in rows], np.float64), port=port, name=name)


def test_frame_wins_needs_score_and_margin():
    s = np.array([[0.97, 0.7, 0.1], [0.6, 0.5, 0.1], [0.9, 0.85, 0.1]])
    assert hc.frame_wins(s, 0.75, 0.10).tolist() == [0, -1, -1]


def test_one_stretch_when_nothing_changes():
    segs = hc.resolve(scores([(t, UNIKA, SLAYER) for t in range(0, 40, 2)]), CHARS)
    assert len(segs) == 1
    assert (segs[0]["P1"]["char"], segs[0]["P2"]["char"]) == ("Unika", "Slayer")
    assert segs[0]["P1"]["source"] == "hud_portrait" and segs[0]["P1"]["confidence"] == 1.0
    assert hc.characters_fields(segs)["characters_source"] == {"P1": "hud_portrait", "P2": "hud_portrait"}


def test_mirror_match_is_read_per_side():
    segs = hc.resolve(scores([(t, MAY, MAY) for t in range(0, 40, 2)]), CHARS)
    assert (segs[0]["P1"]["char"], segs[0]["P2"]["char"]) == ("May", "May")


def test_a_set_changes_characters_between_games():
    """Slycoops vs Leftover: Unika vs Slayer, then May vs Happy Chaos after the game break."""
    rows = [(t, UNIKA, SLAYER) for t in range(20, 310, 2)] + [(t, MAY, CHAOS) for t in range(530, 610, 2)]
    segs = hc.resolve(scores(rows), CHARS)
    assert [(g["P1"]["char"], g["P2"]["char"]) for g in segs] == [("Unika", "Slayer"), ("May", "Happy Chaos")]
    assert segs[0]["end_s"] == segs[1]["start_s"] and 308 < segs[1]["start_s"] < 530
    f = hc.characters_fields(segs)
    assert f["characters"] == {"P1": "Unika", "P2": "Slayer"}             # the most seen
    assert hc.segment_at(segs, 100.0)["P2"]["char"] == "Slayer"
    assert hc.segment_at(segs, 600.0)["P2"]["char"] == "Happy Chaos"


def test_both_sides_changing_at_a_break_is_one_cut():
    """The two sides' last readable frames differ by a few seconds; no empty stretch between them."""
    rows = [(t, UNIKA, SLAYER) for t in range(0, 100, 2)]
    rows += [(100, MAY, None), (102, MAY, None)]
    rows += [(t, MAY, CHAOS) for t in range(104, 200, 2)]
    segs = hc.resolve(scores(rows), CHARS)
    assert [(g["P1"]["char"], g["P2"]["char"]) for g in segs] == [("Unika", "Slayer"), ("May", "Happy Chaos")]
    assert all(g["frames"] > 0 for g in segs)


def test_a_few_misreads_open_no_stretch():
    rows = [(t, UNIKA, SLAYER) for t in range(0, 100, 2)]
    rows[20] = (40, MAY, CHAOS)
    rows[21] = (42, MAY, CHAOS)
    segs = hc.resolve(scores(rows), CHARS)
    assert len(segs) == 1 and segs[0]["P1"]["char"] == "Unika"
    assert segs[0]["P1"]["confidence"] == pytest.approx(48 / 50, abs=1e-3)


def test_falls_back_from_portrait_to_name_to_visual():
    n = 20
    sc = scores([(t, UNIKA, SLAYER) for t in range(n)])
    sc["port"][:, 0] = 0.5                         # P1's portrait unreadable: its name decides
    sc["port"][:, 1] = 0.5
    sc["name"][:, 1] = 0.3                         # P2: nothing on the HUD, so the visual vote
    segs = hc.resolve(sc, CHARS, lambda a, b: ("Unika", "Happy Chaos"))
    assert segs[0]["P1"]["source"] == "hud_name" and segs[0]["P1"]["char"] == "Unika"
    assert segs[0]["P2"]["source"] == "visual" and segs[0]["P2"]["char"] == "Happy Chaos"
    assert segs[0]["P2"]["confidence"] == 0.0


def test_a_confident_hud_beats_the_visual_vote():
    segs = hc.resolve(scores([(t, UNIKA, SLAYER) for t in range(0, 40, 2)]), CHARS, lambda a, b: ("Unika", "Happy Chaos"))
    assert segs[0]["P2"]["char"] == "Slayer" and segs[0]["P2"]["visual"] == "Happy Chaos"


def test_too_few_frames_are_not_confident():
    segs = hc.resolve(scores([(t, UNIKA, SLAYER) for t in range(0, 4, 2)]), CHARS, lambda a, b: ("May", "May"))
    assert segs[0]["P1"]["source"] == "visual" and segs[0]["P1"]["char"] == "May"


def test_portrait_and_name_conflict_is_flagged():
    sc = scores([(t, UNIKA, SLAYER) for t in range(0, 40, 2)])
    sc["name"][:, 1] = 0.3
    sc["name"][:, 1, MAY] = 0.9
    g = hc.resolve(sc, CHARS)[0]["P2"]
    assert g["conflict"] and g["char"] == "Slayer" and g["name"] == "May"


def test_no_frames_at_all():
    empty = dict(t=np.zeros(0), port=np.zeros((0, 2, 4)), name=np.zeros((0, 2, 4)))
    segs = hc.resolve(empty, CHARS, lambda a, b: ("Unika", "Slayer"))
    assert len(segs) == 1 and segs[0]["P1"]["char"] == "Unika" and segs[0]["P1"]["source"] == "visual"
    assert hc.visual_reading("Sol")["source"] == "visual"


def test_sample_times_spreads_over_the_video(tmp_path):
    t = hc.sample_times(None, 100.0, every=2.0)
    assert len(t) == 50 and t[0] < 2 and t[-1] > 98
    assert len(hc.sample_times(None, 100000.0, every=2.0)) == hc.MAX_SAMPLES
    assert len(hc.sample_times(None, 0.0)) == 0


def test_boxes_stay_on_the_games_hud():
    """Never a stream overlay: the score banner is at the top, between x 506 and 1416, above y 58; every box is outside it."""
    overlay = (506, 0, 1416, 58)
    for box in (hc.PORT_BOX, hc.NAME_BOX):
        for x0, y0, x1, y1 in (box, (hc.W - box[2], box[1], hc.W - box[0], box[3])):
            assert x1 <= overlay[0] or x0 >= overlay[2] or y0 >= overlay[3], (x0, y0, x1, y1)
    # the overlay mask the blackout tests use, when this checkout has it
    mask = ROOT / "output" / "blackout" / "mask.json"
    if mask.exists():
        for r in json.loads(mask.read_text()).values():
            if isinstance(r, dict):
                for box in (hc.PORT_BOX, hc.NAME_BOX):
                    for x0, y0, x1, y1 in (box, (hc.W - box[2], box[1], hc.W - box[0], box[3])):
                        assert x1 <= r["x"] or x0 >= r["x"] + r["w"] or y1 <= r["y"] or y0 >= r["y"] + r["h"], (r, (x0, y0, x1, y1))


def test_config_health_bar_is_where_the_name_box_is():
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["rois"]
    x0, y0, x1, y1 = hc.NAME_BOX
    bar = cfg["p1_health_bar"]
    assert y0 <= bar["y1"] and bar["y2"] <= y1 and x1 > bar["x1"]


# ---- the mid-combo rule ----------------------------------------------------------------------------------------

def counter(seq, side=1):
    """A string like '555|666.' -> (readings, index of '|'): digits (hex) = counter on that side, '.' = none."""
    i = seq.index("|")
    s = seq.replace("|", "")
    vals = [int(c, 16) if c != "." else 0 for c in s]
    cs = [side if v else None for v in vals]
    return vo.counter_readings(cs, vals, side), i


@pytest.mark.parametrize("seq,why", [
    ("5555555555|666666", "the 623S at 3:04 of Slycoops vs Leftover: five, then six"),
    ("44..444444|555555", "a reading gap before it"),
    ("3333333333|....555555", "a multi-hit move adds two"),
    ("6666......|.777777", "the HUD's number hidden for a moment"),
    ("2222222222|33333", "two to three straight away"),
    ("7777777777|.8888888", "seven to eight"),
])
def test_a_rising_counter_is_the_same_combo(seq, why):
    n, i = counter(seq)
    assert vo.continues_combo(n, i), why


@pytest.mark.parametrize("seq,why", [
    ("..........|......", "no counter at all: a plain opener"),
    ("5555555555|555555", "the old counter lingers: a one-hit opener in its last second"),
    ("5555555555|...2222", "the counter starts over: a new combo in the linger"),
    ("22222222222|....222222222333", "a new combo after a lingering two: two, then three, later on"),
    ("5555555......................|...666", "the last number is too long ago"),
    ("3333333333|99999", "a misread far above the count"),
    ("..........|666666", "the first number after a start is its own"),
])
def test_a_new_combo_is_kept(seq, why):
    n, i = counter(seq)
    assert not vo.continues_combo(n, i), why


def test_the_other_side_doesnt_count():
    vals = [5] * 10 + [6] * 6
    n = vo.counter_readings([2] * 16, vals, 1)
    assert not vo.continues_combo(n, 10)


def test_one_step_misread_is_ignored():
    n = vo.counter_readings([1] * 8, [3, 3, 3, 0, 9, 0, 3, 3], 1)
    assert n.tolist() == [3, 3, 3, 0, 0, 0, 3, 3]


# ---- what the result says ------------------------------------------------------------------------------------------

def test_lab_result_carries_the_sources():
    seg = dict(start_s=0.0, end_s=None, frames=50, P1=hc.visual_reading("Sol"), P2=hc.visual_reading("Ky"))
    doc = dict(video="x", openers=[], dropped_in_combo=[dict(time_s=1.0, attacker="P1", detector_score=0.5)], **hc.characters_fields([seg]))
    r = la.assemble("v.mp4", 10.0, 30.0, doc)
    assert r["characters"] == {"P1": "Sol", "P2": "Ky"}
    assert r["characters_source"] == {"P1": "visual", "P2": "visual"} and r["characters_confidence"] == {"P1": 0.0, "P2": 0.0}
    assert r["character_segments"][0]["P1"]["char"] == "Sol"
    assert r["dropped_in_combo"][0]["time_s"] == 1.0
    assert "character_segments" not in r["facts"]             # one stretch: nothing to say


def test_facts_name_every_character_of_a_set():
    a = dict(hc.visual_reading("Unika"))
    b = dict(hc.visual_reading("Slayer"))
    c = dict(hc.visual_reading("May"))
    d = dict(hc.visual_reading("Happy Chaos"))
    segs = [dict(start_s=0.0, end_s=320.0, frames=100, P1=a, P2=b), dict(start_s=320.0, end_s=None, frames=30, P1=c, P2=d)]
    doc = dict(video="x", openers=[], **hc.characters_fields(segs))
    f = ls.match_facts(doc)
    assert [g["P2"] for g in f["character_segments"]] == ["Slayer", "Happy Chaos"]
    md = ls.render_md(f, "en")
    assert "May" in md and "Happy Chaos" in md
    assert not [e for e in ls.check_answer("May and Happy Chaos played the last game.", f)["errors"] if e["kind"] == "character_not_in_match"]
    assert [e for e in ls.check_answer("Sol played.", f)["errors"] if e["kind"] == "character_not_in_match"]


def test_transcript_segments_are_on_the_60_fps_timeline():
    a, b = hc.visual_reading("Unika"), hc.visual_reading("Slayer")
    segs = [dict(start_s=0.0, end_s=320.0, P1=a, P2=b), dict(start_s=320.0, end_s=None, P1=b, P2=a)]
    assert la.transcript_segments(segs) == [(0, 19200, ("Unika", "Slayer")), (19200, None, ("Slayer", "Unika"))]
    assert la.transcript_segments(None) is None
