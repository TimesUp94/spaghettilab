"""Language layer (tools/ggst_logger/model/lang_summary.py): match facts from synthetic chain / transcript /
HUD output, the prompts, the providers (nothing is sent anywhere) and the grounding checker on hand-written
good and bad answers. Fast: no models, no data files beyond the repo's moves.json."""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ggst_logger" / "model"))
import lang_summary as ls  # noqa: E402

DT = 1 / 30


def _hud():
    """Two rounds at 30 Hz. R1 3-31 s: P1 5H at 10 s takes P2 to 0.7, P2 2D at 20 s takes P1 to 0.8, a one-frame
    misread 0 on P1 at 15 s, P1 Throw at 30 s KOs P2 at 31 s. R2 36-42 s: P1 236K at 40 s KOs P2 at 42 s."""
    cols = {k: [] for k in ("timestamp_ms", "p1_health_pct", "p2_health_pct", "timer_value", "p1_rounds_won", "p2_rounds_won")}
    t = 0.0
    while t < 46:
        if t < 3:
            p1 = p2 = tm = None
            h = (None, None)
        elif t < 34:
            p1 = 1.0 if t < 20 else max(0.8, 1.0 - (t - 20) * 0.4)
            p2 = 1.0 if t < 10 else max(0.7, 1.0 - (t - 10) * 0.4)
            if t >= 30:
                p2 = max(0.0, 0.7 - (t - 30) * 0.7)
            if abs(t - 15) < DT / 2:
                p1 = 0.0                     # misread
            tm = None if t >= 31 else 99 - int(t - 3)
            h = (0, 0)
        elif t < 36:
            p1 = p2 = tm = None
            h = (1, 0)
        else:
            p1 = 1.0
            p2 = 1.0 if t < 40 else max(0.0, 1.0 - (t - 40) * 0.5)
            tm = None if t >= 42 else 99 - int(t - 36)
            h = (1, 0) if t < 44 else (2, 0)
        cols["timestamp_ms"].append(t * 1000)
        cols["p1_health_pct"].append(p1)
        cols["p2_health_pct"].append(p2)
        cols["timer_value"].append(tm)
        cols["p1_rounds_won"].append(h[0])
        cols["p2_rounds_won"].append(h[1])
        t += DT
    return ls.Hud.from_columns(cols)


def _op(t, side, move, cat="normal", move_p=0.95, hk="none"):
    chars = {"P1": "Sol", "P2": "Ky"}
    return dict(time_s=t, video_frame=int(t * 60), attacker=side, attacker_char=chars[side], defender_char=chars[ls.other(side)],
                detector_score=0.99, category=cat, category_p=0.95, move=move, move_p=move_p,
                move_top3=[(move, move_p), ("5K", 0.3), ("2S", 0.05)], normal=None, hit_kind=hk, hit_kind_p=0.9)


OPENERS = dict(characters={"P1": "Sol", "P2": "Ky"}, openers=[
    _op(10.0, "P1", "5H", hk="counter"), _op(20.0, "P2", "2D", move_p=0.3), _op(30.0, "P1", "Throw", cat="throw"),
    _op(40.0, "P1", "236K", cat="special", hk="punish")])


def _tr(t, side, move, cat="normal", outcome="hit", p=0.9):
    ch = {"P1": "Sol", "P2": "Ky"}[side]
    return dict(replay="x", side=side, start_frame=int(round(t * 60)), score=0.9, category=cat, char_move=f"{ch}:{move}",
                char=ch, normal=None, outcome=outcome, p_cat=0.9, p_move=p)


TRANSCRIPT = [
    _tr(10.0, "P1", "5H", outcome="counter"), _tr(12.3, "P1", "c.S", outcome="block"), _tr(12.8, "P1", "2S", outcome="block"),
    _tr(13.3, "P2", "5P", outcome="counter"), _tr(20.0, "P2", "2D"), _tr(23.0, "P2", "6P", outcome="block"),
    _tr(24.5, "P1", "5K", outcome="whiff"), _tr(30.0, "P1", "Throw", cat="throw"), _tr(40.0, "P1", "236K", cat="special"),
]


@pytest.fixture(scope="module")
def doc():
    return ls.match_facts(OPENERS, TRANSCRIPT, _hud(), match_id="synthetic")


def test_timestamps_round_trip():
    assert ls.ts(83.43) == "1:23.4"
    assert ls.ts(5) == "0:05.0"
    assert ls.parse_ts("1:23.4") == pytest.approx(83.4)


def test_rounds_from_hud(doc):
    r = doc["rounds"]
    assert [x["winner"] for x in r] == ["P1", "P1"]
    assert [x["end"] for x in r] == ["KO", "KO"]
    assert r[0]["start_s"] == pytest.approx(3.0, abs=0.1) and r[0]["end_s"] == pytest.approx(31.0, abs=0.1)
    assert r[0]["hearts_agree"] is True and not r[0]["uncertain"]      # the misread 0 at 15 s is not a KO
    assert doc["result"] == dict(winner="P1", score="2-0", rounds=2, uncertain=False)


def test_openers_damage_and_uncertainty(doc):
    o = {f["move"]: f for f in doc["openers"]}
    assert o["5H"]["damage_pct"] == 30 and not o["5H"]["ko"] and o["5H"]["hit_kind"] == "counter"
    assert o["2D"]["damage_pct"] == 20 and o["2D"]["uncertain"] and "move p=0.30" in o["2D"]["why_uncertain"]
    assert o["2D"]["alternatives"] == ["5K"]
    assert o["Throw"]["damage_pct"] == 70 and o["Throw"]["ko"]
    assert o["236K"]["ko"] and o["236K"]["move_name"] == "Bandit Revolver" and o["236K"]["round"] == 2
    p1 = doc["players"]["P1"]
    assert p1["openers"] == 3 and p1["damage_dealt_pct"] == 200 and p1["by_hit_kind"]["counter"] == 1
    assert doc["players"]["P2"]["openers_uncertain"] == 1
    assert all(r["what"] != "2D" for r in doc["players"]["P2"]["top_opening_moves"])   # uncertain: not counted


def test_transcript_habits(doc):
    t1, t2 = doc["transcript"]["P1"], doc["transcript"]["P2"]
    assert t1["attacks_seen"] == 6 and t1["attack_outcomes_pct"]["whiff"] == 17
    # after P1's 5H combo (ends ~10.8 s): P1's next move is the blocked c.S, P2's the 5P counter (after blocking too)
    assert t1["habits"]["after_own_combo"][0]["what"] == "c.S (block)"
    assert t1["habits"]["pressure_strings"][0]["what"] == "c.S > 2S"
    assert t2["habits"]["after_blocking"][0]["what"] == "5P (counter)"
    assert t1["counter_hit_by"][0]["what"].startswith("5P")


@pytest.mark.parametrize("lang", ["en", "it"])
def test_index_is_exactly_what_is_rendered(doc, lang):
    md = ls.render_md(doc, lang)
    shown = set(re.findall(r"\d+:\d\d\.\d", md))
    assert shown == {x["t"] for x in doc["index"]["times"]}


@pytest.mark.parametrize("lang", ["en", "it"])
def test_prompt(doc, lang, tmp_path):
    p = ls.build_prompt(doc, lang, player="P2")
    assert "frame data" in p["system"]
    assert ("[1:23.4]" in p["system"]) and "P2 (Ky)" in p["user"] and "[0:10.0]" in p["user"]
    assert ls.provider("none").complete(p, tmp_path) is None
    assert json.loads((tmp_path / f"prompt_{lang}.json").read_text(encoding="utf-8"))["user"] == p["user"]
    assert 0 < ls.estimate_tokens(p["system"] + p["user"]) < 3000


def test_claude_provider_never_runs_without_consent(doc, monkeypatch):
    monkeypatch.delenv("GGST_LANG_ALLOW_NETWORK", raising=False)
    p = ls.build_prompt(doc, "en")
    with pytest.raises(RuntimeError, match="disabled"):
        ls.provider("claude").complete(p)
    with pytest.raises(RuntimeError, match="disabled"):
        ls.provider("claude", enabled=True).complete(p)          # the env switch is needed too
    monkeypatch.setenv("GGST_LANG_ALLOW_NETWORK", "1")
    with pytest.raises(RuntimeError, match="disabled"):
        ls.provider("claude").complete(p)                         # and the explicit flag


GOOD = """## Summary
P1 Sol won 2-0: a Throw KO ended round 1 [0:30.0] and 236K (Bandit Revolver) on punish ended round 2 [0:40.0].

## Coaching notes
1. Sol's 5H counter hit opened a combo for 30% [0:10.0]; Ky should look at what he pressed there.
2. Ky's only combo came from 2D [0:20.0], which the model is not sure about.
3. Ky's 5P after blocking c.S > 2S got a counter hit [0:13.3].
"""

GOOD_IT = """## Riepilogo
Ha vinto P1 Sol 2-0: una Throw ha chiuso il round 1 [0:30.0] e 236K su punish il round 2 [0:40.0].

## Note di coaching
1. Il 5H counter di Sol ha aperto una combo del 30% [0:10.0].
2. L'unica combo di Ky è partita da 2D [0:20.0], ma la mossa è incerta.
3. Dopo aver parato c.S > 2S, il 5P di Ky ha fatto counter [0:13.3].
"""

BAD = """## Summary
Sol won with Gun Flame [0:25.5] and Volcanic Viper at [1:10.0].

## Coaching notes
1. 5H is +2 on block and has 10 frames of startup, so Ky should mash.
2. Sol landed 64% of his attacks; 2K at [0:10.0] opened the first combo.
3. Ky should play like May would [0:20.0].
"""


@pytest.mark.parametrize("answer", [GOOD, GOOD_IT])
def test_check_accepts_grounded_answers(doc, answer):
    r = ls.check_answer(answer, doc)
    assert r["ok"], r["errors"]
    assert r["timestamps_cited"] >= 5


def test_check_flags_ungrounded_answer(doc):
    r = ls.check_answer(BAD, doc)
    assert not r["ok"]
    kinds = collections_counter(e["kind"] for e in r["errors"])
    texts = {e["text"] for e in r["errors"]}
    assert "1:10.0" in texts                                 # beyond the match
    assert {"Gun Flame", "Volcanic Viper", "2K"} <= texts    # moves not in the facts
    assert "May" in texts and kinds["character_not_in_match"] == 1
    assert kinds["frame_data_claim"] >= 2                    # "+2 on block", "10 frames", "startup"
    assert "64%" in texts
    assert kinds["note_without_timestamp"] == 1
    assert kinds["move_time_mismatch"] >= 1                  # 2K at 0:10.0, where the facts have 5H


def test_check_wrong_move_at_a_real_time(doc):
    r = ls.check_answer("## Coaching notes\n1. The Throw at [0:10.0] started it.\n2. a [0:20.0]\n3. b [0:30.0]\n", doc)
    assert [e["kind"] for e in r["errors"]] == ["move_time_mismatch"]


def _moves_doc(doc, *moves):
    """The fixture's facts with these moves listed for P1 (the real run's Slayer and Unika moves)."""
    d = json.loads(json.dumps(doc))
    ix = d["index"]
    ix["moves"]["Sol"] = sorted(set(ix["moves"]["Sol"]) | set(moves))
    ix["times"] = ix["times"] + [dict(t_s=44.0 + i, char="Sol", move=m) for i, m in enumerate(moves)]
    return d


@pytest.mark.parametrize("written", ["214P/K~H", "214P/K", "214P", "214K", "214P~H", "214K~K", "[4]6S", "41236K~4~H", "41236K", "5H H"])
def test_check_takes_a_facts_move_however_it_is_written(doc, written):
    """A real run's two summaries were withheld for quoting the facts' own 214P/K~H, which the check cut down to 214P."""
    d = _moves_doc(doc, "214P/K~H", "214P/K~K", "[4]6S", "41236K~4~H", "5H H")
    r = ls.check_answer(f"## Coaching notes\n1. Try {written} here.\n", d)
    assert "move_not_in_facts" not in {e["kind"] for e in r["errors"]}


@pytest.mark.parametrize("written", ["214H", "214P~D", "236P/K", "6S"])
def test_check_still_catches_a_move_not_in_the_facts(doc, written):
    d = _moves_doc(doc, "214P/K~H", "[4]6S")
    r = ls.check_answer(f"## Coaching notes\n1. Try {written} here.\n", d)
    assert {"kind": "move_not_in_facts", "text": written} in r["errors"]


def test_move_time_check_takes_the_short_form(doc):
    d = _moves_doc(doc, "214P/K~H")
    t = d["index"]["times"][-1]["t_s"]
    r = ls.check_answer(f"## Coaching notes\n1. The 214P at [{ls.ts(t)}] started it.\n", d)
    assert "move_time_mismatch" not in {e["kind"] for e in r["errors"]}


def test_repair_lists_a_problem_once(doc):
    p = ls.build_prompt(doc, "en")
    e = dict(kind="move_not_in_facts", text="214P")
    assert ls.repair_prompt(p, "draft", [e, e, e])["user"].count("move_not_in_facts: 214P") == 1


def collections_counter(it):
    import collections
    return collections.Counter(it)


def test_openers_only_first_version():
    """The first version runs on the opener chain alone (no transcript, no HUD): no rounds, no damage, no habits."""
    d = ls.match_facts(OPENERS)
    assert d["rounds"] == [] and d["result"] is None and "transcript" not in d
    assert "damage_pct" not in d["openers"][0]
    assert "no_hud" in d["caveats"] and "Nessuna lettura" in ls.render_md(d, "it")
    md = ls.render_md(d, "it")
    assert "[0:40.0]" in md and "236K (Bandit Revolver)" in md
    assert ls.check_answer("## Note di coaching\n1. 236K [0:40.0]\n2. 5H [0:10.0]\n3. Throw [0:30.0]\n", d)["ok"]


def _times(doc, n=2):
    return [x["t"] if "t" in x else ls.ts(x["t_s"]) for x in doc["index"]["times"][:n]]


def test_check_multiline_notes_and_lead_in(doc):
    """Continuation lines and sub-bullets belong to their numbered note; a lead-in before "1." is no note."""
    a, b = _times(doc)
    answer = (f"## Summary\nA short summary [{a}].\n\n## Coaching notes\nFor both players:\n\n"
              f"1. What happened at [{a}].\n   Why it matters goes on a second line.\n   - Try something else.\n"
              f"2. **Second note** at [{b}].\n\n   Its advice is in another paragraph.\n"
              f"3. Third note [{b}].\n")
    chk = ls.check_answer(answer, doc)
    assert not [e for e in chk["errors"] if e["kind"] == "note_without_timestamp"], chk["errors"]
    assert not [w for w in chk["warnings"] if w["kind"] == "note_count"]
    # a numbered note with no timestamp at all still fails
    chk = ls.check_answer(f"## Coaching notes\n1. Fine [{a}].\n2. No time here.\n", doc)
    assert [e for e in chk["errors"] if e["kind"] == "note_without_timestamp"]


def test_check_percentages_derived_from_counts(doc):
    counts = [c for c in doc["index"]["counts"] if c > 0]
    a, b = min(counts), max(counts)
    if a == b:
        pytest.skip("needs two different counts")
    share = round(100 * a / b)
    if any(abs(share - p) <= 1 for p in doc["index"]["percentages"]):
        pytest.skip("that share is a figure of the facts")
    chk = ls.check_answer(f"Used {share}% of the time.", doc)
    assert not [e for e in chk["errors"] if e["kind"] == "unsupported_percentage"]
    assert [w for w in chk["warnings"] if w["kind"] == "derived_percentage"]
    # a figure no two counts can give stays an error
    bad = next(v for v in (97.3, 83.7, 71.3, 61.9, 43.1, 37.7) if not any(abs(v - p) <= 1 for p in doc["index"]["percentages"])
               and not any(0 < x <= y and abs(v - 100 * x / y) <= 1 for y in doc["index"]["counts"] for x in doc["index"]["counts"]))
    assert [e for e in ls.check_answer(f"Used {bad}% of the time.", doc)["errors"] if e["kind"] == "unsupported_percentage"]


def test_check_italian_behind_is_not_frame_data(doc):
    assert not [e for e in ls.check_answer("P1 era in svantaggio nel secondo round.", doc)["errors"] if e["kind"] == "frame_data_claim"]
    assert [e for e in ls.check_answer("Hai 3 di svantaggio di frame.", doc)["errors"] if e["kind"] == "frame_data_claim"]


def test_repair_prompt_lists_the_problems(doc):
    p = ls.build_prompt(doc, "it")
    r = ls.repair_prompt(p, "BOZZA", [dict(kind="unsupported_percentage", text="57%")])
    assert r["system"] == p["system"] and r["user"].startswith(p["user"]) and "BOZZA" in r["user"] and "57%" in r["user"]
