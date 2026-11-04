"""lab_analyze.py's mapping and assembly (no torch, no models): transcript rows -> lab_result events, the
opener link, the state smoothing, the facts / prompts assembly, the committed UI sample against the
contract (LAB_RESULT.md), and the search presets against search_index.py's query semantics."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "tools" / "ggst_logger" / "model"
sys.path.insert(0, str(MODEL))
import lab_analyze as la  # noqa: E402
import lang_summary as ls  # noqa: E402
from tests.test_lang_summary import _hud  # noqa: E402

FIXTURE = MODEL / "fixtures" / "lab_result_sample.json"
EVENT_KEYS = {"t", "frame", "side", "char", "category", "move", "move_p", "normal", "outcome", "outcome_p", "opener", "hit_kind"}
TOP_KEYS = {"schema", "video", "duration_s", "fps", "characters", "capabilities", "openers", "transcript", "states", "facts",
            "prompts", "timings_s"}


def pred(side, frame, cm, cat="normal", outcome="hit", normal="H/stand", p_move=0.9, p_out=0.8):
    return dict(replay="x", side=side, start_frame=frame, score=0.9, category=cat, char_move=cm, char=cm.split(":")[0] if cm else "Sol",
                normal=normal if cat == "normal" else None, outcome=outcome, p_cat=0.9, p_move=p_move, p_out=p_out)


def opener(side, t, move="5H", hit_kind="counter", char="Sol"):
    return dict(time_s=t, video_frame=int(t * 60), attacker=side, attacker_char=char, defender_char="Ky", detector_score=1.0,
                category="normal", category_p=0.9, move=move, move_p=0.9, move_top3=[[move, 0.9]], normal="H/stand",
                hit_kind=hit_kind, hit_kind_p=0.95)


def test_map_transcript_fields():
    rows = [pred("P2", 300, "Ky:236S", cat="special", outcome="block", p_out=0.61),
            pred("P1", 120, "Sol:5H", outcome="whiff"), pred("P1", 200, None, cat="other", outcome="none", p_out=None)]
    ev = la.map_transcript(rows, [])
    assert [e["frame"] for e in ev] == [120, 200, 300]
    assert all(set(e) == EVENT_KEYS for e in ev)
    a, b, c = ev
    assert a == dict(t=2.0, frame=120, side="P1", char="Sol", category="normal", move="5H", move_p=0.9, normal="H/stand",
                     outcome="whiff", outcome_p=0.8, opener=False, hit_kind=None)
    assert b["move"] == "(rare)" and b["outcome_p"] is None and b["normal"] is None
    assert c["normal"] is None and c["category"] == "special" and c["outcome_p"] == 0.61


def test_link_openers_rule():
    ev = la.map_transcript([pred("P1", 100, "Sol:5K"), pred("P1", 130, "Sol:5H"), pred("P1", 160, "Sol:6H"),
                            pred("P2", 140, "Ky:5P")], [])
    # opener at frame 141: candidates P1 events 101..145 -> 130 is the nearest before it; P2's 140 is the other side
    la.link_openers(ev, [opener("P1", 141 / 60)])
    assert [(e["frame"], e["opener"], e["hit_kind"]) for e in ev] == [(100, False, None), (130, True, "counter"),
                                                                      (140, False, None), (160, False, None)]
    # an opener 4 frames before a start still takes it; 5 frames before takes nothing
    ev = la.map_transcript([pred("P2", 200, "Ky:5P")], [])
    la.link_openers(ev, [opener("P2", 196 / 60, hit_kind="punish")])
    assert ev[0]["opener"] and ev[0]["hit_kind"] == "punish"
    ev = la.map_transcript([pred("P2", 200, "Ky:5P")], [])
    la.link_openers(ev, [opener("P2", 195 / 60)])
    assert not ev[0]["opener"]


def test_smooth_states():
    n = 6
    p = np.zeros((2, n, 6), np.float32)
    p[0, :, 0] = 1                     # P1 neutral, with a one-tubelet knockdown blip
    p[0, 3] = 0
    p[0, 3, 4] = 1
    p[1, :, 3] = 1                     # P2 hitstun throughout
    s = la.smooth_states(p)
    assert s["hz"] == 7.5 and len(s["P1"]) == len(s["P2"]) == n
    assert s["P1"] == ["neutral"] * n and s["P2"] == ["hitstun"] * n


def test_assemble_without_transcript():
    ops = dict(characters={"P1": "Sol", "P2": "Ky"}, openers=[opener("P1", 10.0), opener("P1", 40.0, move="236K", hit_kind="none")])
    doc = la.assemble("m.mp4", 46.0, 30.0, ops, None, None, _hud(), dict(hud=1.0))
    assert set(doc) == TOP_KEYS and doc["schema"] == "spaghetti-lab-result/1"
    assert doc["capabilities"] == {"openers": True, "transcript": False, "facts": True}
    assert doc["transcript"] is None and doc["states"] is None and doc["openers"] == ops["openers"]
    f = doc["facts"]
    assert f["schema"] == "ggst-match-facts/1" and f["match_id"] is None and f["sources"]["transcript"] is False
    assert set(f["md"]) == {"it", "en"} and f["md"]["en"] == ls.render_md({k: v for k, v in f.items() if k != "md"}, "en")
    assert doc["prompts"]["en"].startswith("[system]\n") and "[user]\n" in doc["prompts"]["it"]
    assert doc["timings_s"]["hud"] == 1.0 and "facts" in doc["timings_s"]
    json.dumps(doc)


def test_assemble_with_transcript():
    ops = dict(characters={"P1": "Sol", "P2": "Ky"}, openers=[opener("P1", 10.0)])
    rows = [pred("P1", 598, "Sol:5H", outcome="counter"), pred("P2", 900, "Ky:6P", outcome="block")]
    states = la.smooth_states(np.ones((2, 5, 6), np.float32))
    doc = la.assemble("m.mp4", 46.0, 60.0, ops, rows, states, _hud())
    assert doc["capabilities"]["transcript"] is True and doc["states"] is states
    ev = doc["transcript"]["events"]
    assert ev[0]["opener"] and ev[0]["hit_kind"] == "counter" and not ev[1]["opener"]
    assert doc["facts"]["sources"]["transcript"] is True and "transcript" in doc["facts"]


@pytest.fixture(scope="module")
def sample():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_sample_matches_contract(sample):
    assert set(sample) == TOP_KEYS and sample["schema"] == "spaghetti-lab-result/1"
    assert sample["capabilities"] == {"openers": True, "transcript": True, "facts": True}
    ev = sample["transcript"]["events"]
    assert ev and all(set(e) == EVENT_KEYS for e in ev)
    assert [e["frame"] for e in ev] == sorted(e["frame"] for e in ev)
    for e in ev:
        assert e["side"] in ("P1", "P2") and abs(e["t"] - e["frame"] / 60) < 1e-3
        assert e["outcome"] in ("counter", "hit", "block", "whiff", "interrupted", "teched", "none")
        assert e["category"] in ("normal", "special", "super", "throw", "other")
        assert 0 <= e["move_p"] <= 1 and (e["outcome_p"] is None or 0 <= e["outcome_p"] <= 1)
        assert (e["hit_kind"] is not None) == e["opener"]
    st = sample["states"]
    assert st["hz"] == 7.5 and len(st["P1"]) == len(st["P2"]) > 0 and set(st["P1"] + st["P2"]) <= set(la.CLS)
    assert set(sample["facts"]["md"]) == set(sample["prompts"]) == {"it", "en"}
    assert sample["facts"]["schema"] == "ggst-match-facts/1"
    assert sample["characters"] == sample["facts"]["characters"]
    assert all({"time_s", "attacker", "move", "hit_kind"} <= set(o) for o in sample["openers"])


def test_presets_mirror_search_index(sample):
    """LAB_RESULT.md's presets (lab_analyze.preset_*) return what search_index.query returns on the same events."""
    import search_index as si
    ev = sample["transcript"]["events"]
    states = sample["states"]
    cls = np.array([[la.CLS.index(c) for c in states[s]] for s in ("P1", "P2")])
    chars = sample["characters"]
    rows = [dict(e, replay="x", char_move=f"{e['char']}:{e['move']}", opp_char=chars["P2" if e["side"] == "P1" else "P1"])
            for e in ev]
    ix = si._link(rows, cls)

    def key(rs):
        return sorted((r["side"], r["frame"]) for r in rs)
    for side in ("P1", "P2"):
        got = la.preset_after_block(ev, side)
        assert key(r for _, r in got) == key(si.query(ix, side=side, after=dict(who="own", outcome="block", within=40)))
        got = la.preset_after_kd(ev, states, side)
        want = [r for r in ix if r["side"] == side and r["after_kd"] is not None]
        assert key(r for r, _ in got) == key(want) and [d for _, d in got] == [r["after_kd"] for r in sorted(want, key=lambda r: r["frame"])]
        victim = "P2" if side == "P1" else "P1"
        assert key(la.preset_counter_by(ev, victim)) == key(si.query(ix, side=side, outcome="counter"))
        for oc in ("whiff", "block", "connected", "counter"):
            assert key(la.preset_move_outcome(ev, side, "5H", oc)) == key(si.query(ix, side=side, move="5H", outcome=oc))
    assert any(la.preset_after_kd(ev, states, s) for s in ("P1", "P2"))     # the sample exercises every preset
    assert any(la.preset_after_block(ev, s) for s in ("P1", "P2"))
    assert la.preset_counter_by(ev, "P1") or la.preset_counter_by(ev, "P2")


def test_opener_fallback_events(sample):
    oe = la.opener_events(sample["openers"])
    assert len(oe) == len(sample["openers"]) and all(set(e) == EVENT_KEYS for e in oe)
    assert all(e["opener"] and e["outcome"] in ("hit", "counter") for e in oe)
    assert la.search_events(dict(sample, transcript=None)) == oe
    assert la.preset_after_block(oe, "P1") == [] and la.preset_after_block(oe, "P2") == []


def test_no_paths_or_job_ids_in_the_result():
    """On the server the video is <workdir>/<job id>/video.mp4: neither the folder nor the job id may reach the
    result; the title is the user's file name, or nothing (the heading then shows only the characters)."""
    ops = dict(characters={"P1": "Sol", "P2": "Ky"}, openers=[opener("P1", 10.0)])
    job = "0123456789abcdef0123456789abcdef"
    video = f"/srv/opener_service/work/{job}/video.mp4"
    doc = la.assemble(video, 46.0, 30.0, ops, None, None, _hud())
    text = json.dumps(doc, ensure_ascii=False)
    assert job not in text and "/srv" not in text and doc["video"] == "video.mp4"
    assert doc["facts"]["md"]["en"].splitlines()[0].endswith("P1 Sol vs P2 Ky")
    doc = la.assemble(video, 46.0, 30.0, ops, None, None, _hud(), match_id=r"C:\Users\me\Tappa 4 [GF].mp4")
    assert doc["facts"]["match_id"] == "Tappa 4 GF" and "Users" not in json.dumps(doc)
    assert "(Tappa 4 GF)" in doc["prompts"]["it"]


def test_ai_summary_grounding_gate():
    """A grounded answer is kept, an answer with invented facts is withheld, a provider error is reported;
    nothing here touches the network (a fake provider)."""
    import json as _json
    from pathlib import Path as _P
    import lab_analyze as la
    fx = _P(la.__file__).parent / "fixtures" / "lab_result_sample.json"
    doc = _json.loads(fx.read_text(encoding="utf-8"))
    lang_dir = _P(r"G:\Projects\replanal-data\ggst_dataset\lang\inputs_20261022_181805")
    good = (lang_dir / "answer_example_good_en.md")
    bad = (lang_dir / "answer_example_bad_en.md")
    if not (good.exists() and bad.exists()):
        import pytest
        pytest.skip("example answers not on this machine")

    class Fake:
        model = "fake"
        def __init__(self, answers): self.answers = answers
        def complete(self, prompt, out_dir=None):
            a = self.answers[prompt["lang"]]
            if isinstance(a, Exception):
                raise a
            return a

    res = la.ai_summary(doc, "claude", langs=("en",), prov=Fake({"en": good.read_text(encoding="utf-8")}))
    assert res["langs"]["en"]["grounded"] and res["langs"]["en"]["text"]
    res = la.ai_summary(doc, "claude", langs=("en",), prov=Fake({"en": bad.read_text(encoding="utf-8")}))
    assert not res["langs"]["en"]["grounded"] and res["langs"]["en"]["text"] is None and res["langs"]["en"]["errors"]
    res = la.ai_summary(doc, "claude", langs=("en",), prov=Fake({"en": RuntimeError("refused: cyber")}))
    assert not res["langs"]["en"]["grounded"] and "refused" in res["langs"]["en"]["errors"][0]
    assert la.ai_summary(doc, "none") is None
