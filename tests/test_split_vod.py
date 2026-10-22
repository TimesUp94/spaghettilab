"""VOD splitter (scripts/split_vod.py): HUD gameplay detection, name picking
and set grouping, on synthetic scans (no video or OCR needed)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sv():
    spec = importlib.util.spec_from_file_location("split_vod", ROOT / "scripts" / "split_vod.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["split_vod"] = mod  # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


def _round(start_value: int, secs: int) -> list[int]:
    return [max(start_value - i, 0) for i in range(secs)]


# ── Gameplay from the timer ─────────────────────────────────────────────────

def test_isolated_timer_reads_are_rejected(sv):
    timer = [None] * 30
    timer[5] = 42            # lone false read (lobby / stream scene)
    timer[20] = 99           # two reads that disagree with a countdown
    timer[22] = 99 - 30
    assert not any(sv.valid_timer_seconds(timer))


def test_countdown_and_frozen_timer_are_accepted(sv):
    timer = [None] * 5 + _round(99, 20) + [57] * 6 + [None] * 5
    valid = sv.valid_timer_seconds(timer)
    assert all(valid[5:31])
    assert not any(valid[:5]) and not any(valid[31:])


def test_round_reset_to_99_is_not_needed_to_link_rounds(sv):
    # Two rounds 8 s apart (KO cinematic + next round intro, HUD hidden)
    timer = _round(99, 40) + [None] * 8 + _round(99, 40)
    scan = sv.Scan(timer=timer, hearts=[None] * len(timer))
    segs = sv.build_segments(scan, merge_gap=12, min_hud=15)
    assert len(segs) == 1
    assert segs[0].start == 0 and segs[0].end == len(timer)


def test_short_hud_runs_are_dropped_and_long_gaps_split(sv):
    timer = _round(99, 40) + [None] * 60 + _round(80, 8) + [None] * 60 + _round(99, 30)
    scan = sv.Scan(timer=timer, hearts=[None] * len(timer))
    segs = sv.build_segments(scan, merge_gap=12, min_hud=15)
    assert [(s.start, s.end) for s in segs] == [(0, 40), (168, 198)]


def test_timer_digits_roi_matches_pipeline_config(sv):
    assert sv.timer_digits_roi((110, 180, 920, 1000)) == (115, 170, 920, 1000)


# ── Hearts ──────────────────────────────────────────────────────────────────

def test_stable_hearts_ignores_single_misreads(sv):
    h = [(0, 0)] * 3 + [(1, 0)] + [(0, 0)] * 2 + [None, (0, 1), (0, 1), (1, 1), None, (1, 1)]
    assert [st for _, st in sv.stable_hearts(h)] == [(0, 0), (0, 1), (1, 1)]


def test_game_count_from_hearts(sv):
    seg_a = sv.Segment(0, 100, hearts=[(0, 0), (0, 1), (1, 1), (0, 0), (1, 0)])
    seg_b = sv.Segment(150, 250, hearts=[(0, 0), (0, 1)])
    seg_c = sv.Segment(260, 300, hearts=[(1, 1)])  # resumes the game in progress
    seg_d = sv.Segment(330, 400, hearts=[(1, 0)])  # new game, its 0-0 start missed
    ds = sv.DetectedSet(segments=[seg_a, seg_b, seg_c, seg_d])
    assert ds.game_count == 4


# ── Names ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["Giocatore 1", "Glocatore", "Giocatore z", "Giacatore 2",
                                  "Player 2", "Joueur 1", "Spieler 2", "プレイヤー1"])
def test_player_label_is_recognised(sv, text):
    assert sv.is_label(text)


@pytest.mark.parametrize("text", ["DeeSlyCoops", "Salkazko", "Revill", "Dona", "Happy Meal", "umadacchi"])
def test_names_are_not_labels(sv, text):
    assert not sv.is_label(text)


def _box(x0, x1, y0, y1, text, conf=0.9):
    return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, conf)


def test_pick_name_skips_label_and_icon(sv):
    results = [
        _box(2, 50, 4, 44, "D", 0.83),             # platform icon
        _box(60, 220, 6, 50, "HLScarecrow", 0.98),
        _box(300, 372, 0, 30, "Giocatore", 1.0),  # label sliver
    ]
    assert sv.pick_name(results) == "HLScarecrow"


@pytest.mark.parametrize("text", ["Payer", "Players", "Plaver", "Spieler", "Jogadora"])
def test_label_like_handles_are_kept_as_names(sv, text):
    # On the name line (the label sits above it, outside the quad) a
    # label-like word is a player's handle
    assert sv.pick_name([_box(2, 40, 8, 40, "D", 0.8), _box(50, 200, 8, 42, text)]) == text
    assert sv.pick_name([_box(50, 200, 8, 42, text)]) == text


def test_label_with_its_number_or_above_the_name_is_dropped(sv):
    assert sv.pick_name([_box(10, 120, 0, 20, "Giocatore 1")]) == ""
    assert sv.pick_name([_box(10, 120, 0, 18, "Player"), _box(30, 200, 20, 50, "Revill")]) == "Revill"


def test_pick_name_joins_words_and_drops_icon_token(sv):
    results = [_box(40, 120, 10, 40, "C Dave"), _box(130, 230, 10, 40, "Fennoy")]
    assert sv.pick_name(results) == "Dave Fennoy"
    assert sv.pick_name([_box(40, 200, 10, 40, "CI umadacchi")]) == "umadacchi"


def test_vote_names_prefers_the_consistent_read(sv):
    name, support = sv.vote_names(["Salkazko", "Salkazko", "Salkazk0", "OGlsla", "", "Salkazko"])
    assert name == "Salkazko" and support == 4


def test_name_key_ignores_punctuation_and_ocr_lookalikes(sv):
    assert sv.name_key("Junker_Boy") == sv.name_key("Junker Boy") == sv.name_key("JunkerBoy")
    assert sv.name_key("U.M.A.musume") == sv.name_key("UMAmusume")
    assert sv.name_key("Left0ver") == sv.name_key("LeftOver")
    assert sv.name_key("Revill") == sv.name_key("ReviII")
    assert sv.name_key("Dona") != sv.name_key("Dana")


def test_vote_names_counts_spellings_of_one_name_together(sv):
    # Five reads of one name in two spellings beat four reads of another
    reads = ["Junker Boy"] * 3 + ["Junker_Boy"] * 2 + ["Salkazko"] * 4
    assert sv.vote_names(reads) == ("Junker_Boy", 5)
    # OCR drops the underscore more often than it reads it, and never adds
    # one: a spelling with it wins once a quarter of the reads have it
    assert sv.vote_names(["Junker Boy"] * 6 + ["Junker_Boy"] * 2)[0] == "Junker_Boy"
    assert sv.vote_names(["Junker Boy"] * 7 + ["Junker_Boy"])[0] == "Junker Boy"
    assert sv.vote_names(["UMAmusume"] * 5 + ["U.M.A.musume"] * 2)[0] == "U.M.A.musume"
    # A stray mark at an edge is noise, not part of the name
    assert sv.vote_names(["TimesUp"] * 5 + ["TimesUp."] * 3)[0] == "TimesUp"


def test_clean_text_joins_an_underscore_read_with_a_space(sv):
    assert sv.clean_text("Junker_ Boy") == "Junker_Boy"
    assert sv.clean_text("Junker _Boy ") == "Junker_Boy"
    assert sv.clean_text("Dave  Fennoy") == "Dave Fennoy"
    # also across the boxes pick_name joins
    assert sv.pick_name([_box(40, 120, 10, 40, "Junker_"), _box(130, 200, 10, 40, "Boy")]) == "Junker_Boy"


def test_names_relation(sv):
    assert sv.names_relation(("Dona", "umadacchi"), ("Dona", "umadacch")) == "same"
    assert sv.names_relation(("Dona", "umadacchi"), ("umadacchi", "Dona")) == "same"   # side swap
    assert sv.names_relation(("Revill", "Salkazko"), ("U.M.A.musume", "Salkazko")) == "different"
    assert sv.names_relation(("", ""), ("Revill", "Dona")) == "unknown"
    assert sv.names_relation(("Revill", ""), ("Revill", "Dona")) == "same"
    # A swap with a name unread: not a new set by itself, the look decides
    assert sv.names_relation(("Alice", "Bob"), ("Bob", "")) == "unknown"
    assert sv.names_relation(("Alice", "Bob"), ("Carol", "Alice")) == "different"


def test_ocr_failures_degrade_to_the_look_fallback(sv, monkeypatch):
    import builtins
    import types

    def fresh():
        monkeypatch.setattr(sv, "_ocr_reader", None)
        monkeypatch.setattr(sv, "_ocr_error", None)

    # torch failing to load its DLLs on import
    fresh()
    real_import = builtins.__import__

    def broken_import(name, *a, **k):
        if name == "easyocr":
            raise OSError("[WinError 126] The specified module could not be found: c10.dll")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert sv._get_ocr_reader() is None and "c10.dll" in sv._ocr_error
    assert sv.ocr_name_strip(np.zeros((50, 512), np.uint8)) == ""
    monkeypatch.setattr(builtins, "__import__", real_import)

    # the first run's model download failing without internet
    fresh()
    fake = types.ModuleType("easyocr")

    def no_net(*a, **k):
        raise ConnectionError("<urlopen error [Errno 11001] getaddrinfo failed>")
    fake.Reader = no_net
    monkeypatch.setitem(sys.modules, "easyocr", fake)
    assert sv._get_ocr_reader() is None and "internet" in sv._ocr_error


# ── Set grouping ────────────────────────────────────────────────────────────

def _seg(sv, start, end, p1, p2, hearts=((0, 0),)):
    s = sv.Segment(float(start), float(end), hearts=list(hearts))
    s.p1_name, s.p2_name = p1, p2
    s.p1_votes, s.p2_votes = [p1] * 5, [p2] * 5
    return s


def test_game_breaks_do_not_split_a_set(sv):
    segs = [
        _seg(sv, 0, 300, "Dona", "U.M.A.musume"),
        _seg(sv, 400, 700, "Dona", "U.M.A.musume"),      # 100 s lobby between games
        _seg(sv, 760, 900, "Dona", "U.M.A.musumc"),       # OCR noise
        _seg(sv, 1100, 1400, "Revill", "EneroneJR"),
    ]
    groups = sv.group_segments(segs)
    assert [len(g) for g in groups] == [3, 1]


def test_new_players_start_a_new_set_even_after_a_short_gap(sv):
    segs = [_seg(sv, 0, 300, "Dona", "umadacchi"), _seg(sv, 320, 600, "Dave Fennoy", "Salkazko")]
    assert [len(g) for g in sv.group_segments(segs)] == [1, 1]


def test_segment_opening_mid_game_continues_the_set(sv):
    segs = [
        _seg(sv, 0, 300, "Dona", "umadacchi", hearts=[(0, 0), (0, 1)]),
        _seg(sv, 330, 400, "", "", hearts=[(0, 1), (1, 1)]),   # unreadable names
    ]
    assert [len(g) for g in sv.group_segments(segs)] == [2]


def test_readable_different_names_beat_a_hearts_misread(sv):
    # A new set whose first frames read as 0-1 is still a new set
    segs = [
        _seg(sv, 0, 300, "Dave Fennoy", "Salkazko", hearts=[(0, 0), (0, 1)]),
        _seg(sv, 480, 700, "Dona", "U.M.A.musume", hearts=[(0, 1), (1, 1)]),
    ]
    assert [len(g) for g in sv.group_segments(segs)] == [1, 1]


def test_unreadable_names_fall_back_to_gap(sv):
    segs = [_seg(sv, 0, 300, "", ""), _seg(sv, 330, 600, "", ""), _seg(sv, 900, 1200, "", "")]
    for s in segs:
        s.signature = None
    assert [len(g) for g in sv.group_segments(segs)] == [2, 1]


def _text_strip(text, slant=0.0, width=256):
    import cv2
    img = np.zeros((25, width), np.uint8)
    cv2.putText(img, text, (20, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 230, 2)
    if slant:
        m = np.float32([[1, slant, -slant * 12], [0, 1, 0]])
        img = cv2.warpAffine(img, m, (width, 25))
    return img


def _look_seg(sv, start, p1_text, p2_text, p2_slant=0.2):
    s = sv.Segment(float(start), float(start + 100))
    tags = [sv.binarize_strip(_text_strip(p1_text)), sv.binarize_strip(_text_strip(p2_text, p2_slant, 190))]
    s.signature = sv._signature_of(tags)
    s.tag_shapes = (sv._tag_shape(tags[0]), sv._tag_shape(tags[1]))
    return s


def test_side_swap_by_look_keeps_the_set(sv):
    # No names read; the players swap sides between games (P2's name is
    # drawn slanted).  Straight the tags look different, swapped they match.
    a = _look_seg(sv, 0, "Revill", "Salkazko")
    b = _look_seg(sv, 300, "Salkazko", "Revill")
    c = _look_seg(sv, 600, "HLScarecrow", "Dona")
    assert sv.signature_distance(a.signature, b.signature) > sv.SIGNATURE_SAME
    assert sv.swapped_look_same(a, b)
    assert not sv.swapped_look_same(b, c)
    assert [len(g) for g in sv.group_segments([a, b, c])] == [2, 1]


def test_signature_threshold_follows_the_vods_noise(sv):
    segs = [sv.Segment(0, 10) for _ in range(5)]
    assert sv.signature_threshold(segs) == sv.SIGNATURE_SAME            # no estimate
    for s in segs:
        s.signature_noise = 0.005
    assert sv.signature_threshold(segs) == pytest.approx(0.05)
    for s in segs:
        s.signature_noise = 0.02                                        # a very noisy encode
    assert sv.signature_threshold(segs) == sv.SIGNATURE_SAME_MAX


def test_binarize_is_independent_of_the_encodes_levels(sv):
    strip = _text_strip("Salkazko")
    limited = (16 + strip.astype(np.float32) * 219 / 255).astype(np.uint8)
    dark = (255 * (strip / 255.0) ** 1.6).astype(np.uint8)
    for v in (limited, dark):
        assert np.mean(np.abs(sv.binarize_strip(v) - sv.binarize_strip(strip))) < 0.01


def test_set_names_align_side_swaps(sv):
    segs = [_seg(sv, 0, 300, "Dona", "umadacchi"), _seg(sv, 400, 700, "umadacchi", "Dona")]
    assert sv.set_names(segs) == ("Dona", "umadacchi")
    segs = [_seg(sv, 0, 300, "Dona", "umadacchi"), _seg(sv, 400, 700, "umadacchi", "")]
    segs[1].p2_votes = []
    assert sv.set_names(segs) == ("Dona", "umadacchi")


def _set(sv, start, p1_reads, p2_reads):
    seg = sv.Segment(float(start), float(start + 300))
    seg.p1_votes, seg.p2_votes = list(p1_reads), list(p2_reads)
    seg.p1_name, seg.p2_name = sv.vote_names(seg.p1_votes)[0], sv.vote_names(seg.p2_votes)[0]
    return [seg]


def test_one_player_keeps_one_spelling_across_the_vod(sv):
    # OCR keeps the underscore on one side and drops it on the other: every
    # file gets the spelling read most over the whole VOD
    groups = [
        _set(sv, 0, ["FFShadey"] * 9, ["Junker_Boy"] * 8 + ["Junker Boy"]),
        _set(sv, 1000, ["VituzULF"] * 9, ["Junker_Boy"] * 9),
        _set(sv, 2000, ["Junker Boy"] * 7 + ["Junker_Boy"] * 2, ["TimesUp"] * 9),
    ]
    sets = sv.group_sets(groups)
    assert [ds.p2_name for ds in sets[:2]] + [sets[2].p1_name] == ["Junker_Boy"] * 3
    assert sv.set_label(sets[2]) == "03_Junker_Boy_vs_TimesUp"


def test_a_rare_one_letter_misread_takes_the_common_spelling(sv):
    groups = [
        _set(sv, 0, ["UMAmusume"] * 9, ["RicBionicle"] * 9),
        _set(sv, 1000, ["Dona"] * 9, ["UMAmusumi"] * 6 + ["UMAmusume"] * 3),
        _set(sv, 2000, ["UMAmusume"] * 9, ["Salkazko"] * 9),
    ]
    assert sv.group_sets(groups)[1].p2_name == "UMAmusume"


def test_similar_names_of_two_players_stay_apart(sv):
    # Short names never merge, nor do names that face each other in a set
    groups = [
        _set(sv, 0, ["Dona"] * 9, ["Salkazko"] * 9),
        _set(sv, 1000, ["Dana"] * 2, ["Revill"] * 9),
        _set(sv, 2000, ["Kazuma99"] * 9, ["Kazuma98"] * 3),
    ]
    sets = sv.group_sets(groups)
    assert sets[1].p1_name == "Dana"
    assert (sets[2].p1_name, sets[2].p2_name) == ("Kazuma99", "Kazuma98")


def test_a_long_name_never_merges_into_a_short_one(sv):
    # "Revill" read rarely, "Revil" (a different, 5-letter handle) often:
    # both keys must be long enough to merge.
    groups = [
        _set(sv, 0, ["Revil"] * 9, ["Salkazko"] * 9),
        _set(sv, 1000, ["Revill"] * 4, ["TimesUp"] * 9),
    ]
    assert sv.group_sets(groups)[1].p1_name == "Revill"


def test_misread_chain_ends_at_the_common_spelling(sv):
    # "UMAmusumii" -> "UMAmusumi" -> "UMAmusume": each variant gets the
    # spelling of the name read most, not an intermediate misread.
    groups = [
        _set(sv, 0, ["UMAmusume"] * 15, ["RicBionicle"] * 9),
        _set(sv, 1000, ["Dona"] * 9, ["UMAmusumi"] * 6),
        _set(sv, 2000, ["UMAmusumii"] * 2, ["Salkazko"] * 9),
    ]
    sets = sv.group_sets(groups)
    assert sets[1].p2_name == "UMAmusume"
    assert sets[2].p1_name == "UMAmusume"


# ── Tools ───────────────────────────────────────────────────────────────────

def test_find_tool_honours_env_override(sv, tmp_path, monkeypatch):
    exe = tmp_path / "myffmpeg.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("SPAGHETTI_FFMPEG", str(exe))
    assert sv.find_tool("ffmpeg") == str(exe)


def test_name_strip_is_upscaled_quad(sv):
    q = sv.P1_NAME_QUAD
    x0, y0, x1, y1 = sv.quad_bbox(q)
    crop = np.zeros((y1 - y0, x1 - x0), np.uint8)
    strip = sv.name_strip(crop, q)
    assert strip.shape == (54, 512)


def test_name_quads_match_the_app_defaults(sv):
    # RoiPicker's DEFAULT_ROIS and the splitter must agree
    import re
    src = (ROOT / "app" / "src" / "components" / "RoiPicker.tsx").read_text(encoding="utf-8")
    for key, quad in (("p1_name", sv.P1_NAME_QUAD), ("p2_name", sv.P2_NAME_QUAD)):
        m = re.search(key + r": \{ tl: \[(\d+), (\d+)\], tr: \[(\d+), (\d+)\], br: \[(\d+), (\d+)\], bl: \[(\d+), (\d+)\] \}", src)
        assert m and tuple(int(v) for v in m.groups()) == quad, key
