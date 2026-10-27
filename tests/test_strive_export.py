"""Tests for replanal.strive_export: screen recognition on saved screenshots,
row/date parsing, FFmpeg/encoder selection (mocked) and the export state
machine driven by a simulated game (no game, no keys, no FFmpeg)."""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from replanal.strive_export import capture, exporter, rows, screens
from replanal.strive_export.exporter import (Cancel, ExportError, Exporter, Options, UnsafeKey,
                                             key_allowed)
from replanal.strive_export.rows import RowInfo
from replanal.strive_export.screens import Classifier, RefSet, Screen

REFS = Path(screens.__file__).parent / "refs" / "it"
DATA = Path(__file__).parent / "data" / "strive_export"


def load(p) -> np.ndarray:
    img = cv2.imread(str(p))
    assert img is not None, p
    return img


def classify(img, lang="auto"):
    return Classifier(RefSet([]), lang=lang).classify(img)


# ── screen recognition on the shipped references ─────────────────────────────

@pytest.mark.parametrize("name, expect", [
    ("title", dict(name="title")),
    ("main", dict(name="main", tab=0, ready=False)),
    ("collection", dict(name="main", tab=6, ready=True)),
    ("saved_list", dict(name="saved_list", tab=2, row_y=152)),
    ("replay_menu", dict(name="replay_menu", item=0, items=5)),
    ("replay_end", dict(name="replay_end", item=0, items=3)),
    ("server", dict(name="server")),
])
def test_reference_screens(name, expect):
    s = classify(load(REFS / f"{name}.jpg"))
    for k, v in expect.items():
        assert getattr(s, k) == v, (name, s)


def test_auto_language_detects_italian():
    c = Classifier(RefSet([]), lang="auto")
    assert c.classify(load(REFS / "saved_list.jpg")).lang == "it"
    assert c.lang == "it"


def test_live_1080p_captures():
    s = classify(load(DATA / "live_saved_list_1080.jpg"))
    assert (s.name, s.tab, s.row_y) == ("saved_list", 2, 371)   # scrolled list, ranked rows
    s = classify(load(DATA / "live_replay_end_1080.jpg"))       # over a different stage
    assert (s.name, s.item, s.items) == ("replay_end", 0, 3)


def test_live_letterboxed_window():
    # a 1196x1076 window: the game draws 16:9 between black bars, and its UI
    # text comes out ~4 % narrower than in the 1080p references
    img = load(DATA / "live_replay_menu_1196x1076.jpg")
    for lang in ("auto", "it"):
        s = classify(img, lang=lang)
        assert (s.name, s.item, s.items) == ("replay_menu", 0, 5), (lang, s)


@pytest.mark.parametrize("name, allowed", [
    ("neg_gameplay_a", {"unknown"}),
    ("neg_gameplay_b", {"unknown"}),
    ("neg_charsel", {"unknown"}),
    ("neg_side", {"unknown"}),
    ("neg_training", {"unknown"}),
    ("neg_filter_menu", {"unknown"}),         # 13-item filter pop-up
    ("neg_search_menu", {"unknown"}),         # 3-item pop-up over the replay room: not the end menu
    ("neg_search_end", {"unknown"}),          # online replay's 5-item end menu over the game
    ("neg_search_list", {"room"}),            # replay room, search tab
    ("neg_search_cond", {"room", "unknown"}),
    ("neg_dojo", {"main"}),                   # main menu, DOJO tab
])
def test_other_screens_are_never_actionable(name, allowed):
    s = classify(load(DATA / f"{name}.jpg"))
    assert s.name in allowed, s
    assert not key_allowed(s, "u") or s.name == "saved_list"
    if s.name == "main":
        assert s.tab != screens.COLLECTION_TAB and not s.ready
    if s.name == "room":
        assert s.tab != screens.SAVED_TAB


def test_empty_capture_is_unknown():
    c = Classifier(RefSet([]))
    assert c.classify(None).name == "unknown"
    assert c.classify(np.zeros((0, 0, 3), np.uint8)).name == "unknown"
    assert c.classify(np.zeros((540, 960, 3), np.uint8)).name == "unknown"


# ── perturbations ────────────────────────────────────────────────────────────

def _jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


PERTURB = {
    "darker": lambda im: np.clip(im.astype(np.float32) * 0.9, 0, 255).astype(np.uint8),
    "brighter": lambda im: np.clip(im.astype(np.float32) * 1.08 + 4, 0, 255).astype(np.uint8),
    "noise": lambda im: np.clip(im.astype(np.float32) + np.random.default_rng(1).normal(0, 4, im.shape),
                                0, 255).astype(np.uint8),
    "jpeg60": lambda im: _jpeg(im, 60),
    "720p_window": lambda im: cv2.resize(cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA), (1920, 1080)),
    "1440p_window": lambda im: cv2.resize(im, (2560, 1440), interpolation=cv2.INTER_CUBIC),
    "blur": lambda im: cv2.GaussianBlur(im, (3, 3), 0),
}


@pytest.mark.parametrize("perturb", sorted(PERTURB))
@pytest.mark.parametrize("name", ["title", "main", "collection", "saved_list", "replay_menu", "replay_end"])
def test_classification_survives_perturbations(name, perturb):
    base = load(REFS / f"{name}.jpg")
    want = classify(base)
    big = cv2.resize(base, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    got = classify(PERTURB[perturb](big))
    assert (got.name, got.tab, got.item, got.items, got.ready) == \
        (want.name, want.tab, want.item, want.items, want.ready), (name, perturb, got, got.scores)


# ── language independence: replace the text, keep the layout ────────────────

def _retext(img, box, text):
    """Paint over a text region with its own background colour and write
    other words there (what another UI language would look like)."""
    out = img.copy()
    x0, y0, x1, y1 = (int(v) for v in box)
    bg = np.median(out[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    out[y0:y1, x0:x1] = bg
    cv2.putText(out, text, (x0 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return out


FOREIGN = {
    "title": [((250, 372, 710, 412), "Press any button   Exit game")],
    "main": [((60, 478, 900, 499), "NETWORK   3V3   LOCAL   DOJO   STORY   EDIT   COLLECTION   SHOP   EXIT")],
    "collection": [((60, 478, 900, 499), "NETWORK   3V3   LOCAL   DOJO   STORY   EDIT   COLLECTION   SHOP   EXIT"),
                   ((660, 284, 760, 300), "REPLAY")],
    "saved_list": [((60, 104, 340, 146), "Saved Replays  Sort by match date")],
    "replay_menu": [((250, 224, 600, 243), "View Replay"), ((250, 250, 600, 270), "Lock Replay"),
                    ((250, 276, 600, 296), "Edit Comment"), ((250, 303, 600, 323), "Delete Replay")],
    "replay_end": [((250, 250, 600, 269), "Return to Replay List"), ((250, 277, 600, 296), "Replay again"),
                   ((250, 303, 600, 323), "Main Menu")],
}


@pytest.mark.parametrize("name", sorted(FOREIGN))
def test_other_language_text_is_recognised_by_layout(name):
    img = load(REFS / f"{name}.jpg")
    for box, text in FOREIGN[name]:
        img = _retext(img, box, text)
    want = classify(load(REFS / f"{name}.jpg"))
    got = classify(img, lang="auto")
    assert (got.name, got.tab, got.item, got.items) == (want.name, want.tab, want.item, want.items), got
    assert got.lang is None   # recognised without any language's text


@pytest.mark.parametrize("name", ["title", "saved_list", "replay_menu", "replay_end"])
def test_known_language_text_check_rejects_other_labels(name):
    """With --lang it the labels must match the Italian references too."""
    img = load(REFS / f"{name}.jpg")
    for box, text in FOREIGN[name]:
        img = _retext(img, box, text)
    got = classify(img, lang="it")
    assert got.name not in (name, "saved_list"), got


def _move_menu_highlight(img, to_item, first_bar=222, pitch=26.5, n=5):
    """Repaint a pop-up so another item carries the red highlight."""
    out = img.copy()
    grey = np.array([56, 56, 56], np.uint8)
    red = np.array([0, 0, 192], np.uint8)
    for i in range(n):
        y0 = int(round(first_bar + pitch * i))
        out[y0 + 1:y0 + 21, 247:704] = red if i == to_item else grey
    return out


@pytest.mark.parametrize("item", [1, 2, 3, 4])
def test_entry_menu_with_highlight_off_view_never_confirms(item):
    img = _move_menu_highlight(load(REFS / "replay_menu.jpg"), item)
    s = classify(img, lang="auto")
    assert s.name in ("replay_menu", "unknown")
    if s.name == "replay_menu":
        assert s.item == item
    assert not key_allowed(s, "u")


def test_ranked_tab_never_confirms():
    s = classify(load(REFS / "main.jpg"))
    assert s.tab == screens.RANKED_TAB
    assert not key_allowed(s, "u")
    assert key_allowed(s, "d")


def test_key_allow_list():
    assert key_allowed(Screen("title"), "u")
    assert not key_allowed(Screen("title"), "enter")
    assert not key_allowed(Screen("main", tab=6, ready=False), "u")
    assert key_allowed(Screen("main", tab=6, ready=True), "u")
    for t in range(9):
        if t != 6:
            assert not key_allowed(Screen("main", tab=t, ready=True), "u")
    assert key_allowed(Screen("room", tab=0), "l") and not key_allowed(Screen("room", tab=0), "u")
    assert key_allowed(Screen("saved_list", tab=2, row_y=152), "u")
    assert not key_allowed(Screen("saved_list", tab=2, row_y=152), "m")   # sort order
    assert key_allowed(Screen("replay_menu", item=0, items=5), "u")
    for k in "ws":   # never move inside the entry menu (Delete is three down)
        assert not key_allowed(Screen("replay_menu", item=0, items=5), k)
    assert not key_allowed(Screen("replay_menu", item=0, items=3), "u")
    assert not key_allowed(Screen("replay_end", item=1, items=3), "u")
    for k in ("u", "enter", "esc", "s", "w"):
        assert not key_allowed(Screen("unknown"), k)
        assert not key_allowed(Screen("server"), k)


# ── rows: dates, names, file names ───────────────────────────────────────────

@pytest.mark.parametrize("text, order, want, amb", [
    ("Data dell'incontro: 20/10/2026", "DMY", dt.date(2026, 10, 20), False),
    ("Match Date: 10/20/2026", "DMY", dt.date(2026, 10, 20), False),   # 20 can only be the day
    ("Match Date: 10/03/2026", "MDY", dt.date(2026, 10, 3), True),
    ("10/03/2026", "DMY", dt.date(2026, 3, 10), True),
    ("2026/10/20", "DMY", dt.date(2026, 10, 20), False),
    ("2026-01-05 12:00:00", "MDY", dt.date(2026, 1, 5), False),
    ("05.05.2026", "DMY", dt.date(2026, 5, 5), False),
    ("20810/2026", "DMY", None, False),     # OCR glued a digit in: no date
    ("31/02/2026", "DMY", None, False),
    ("20/10/1999", "DMY", None, False),     # before Strive existed
    ("Ver. 5.03", "DMY", None, False),
])
def test_parse_date(text, order, want, amb):
    d, a = rows.parse_date(text, order)
    assert d == want
    if want is not None:
        assert a == amb


@pytest.mark.parametrize("text, want", [
    ("Data dell'incontro: 20/10/2026 22:10:28 Ver. 5.03", (dt.date(2026, 10, 20), dt.time(22, 10, 28))),
    ("D Data dell'incontro: 20/10/2026 00.23.27 Ver. 5.03", (dt.date(2026, 10, 20), dt.time(0, 23, 27))),
    ("Data dell'incontro: 20/10/2026 21 .55.52 Ver5,03", (dt.date(2026, 10, 20), dt.time(21, 55, 52))),
    ("20/10/2026 22,06.58", (dt.date(2026, 10, 20), dt.time(22, 6, 58))),
    ("PCData dell incontro: 27/10/2026 005538 Ver5.03", (dt.date(2026, 10, 27), dt.time(0, 55, 38))),
    ("27/10/2026 Ver 503000", (dt.date(2026, 10, 27), None)),   # six digits but not a time of day
    ("nothing here", (None, None)),
])
def test_parse_row_text(text, want):
    d, t, _ = rows.parse_row_text(text, "DMY")
    assert (d, t) == want


def test_parse_since():
    today = dt.date(2026, 10, 23)
    assert rows.parse_since("today", today) == today
    assert rows.parse_since("Yesterday", today) == dt.date(2026, 10, 22)
    assert rows.parse_since("2026-10-01") == dt.date(2026, 10, 1)
    with pytest.raises(ValueError):
        rows.parse_since("23/10/2026")


def test_names_and_file_names():
    assert rows.clean_name(["Ci TimesUp"]) == "TimesUp"
    assert rows.clean_name(["CI", "JArusty"]) == "JArusty"
    assert rows.clean_name(["Wowwie Zowwie"]) == "Wowwie Zowwie"
    assert rows.clean_name([]) is None
    info = RowInfo(dt.date(2026, 10, 20), dt.time(22, 10, 28), "Times/Up?", "J4 rusty")
    assert rows.filename_for(info, 3) == "2026-10-20_22-10-28_TimesUp_vs_J4_rusty.mp4"
    assert rows.filename_for(RowInfo(dt.date(2026, 10, 20), dt.time(1, 2, 3)), 0) == "2026-10-20_01-02-03.mp4"
    assert rows.filename_for(RowInfo(dt.date(2026, 10, 20)), 7) == "replay_007.mp4"
    assert rows.filename_for(None, 12) == "replay_012.mp4"
    assert rows.parse_count("100 /100") == 100
    assert rows.parse_count("ti 37 / 100") == 37
    assert rows.parse_count("1007100") is None
    assert rows.parse_count("") is None


def test_ocr_reads_a_live_row():
    reader = rows.RowReader()
    if not reader.available:
        pytest.skip(f"easyocr unavailable: {reader.error}")
    img = load(DATA / "live_saved_list_1080.jpg")
    info = reader.read_row(img, 371)
    assert (info.date, info.time) == (dt.date(2026, 10, 20), dt.time(0, 23, 27))
    assert (info.p1, info.p2) == ("TimesUp", "Daiko")


# ── FFmpeg / encoder selection (mocked) ──────────────────────────────────────

def _info(path, filters=(), encoders=()):
    return capture.FfmpegInfo(path, "ffmpeg version x", set(filters), set(encoders))


def test_choose_ffmpeg_prefers_window_capture():
    infos = {"old": _info("old", ["ddagrab"]), "new": _info("new", ["ddagrab", "gfxcapture"])}
    ch = capture.choose_ffmpeg(["old", "new"], probe_fn=infos.get)
    assert (ch.ffmpeg.path, ch.method, ch.warning) == ("new", "gfxcapture", None)


def test_choose_ffmpeg_falls_back_to_ddagrab_with_a_warning():
    ch = capture.choose_ffmpeg(["old"], probe_fn=lambda p: _info(p, ["ddagrab"]))
    assert ch.method == "ddagrab" and "visible" in ch.warning
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        capture.choose_ffmpeg([], probe_fn=lambda p: _info(p))
    with pytest.raises(RuntimeError, match="capture the screen"):
        capture.choose_ffmpeg(["x"], probe_fn=lambda p: _info(p, ["scale"]))


def test_probe_parses_ffmpeg_listings():
    out = {
        "-version": b"ffmpeg version 9.0.2-essentials_build Copyright\n",
        "-filters": b" .. ddagrab           |->V       Grab\n .. gfxcapture        |->V       Capture\n",
        "-encoders": b" V....D h264_mf              H264 via MediaFoundation\n V....D libx264  x\n",
    }
    run = lambda args, **kw: SimpleNamespace(stdout=out[args[-1]], returncode=0)
    info = capture.probe("ff", run=run)
    assert info.gfxcapture and info.ddagrab
    assert info.encoders == {"h264_mf", "libx264"}
    assert capture.major_version(info) == 9


def _fake_run(ok):
    calls = []

    def run(args, **kw):
        enc = args[args.index("-c:v") + 1]
        calls.append(enc)
        return SimpleNamespace(returncode=0 if enc in ok else 1, stdout=b"", stderr=b"")
    return run, calls


def test_encoder_order_nvenc_then_mf_then_x264():
    info = _info("ff", encoders=["h264_nvenc", "h264_mf", "libx264"])
    run, calls = _fake_run({"h264_nvenc", "h264_mf", "libx264"})
    assert capture.pick_encoder(info, run=run)[0] == "h264_nvenc"
    run, calls = _fake_run({"h264_mf", "libx264"})       # e.g. a driver too old for NVENC
    assert capture.pick_encoder(info, run=run)[0] == "h264_mf"
    assert calls == ["h264_nvenc", "h264_mf"]
    run, calls = _fake_run({"libx264"})
    assert capture.pick_encoder(info, run=run)[0] == "libx264"


def test_encoder_not_built_in_is_not_tried_and_none_working_raises():
    run, calls = _fake_run({"libx264"})
    assert capture.pick_encoder(_info("ff", encoders=["libx264"]), run=run)[0] == "libx264"
    assert calls == ["libx264"]
    run, _ = _fake_run(set())
    with pytest.raises(RuntimeError, match="No H.264 encoder"):
        capture.pick_encoder(_info("ff", encoders=["h264_mf", "libx264"]), run=run)
    run, calls = _fake_run({"h264_mf", "libx264"})
    assert capture.pick_encoder(_info("ff", encoders=["h264_mf", "libx264"]), run=run, prefer="libx264")[0] == "libx264"


def test_source_filters():
    g = capture.source_filter("gfxcapture", 60)
    assert g.startswith("gfxcapture=window_exe=GGST-Win64-Shipping") and g.endswith("format=nv12")
    d = capture.source_filter("ddagrab", 60, rect=(100, 50, 1920, 1080), monitor=(1, 100, 50))
    assert "output_idx=1" in d and "offset_x=100" in d and "video_size=1920x1080" in d
    assert capture.source_filter("gfxcapture", 10, still=True).endswith("format=bgra")


def test_window_on_monitor():
    from replanal.strive_export.winio import window_on_monitor
    mons = [(0, 0, 2560, 1440), (2560, 0, 4480, 1080)]
    assert window_on_monitor((2560, 0, 1920, 1080), mons) == (1, 0, 0)
    assert window_on_monitor((320, 180, 1920, 1080), mons) == (0, 320, 180)


# ── state machine on a simulated game ────────────────────────────────────────

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeRec:
    def __init__(self, game, path):
        self.game, self.path, self.stopped = game, path, False

    def alive(self):
        return not self.stopped

    def stop(self):
        if not self.stopped:
            self.stopped = True
            Path(self.path).write_bytes(b"mp4")
            self.game.log.append(("stop_rec", Path(self.path).name))
        return 0, ""


class SimGame:
    """A small model of Strive's menus. Records every key with the true state
    it landed on, and flags the forbidden ones."""

    def __init__(self, n=5, state="title", cursor=0, play_grabs=4, menu_item=0, wrap=False,
                 attract_grabs=0, swallow_list_u=0, dates=None, on_play=None, ready_delay=0):
        self.n, self.state, self.cursor = n, state, cursor
        self.tab, self.room_tab = 0, 0
        self.play_grabs, self.play_left = play_grabs, 0
        self.menu_item, self.wrap = menu_item, wrap
        self.attract_left = attract_grabs
        self.swallow_list_u = swallow_list_u
        self.on_play = on_play
        self.ready_delay = ready_delay
        self.ready_left = 0
        self.dates = dates or [dt.datetime(2026, 10, 20, 22, 10) - dt.timedelta(hours=7 * i) for i in range(n)]
        self.keys, self.log, self.violations = [], [], []

    # what the screen shows
    def screen(self) -> Screen:
        st = self.state
        if st == "title":
            return Screen("title")
        if st == "main":
            return Screen("main", tab=self.tab, ready=self.tab == 6 and self.ready_left <= 0)
        if st == "room":
            if self.room_tab == 2:
                return Screen("saved_list", tab=2, row_y=152 + 46 * min(self.cursor, 6))
            return Screen("room", tab=self.room_tab)
        if st == "entry_menu":
            return Screen("replay_menu", item=self.menu_item, items=5)
        if st == "end":
            return Screen("replay_end", item=0, items=3)
        return Screen("unknown")   # attract video, loading, replay playing

    def grab(self):
        if self.state == "attract":
            self.attract_left -= 1
            if self.attract_left <= 0:
                self.state = "title"
        elif self.state == "playing":
            self.play_left -= 1
            if self.play_left <= 0:
                self.state = "end"
        if self.state == "main" and self.tab == 6:
            self.ready_left -= 1
        img = np.zeros((540, 960, 3), np.uint8)
        x0, y0, x1, y1 = screens.LIST_BOX
        img[y0:y1, x0:x1] = 20 + (self.cursor * 37) % 200
        return img

    def press(self, key):
        st, scr = self.state, self.screen()
        self.keys.append(key)
        self.log.append(("key", key, str(scr)))
        if st in ("attract", "playing"):
            self.violations.append(f"{key} on {st}")
        if st == "main" and key == "u" and self.tab != 6:
            self.violations.append(f"confirm on main tab {self.tab}")
        if st == "entry_menu" and key == "u" and self.menu_item != 0:
            self.violations.append(f"confirm on entry menu item {self.menu_item} (3 = DELETE)")
        if st == "entry_menu" and key != "u":
            self.violations.append(f"{key} in the entry menu")
        # transitions
        if st == "title" and key == "u":
            self.state = "main"
        elif st == "main":
            if key == "d":
                self.tab = min(8, self.tab + 1)
                self.ready_left = self.ready_delay
            elif key == "a":
                self.tab = max(0, self.tab - 1)
            elif key == "u" and self.tab == 6:
                self.state, self.room_tab = "room", 0
        elif st == "room" and self.room_tab != 2:
            if key == "l":
                self.room_tab += 1
            elif key == "k":
                self.room_tab -= 1
        elif st == "room":
            if key == "u":
                if self.swallow_list_u > 0:
                    self.swallow_list_u -= 1
                else:
                    self.state = "entry_menu"
            elif key == "s":
                if self.cursor < self.n - 1:
                    self.cursor += 1
                elif self.wrap:
                    self.cursor = 0
            elif key == "w":
                if self.cursor > 0:
                    self.cursor -= 1
                elif self.wrap:
                    self.cursor = self.n - 1
        elif st == "entry_menu" and key == "u" and self.menu_item == 0:
            self.state, self.play_left = "playing", self.play_grabs
            if self.on_play:
                self.on_play(self)
        elif st == "end" and key == "u":
            self.state = "room"

    def start_recording(self, path):
        self.log.append(("start_rec", Path(path).name))
        return FakeRec(self, path)


class SimClassifier:
    def __init__(self, game):
        self.game = game

    def classify(self, img):
        return self.game.screen() if img is not None else Screen("unknown")


class SimReader:
    def __init__(self, game, garble=()):
        self.game, self.garble = game, set(garble)
        self.reads = 0

    def read_row(self, full, row_y):
        self.reads += 1
        i = self.game.cursor
        if i in self.garble:
            return RowInfo(None, None, None, None, raw="???")
        d = self.game.dates[i]
        return RowInfo(d.date(), d.time(), f"P{i}a", f"P{i}b")

    def read_count(self, full):
        return self.game.n


def run_sim(game, tmp_path, reader=True, events=None, cancel=None, **opt):
    opts = Options(out_dir=str(tmp_path / "out"), nav_timeout=600, poll=1.0, **opt)
    ev = events if events is not None else []
    ex = Exporter(game, SimClassifier(game), SimReader(game) if reader is True else reader, opts,
                  emit=lambda e, **kw: ev.append((e, kw)), clock=FakeClock(), cancel=cancel)
    return ex.run(), ev


def test_full_export_from_the_title(tmp_path):
    g = SimGame(n=5)
    summary, ev = run_sim(g, tmp_path, count=3)
    assert g.violations == []
    assert summary["reason"] == "count reached"
    names = [Path(p).name for p in summary["exported"]]
    assert names == ["2026-10-20_22-10-00_P0a_vs_P0b.mp4", "2026-10-20_15-10-00_P1a_vs_P1b.mp4",
                     "2026-10-20_08-10-00_P2a_vs_P2b.mp4"]
    # title, 6 x right to Collection, confirm, 2 tabs right, then per replay:
    # open entry, view, back to list, and a step down between replays.
    assert g.keys == ["u"] + ["d"] * 6 + ["u"] + ["l", "l"] + \
        ["u", "u", "u", "s", "u", "u", "u", "s", "u", "u", "u"]
    man = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert [e["file"] for e in man["entries"]] == names
    assert man["entries"][0]["p1"] == "P0a" and man["entries"][0]["partial"] is False
    # recording starts right after "View replay" and stops on the end menu
    recs = [x for x in g.log if x[0] in ("start_rec", "stop_rec")]
    assert recs[:2] == [("start_rec", names[0]), ("stop_rec", names[0])]
    assert [e for e, _ in ev].count("recorded") == 3


def test_waits_through_the_attract_video_without_pressing(tmp_path):
    g = SimGame(n=2, state="attract", attract_grabs=6)
    summary, _ = run_sim(g, tmp_path, count=1)
    assert g.violations == [] and summary["reason"] == "count reached"
    assert g.keys[0] == "u"


def test_waits_for_the_collection_submenu(tmp_path):
    g = SimGame(n=2, ready_delay=3)
    run_sim(g, tmp_path, count=1)
    assert g.violations == []


def test_never_confirms_delete(tmp_path):
    g = SimGame(n=3, menu_item=3)   # the entry menu opens with "Delete replay" highlighted
    with pytest.raises(ExportError, match="View replay"):
        run_sim(g, tmp_path, count=1)
    assert g.violations == []
    assert g.state == "entry_menu"
    assert not any(x[0] == "start_rec" for x in g.log)


def test_entry_menu_open_at_start_is_left_alone(tmp_path):
    g = SimGame(n=3, state="entry_menu")
    with pytest.raises(ExportError, match="menu is open"):
        run_sim(g, tmp_path, count=1)
    assert g.keys == []


def test_swallowed_open_is_retried_while_the_list_shows(tmp_path):
    g = SimGame(n=2, state="room", swallow_list_u=2)
    g.room_tab = 2
    summary, _ = run_sim(g, tmp_path, count=1)
    assert g.violations == [] and len(summary["exported"]) == 1


def test_dry_run_lists_without_opening(tmp_path):
    g = SimGame(n=4)
    summary, ev = run_sim(g, tmp_path, dry_run=True)
    assert summary["reason"] == "end of the list" and summary["listed"] == 4
    assert summary["exported"] == []
    after_list = g.keys[g.keys.index("l") + 2:]
    assert set(after_list) == {"s"}
    assert [kw["index"] for e, kw in ev if e == "entry"] == [0, 1, 2, 3]


def test_end_of_list_wrapping_stops(tmp_path):
    g = SimGame(n=3, wrap=True)
    summary, _ = run_sim(g, tmp_path, dry_run=True)
    assert summary["reason"] == "end of the list" and summary["listed"] == 3


def test_since_stops_at_older_replays(tmp_path):
    days = [dt.datetime(2026, 10, 23, 1, 0), dt.datetime(2026, 10, 23, 0, 30),
            dt.datetime(2026, 10, 22, 23, 50), dt.datetime(2026, 10, 22, 23, 0)]
    g = SimGame(n=4, dates=days)
    summary, _ = run_sim(g, tmp_path, since=dt.date(2026, 10, 23))
    assert summary["reason"].startswith("reached replays older")
    assert len(summary["exported"]) == 2


def test_since_keeps_going_when_a_date_is_unreadable(tmp_path):
    days = [dt.datetime(2026, 10, 23, 1, 0), dt.datetime(2026, 10, 23, 0, 30),
            dt.datetime(2026, 10, 22, 23, 50)]
    g = SimGame(n=3, dates=days)
    summary, ev = run_sim(g, tmp_path, reader=SimReader(g, garble={1}), since=dt.date(2026, 10, 23))
    assert len(summary["exported"]) == 2
    assert Path(summary["exported"][1]).name == "replay_001.mp4"
    assert any(e == "warning" and "could not read the date" in kw["message"] for e, kw in ev)


def test_start_offset_and_skip_existing(tmp_path):
    g = SimGame(n=5)
    out = tmp_path / "out"
    out.mkdir()
    (out / "2026-10-20_08-10-00_P2a_vs_P2b.mp4").write_bytes(b"old")
    summary, ev = run_sim(g, tmp_path, start=1, count=3)
    assert [Path(p).name for p in summary["exported"]] == ["2026-10-20_15-10-00_P1a_vs_P1b.mp4",
                                                          "2026-10-20_01-10-00_P3a_vs_P3b.mp4"]
    assert [Path(p).name for p in summary["skipped"]] == ["2026-10-20_08-10-00_P2a_vs_P2b.mp4"]
    assert g.violations == []


def test_started_on_the_list_goes_to_the_top_first(tmp_path):
    g = SimGame(n=5, state="room", cursor=3)
    g.room_tab = 2
    summary, ev = run_sim(g, tmp_path, count=1)
    assert g.keys[:3] == ["w", "w", "w"]
    assert Path(summary["exported"][0]).name.startswith("2026-10-20_22-10-00")


def test_soft_cancel_finishes_the_current_replay(tmp_path):
    cancel = Cancel()
    g = SimGame(n=5, on_play=lambda game: cancel.request(now=False))
    summary, _ = run_sim(g, tmp_path, cancel=cancel, count=5)
    assert summary["reason"] == "cancelled"
    assert len(summary["exported"]) == 1
    assert g.state == "room"      # back on the list
    assert g.violations == []


def test_hard_stop_finalises_the_recording(tmp_path):
    cancel = Cancel()
    g = SimGame(n=5, play_grabs=50, on_play=lambda game: cancel.request(now=True))
    summary, _ = run_sim(g, tmp_path, cancel=cancel, count=5)
    assert summary["reason"] == "stopped"
    man = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert man["entries"][0]["partial"] is True
    assert g.log[-1][0] == "stop_rec"
    assert g.keys[-1] == "u" and g.state == "playing"   # last key was 'View replay'; nothing after


def test_a_stopped_replay_is_recorded_again(tmp_path):
    cancel = Cancel()
    g = SimGame(n=5, play_grabs=50, on_play=lambda game: cancel.request(now=True))
    first, _ = run_sim(g, tmp_path, cancel=cancel, count=5)
    g2 = SimGame(n=5)
    summary, _ = run_sim(g2, tmp_path, count=1)
    assert summary["exported"] == first["exported"] and summary["skipped"] == []
    man = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert [e["partial"] for e in man["entries"]] == [False]


def test_manifest_merges_with_an_earlier_run(tmp_path):
    g = SimGame(n=5)
    run_sim(g, tmp_path, count=1)
    g2 = SimGame(n=5)
    run_sim(g2, tmp_path, start=1, count=1)
    man = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert len(man["entries"]) == 2


def test_random_screens_never_get_a_forbidden_key(tmp_path):
    """Fuzz: the game shows random screens (dangerous ones included) no matter
    what is pressed; every key that goes out must be allowed on the screen
    that was just recognised."""
    pool = [Screen("title"), Screen("server"), Screen("unknown"),
            Screen("main", tab=0), Screen("main", tab=8), Screen("main", tab=6, ready=False),
            Screen("main", tab=6, ready=True), Screen("room", tab=0), Screen("room", tab=4),
            Screen("saved_list", tab=2, row_y=152), Screen("saved_list", tab=2, row_y=198),
            Screen("replay_menu", item=0, items=5), Screen("replay_menu", item=3, items=5),
            Screen("replay_menu", item=4, items=5), Screen("replay_end", item=0, items=3),
            Screen("replay_end", item=1, items=3), Screen("unknown", item=0, items=3)]

    class RandomGame:
        def __init__(self, seed):
            self.rng = random.Random(seed)
            self.cur = Screen("unknown")
            self.bad = []

        def grab(self):
            self.cur = self.rng.choice(pool)
            return np.full((540, 960, 3), self.rng.randrange(255), np.uint8)

        def press(self, key):
            if not key_allowed(self.cur, key):
                self.bad.append((str(self.cur), key))
            if self.cur.name == "replay_menu" and key == "u" and self.cur.item != 0:
                self.bad.append(("DELETE", key))
            if self.cur.name == "main" and key == "u" and self.cur.tab == 0:
                self.bad.append(("RANKED", key))

        def start_recording(self, path):
            return FakeRec(SimpleNamespace(log=[]), path)

    for seed in range(60):
        g = RandomGame(seed)
        cls = SimpleNamespace(classify=lambda img, g=g: g.cur)
        opts = Options(out_dir=str(tmp_path / f"o{seed}"), count=3, nav_timeout=120, poll=1.0, max_seconds=60)
        ex = Exporter(g, cls, None, opts, clock=FakeClock())
        try:
            ex.run()
        except ExportError:
            pass
        assert g.bad == [], (seed, g.bad)


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_defaults_to_export():
    from replanal.strive_export.__main__ import parse_args
    a = parse_args(["--out", "x", "--count", "3"])
    assert (a.cmd, a.count, a.all) == ("export", 3, False)
    a = parse_args(["export", "--out", "x", "--since", "today"])
    assert a.count is None and a.since == "today"
    with pytest.raises(SystemExit):
        parse_args(["--out", "x"])                       # how many?
    with pytest.raises(SystemExit):
        parse_args(["--out", "x", "--count", "2", "--all"])
    assert parse_args(["calibrate", "--lang", "en"]).cmd == "calibrate"


def test_cli_fails_clearly_off_windows(monkeypatch, capsys):
    from replanal.strive_export import winio
    from replanal.strive_export.__main__ import main
    monkeypatch.setattr(winio.sys, "platform", "linux")
    assert main(["--out", "x", "--count", "1", "--json-progress"]) == 2
    line = capsys.readouterr().out.strip().splitlines()[-1]
    msg = json.loads(line)
    assert msg["event"] == "error" and "Windows only" in msg["message"]


# ── key input ────────────────────────────────────────────────────────────────

class FakeUser32:
    """Records what winio asks of user32; nothing reaches a real window."""

    def __init__(self, fg=0):
        self.calls, self.fg = [], fg

    def MapVirtualKeyW(self, scan, kind):
        assert kind == 3
        return {0x16: 0x55, 0x1F: 0x53, 0xE048: 0x26}[scan]   # U, S, Up

    def PostMessageW(self, hwnd, msg, vk, lp):
        self.calls.append(("post", hwnd, msg, vk, lp))
        return 1

    def SendInput(self, *a):
        self.calls.append(("sendinput",))

    def SetForegroundWindow(self, hwnd):
        self.calls.append(("setfg", hwnd))

    def GetForegroundWindow(self):
        return self.fg

    def IsIconic(self, hwnd):
        return False


def _fake_winio(monkeypatch, fg=0):
    from replanal.strive_export import winio
    u = FakeUser32(fg)
    monkeypatch.setattr(winio, "_u32", lambda: u)
    monkeypatch.setattr(winio, "find_window", lambda: 42)
    monkeypatch.setattr(winio.time, "sleep", lambda s: None)
    return winio, u


def test_background_keys_are_posted_to_the_game_window_only(monkeypatch):
    winio, u = _fake_winio(monkeypatch, fg=7)   # another app is in front
    winio.press("u s:100 up", verify_focus=True)
    assert all(c[0] == "post" and c[1] == 42 for c in u.calls), u.calls   # no SendInput, no focus
    msgs = [(m, vk, lp) for _, _, m, vk, lp in u.calls]
    assert msgs == [
        (0x100, 0x55, 0x00160001), (0x101, 0x55, 0xC0160001),
        (0x100, 0x53, 0x001F0001), (0x101, 0x53, 0xC01F0001),
        (0x100, 0x26, 0x01480001), (0x101, 0x26, 0xC1480001),   # extended: bit 24
    ]


def test_background_keys_need_the_game_window(monkeypatch):
    winio, u = _fake_winio(monkeypatch)
    monkeypatch.setattr(winio, "find_window", lambda: 0)
    with pytest.raises(winio.FocusError):
        winio.press("u")
    assert u.calls == []


def test_foreground_keys_refuse_another_window_in_front(monkeypatch):
    winio, u = _fake_winio(monkeypatch, fg=7)
    monkeypatch.setattr(winio, "send_scan", lambda c, up: u.calls.append(("scan", c, up)))
    with pytest.raises(winio.FocusError):
        winio.press("u", verify_focus=True, background=False)
    keys = lambda: [c for c in u.calls if c[0] == "post" or (c[0] == "scan" and c[1] != winio.ALT)]
    assert keys() == []          # only focus()'s Alt taps went out
    u.fg = 42
    winio.press("u", verify_focus=True, background=False)
    assert keys() == [("scan", 0x16, False), ("scan", 0x16, True)]


def test_live_game_posts_keys_unless_foreground(monkeypatch):
    from replanal.strive_export.__main__ import LiveGame, parse_args
    from replanal.strive_export import winio
    seen = []
    monkeypatch.setattr(winio, "press", lambda key, **kw: seen.append((key, kw["background"])))
    assert parse_args(["--out", "x", "--count", "1"]).foreground is False
    assert parse_args(["--out", "x", "--count", "1", "--foreground"]).foreground is True
    LiveGame(cap=None).press("u")
    LiveGame(cap=None, background=False).press("u")
    assert seen == [("u", True), ("u", False)]


def test_find_window_skips_other_apps_named_after_the_game(monkeypatch):
    from replanal.strive_export import winio
    titles = {1: "GUILTY GEAR -STRIVE- Sol combos - YouTube", 2: "Guilty Gear -Strive-  "}
    exes = {1: "chrome", 2: "GGST-Win64-Shipping"}

    class U:
        def EnumWindows(self, cb, _):
            for h in titles:
                cb(h, 0)

        def GetWindowTextW(self, h, buf, n):
            buf.value = titles[h]

        def IsWindowVisible(self, h):
            return True

    monkeypatch.setattr(winio, "_u32", lambda: U())
    monkeypatch.setattr(winio, "_window_exe", lambda h: exes[h])
    assert winio.find_window() == 2
