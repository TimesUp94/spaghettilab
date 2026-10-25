"""The export state machine: title screen -> saved replays -> record each one.

Everything that touches the game goes through a ``game`` object (``grab()``,
``press(key)``, ``start_recording(path)``) so the logic can be driven by a
scripted sequence of screens in tests.

Safety rule: a key is pressed only right after the screen was recognised, and
only a key from that screen's allow-list (:func:`key_allowed`). Confirm is
allowed on exactly three screens: the title, the main menu's Collection tab
with its Replay entry highlighted, and item 0 of the replay entry menu
("View replay") or of the end-of-replay menu ("Back to the replay list"). On
anything unrecognised the exporter waits; it never presses a key to find out.
That keeps ranked matchmaking (the main menu's first tab) and "Delete replay"
(item 3 of the entry menu) out of reach.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .rows import RowInfo, filename_for
from .screens import (COLLECTION_TAB, END_MENU_ITEMS, ENTRY_MENU_ITEMS, LIST_BOX, SAVED_TAB,
                      Screen, crop, normalize, thumb)


class ExportError(RuntimeError):
    pass


class UnsafeKey(ExportError):
    """A key that the current screen's allow-list does not permit: the export
    stops instead of pressing it (e.g. the replay menu's highlight is not on
    'View replay')."""


class Stopped(Exception):
    """The user asked to stop now."""


def key_allowed(screen: Screen, key: str) -> bool:
    n = screen.name
    if n == "title":
        return key == "u"
    if n == "main":
        if key in ("a", "d"):
            return True
        return key == "u" and screen.tab == COLLECTION_TAB and screen.ready
    if n == "room":
        return key in ("k", "l")
    if n == "saved_list":
        return key in ("u", "s", "w")
    if n == "replay_menu":
        return key == "u" and screen.item == 0 and screen.items == ENTRY_MENU_ITEMS
    if n == "replay_end":
        return key == "u" and screen.item == 0 and screen.items == END_MENU_ITEMS
    return False


class Clock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, s: float) -> None:
        time.sleep(s)


class Cancel:
    """``soft``: finish the current replay, then stop. ``hard``: stop now,
    finalising the current recording."""

    def __init__(self):
        self.soft = threading.Event()
        self.hard = threading.Event()

    def request(self, now: bool = False) -> None:
        self.soft.set()
        if now:
            self.hard.set()


@dataclass
class Options:
    out_dir: str
    count: int | None = None          # None: all
    since: dt.date | None = None
    start: int = 0
    dry_run: bool = False
    skip_existing: bool = True
    max_seconds: int = 900
    poll: float = 2.0
    nav_timeout: float = 300.0
    meta: dict = field(default_factory=dict)   # capture/encoder, for the manifest


@dataclass
class ListSig:
    row_y: int
    thumb: np.ndarray


def _sig(screen: Screen, img) -> ListSig:
    return ListSig(screen.row_y, thumb(crop(normalize(img), LIST_BOX), (83, 32)))


def _changed(a: ListSig, b: ListSig) -> bool:
    return abs(a.row_y - b.row_y) > 4 or float(np.mean(np.abs(a.thumb - b.thumb))) > 4.0


class Exporter:
    def __init__(self, game, classifier, reader, opts: Options, emit=None,
                 clock: Clock | None = None, cancel: Cancel | None = None):
        self.game = game
        self.cls = classifier
        self.reader = reader
        self.opts = opts
        self.emit = emit or (lambda event, **kw: None)
        self.clock = clock or Clock()
        self.cancel = cancel or Cancel()
        self.entries: list[dict] = []
        self._last_screen = None

    # ── primitives ──────────────────────────────────────────────────────────
    def look(self) -> tuple[Screen, np.ndarray | None]:
        img = None
        for _ in range(5):  # a capture occasionally comes back empty
            img = self.game.grab()
            if img is not None:
                break
            self.clock.sleep(0.5)
        s = self.cls.classify(img)
        if str(s) != self._last_screen:
            self._last_screen = str(s)
            self.emit("screen", screen=s.name, detail=str(s), lang=s.lang)
        return s, img

    def press(self, screen: Screen, key: str) -> None:
        if not key_allowed(screen, key):
            raise UnsafeKey(f"refusing to press {key!r} on {screen}: not a key this screen "
                            "allows; stopping")
        self.game.press(key)

    def wait_for(self, names, timeout: float, poll: float = 1.0) -> tuple[Screen, np.ndarray]:
        names = (names,) if isinstance(names, str) else tuple(names)
        t0 = self.clock.now()
        while True:
            s, img = self.look()
            if s.name in names:
                return s, img
            if self.clock.now() - t0 > timeout:
                raise ExportError(f"timed out waiting for {' / '.join(names)} (screen: {s})")
            self.clock.sleep(poll)

    def _check_hard(self):
        if self.cancel.hard.is_set():
            raise Stopped()

    # ── navigation ──────────────────────────────────────────────────────────
    def reach_saved_list(self) -> bool:
        """Get to the saved replays list. Returns True when the list was
        opened from the main menu (cursor on the newest replay)."""
        t0 = self.clock.now()
        via_menu = False
        waiting_since = None
        while self.clock.now() - t0 < self.opts.nav_timeout:
            self._check_hard()
            if self.cancel.soft.is_set():
                raise Stopped()
            s, _ = self.look()
            if s.name != "unknown":
                waiting_since = None
            if s.name == "saved_list":
                return via_menu
            if s.name == "title":
                self.press(s, "u")
                self.clock.sleep(6)
            elif s.name == "server":
                self.clock.sleep(2)
            elif s.name == "main":
                if s.tab is None:
                    self.clock.sleep(1)
                elif s.tab < COLLECTION_TAB:
                    self.press(s, "d")
                    self.clock.sleep(1.2)
                elif s.tab > COLLECTION_TAB:
                    self.press(s, "a")
                    self.clock.sleep(1.2)
                elif s.ready:
                    self.press(s, "u")
                    via_menu = True
                    self.clock.sleep(8)
                else:
                    self.clock.sleep(1)   # the sub-menu is still sliding in
            elif s.name == "room":
                if s.tab is None or s.tab == SAVED_TAB:
                    self.clock.sleep(1.5)  # a tab is loading
                elif s.tab < SAVED_TAB:
                    self.press(s, "l")
                    self.clock.sleep(4)
                else:
                    self.press(s, "k")
                    self.clock.sleep(4)
            elif s.name == "replay_end":
                self.press(s, "u")
                self.clock.sleep(6)
            elif s.name == "replay_menu":
                raise ExportError("A replay's menu is open in the game. Close it by hand "
                                  "(do not pick Delete) and start the export again.")
            else:
                if waiting_since is None:
                    waiting_since = self.clock.now()
                    self.emit("status", message="Waiting for a known screen (go to the title "
                                                "screen or the main menu)...")
                self.clock.sleep(2)
        raise ExportError("Could not reach the saved replays list. Start with the game at the "
                          "title screen or the main menu, with keyboard controls.")

    def stable_list(self, timeout: float = 30) -> tuple[Screen, np.ndarray, ListSig]:
        """The list once two looks in a row agree (portraits load in late)."""
        t0 = self.clock.now()
        prev = None
        while True:
            s, img = self.wait_for("saved_list", timeout=max(1.0, timeout - (self.clock.now() - t0)))
            sig = _sig(s, img)
            if prev is not None and not _changed(prev, sig):
                return s, img, sig
            prev = sig
            if self.clock.now() - t0 > timeout:
                return s, img, sig
            self.clock.sleep(0.7)

    def step(self, key: str = "s") -> str:
        """Move the list cursor one entry: 'moved', 'wrapped' or 'stuck'."""
        s, _, before = self.stable_list()
        for _ in range(3):
            self.press(s, key)
            self.clock.sleep(0.6)
            s, _, after = self.stable_list()
            if key == "s" and after.row_y < before.row_y - 4:
                return "wrapped"
            if key == "w" and after.row_y > before.row_y + 4:
                return "wrapped"
            if _changed(before, after):
                return "moved"
        return "stuck"

    def go_top(self) -> None:
        """Move the cursor to the newest replay (when the export starts on the list)."""
        for _ in range(130):
            r = self.step("w")
            if r == "stuck":
                return
            if r == "wrapped":   # went from the first entry to the last
                if self.step("s") != "wrapped":
                    raise ExportError("could not return to the top of the replay list")
                return
        raise ExportError("could not reach the top of the replay list")

    # ── one replay ──────────────────────────────────────────────────────────
    def open_entry(self) -> Screen:
        """Open the selected entry's menu. A press while the list is still
        loading is swallowed: retry, but only while the list still shows."""
        for _ in range(5):
            s, _ = self.wait_for("saved_list", timeout=30)
            self.press(s, "u")
            self.clock.sleep(2)
            s, _ = self.look()
            for _ in range(4):  # the menu fades in: its labels can still be faint
                if s.name != "unknown":
                    break
                self.clock.sleep(1)
                s, _ = self.look()
            if s.name == "replay_menu":
                return s
            if s.name != "saved_list":
                raise ExportError(f"unexpected screen after opening a replay entry: {s}")
        raise ExportError("the replay entry menu did not open")

    def start_replay(self, menu: Screen) -> None:
        """Confirm 'View replay' (item 0) and make sure the menu went away."""
        s = menu
        for _ in range(3):
            self.press(s, "u")
            self.clock.sleep(1.0)
            s, _ = self.look()
            if s.name != "replay_menu":
                return
        raise ExportError("'View replay' did not start the replay")

    def record_one(self, index: int, info: RowInfo | None, path: str) -> dict:
        menu = self.open_entry()
        if not key_allowed(menu, "u"):
            raise ExportError(f"the replay menu's highlight is not on 'View replay' ({menu}); "
                              "not confirming anything")
        self.start_replay(menu)
        rec = self.game.start_recording(path)
        self.emit("recording", index=index, file=os.path.basename(path))
        t0 = self.clock.now()
        partial = timed_out = False
        try:
            while True:
                self.clock.sleep(self.opts.poll)
                if self.cancel.hard.is_set():
                    partial = True
                    break
                if not rec.alive():
                    rc, err = rec.stop()
                    raise ExportError(f"ffmpeg stopped while recording (exit {rc}): {err[-400:]}")
                s, _ = self.look()
                if s.name == "replay_end":
                    break
                if self.clock.now() - t0 > self.opts.max_seconds:
                    partial = timed_out = True
                    break
        finally:
            rc, err = rec.stop()   # finalise the mp4 whatever happened
        if timed_out:
            raise ExportError(f"the replay did not end within {self.opts.max_seconds} s "
                              f"(recording kept: {path})")
        duration = round(self.clock.now() - t0, 1)
        entry = {"index": index, "file": os.path.basename(path), **(info.as_dict() if info else {}),
                 "duration_s": duration, "partial": partial,
                 "recorded_at": dt.datetime.now().isoformat(timespec="seconds"), **self.opts.meta}
        if rc not in (0, None) and err:
            self.emit("warning", message=f"ffmpeg exit {rc}: {err[-300:]}")
        self.emit("recorded", **entry)
        if partial:
            raise Stopped(entry)
        # Back to the list ("Back to the replay list" is item 0).
        for _ in range(3):
            s, _ = self.wait_for("replay_end", timeout=15)
            self.press(s, "u")
            self.clock.sleep(6)
            s, _ = self.look()
            if s.name != "replay_end":
                break
        self.wait_for("saved_list", timeout=60)
        return entry

    # ── manifest ────────────────────────────────────────────────────────────
    def manifest_path(self) -> str:
        return os.path.join(self.opts.out_dir, "manifest.json")

    def partial_files(self) -> set[str]:
        """Files the manifest lists as cut short (a stop, or the time limit)."""
        try:
            with open(self.manifest_path(), encoding="utf-8") as f:
                entries = json.load(f).get("entries", [])
        except (OSError, ValueError, AttributeError):
            return set()
        return {e.get("file") for e in entries if isinstance(e, dict) and e.get("partial")}

    def write_manifest(self) -> None:
        path = self.manifest_path()
        old = []
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f).get("entries", [])
        except (OSError, ValueError, AttributeError):
            pass
        mine = {e["file"] for e in self.entries}
        merged = [e for e in old if e.get("file") not in mine] + self.entries
        doc = {"tool": "replanal.strive_export", "version": 1,
               "updated": dt.datetime.now().isoformat(timespec="seconds"), "entries": merged}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    # ── main loop ───────────────────────────────────────────────────────────
    def _read_row(self, s: Screen, img) -> RowInfo | None:
        if self.reader is None or img is None:
            return None
        from .screens import full_res
        try:
            return self.reader.read_row(full_res(img), s.row_y)
        except Exception as e:  # OCR trouble never stops an export
            self.emit("warning", message=f"could not read the list row: {e}")
            return None

    def _target(self, info: RowInfo | None, index: int) -> tuple[str, bool]:
        """(output path, already exported)."""
        name = filename_for(info, index)
        path = os.path.join(self.opts.out_dir, name)
        by_date = not name.startswith("replay_")
        if os.path.exists(path):
            if by_date and name in self.partial_files():
                return path, False   # an earlier export was stopped mid-replay: record it again
            if by_date and self.opts.skip_existing:
                return path, True
            stem, ext = os.path.splitext(path)
            k = 1
            while os.path.exists(f"{stem}_{k}{ext}"):
                k += 1
            path = f"{stem}_{k}{ext}"
        return path, False

    def run(self) -> dict:
        o = self.opts
        if not o.dry_run:
            os.makedirs(o.out_dir, exist_ok=True)
        exported, skipped, listed = [], [], 0
        reason = "done"
        try:
            via_menu = self.reach_saved_list()
            if not via_menu:
                self.emit("status", message="Already on the replay list: moving to the newest replay")
                self.go_top()
            s, img, _ = self.stable_list()
            if self.reader is not None and img is not None:
                from .screens import full_res
                total = self.reader.read_count(full_res(img))
                if total is not None:
                    self.emit("list", total=total)
            for k in range(o.start):
                self._check_hard()
                if self.step("s") != "moved":
                    raise ExportError(f"the list has only {k + 1} replays (--start {o.start})")
            index = o.start
            prev: RowInfo | None = None
            while True:
                if o.count is not None and listed >= o.count:
                    reason = "count reached"
                    break
                if self.cancel.soft.is_set():
                    reason = "cancelled"
                    break
                s, img, _ = self.stable_list()
                info = self._read_row(s, img)
                if info is not None and prev is not None and info.stamp and prev.stamp \
                        and info.stamp > prev.stamp + dt.timedelta(minutes=1):
                    self.emit("warning", message=f"row {index} reads as newer than the one above it "
                                                 f"({info.stamp} > {prev.stamp}); a misread?")
                if o.since is not None:
                    if info is None or info.date is None:
                        self.emit("warning", message=f"could not read the date of replay {index}; "
                                                     "exporting it anyway")
                    elif info.date < o.since:
                        # Re-read before stopping: one misread digit must not end the export.
                        s2, img2, _ = self.stable_list()
                        info2 = self._read_row(s2, img2)
                        if info2 is not None and info2.date is not None and info2.date < o.since:
                            reason = f"reached replays older than {o.since.isoformat()}"
                            break
                        self.emit("warning", message=f"replay {index}: date reads differ "
                                                     f"({info.date} / {info2.date if info2 else None}); "
                                                     "exporting it")
                path, exists = self._target(info, index)
                self.emit("entry", index=index, file=os.path.basename(path),
                          **(info.as_dict() if info else {}))
                listed += 1
                if o.dry_run:
                    pass
                elif exists:
                    skipped.append(path)
                    self.emit("skipped", index=index, file=os.path.basename(path), reason="already exported")
                else:
                    entry = None
                    try:
                        entry = self.record_one(index, info, path)
                    except Stopped as st:
                        if st.args:
                            self.entries.append(st.args[0])
                            exported.append(path)
                            self.write_manifest()
                        raise
                    self.entries.append(entry)
                    exported.append(path)
                    self.write_manifest()
                if info is not None:
                    prev = info
                if o.count is not None and listed >= o.count:
                    reason = "count reached"
                    break
                if self.cancel.soft.is_set():
                    reason = "cancelled"
                    break
                r = self.step("s")
                if r != "moved":
                    reason = "end of the list"
                    break
                index += 1
        except Stopped:
            reason = "stopped"
        summary = {"reason": reason, "listed": listed, "exported": exported, "skipped": skipped,
                   "manifest": self.manifest_path() if self.entries else None}
        self.emit("done", **summary)
        return summary
