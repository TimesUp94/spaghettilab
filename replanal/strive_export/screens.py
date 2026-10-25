"""Recognise Strive's menu screens from a window capture.

Every check runs on the capture scaled to 960x540 (the size the reference
screenshots are stored at).  A screen is recognised in up to three layers:

1. **Structure** (language independent, no reference image): colours and
   layout that the game draws the same in every UI language. The red
   highlight of the selected tab in the main menu's icon row, the replay
   room's red header banner, its tab icon row and the selected list row, and
   the grey item bars of a pop-up menu with the red bar on the highlighted
   item, counted.
2. **Text-free reference regions** (language independent): small crops of a
   reference screenshot that hold no text: the GUILTY GEAR logo, the main
   menu's tab icons, the replay room's film icon, the pop-up menu's header
   picture. They are read from ``refs/<lang>/`` when the language has its own
   references, else from the shipped Italian ones.
3. **Text regions** (language specific): the title screen's prompt, the tab
   labels, the replay room's "Saved replays" heading, the pop-up menus' item
   labels. They are compared only against references of the UI language, so
   they add a check once the language is known (``it`` ships; ``calibrate``
   records others) and are skipped otherwise.

Which pop-up is which is told by its item count, not its labels: the replay
entry menu has five items (view, lock, edit comment, delete, close) over the
replay room; the end-of-replay menu has three (back to the list, play again,
main menu) over the game.  Confirm is only ever pressed on item 0 of either.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540

# Main menu: nine tabs in the bottom icon row, the selected one on a dark red square.
MAIN_TAB_X0, MAIN_TAB_PITCH, MAIN_TABS = 117, 90.4, 9
MAIN_TAB_Y = (428, 492)
COLLECTION_TAB = 6            # RETE, 3v3, LOCALE, DOJO, STORIA, MODIFICA, RACCOLTA, NEGOZIO, ESCI
RANKED_TAB = 0                # the network tab: confirm there starts ranked matchmaking

# Replay room: five tab icons under the header; "saved replays" is the third.
ROOM_TAB_XS = (400, 440, 480, 520, 560)
ROOM_TAB_Y = (112, 136)
SAVED_TAB = 2
ROOM_BANNER = (140, 30, 640, 100)          # x0, y0, x1, y1
ROOM_ICON = (62, 40, 118, 90)              # film icon in the banner
LIST_BOX = (60, 145, 890, 470)             # replay rows
LIST_COUNT_BOX = (800, 70, 905, 102)       # "100 / 100" counter (OCR, full res)

# Pop-up menus: grey item bars, the highlighted one red; sampled right of the labels.
MENU_SAMPLE_X = (610, 670)
MENU_SCAN_Y = (100, 470)
MENU_ICON_DX = (243, 292)                  # header picture, relative to the first bar
MENU_ICON_DY = (-51, -8)
ENTRY_MENU_ITEMS = 5                       # view, lock, edit comment, DELETE, close
END_MENU_ITEMS = 3                         # back to list, play again, main menu

TITLE_LOGO = (270, 222, 690, 318)
SERVER_ICONS = [(230, 188, 283, 232), (600, 235, 728, 340)]

# Text (language-specific) regions, compared only against same-language refs.
TEXT_REGIONS = {
    "title": [(250, 372, 710, 412)],
    "main": [(60, 478, 900, 499)],
    "collection": [(60, 478, 900, 499)],
    "saved_list": [(60, 104, 340, 146)],
    "server": [(300, 270, 680, 305)],
}
# Menu labels: one box per item bar, relative to the first bar (the dialog is centred).
MENU_TEXT_X = (250, 600)
MENU_PITCH = 26.5

REF_NAMES = ("title", "main", "collection", "saved_list", "replay_menu", "replay_end", "server")

STRUCT_THR = 22.0   # text-free region diff (grey thumbnail, 0-255)
TEXT_NCC = 0.85     # text regions: correlation with the same-language reference
TEXT_SHIFT = 3      # px of misalignment the text check tolerates (960x540)
TEXT_SCALES = (0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08)  # text sizes it tries


@dataclass
class Screen:
    """What is on screen. ``name`` is one of: title, server, main, room,
    saved_list, replay_menu, replay_end, unknown."""
    name: str
    tab: int | None = None          # main menu / replay room tab index
    item: int | None = None         # highlighted pop-up item
    items: int | None = None        # pop-up item count
    row_y: int | None = None        # top of the selected list row (960x540 space)
    ready: bool = False             # main: collection tab with its REPLAY entry highlighted
    lang: str | None = None         # language whose text regions matched, if checked
    detail: str = ""
    scores: dict = field(default_factory=dict)

    def __str__(self) -> str:
        bits = [self.name]
        if self.tab is not None:
            bits.append(f"tab={self.tab}")
        if self.items is not None:
            bits.append(f"item={self.item}/{self.items}")
        if self.row_y is not None:
            bits.append(f"row_y={self.row_y}")
        return " ".join(bits)


# ── image helpers ────────────────────────────────────────────────────────────

def game_area(img: np.ndarray) -> np.ndarray:
    """The centred 16:9 picture: in a window of another shape the game draws
    black bars above and below (or left and right)."""
    h, w = img.shape[:2]
    gw, gh = min(w, round(h * 16 / 9)), min(h, round(w * 9 / 16))
    if w - gw < 4 and h - gh < 4:
        return img
    x, y = (w - gw) // 2, (h - gh) // 2
    return img[y:y + gh, x:x + gw]


def normalize(img: np.ndarray) -> np.ndarray:
    """The capture at 960x540 BGR (through the same area-downscale as the refs)."""
    img = game_area(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def full_res(img: np.ndarray) -> np.ndarray:
    """The capture at 1920x1080 BGR, for OCR."""
    img = game_area(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[:2] != (1080, 1920):
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return img


def crop(img, box):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]


def dark_red_frac(c: np.ndarray) -> float:
    """Share of pixels in the menu-highlight dark red (about BGR 10,20,110)."""
    if c.size == 0:
        return 0.0
    c = c.astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    return float(np.mean((r > 85) & (r > 2.5 * g) & (r > 2.5 * b)))


def bright_red_frac(c: np.ndarray) -> float:
    """Share of pixels in the bright UI red (about BGR 0,0,192)."""
    if c.size == 0:
        return 0.0
    c = c.astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    return float(np.mean((r > 150) & (g < 60) & (b < 60)))


def thumb(c: np.ndarray, size=(96, 32)) -> np.ndarray:
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _text_gray(img: np.ndarray, box) -> np.ndarray:
    c = crop(img, box)
    if c.size == 0:
        return np.zeros((0, 0), np.float32)
    return cv2.GaussianBlur(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)


def _ink_cols(g: np.ndarray) -> np.ndarray:
    """The columns that hold the region's text, with a little padding."""
    col = g.max(axis=0) - np.median(g)
    xs = np.where(col > 0.4 * col.max())[0] if col.max() > 8 else []
    if len(xs) == 0:
        return g
    return g[:, max(0, xs[0] - 4):xs[-1] + 5]


def _text_match(a, b, box, box_b) -> float:
    """How well a's text, at a slightly different size or position, is found
    in b's region."""
    t0 = _ink_cols(_text_gray(a, box))
    x0, y0, x1, y1 = box_b
    m = TEXT_SHIFT
    g = _text_gray(b, (x0 - m, y0 - m, x1 + m, y1 + m))
    if t0.size == 0 or g.size == 0 or t0.std() < 1e-3:
        return 0.0
    best = -1.0
    for s in TEXT_SCALES:
        t = cv2.resize(t0, (max(4, round(t0.shape[1] * s)), t0.shape[0]), interpolation=cv2.INTER_LINEAR)
        if t.shape[0] <= g.shape[0] and t.shape[1] <= g.shape[1]:
            best = max(best, float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max()))
    return best


def text_similarity(a: np.ndarray, b: np.ndarray, box, box_b=None) -> float:
    """Correlation of two text regions at full 960x540 detail (1 = same
    words; a different wording scores far lower even on the same background).
    A window of another size renders the UI text a few % smaller or larger and a
    pixel or two off, so each side's text is searched in the other's region at a
    few sizes; the lower of the two directions counts, so a label is not taken
    for a longer one that starts the same."""
    box_b = box_b or box
    return min(_text_match(a, b, box, box_b), _text_match(b, a, box_b, box))


def region_diff(a: np.ndarray, b: np.ndarray, box, box_b=None) -> float:
    ca, cb = crop(a, box), crop(b, box_b or box)
    if ca.size == 0 or cb.size == 0:
        return 255.0
    return float(np.mean(np.abs(thumb(ca) - thumb(cb))))


# ── structural (language-independent) features ──────────────────────────────

def main_tab_index(img: np.ndarray) -> int | None:
    """Index of the main menu's selected tab: exactly one icon slot on the dark
    red square, none of the others."""
    fr = [dark_red_frac(crop(img, (MAIN_TAB_X0 + MAIN_TAB_PITCH * i - 32, MAIN_TAB_Y[0],
                                   MAIN_TAB_X0 + MAIN_TAB_PITCH * i + 32, MAIN_TAB_Y[1])))
          for i in range(MAIN_TABS)]
    hits = [i for i, f in enumerate(fr) if f > 0.5]
    if len(hits) != 1 or sorted(fr)[-2] > 0.3:
        return None
    return hits[0]


def main_icons(img: np.ndarray) -> list[np.ndarray]:
    """Grey crops of the nine tab icons (labels below them excluded)."""
    out = []
    for i in range(MAIN_TABS):
        cx = MAIN_TAB_X0 + MAIN_TAB_PITCH * i
        c = cv2.cvtColor(crop(img, (cx - 30, 430, cx + 30, 478)), cv2.COLOR_BGR2GRAY)
        out.append(cv2.GaussianBlur(c, (3, 3), 0).astype(np.float32))
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two same-size crops (-1..1)."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def room_banner(img: np.ndarray) -> bool:
    return bright_red_frac(crop(img, ROOM_BANNER)) > 0.7


def room_banner_dimmed(img: np.ndarray) -> bool:
    """The replay room's banner under a pop-up's dark veil: dark red, not bright."""
    c = crop(img, ROOM_BANNER).astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    frac = float(np.mean((r > 45) & (r < 150) & (r > 2.5 * g) & (r > 2.5 * b)))
    return frac > 0.6


def room_tab_index(img: np.ndarray) -> int | None:
    fr = [bright_red_frac(crop(img, (x - 14, ROOM_TAB_Y[0], x + 14, ROOM_TAB_Y[1])))
          for x in ROOM_TAB_XS]
    hits = [i for i, f in enumerate(fr) if f > 0.15]
    if len(hits) != 1 or sorted(fr)[-2] > 0.08:
        return None
    return hits[0]


def list_selected_row(img: np.ndarray) -> int | None:
    """Top y of the bright red (selected) replay row, or None."""
    x0, y0, x1, y1 = LIST_BOX
    # Sampled left and right of the portraits / rank badges in the middle.
    rows = [bright_red_frac(np.hstack([img[y:y + 1, x0:240], img[y:y + 1, 600:x1 - 40]])) > 0.6
            for y in range(y0, y1)]
    runs, start = [], None
    for k, on in enumerate(rows + [False]):
        if on and start is None:
            start = k
        elif not on and start is not None:
            runs.append((start, k))
            start = None
    good = [(a, b) for a, b in runs if 28 <= b - a <= 48]
    if len(good) != 1:
        return None
    return y0 + good[0][0]


def menu_bars(img: np.ndarray) -> list[tuple[int, int, str]]:
    """Horizontal bars in the pop-up menu column: (y0, y1, 'R' | 'G')."""
    x0, x1 = MENU_SAMPLE_X
    kinds = []
    for y in range(*MENU_SCAN_Y):
        row = img[y, x0:x1].astype(np.int16)
        b, g, r = np.median(row, axis=0)
        spread = float(row.std())
        if r >= 150 and g <= 50 and b <= 50:
            kinds.append("R")
        elif 44 <= min(b, g, r) and max(b, g, r) <= 72 and max(b, g, r) - min(b, g, r) <= 8 and spread <= 10:
            kinds.append("G")
        else:
            kinds.append(".")
    bars, start = [], None
    for k, kind in enumerate(kinds + ["."]):
        if start is not None and kind != kinds[start]:
            if k - start >= 12:
                bars.append((MENU_SCAN_Y[0] + start, MENU_SCAN_Y[0] + k, kinds[start]))
            start = None
        if start is None and kind != ".":
            start = k
    return bars


def menu_info(img: np.ndarray) -> tuple[int, int, int] | None:
    """(item count, highlighted index, top y of the first bar) of a pop-up
    menu, or None. The bars must be evenly spaced, 16-30 px tall, with
    exactly one red."""
    bars = menu_bars(img)
    groups, cur = [], []
    for bar in bars:
        if cur and bar[0] - cur[-1][1] > 10:
            groups.append(cur)
            cur = []
        cur.append(bar)
    if cur:
        groups.append(cur)
    for grp in groups:
        if len(grp) < 2 or not all(16 <= b1 - b0 <= 30 for b0, b1, _ in grp):
            continue
        pitches = [grp[i + 1][0] - grp[i][0] for i in range(len(grp) - 1)]
        if min(pitches) < 22 or max(pitches) > 32 or max(pitches) - min(pitches) > 4:
            continue
        reds = [i for i, (_, _, k) in enumerate(grp) if k == "R"]
        if len(reds) != 1:
            continue
        return len(grp), reds[0], grp[0][0]
    return None


# ── references ──────────────────────────────────────────────────────────────

PACKAGE_REFS = Path(__file__).resolve().parent / "refs"
DEFAULT_LANG = "it"


def user_refs_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(base) / "SpaghettiLab" / "strive_refs"


class RefSet:
    """Reference screenshots per UI language: ``<dir>/<lang>/<screen>.jpg``
    plus an optional ``profile.json`` (``date_order``). Directories are
    searched in order; the package's own ``refs/`` (Italian) comes last."""

    def __init__(self, dirs: list[Path] | None = None):
        self.dirs = [Path(d) for d in (dirs or [])] + [user_refs_dir(), PACKAGE_REFS]
        self._cache: dict[tuple[str, str], np.ndarray | None] = {}

    def languages(self) -> list[str]:
        out = []
        for d in self.dirs:
            if d.is_dir():
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and any(sub.glob("*.jpg")) and sub.name not in out:
                        out.append(sub.name)
        return out

    def get(self, lang: str, name: str) -> np.ndarray | None:
        key = (lang, name)
        if key not in self._cache:
            img = None
            for d in self.dirs:
                p = d / lang / f"{name}.jpg"
                if p.is_file():
                    img = cv2.imread(str(p))
                    if img is not None:
                        img = normalize(img)
                        break
            self._cache[key] = img
        return self._cache[key]

    def struct_ref(self, lang: str | None, name: str) -> np.ndarray | None:
        """A reference for text-free regions: the language's own if it has
        one, else the shipped default."""
        own = self.get(lang, name) if lang else None
        return own if own is not None else self.get(DEFAULT_LANG, name)

    def profile(self, lang: str | None) -> dict:
        for d in self.dirs:
            p = d / (lang or DEFAULT_LANG) / "profile.json"
            if p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
        return {}


# ── classifier ──────────────────────────────────────────────────────────────

class Classifier:
    """Turn a capture into a :class:`Screen`.

    ``lang``: a UI language code whose text regions must also match, or
    ``"auto"`` (default): the first screen whose text matches one of the
    available reference languages fixes the language; until then, and when
    none matches, only the language-independent checks are used.
    """

    def __init__(self, refs: RefSet | None = None, lang: str = "auto"):
        self.refs = refs or RefSet()
        self.lang = None if lang in (None, "", "auto") else lang
        self.auto = self.lang is None
        self._tried_auto: set[str] = set()

    # text checks -------------------------------------------------------------
    def _text_ok(self, img, ref_name, boxes, ref_boxes=None) -> tuple[bool, str | None]:
        """(passed, language). Passes when no language is known yet / no
        same-language reference exists; with ``auto`` it tries every
        reference language and fixes the first match."""
        langs = [self.lang] if self.lang else (self.refs.languages() if self.auto else [])
        for lang in langs:
            ref = self.refs.get(lang, ref_name)
            if ref is None:
                continue
            rboxes = ref_boxes(ref) if callable(ref_boxes) else (ref_boxes or boxes)
            if rboxes is None:
                continue
            sim = min(text_similarity(img, ref, b, rb) for b, rb in zip(boxes, rboxes))
            if sim >= TEXT_NCC:
                if self.auto and self.lang is None:
                    self.lang = lang
                return True, lang
            if self.lang:  # known language, its reference disagrees
                return False, lang
        return True, None

    # screen checks -----------------------------------------------------------
    def _menu(self, img) -> Screen | None:
        info = menu_info(img)
        if info is None:
            return None
        n, sel, top = info
        ref = self.refs.struct_ref(self.lang, "replay_menu")
        icon = (MENU_ICON_DX[0], top + MENU_ICON_DY[0], MENU_ICON_DX[1], top + MENU_ICON_DY[1])
        ref_info = menu_info(ref) if ref is not None else None
        if ref_info is None:
            return None
        rt = ref_info[2]
        ref_icon = (MENU_ICON_DX[0], rt + MENU_ICON_DY[0], MENU_ICON_DX[1], rt + MENU_ICON_DY[1])
        d_icon = region_diff(img, ref, icon, ref_icon)
        scores = {"menu_icon": round(d_icon, 1)}
        if d_icon >= STRUCT_THR:
            return Screen("unknown", detail=f"pop-up with {n} items but an unknown header", scores=scores)
        if n == ENTRY_MENU_ITEMS and room_banner_dimmed(img):
            name = "replay_menu"
        elif n == END_MENU_ITEMS and not room_banner_dimmed(img):
            name = "replay_end"
        else:
            return Screen("unknown", items=n, item=sel, detail=f"pop-up menu with {n} items", scores=scores)
        x0, x1 = MENU_TEXT_X
        boxes = [(x0, top + MENU_PITCH * i + 3, x1, top + MENU_PITCH * i + 19) for i in range(n)]

        def ref_boxes(ref):
            ri = menu_info(ref)
            if ri is None or ri[0] != n:
                return None
            return [(x0, ri[2] + MENU_PITCH * i + 3, x1, ri[2] + MENU_PITCH * i + 19) for i in range(n)]

        ok, lang = self._text_ok(img, name, boxes, ref_boxes)
        if not ok:
            return Screen("unknown", items=n, item=sel, detail=f"{name} layout but its labels differ", scores=scores)
        return Screen(name, items=n, item=sel, lang=lang, scores=scores)

    def _title(self, img) -> Screen | None:
        ref = self.refs.struct_ref(self.lang, "title")
        if ref is None:
            return None
        d = region_diff(img, ref, TITLE_LOGO)
        if d >= STRUCT_THR:
            return None
        ok, lang = self._text_ok(img, "title", TEXT_REGIONS["title"])
        if not ok:
            return None
        return Screen("title", lang=lang, scores={"logo": round(d, 1)})

    def _server(self, img) -> Screen | None:
        ref = self.refs.struct_ref(self.lang, "server")
        if ref is None:
            return None
        d = max(region_diff(img, ref, b) for b in SERVER_ICONS)
        if d >= STRUCT_THR:
            return None
        return Screen("server", scores={"icons": round(d, 1)})

    def _main(self, img) -> Screen | None:
        tab = main_tab_index(img)
        if tab is None:
            return None
        # Unselected tab icons must match the reference icon row.
        ref_main = self.refs.struct_ref(self.lang, "main")
        ref_coll = self.refs.struct_ref(self.lang, "collection")
        if ref_main is None or ref_coll is None:
            return None
        cur = main_icons(img)
        refm, refc = main_icons(ref_main), main_icons(ref_coll)
        # Each unselected icon against a reference where it is unselected too.
        sims = [ncc(cur[i], refm[i] if i != 0 else refc[i]) for i in range(MAIN_TABS) if i != tab]
        sim = float(np.mean(sims))
        scores = {"icons": round(sim, 2)}
        if sim < 0.6:
            return None
        ready = False
        if tab == COLLECTION_TAB:
            # The collection sub-menu, its first entry (REPLAY) highlighted: the
            # confirm-key badge sits beside it.
            d = region_diff(img, ref_coll, (615, 278, 790, 306))
            scores["replay_entry"] = round(d, 1)
            ready = d < STRUCT_THR
        ref_name = {0: "main", COLLECTION_TAB: "collection"}.get(tab)
        lang = None
        if ref_name:
            ok, lang = self._text_ok(img, ref_name, TEXT_REGIONS[ref_name])
            if not ok:
                return None
        return Screen("main", tab=tab, ready=ready, lang=lang, scores=scores)

    def _room(self, img) -> Screen | None:
        if not room_banner(img):
            return None
        ref = self.refs.struct_ref(self.lang, "saved_list")
        if ref is None:
            return None
        d = region_diff(img, ref, ROOM_ICON)
        scores = {"film_icon": round(d, 1)}
        if d >= STRUCT_THR:
            return None
        tab = room_tab_index(img)
        if tab == SAVED_TAB:
            ok, lang = self._text_ok(img, "saved_list", TEXT_REGIONS["saved_list"])
            if not ok:
                return Screen("room", tab=tab, detail="saved tab but its heading differs", scores=scores)
            row = list_selected_row(img)
            if row is not None:
                return Screen("saved_list", tab=tab, row_y=row, lang=lang, scores=scores)
        return Screen("room", tab=tab, scores=scores)

    def classify(self, capture: np.ndarray | None) -> Screen:
        if capture is None or capture.size == 0 or min(capture.shape[:2]) < 100:
            return Screen("unknown", detail="empty capture")
        img = normalize(capture)
        for check in (self._menu, self._room, self._main, self._title, self._server):
            s = check(img)
            if s is not None:
                return s
        return Screen("unknown")
