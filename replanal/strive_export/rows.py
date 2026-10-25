"""Read the replay list: match date/time and player names of the selected row.

The row text is read with EasyOCR (the ``ocr`` extra) from the full-resolution
capture. Only digits and separators matter for the date, so the language of
the label in front of it ("Data dell'incontro:", "Match Date:") does not;
the order of day, month and year does, and is a per-language setting
(``date_order``: DMY, MDY or YMD) that is guessed when a part can only be
one thing (a 4-digit first part is a year, a part over 12 is a day).
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from dataclasses import dataclass

import numpy as np

_DATE_RE = re.compile(r"(?<!\d)(\d{1,4})\s?[/.\-]\s?(\d{1,2})\s?[/.\-]\s?(\d{2,4})(?!\d)")
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2})\s?[:.,;]\s?(\d{2})\s?[:.,;]\s?(\d{2})(?!\d)")


@dataclass
class RowInfo:
    date: dt.date | None = None
    time: dt.time | None = None
    p1: str | None = None
    p2: str | None = None
    raw: str = ""
    date_ambiguous: bool = False

    def as_dict(self) -> dict:
        return {
            "date": self.date.isoformat() if self.date else None,
            "time": self.time.strftime("%H:%M:%S") if self.time else None,
            "p1": self.p1, "p2": self.p2,
        }

    @property
    def stamp(self) -> dt.datetime | None:
        if self.date is None:
            return None
        return dt.datetime.combine(self.date, self.time or dt.time(0, 0))


def _mk_date(y, m, d) -> dt.date | None:
    if y < 100:
        y += 2000
    if not 2021 <= y <= dt.date.today().year + 1:   # Strive came out in 2021
        return None
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def parse_date(text: str, order: str = "DMY") -> tuple[dt.date | None, bool]:
    """(date, ambiguous). ``order`` is used only when the parts don't settle it."""
    m = _DATE_RE.search(text)
    if not m:
        return None, False
    a, b, c = m.group(1), m.group(2), m.group(3)
    if len(a) == 4:                                  # YYYY/MM/DD
        return _mk_date(int(a), int(b), int(c)), False
    if len(c) not in (2, 4):
        return None, False
    x, y, year = int(a), int(b), int(c)
    if x > 12 and y <= 12:
        return _mk_date(year, y, x), False           # DMY for sure
    if y > 12 and x <= 12:
        return _mk_date(year, x, y), False           # MDY for sure
    ambiguous = x != y
    if order.upper() == "MDY":
        return _mk_date(year, x, y), ambiguous
    return _mk_date(year, y, x), ambiguous


def parse_time(text: str) -> dt.time | None:
    for m in _TIME_RE.finditer(text):
        h, mi, s = (int(g) for g in m.groups())
        if h < 24 and mi < 60 and s < 60:
            return dt.time(h, mi, s)
    return None


def parse_row_text(text: str, order: str = "DMY") -> tuple[dt.date | None, dt.time | None, bool]:
    date, amb = parse_date(text, order)
    rest = _DATE_RE.sub(" ", text, count=1) if date else text
    return date, parse_time(rest), amb


def parse_since(value: str, today: dt.date | None = None) -> dt.date:
    """YYYY-MM-DD, or today / yesterday."""
    today = today or dt.date.today()
    v = value.strip().lower()
    if v == "today":
        return today
    if v == "yesterday":
        return today - dt.timedelta(days=1)
    return dt.date.fromisoformat(v)


def clean_name(tokens: list[str]) -> str | None:
    """A player name from OCR tokens: drop the platform icon read as 1-2
    junk letters ("Ci", "D") in front, keep the rest."""
    toks = [t.strip() for t in tokens if t and t.strip()]
    while len(toks) > 1 and len(toks[0]) <= 2:
        toks.pop(0)
    if not toks:
        return None
    words = " ".join(toks).split()
    if len(words) > 1 and len(words[0]) <= 2 and not words[0].isdigit():
        words = words[1:]
    name = " ".join(words).strip()
    return name or None


_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_part(s: str, limit: int = 40) -> str:
    s = _BAD_CHARS.sub("", s).strip().replace(" ", "_")
    return s[:limit].rstrip("._") or "unknown"


def filename_for(info: RowInfo | None, index: int) -> str:
    """``2026-10-20_22-10-28_TimesUp_vs_J4rusty.mp4`` when the row was read,
    else ``replay_003.mp4`` (index in the list, from 0)."""
    if info is None or info.date is None or info.time is None:
        return f"replay_{index:03d}.mp4"
    parts = [info.date.isoformat(), info.time.strftime("%H-%M-%S")]
    if info.p1 or info.p2:
        parts.append(f"{safe_part(info.p1 or 'P1')}_vs_{safe_part(info.p2 or 'P2')}")
    return "_".join(parts) + ".mp4"


# Row layout (1920x1080, relative to the selected row's top y): the right part
# holds "<label>: <date> <time>  Ver. x.yy"; the bottom line the two names.
ROW_RIGHT = (1080, 1700, 0, 40)
ROW_P1 = (516, 700, 40, 84)
ROW_P2 = (990, 1520, 40, 84)
COUNT_BOX = (1600, 150, 1820, 200)


class RowReader:
    """EasyOCR on the list rows. ``available`` is False (with ``error`` set)
    when easyocr can't be loaded; the export then names files by index."""

    def __init__(self, date_order: str = "DMY", enabled: bool = True):
        self.date_order = date_order
        self.enabled = enabled
        self._reader = None
        self.error: str | None = None if enabled else "OCR disabled"

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._reader is None and self.error is None:
            try:
                import easyocr
                try:
                    import torch
                    gpu = bool(torch.cuda.is_available())
                except Exception:
                    gpu = False
                self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            except ImportError:
                self.error = (f'easyocr is not installed, so dates and names in the replay list '
                              f'can\'t be read. Install it with: "{sys.executable}" -m pip install easyocr')
            except Exception as e:  # torch DLLs, first-run model download offline
                self.error = f"easyocr could not start ({type(e).__name__}: {e})"
        return self._reader

    def _text(self, img: np.ndarray) -> list[str]:
        r = self._load()
        if r is None or img.size == 0:
            return []
        try:
            return [t for t in r.readtext(img, detail=0)]
        except Exception:
            return []

    def read_row(self, full: np.ndarray, row_y960: int) -> RowInfo:
        """Read the row whose top is at ``row_y960`` (960x540 space) from the
        1920x1080 capture."""
        y = row_y960 * 2

        def box(b):
            x0, x1, a, c = b
            return full[max(0, y + a):y + c, x0:x1]

        right = " ".join(self._text(box(ROW_RIGHT)))
        date, time, amb = parse_row_text(right, self.date_order)
        p1 = clean_name(self._text(box(ROW_P1)))
        p2 = clean_name(self._text(box(ROW_P2)))
        return RowInfo(date, time, p1, p2, raw=right, date_ambiguous=amb)

    def read_count(self, full: np.ndarray) -> int | None:
        """The number of saved replays from the "100 / 100" counter (the big
        first number), or None."""
        x0, y0, x1, y1 = COUNT_BOX
        return parse_count(" ".join(self._text(full[y0:y1, x0:x1])))


def parse_count(text: str) -> int | None:
    """First number of "100 / 100" as OCR reads it ("100 /100", "100 _ 7100")."""
    for tok in re.findall(r"\d+", text):
        if len(tok) <= 3:
            return int(tok)
        return None
    return None
