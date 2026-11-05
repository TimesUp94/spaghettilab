"""Training pause menu -> "Impostazioni specifiche personaggi": read and change one named row.

Offline, research only. The screen is recognised from a real capture ($GGST_INPUT_LOG_DIR/ref/char_settings.png);
only ALLOWED keys are ever pressed on it, and A/D only after the highlighted row has been read back as the
row meant. Rows are found by their text because the list scrolls.
"""
import difflib
import os
import re
import time

import cv2
import numpy as np

import ggst
import train_sweep as ts

TITLE = (590, 60, 1300, 125)
LABEL_X = (500, 945)
VALUE_X = (985, 1365)
ROWS_Y = (150, 1000)
ROW_H = 53
ALLOWED = {"w", "s", "a", "d", "o"}           # move, change a value, O = back
SPELL_KEYS = {"w", "s", "u", "o"}             # the SPELL list: move, choose, back; no A/D
_reader = None


def key_allowed(key):
    return key in ALLOWED


def press(key, hold=70, after=0.4):
    if not key_allowed(key):
        raise SystemExit(f"key '{key}' is not allowed on the character settings screen")
    if not on_screen():
        raise SystemExit("not on the character settings screen; no key sent")
    ggst.keys(f"{key}:{hold}", 200)
    time.sleep(after)


def title_diff(im, ref):
    x0, y0, x1, y1 = TITLE
    return float(np.mean(np.abs(im[y0:y1, x0:x1].astype(int) - ref[y0:y1, x0:x1])))


def on_screen():
    ref = cv2.imread(os.path.join(ts.REFS, "char_settings.png"))
    return ref is not None and title_diff(ts.capture(), ref) < 12


def _ocr(crop):
    global _reader
    if _reader is None:
        import easyocr      # slow import, only when needed
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return " ".join(t for _, t, c in _reader.readtext(crop) if c > 0.2).strip()


def norm(s):
    # OCR reads the menu's 1 as a 7 ("Liv. 7 (fisso)"), so they compare alike
    return re.sub(r"[^a-z0-9]", "", (s or "").lower()).replace("7", "1")


def highlighted():
    """(row text, value text) of the red row, or (None, None)."""
    im = ts.capture()
    x0, x1 = LABEL_X
    band = im[ROWS_Y[0]:ROWS_Y[1], x0:x1]
    red = (band[:, :, 2] > 150) & (band[:, :, 1] < 40) & (band[:, :, 0] < 40)
    rows = np.where(red.mean(axis=1) > 0.5)[0]
    if not len(rows):
        return None, None
    y0, y1 = rows[0] + ROWS_Y[0], rows[-1] + ROWS_Y[0]
    label = _ocr(cv2.resize(im[y0:y1, x0:x1], None, fx=2, fy=2))
    value = _ocr(cv2.resize(im[y0:y1, VALUE_X[0]:VALUE_X[1]], None, fx=2, fy=2))
    return label, value


def go_to(label, tries=45):
    """Walk the list until the highlighted row reads as `label`."""
    want = norm(label)
    for _ in range(tries):
        cur, _v = highlighted()
        if cur and difflib.SequenceMatcher(None, norm(cur), want).ratio() >= 0.8:
            return cur
        press("s")
    raise SystemExit(f"row '{label}' not found; no value changed")


def options(label, max_values=12):
    """Values a row offers, leaving it where it was. Steps right, then back the same number of steps."""
    go_to(label)
    first = highlighted()[1]
    seen, steps = [first], 0
    for _ in range(max_values):
        press("d")
        steps += 1
        v = highlighted()[1]
        if norm(v) == norm(first):
            return seen, steps
        seen.append(v)
    for _ in range(steps):
        press("a")
    return seen, 0


def spell_screen():
    """The SPELL list opened from a "Spell: Slot N" row."""
    ref = cv2.imread(os.path.join(ts.REFS, "spell.png"))
    return ref is not None and title_diff(ts.capture(), ref) < 12


def press_spell(key, after=0.25):
    if key not in SPELL_KEYS:
        raise SystemExit(f"key '{key}' is not allowed on the spell list")
    if not spell_screen():
        raise SystemExit("not on the spell list; no key sent")
    ggst.keys(f"{key}:60", 80)
    time.sleep(after)


def open_slot(n):
    """U only on the row read back as that slot; nowhere else on this screen."""
    label = go_to(f"Spell: Slot {n}")
    if not norm(label).endswith(f"slot{n}"):
        raise SystemExit(f"highlighted row is '{label}', not slot {n}; stopping")
    if not on_screen():
        raise SystemExit("not on the character settings screen; no key sent")
    ggst.keys("u:70", 200)
    time.sleep(1.0)
    if not spell_screen():
        raise SystemExit("the spell list did not open")


def spell_highlighted():
    """Text of the red row on the spell list."""
    im = ts.capture()
    band = im[ROWS_Y[0]:ROWS_Y[1], 520:1300]
    red = (band[:, :, 2] > 150) & (band[:, :, 1] < 40) & (band[:, :, 0] < 40)
    rows = np.where(red.mean(axis=1) > 0.5)[0]
    if not len(rows):
        return None
    y0, y1 = rows[0] + ROWS_Y[0], rows[-1] + ROWS_Y[0]
    return _ocr(cv2.resize(im[y0:y1, 500:1370], None, fx=2, fy=2))


def closest(text, opts):
    """Index of the option the OCR text is nearest to: names read differently each time ("Peso 10Ot")."""
    t = norm(text)
    return max(range(len(opts)), key=lambda i: difflib.SequenceMatcher(None, t, norm(opts[i])).ratio())


def step_to(label, opts, index):
    """Put the row on opts[index] by position, one D at a time, from where it is now. Only the row is checked
    strictly: a value's text is a loose match, and the raw OCR is returned so mismatches can be reported."""
    go_to(label)
    cur_label, v = highlighted()
    if difflib.SequenceMatcher(None, norm(cur_label), norm(label)).ratio() < 0.8:
        raise SystemExit(f"highlighted row is '{cur_label}', not '{label}'; stopping")
    for _ in range((index - closest(v, opts)) % len(opts)):
        press("d")
    return highlighted()[1]


def set_value(label, value, max_steps=12):
    """Set the row `label` to `value`, reading the row and value back at each step."""
    go_to(label)
    for _ in range(max_steps):
        cur_label, v = highlighted()
        if difflib.SequenceMatcher(None, norm(cur_label), norm(label)).ratio() < 0.8:
            raise SystemExit(f"highlighted row is '{cur_label}', not '{label}'; stopping")
        if norm(v) == norm(value):
            return v
        press("d")
    raise SystemExit(f"'{label}' never showed '{value}'")
