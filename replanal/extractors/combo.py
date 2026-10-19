"""Combo hit-counter reader for Guilty Gear Strive.

What the counter looks like (checked by eye on set 17, see
output/embed_test/combo_counter/):

* It is drawn on the HUD side of the player DOING the combo: a P1 combo shows
  on the left half of the screen, a P2 combo on the right half, wherever the
  characters stand.
* 2-9 hits: a large, semi-transparent light-grey digit (lower half nearly
  opaque, upper half faded) with the number spelled out in smaller, crisper
  grey text with a dark shadow ("two" ... "nine") sitting in a notch of the
  digit. Text height ~55-125 px depending on the pop-in animation.
* 10+ hits: no word; two huge digits (~600-850 px tall, P1 drawn ~1.2x
  larger than P2) roughly in place on the attacker's side: P1 tens at
  x ~190-260, ones at x ~530-720; P2 tens at x ~960-1100, ones at x
  ~1350-1500. The attacker usually stands in front of them, so one of the
  two digits is often covered.
* The counter pops in (zoom + fade) on each hit and fades out ~0.5-1 s after
  the combo ends. Counts can skip numbers because the video is 30 fps and
  the game runs at 60.

How it is read (template matching, like the timer reader):

* 2-9: grey-level NCC of averaged word templates ("w2".."w9") over a side
  ROI at 1/4 scale and several heights. Reading the word is far more robust
  than the translucent digit, which vanishes on grey/white backgrounds.
* 10+: NCC of glyph templates "g0".."g9" (the font is the same for tens and
  ones, P1 and P2) at 1/8 scale, ignoring the glyph top (under the HUD).
  The tens digit (1-3) and the ones digit are searched independently, each
  with its left/top edge inside a measured slot (BIG_SLOTS). A strong ones
  digit with a covered tens digit reads as 1x; a strong tens "1" with a
  covered ones digit reads as 10 (mode "tens", a lower bound that only keeps
  the combo together). Tens "2" was seen once (a 2x P1 combo in ggstsets/07)
  but too covered to template; "2"/"3" tens use the ones templates and are
  unverified on real 20+ counters. 40+ is not handled.

Templates are built from real frames by scripts/embed/build_combo_templates.py
and stored in config/templates/combo_templates.npz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData, Side

_DEFAULT_TEMPLATES = Path(__file__).resolve().parents[2] / "config" / "templates" / "combo_templates.npz"

WORD_DS = 4  # downscale for word matching
# Word text heights (full-res px) to try.
WORD_HEIGHTS = (56, 62, 68, 75, 83, 92, 102, 113, 125)
WORD_MIN_SCORE = 0.70
WORD_MIN_MARGIN = 0.03  # best word must beat the runner-up word by this much

# ── 10+ hits: the huge two-digit counter ────────────────────────────────────
BIG_DS = 8
BIG_PAD = 0.04  # template = glyph box padded by this fraction of the glyph height
# The top of a big glyph often sits under the HUD bars (and is the faded
# half): only the part below this fraction of the glyph height is matched.
BIG_TOP_CUT = 0.16
# Glyph heights (full-res px) to try: ~600-850 px seen, plus the zoom-in.
BIG_GLYPH_HEIGHTS = (590, 625, 665, 705, 750, 795, 845, 900)
# Tens digits looked for. "1" is seen in many videos; "2" once (a 2x P1
# combo in ggstsets/07 f18500-18600, too covered to template), so 2 and 3
# rely on the ones-digit templates and are unverified.
TENS_DIGITS = (1, 2, 3)
BIG_TENS_MIN_SCORE = 0.60
BIG_TENS_MIN_MARGIN = 0.08
BIG_ONES_MIN_SCORE = 0.60
BIG_ONES_MIN_MARGIN = 0.06
BIG_ONES_MIN_MARGIN_1 = 0.20
# With a very clear tens glyph the ones glyph (often half covered by the
# attacker) may score down to BIG_ONES_RELAXED_MIN_SCORE.
BIG_TENS_STRONG = 0.78
BIG_ONES_RELAXED_MIN_SCORE = 0.50
# Both glyphs barely over their thresholds is not enough: the "LET'S ROCK"
# round-start banner reads as 1 + 1 at ~0.60 each.
BIG_PAIR_MIN_SUM = 1.25
# Ones digit alone (tens glyph hidden by a character / effect): read as
# 1x only when the ones match is this strong. A hidden "2" tens digit is
# misread as 1 then; 20+ combos are rare.
BIG_ONES_ALONE_MIN_SCORE = 0.75
BIG_ONES_ALONE_MIN_MARGIN = 0.10
# Tens "1" alone (ones digit covered): reported as 10, a lower bound, so the
# combo is not lost and its damage is measured over its whole length.
BIG_TENS_ALONE_MIN_SCORE = 0.80
BIG_TENS_ALONE_MIN_MARGIN = 0.25
# Where each glyph's left edge and top edge may sit, relative to the
# *_combo_big ROI's top-left (full-res px). Measured on set 17 plus 12 other
# ggstsets/league videos (same overlay; P1 ROI starts at x=40, P2 at x=860).
BIG_SLOTS = {
    # side: (tens (x range, y range), ones (x range, y range))
    Side.P1: (((10, 290), (20, 200)), ((440, 760), (20, 200))),
    Side.P2: (((60, 300), (40, 200)), ((470, 670), (60, 210))),
}

WORD_ROI = {Side.P1: "p1_combo_counter", Side.P2: "p2_combo_counter"}
BIG_ROI = {Side.P1: "p1_combo_big", Side.P2: "p2_combo_big"}


def _gray_small(bgr: np.ndarray, ds: int) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = g.shape[:2]
    return cv2.resize(g, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)


def load_templates(path: Path | None = None) -> dict[str, np.ndarray]:
    path = Path(path or _DEFAULT_TEMPLATES)
    if not path.exists():
        return {}
    data = np.load(str(path))
    return {k: data[k].astype(np.float32) for k in data.files}


@dataclass
class Reading:
    count: int
    score: float
    margin: float
    mode: str  # "word" or "big"
    x: int = 0  # match position in the (downscaled) ROI, for debugging
    y: int = 0


class _Matcher:
    """Multi-scale NCC of a set of labelled templates against one image."""

    def __init__(self, templates: dict[int, np.ndarray], heights: tuple[int, ...], ds: int):
        self.ds = ds
        self.scaled: list[tuple[int, np.ndarray]] = []
        for count, t in templates.items():
            for h in heights:
                th = max(4, int(round(h / ds)))
                tw = max(4, int(round(t.shape[1] * th / t.shape[0])))
                self.scaled.append((count, cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA).astype(np.uint8)))

    def best(self, small: np.ndarray) -> tuple[dict[int, tuple[float, int, int]], ...]:
        per: dict[int, tuple[float, int, int]] = {}
        for count, t in self.scaled:
            if t.shape[0] > small.shape[0] or t.shape[1] > small.shape[1]:
                continue
            r = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if not np.isfinite(mx):
                continue
            if mx > per.get(count, (-2.0, 0, 0))[0]:
                per[count] = (float(mx), loc[0], loc[1])
        return (per,)


def _pick(
    per: dict[int, tuple[float, int, int]],
    min_score: float,
    min_margin: float,
    mode: str,
    expect_xy: tuple[int, int] | None = None,
    tol: int = 0,
) -> Reading | None:
    if not per:
        return None
    if expect_xy is not None:
        per = {c: v for c, v in per.items() if abs(v[1] - expect_xy[0]) <= tol and abs(v[2] - expect_xy[1]) <= tol}
        if not per:
            return None
    ranked = sorted(per.items(), key=lambda kv: -kv[1][0])
    count, (score, x, y) = ranked[0]
    second = ranked[1][1][0] if len(ranked) > 1 else -1.0
    if score < min_score or score - second < min_margin:
        return None
    return Reading(count=count, score=score, margin=score - second, mode=mode, x=x, y=y)


class _BigMatcher:
    """Two-digit big counter: best glyph in the tens slot and, independently,
    in the ones slot (a character often covers one of the two)."""

    def __init__(self, glyphs: dict[str, np.ndarray]):
        self.scaled: dict[str, list[np.ndarray]] = {}  # key -> one template per height
        for d, t in glyphs.items():
            lst = []
            for h in BIG_GLYPH_HEIGHTS:
                th = max(4, int(round(h * (1 + 2 * BIG_PAD) / BIG_DS)))
                tw = max(4, int(round(t.shape[1] * th / t.shape[0])))
                c0 = int(round(th * (BIG_PAD + BIG_TOP_CUT) / (1 + 2 * BIG_PAD)))
                lst.append(cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA).astype(np.uint8)[c0:])
            self.scaled[d] = lst
        self.tens = [k for k in self.scaled if int(k[0]) in TENS_DIGITS]

    def _slot(self, small: np.ndarray, keys, slot) -> dict[int, float]:
        """Best NCC per digit (over its template variants) with the glyph's
        left / top edge inside slot."""
        (xa, xb), (ya, yb) = slot
        sh, sw = small.shape[:2]
        per: dict[int, float] = {}
        for k, h in enumerate(BIG_GLYPH_HEIGHTS):
            px, py = BIG_PAD * h, BIG_TOP_CUT * h  # template top-left -> glyph left / top
            x0, y0 = max(0, int((xa - px) / BIG_DS)), max(0, int((ya + py) / BIG_DS))
            for key in keys:
                t = self.scaled[key][k]
                d = int(key[0])
                th, tw = t.shape
                x1 = min(sw, int(np.ceil((xb - px) / BIG_DS)) + tw)
                y1 = min(sh, int(np.ceil((yb + py) / BIG_DS)) + th)
                if y1 - y0 < th or x1 - x0 < tw:
                    continue
                r = cv2.matchTemplate(small[y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, _ = cv2.minMaxLoc(r)
                if np.isfinite(mx) and mx > per.get(d, -2.0):
                    per[d] = float(mx)
        return per

    def read(self, roi_bgr: np.ndarray, side: Side) -> tuple[Reading | None, dict]:
        """Returns (reading or None, debug dict with the best score per digit).

        Count 10 with mode "tens" = a clear tens "1" whose ones digit could not
        be read (covered by the attacker): a 10+ counter, 10 is a lower bound."""
        if not self.tens:
            return None, {}
        small = _gray_small(roi_bgr, BIG_DS)
        tens_slot, ones_slot = BIG_SLOTS[side]
        tens = self._slot(small, self.tens, tens_slot)
        ones = self._slot(small, list(self.scaled), ones_slot)
        dbg: dict = {"tens": tens, "ones": ones}
        rt = sorted(tens.items(), key=lambda kv: -kv[1])
        t_digit, t_score = rt[0] if rt else (1, -1.0)
        t_margin = t_score - (rt[1][1] if len(rt) > 1 else -1.0)
        ranked = sorted(ones.items(), key=lambda kv: -kv[1])
        o_digit, o_score = ranked[0] if ranked else (0, -1.0)
        o_margin = o_score - (ranked[1][1] if len(ranked) > 1 else -1.0)
        # a plain "1" bar also fits the straight strokes of 0/4/8: ones = 1
        # needs a clear lead
        min_margin = BIG_ONES_MIN_MARGIN_1 if o_digit == 1 else BIG_ONES_MIN_MARGIN
        o_min = BIG_ONES_RELAXED_MIN_SCORE if t_score >= BIG_TENS_STRONG else BIG_ONES_MIN_SCORE
        if o_score >= o_min and o_margin >= min_margin:
            if t_score >= BIG_TENS_MIN_SCORE and t_margin >= BIG_TENS_MIN_MARGIN and t_score + o_score >= BIG_PAIR_MIN_SUM:
                return Reading(10 * t_digit + o_digit, min(t_score, o_score), o_margin, "big"), dbg
            if o_score >= BIG_ONES_ALONE_MIN_SCORE and o_margin >= BIG_ONES_ALONE_MIN_MARGIN:
                return Reading(10 + o_digit, o_score, o_margin, "big"), dbg
        if t_digit == 1 and t_score >= BIG_TENS_ALONE_MIN_SCORE and t_margin >= BIG_TENS_ALONE_MIN_MARGIN:
            return Reading(10, t_score, t_margin, "tens"), dbg
        return None, dbg


class ComboCounterExtractor(BaseExtractor):
    """Per-frame combo counter: which side shows a counter and its hit number."""

    def __init__(self, templates_path: Path | None = None, stride: int = 2):
        """``stride`` > 1 reads only every stride-th frame and repeats that
        reading on the frames in between (the counter stays up for many
        frames; matching costs ~40 ms per frame on 1080p)."""
        self.stride = max(1, int(stride))
        self._held: tuple[Side, int] | None = None
        self._held_frame = -10**9
        tm = load_templates(templates_path)
        words = {int(k[1:]): v for k, v in tm.items() if k.startswith("w")}
        # "g<digit>[variant]" -> keyed "<digit>[variant]"
        glyphs = {k[1:]: v for k, v in tm.items() if k.startswith("g")}
        self.word = _Matcher(words, WORD_HEIGHTS, WORD_DS)
        self.big = _BigMatcher(glyphs)
        self.enabled = bool(words)
        # Per-side readings from the last frame (for debugging / tests).
        self.last: dict[Side, Reading | None] = {}
        self.last_debug: dict[Side, dict] = {}

    @property
    def required_rois(self) -> list[str]:
        return list(WORD_ROI.values()) + list(BIG_ROI.values())

    def read_word(self, roi_bgr: np.ndarray) -> Reading | None:
        (per,) = self.word.best(_gray_small(roi_bgr, WORD_DS))
        return _pick(per, WORD_MIN_SCORE, WORD_MIN_MARGIN, "word")

    def read_big(self, roi_bgr: np.ndarray, side: Side) -> Reading | None:
        r, dbg = self.big.read(roi_bgr, side)
        self.last_debug[side] = dbg
        return r

    def read_side(self, ctx_rois: dict, side: Side) -> Reading | None:
        best: Reading | None = None
        roi = ctx_rois.get(WORD_ROI[side])
        if roi is not None and roi.size:
            best = self.read_word(roi)
        roi = ctx_rois.get(BIG_ROI[side])
        if roi is not None and roi.size:
            r = self.read_big(roi, side)
            if r is not None and (best is None or r.score > best.score):
                best = r
        return best

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        if not self.enabled:
            return
        if self.stride > 1 and ctx.frame_number % self.stride:
            if self._held is not None and ctx.frame_number - self._held_frame < self.stride:
                data.combo_side, data.combo_count = self._held
            return
        self._held_frame = ctx.frame_number
        self.last = {side: self.read_side(ctx.rois, side) for side in (Side.P1, Side.P2)}
        cands = [(s, r) for s, r in self.last.items() if r is not None]
        if not cands:
            self._held = None
            return
        side, r = max(cands, key=lambda sr: sr[1].score)
        data.combo_side = side
        data.combo_count = r.count
        self._held = (side, r.count)


# ── Combo list ───────────────────────────────────────────────────────────────


@dataclass
class Combo:
    attacker: Side
    frame_start: int
    frame_end: int
    start_ms: float
    end_ms: float
    max_hits: int
    hp_before: float | None  # defender HP (0..1) just before the counter appears
    hp_after: float | None  # defender HP (0..1) just after the counter disappears
    damage: float | None  # hp_before - hp_after (None if either side is unknown)


def _smooth_readings(frames: list[FrameData], min_run: int = 3) -> list[tuple[Side | None, int | None]]:
    """Per-frame (side, count), dropping side readings that last < min_run frames
    (3 frames = 2 independent reads at the extractor's default stride of 2)
    and count readings that disagree with both neighbours (single-frame misreads)."""
    raw = [(f.combo_side, f.combo_count) for f in frames]
    n = len(raw)
    out: list[tuple[Side | None, int | None]] = [(None, None)] * n
    i = 0
    while i < n:
        side = raw[i][0]
        j = i
        while j < n and raw[j][0] == side:
            j += 1
        if side is not None and j - i >= min_run:
            for k in range(i, j):
                out[k] = raw[k]
        i = j
    # single-frame count outliers inside a run: replace by the previous value
    for k in range(1, n - 1):
        s, c = out[k]
        if s is None:
            continue
        ps, pc = out[k - 1]
        ns, nc = out[k + 1]
        if ps == s == ns and pc is not None and pc == nc and c != pc:
            out[k] = (s, pc)
    return _drop_count_glitches(out)


def _drop_count_glitches(
    seq: list[tuple[Side | None, int | None]], max_len: int = 8, max_gap: int = 45
) -> list[tuple[Side | None, int | None]]:
    """Drop short count runs that break the order of their neighbours.

    Within a combo the count only goes up. A short run (< ``max_len`` frames)
    of count c between runs p and n of the same side (each within ``max_gap``
    frames) with p <= n but not p <= c <= n is a misread: "three" read as
    "nine" under occlusion (3, 9, 4) or a big digit misread (17, 11, 18). If
    the count drops (p > n) a new combo may have started, so nothing is
    touched there."""
    out = list(seq)
    runs: list[list] = []  # [side, count, first, last]
    for i, (s, c) in enumerate(out):
        if s is None or c is None:
            continue
        if runs and runs[-1][0] == s and runs[-1][1] == c and i - runs[-1][3] <= max_gap:
            runs[-1][3] = i
        else:
            runs.append([s, c, i, i])
    for k in range(1, len(runs) - 1):
        ps, pc, _, pl = runs[k - 1]
        s, c, f, l = runs[k]
        ns, nc, nf, _ = runs[k + 1]
        if not (ps == s == ns) or f - pl > max_gap or nf - l > max_gap or l - f + 1 >= max_len:
            continue
        if nc == 10 and c > 10:
            continue  # a covered-ones "10" is only a lower bound
        if pc <= nc and not (pc <= c <= nc):
            for i in range(f, l + 1):
                out[i] = (None, None)
            runs[k][1] = pc  # treat as the previous count for the next check
    return out


def _confirmed_max(counts: list[tuple[int, int]], min_run: int = 3) -> int:
    """Highest count that is either held for >= min_run consecutive frames or
    followed later in the combo by a higher reading. One-frame
    misreads (13 -> 14 -> 13) are dropped; if nothing qualifies, the median."""
    runs: list[tuple[int, int]] = []  # (count, run length)
    prev_i = None
    for i, c in counts:
        if runs and runs[-1][0] == c and prev_i is not None and i == prev_i + 1:
            runs[-1] = (c, runs[-1][1] + 1)
        else:
            runs.append((c, 1))
        prev_i = i
    ok = []
    for k, (c, n) in enumerate(runs):
        later = [rc for rc, _ in runs[k + 1 :]]
        if n >= min_run or (later and max(later) > c):
            ok.append(c)
    if ok:
        return max(ok)
    vals = sorted(c for _, c in counts)
    return vals[len(vals) // 2]


def _hp(f: FrameData, side: Side) -> float | None:
    r = f.p1_health if side == Side.P1 else f.p2_health
    return None if r is None else r.health_pct


def derive_combos(
    frames: list[FrameData],
    max_gap_frames: int = 45,
    min_frames: int = 3,
    hp_window: int = 20,
    lookback_frames: int = 45,
    short_gap_frames: int = 10,
    linger_frames: int = 30,
    long_gap_frames: int = 120,
) -> list[Combo]:
    """Group per-frame counter readings into combos.

    A combo is a run of frames where the same side shows a counter, allowing
    gaps of up to ``max_gap_frames`` (occlusion, flashes, the counter fading).
    A new combo starts when the side changes, or when the count falls back to
    2-3 after having been higher (a fresh combo right after the last one).
    A 10+ count just above the last one continues the combo after a gap of up
    to ``long_gap_frames`` (super cinematics hide the counter). After a gap of
    more than ``short_gap_frames`` the count must go up, or stay at the same
    4+ count (the counter was briefly hidden).

    ``max_hits`` comes from ``_confirmed_max`` (drops one-frame misreads).

    Damage = defender HP before the combo minus the lowest defender HP from the
    last counter frame to ``hp_window`` frames after it. The counter only
    appears on the 2nd hit, so "before" is the HIGHEST defender reading in the
    ``lookback_frames`` before the first counter frame (HP only goes down
    during a combo), never reaching back into the previous combo.
    """
    seq = _smooth_readings(frames)
    combos: list[Combo] = []
    cur: dict | None = None

    def close(c: dict):
        if c["last"] - c["first"] + 1 < min_frames or len(c["counts"]) < 2:
            return
        max_hits = _confirmed_max(c["counts"])
        att: Side = c["side"]
        floor = combos[-1].frame_end if combos else -1
        lo = c["first"] - lookback_frames
        while lo < c["first"] and frames[max(lo, 0)].frame_number <= floor:
            lo += 1
        before, after = _defender_hp(frames, att, max(lo, 0), c["first"], c["last"], hp_window)
        dmg = None
        if before is not None and after is not None:
            dmg = max(0.0, before - after)
        combos.append(
            Combo(
                attacker=att,
                frame_start=frames[c["first"]].frame_number,
                frame_end=frames[c["last"]].frame_number,
                start_ms=frames[c["first"]].timestamp_ms,
                end_ms=frames[c["last"]].timestamp_ms,
                max_hits=max_hits,
                hp_before=before,
                hp_after=after,
                damage=dmg,
            )
        )

    for i, (side, count) in enumerate(seq):
        if side is None or count is None:
            continue
        if (
            cur is not None
            and side == cur["side"]
            and count == 10
            and cur["counts"][-1][1] > 10
            and i - cur["last"] <= max_gap_frames
        ):
            # "10" after a higher count is a covered 1x counter (lower bound)
            # of the same combo: it extends the combo but adds no count (so
            # it cannot confirm a one-off misread either)
            cur["last"] = i
            continue
        if cur is not None:
            gap = i - cur["last"]
            last_count = cur["counts"][-1][1]
            held = cur["last"] - cur["hold_start"] + 1
            # a big (10+) count just above the last one continues the combo
            # even after a long gap (super cinematic, round-end slash): a
            # fresh combo is first read at 2-3, never at 10+
            resumed = count >= 10 and last_count < count <= last_count + 3 and gap <= long_gap_frames
            new = (
                side != cur["side"]
                or (gap > max_gap_frames and not resumed)
                # a fresh combo starts at 2-3 hits
                or (count <= 3 and _confirmed_max(cur["counts"]) > count)
                # after the counter vanished for a while, the same combo can
                # only continue upwards (or show the same 4+ count again: a
                # fresh combo would be read at 2-3 first). If the old count had been lingering (the
                # counter stays up ~1 s after a combo ends) a reading of 2-3 is
                # a fresh combo; a higher one is the same combo going on (a
                # combo can hold one count for 1-2 s, e.g. 4 -> 5 after a long
                # juggle, and a new combo is first read at 2-3)
                or (
                    gap > short_gap_frames
                    and (
                        count < last_count
                        or (count == last_count and count <= 3)
                        or (held >= linger_frames and count <= 3)
                    )
                )
            )
            if new:
                close(cur)
                cur = None
        if cur is None:
            cur = {"side": side, "first": i, "last": i, "counts": [(i, count)], "hold_start": i}
        else:
            if count != cur["counts"][-1][1]:
                cur["hold_start"] = i
            cur["last"] = i
            cur["counts"].append((i, count))
    if cur is not None:
        close(cur)
    return combos


def _defender_hp(frames: list[FrameData], attacker: Side, lo: int, first: int, last: int, window: int):
    defender = Side.P2 if attacker == Side.P1 else Side.P1
    vals = [_hp(frames[k], defender) for k in range(lo, first + 1)]
    vals = [v for v in vals if v is not None]
    before = max(vals) if vals else None
    vals = [_hp(frames[k], defender) for k in range(last, min(len(frames), last + 1 + window))]
    vals = [v for v in vals if v is not None]
    after = min(vals) if vals else None
    return before, after


def combos_to_records(combos: list[Combo]) -> list[dict]:
    return [
        {
            "attacker": "P1" if c.attacker == Side.P1 else "P2",
            "frame_start": c.frame_start,
            "frame_end": c.frame_end,
            "start_s": round(c.start_ms / 1000.0, 3),
            "end_s": round(c.end_ms / 1000.0, 3),
            "max_hits": c.max_hits,
            "hp_before": None if c.hp_before is None else round(c.hp_before, 4),
            "hp_after": None if c.hp_after is None else round(c.hp_after, 4),
            "damage": None if c.damage is None else round(c.damage, 4),
        }
        for c in combos
    ]
