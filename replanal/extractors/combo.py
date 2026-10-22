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

from concurrent.futures import ThreadPoolExecutor
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
# A template cut at the ROI bottom (see _BigMatcher._slot) must keep at least
# this fraction of its rows.
BIG_MIN_KEEP = 0.6
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
# "11" from a pair needs more: the banner's vertical strokes match a "1" in
# both slots at ~0.60-0.72 (sum ~1.35) once the ones slot no longer sees the
# bottom of the screen; real 11 pairs measured 1.54+ (a clear ones "1" also
# reads via BIG_ONES_ALONE_MIN_SCORE).
BIG_PAIR_MIN_SUM_11 = 1.45
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
# ggstsets/league videos (game HUD only; P1 ROI starts at x=40, P2 at x=860).
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
            # Lowest template top inside the slot. A glyph placed there can
            # reach below the ROI (which stops above the screen bottom, where
            # streams draw their own logos): its template is cut to the rows
            # that fit, the same cut at every position, so pixels below the
            # ROI never take part in the match.
            y_low = int(np.ceil((yb + py) / BIG_DS))
            for key in keys:
                t = self.scaled[key][k]
                d = int(key[0])
                th, tw = t.shape
                if y_low + th > sh:
                    keep = sh - y_low
                    if keep < BIG_MIN_KEEP * th:
                        continue
                    t = t[:keep]
                    th = keep
                x1 = min(sw, int(np.ceil((xb - px) / BIG_DS)) + tw)
                y1 = min(sh, y_low + th)
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
            pair_min = BIG_PAIR_MIN_SUM_11 if (t_digit, o_digit) == (1, 1) else BIG_PAIR_MIN_SUM
            if t_score >= BIG_TENS_MIN_SCORE and t_margin >= BIG_TENS_MIN_MARGIN and t_score + o_score >= pair_min:
                return Reading(10 * t_digit + o_digit, min(t_score, o_score), o_margin, "big"), dbg
            if o_score >= BIG_ONES_ALONE_MIN_SCORE and o_margin >= BIG_ONES_ALONE_MIN_MARGIN:
                return Reading(10 + o_digit, o_score, o_margin, "big"), dbg
        if t_digit == 1 and t_score >= BIG_TENS_ALONE_MIN_SCORE and t_margin >= BIG_TENS_ALONE_MIN_MARGIN:
            return Reading(10, t_score, t_margin, "tens"), dbg
        return None, dbg


# After the last frame with a counter, every frame is still read for this
# many samples (the counter flickers under occlusion and VFX).
ACTIVE_GRACE = 15


class ComboCounterExtractor(BaseExtractor):
    """Per-frame combo counter: which side shows a counter and its hit number."""

    def __init__(self, templates_path: Path | None = None, stride: int = 3, workers: int = 4):
        """``stride`` > 1 reads only every stride-th frame while no counter is
        shown (matching costs ~60 ms per frame).  Once one is found every
        frame is read until none has been seen for ACTIVE_GRACE samples, and
        the frames the stride skipped just before the find are read too, so
        the output does not depend on which frames the stride lands on (a cut
        of the same video starting one frame later reads the same). A counter
        shown on fewer than ``stride`` samples can be missed; grouping drops
        such blips anyway (``_smooth_readings`` keeps runs of 3+ samples).

        ``workers``: template matching runs in threads (OpenCV releases the
        GIL): 2 = the two sides in parallel, 4 = also the word and the big
        matcher of each side; 1 = all in the calling thread. The readings do
        not depend on it."""
        self.stride = max(1, int(stride))
        # Samples left in which every frame is read (counter seen recently).
        self._active = 0
        # Frames skipped by the stride since the last read: (frame number, rois, data).
        self._skipped: list[tuple[int, dict, FrameData]] = []
        # Spacing between consecutive sampled frame numbers (1 at 30 fps,
        # 2 when a 60 fps video is read every 2nd frame). The stride counts
        # samples, not source frames.
        self._step = 0
        self._prev_frame: int | None = None
        self.workers = max(1, int(workers))
        self._side_pool = ThreadPoolExecutor(2, "combo-side") if self.workers >= 2 else None
        self._big_pool = ThreadPoolExecutor(2, "combo-big") if self.workers >= 4 else None
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
        word_roi = ctx_rois.get(WORD_ROI[side])
        big_roi = ctx_rois.get(BIG_ROI[side])
        has_big = big_roi is not None and big_roi.size
        fut = self._big_pool.submit(self.read_big, big_roi, side) if has_big and self._big_pool else None
        best: Reading | None = None
        if word_roi is not None and word_roi.size:
            best = self.read_word(word_roi)
        r = fut.result() if fut is not None else (self.read_big(big_roi, side) if has_big else None)
        if r is not None and (best is None or r.score > best.score):
            best = r
        return best

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        if not self.enabled:
            return
        if self._prev_frame is not None:
            d = ctx.frame_number - self._prev_frame
            if d > 0 and (self._step == 0 or d < self._step):
                self._step = d
        self._prev_frame = ctx.frame_number
        step = self._step or 1
        idx = ctx.frame_number // step
        if self.stride > 1 and idx % self.stride and self._active <= 0:
            self._skipped.append((ctx.frame_number, ctx.rois, data))
            del self._skipped[: -(self.stride - 1)]
            return
        found = self._read_into(ctx.rois, data)
        skipped, self._skipped = self._skipped, []
        if found:
            if self._active <= 0:
                # The counter may have appeared on frames the stride skipped:
                # read the ones right before this one (oldest first).
                back = [sk for sk in skipped if 0 < ctx.frame_number - sk[0] < self.stride * step]
                for _, rois, d in back:
                    self._read_into(rois, d)
            self._active = ACTIVE_GRACE
        else:
            self._active -= 1

    def _read_into(self, rois: dict, data: FrameData) -> bool:
        if self._side_pool is not None:
            futs = {side: self._side_pool.submit(self.read_side, rois, side) for side in (Side.P1, Side.P2)}
            self.last = {side: f.result() for side, f in futs.items()}
        else:
            self.last = {side: self.read_side(rois, side) for side in (Side.P1, Side.P2)}
        cands = [(s, r) for s, r in self.last.items() if r is not None]
        if not cands:
            return False
        side, r = max(cands, key=lambda sr: sr[1].score)
        data.combo_side = side
        data.combo_count = r.count
        return True


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
    (the extractor reads every frame while a counter is shown) and count
    readings that disagree with both neighbours (single-frame misreads)."""
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


def _pending(f: FrameData, side: Side) -> float | None:
    """Red pending segment (HP already lost in the running combo) of a frame
    whose bar was read cleanly: its own fill reading agrees with the cleaned
    HP. None when the bar was covered or misread (a character or an effect
    over the bar reads as no red)."""
    raw = f.p1_health_raw if side == Side.P1 else f.p2_health_raw
    clean = f.p1_health if side == Side.P1 else f.p2_health
    if raw is None:
        raw = clean
    if raw is None:
        return None
    fill = raw.raw_pct if raw.raw_pct is not None else raw.health_pct
    if clean is not None and abs(fill - clean.health_pct) > CLEAN_READ_TOL:
        return None
    return raw.pending_pct


# ── HUD state (game terms: is the HUD drawn, which round is it) ─────────────

# A run of at least this many samples without a round timer = HUD hidden
# (super cinematic, wallbreak, round-end slash, menus). Shorter runs are
# timer misreads.
HIDDEN_MIN_RUN = 4
# Below this share of frames with a timer the timer is not trusted at all
# (no HUD state: every frame counts as visible, a single round).
TIMER_MIN_SHARE = 0.05
# A timer reading is supported when at least TIMER_SUPPORT_RUN readings
# (itself included) within +-TIMER_SUPPORT_WIN samples show the same value
# (the clock shows each value for a second).  Only supported readings decide
# whether a hidden stretch ended the round: a lone misread next to the gap
# (a covered tens digit read "7" for 97) must not.
TIMER_SUPPORT_RUN = 3
TIMER_SUPPORT_WIN = 15
# Red pending segment at least this wide (fraction of the bar) counts as shown.
PENDING_SHOWN = 0.004
# A frame's red is only trusted when its fill reading is this close to the
# cleaned HP (else the bar was covered or misread).
CLEAN_READ_TOL = 0.02
# A combo that took less than this from the defender dealt no hit: a counter
# still on screen from an earlier combo (or after the KO).
MIN_COMBO_DAMAGE = 0.003
# hp_before: hits of one combo are never this far apart (samples, like
# derive_combos' max_gap_frames); an earlier HP drop belongs to something else.
HIT_PAUSE = 45
# hp_before: the red pending segment off (under RED_CLEAR wide: slivers are
# left while it drains) on this many cleanly read frames in a row means no
# combo was running there.
RED_OFF_RUN = 8
RED_CLEAR = 0.015
# hp_before: a steady bar total (fill + red; frames within TOTAL_STEADY of
# each other) this far above the combo's own on this many cleanly read frames
# in a row is an earlier combo's red. (Only above: the red is sometimes read
# short over a red-ish stage.)
TOTAL_TOL = 0.06
TOTAL_STEADY = 0.02
TOTAL_MISMATCH_RUN = 3
# Share of HUD frames in a gap that must show the defender's red for the
# combo to be bridged across it.
RED_HELD_SHARE = 0.9
# A 10+ count may come back after a gap having gone up by this many hits per
# second of gap (the counter was not read in between).
RESUME_HITS_PER_S = 5.0


@dataclass
class HudState:
    hidden: np.ndarray  # bool per frame: HUD not drawn
    over: np.ndarray  # bool per frame: inside the hidden gap that ended a round
    round_id: np.ndarray  # int per frame: goes up by one when a new round starts


def hud_state(frames: list[FrameData]) -> HudState:
    """Per-frame HUD state from the round timer (game-native HUD only).

    The timer is part of the HUD: it is hidden during super cinematics,
    wallbreaks and the round transition, and only counts down within a round.
    A hidden stretch ended the round when the timer comes back higher than it
    was (a new round starts at 99; the next reading must agree too), or when
    no timer comes back at all.  Before and after the gap the nearest
    supported readings (see TIMER_SUPPORT_RUN) are compared when there are
    any (else the raw ones)."""
    n = len(frames)
    timer = [f.timer_value if f.is_gameplay else None for f in frames]
    hidden = np.zeros(n, dtype=bool)
    over = np.zeros(n, dtype=bool)
    round_id = np.zeros(n, dtype=int)
    n_timer = sum(t is not None for t in timer)
    if n == 0 or n_timer < TIMER_MIN_SHARE * n:
        return HudState(hidden, over, round_id)
    tv = np.array([np.nan if t is None else float(t) for t in timer])
    same = np.zeros(n, dtype=int)
    for d in range(-TIMER_SUPPORT_WIN + 1, TIMER_SUPPORT_WIN):
        lo, hi = max(0, -d), min(n, n - d)
        same[lo:hi] += tv[lo:hi] == tv[lo + d:hi + d]
    sup = same >= TIMER_SUPPORT_RUN
    last_sup = None  # the last supported timer value so far
    rid = 0
    prev = None  # index of the last frame with a timer
    i = 0
    while i < n:
        if timer[i] is not None:
            prev = i
            if sup[i]:
                last_sup = timer[i]
            round_id[i] = rid
            i += 1
            continue
        j = i
        while j < n and timer[j] is None:
            j += 1
        ended = False
        if j - i >= HIDDEN_MIN_RUN:
            hidden[i:j] = True
            if prev is not None:
                if j >= n:
                    ended = True
                else:
                    before = last_sup if last_sup is not None else timer[prev]
                    after = next((timer[k] for k in range(j, min(n, j + 2 * TIMER_SUPPORT_WIN)) if sup[k]), None)
                    if after is not None:
                        ended = after >= before + 2
                    else:
                        nxt = next((timer[k] for k in range(j + 1, min(n, j + 30)) if timer[k] is not None), timer[j])
                        ended = timer[j] >= before + 2 and nxt >= before + 2
        round_id[i:j] = rid
        if ended:
            over[i:j] = True
            rid += 1
        i = j
    return HudState(hidden, over, round_id)


def derive_combos(
    frames: list[FrameData],
    max_gap_frames: int = 45,
    min_frames: int = 3,
    hp_window: int = 3,
    lookback_frames: int = 150,
    short_gap_frames: int = 10,
    linger_frames: int = 30,
    long_gap_frames: int = 120,
    max_bridge_frames: int = 240,
    sample_fps: float = 30.0,
    hud: HudState | None = None,
) -> list[Combo]:
    """Group per-frame counter readings into combos.

    A combo is a run of frames where the same side shows a counter, allowing
    gaps of up to ``max_gap_frames`` (occlusion, flashes, the counter fading).
    Gaps are counted in frames where the HUD is drawn: while the HUD is
    hidden (super cinematic, wallbreak; see ``hud_state``) the combo goes on,
    for at most ``max_bridge_frames`` in all. A new combo starts when the
    side changes, or when the count falls back to 2-3 after having been
    higher (a fresh combo right after the last one). After a gap of more
    than ``short_gap_frames`` the count must go up, or stay at the same 4+
    count (the counter was briefly hidden). Longer gaps are bridged when:

    * the defender's red pending segment stayed on screen all through the
      gap (GGST drains it only after the combo ends) and the count did not
      fall back, or
    * a 10+ count came back higher, by at most RESUME_HITS_PER_S per second of
      the gap, within ``long_gap_frames`` (a fresh combo is first read at
      2-3, never at 10+).

    No combo continues into the next round; a counter seen in the hidden gap
    after a round ended only extends the combo that was running.

    A combo must deal damage: one whose defender lost no HP (less than
    MIN_COMBO_DAMAGE) is a counter still on screen from an earlier combo (it
    lingers ~1 s; after a KO it stays up through the slash), or a misread,
    and is dropped.

    ``max_hits`` comes from ``_confirmed_max`` (drops one-frame misreads).

    The combo ends when its counter goes away (it lingers ~1 s after the last
    hit, so the HP then is final): ``hp_after`` = lowest defender HP within
    ``hp_window`` frames of the last counter frame. Later HP loss (the red
    pending segment draining, chip damage) is not the combo's.

    ``hp_before`` = defender HP before hit 1 (see ``_hp_before``).
    """
    seq = _smooth_readings(frames)
    n = len(frames)
    hud = hud or hud_state(frames)
    hcum = np.cumsum(hud.hidden)
    combos: list[Combo] = []
    kept: list[dict] = []  # group dicts of ``combos`` (same order)
    cur: dict | None = None

    def hidden_between(a: int, b: int) -> int:
        """Hidden-HUD frames strictly between frame indices a < b."""
        return int(hcum[b - 1] - hcum[a]) if b - 1 > a else 0

    def round_start(k: int) -> int:
        r = hud.round_id[k]
        while k > 0 and hud.round_id[k - 1] == r:
            k -= 1
        return k

    def defender_of(side: Side) -> Side:
        return Side.P2 if side == Side.P1 else Side.P1

    def red_held(a: int, b: int, defender: Side) -> bool:
        """The defender's red pending segment stayed on screen between frames
        a and b (on RED_HELD_SHARE of the HUD frames with a reading: single
        frames miss it): the combo never ended."""
        seen = shown = 0
        for k in range(a + 1, b):
            if hud.hidden[k]:
                continue
            q = _pending(frames[k], defender)
            if q is None:
                continue
            seen += 1
            shown += q >= PENDING_SHOWN
        return seen > 0 and shown >= RED_HELD_SHARE * seen

    def build(c: dict) -> Combo:
        att: Side = c["side"]
        defender = defender_of(att)
        first, last = c["first"], c["last"]
        floor = round_start(first)
        prev_on = [k["last"] for k in kept if k["side"] == att and k is not c and k["last"] < first]
        if prev_on:
            floor = max(floor, prev_on[-1] + 1)
        before = _hp_before(frames, defender, first, last, floor, lookback_frames, hud.hidden)
        vals = [_hp(frames[k], defender) for k in range(last, min(n, last + 1 + hp_window))]
        vals = [v for v in vals if v is not None]
        if not vals:
            # no HP on screen when the counter went (round over, HUD gone):
            # the last HP shown during the combo
            vals = [v for k in range(last, first - 1, -1) if (v := _hp(frames[k], defender)) is not None][:1]
        after = min(vals) if vals else None
        dmg = None
        if before is not None and after is not None:
            dmg = max(0.0, before - after)
        return Combo(
            attacker=att,
            frame_start=frames[first].frame_number,
            frame_end=frames[last].frame_number,
            start_ms=frames[first].timestamp_ms,
            end_ms=frames[last].timestamp_ms,
            max_hits=_confirmed_max(c["counts"]),
            hp_before=before,
            hp_after=after,
            damage=dmg,
        )

    def close(c: dict):
        if c["last"] - c["first"] + 1 < min_frames or len(c["counts"]) < 2:
            return
        combo = build(c)
        if combo.damage is not None and combo.damage < MIN_COMBO_DAMAGE:
            return  # no hit landed: a counter still on screen
        kept.append(c)
        combos.append(combo)

    for i, (side, count) in enumerate(seq):
        if side is None or count is None:
            continue
        if cur is not None and hud.round_id[i] != hud.round_id[cur["last"]]:
            close(cur)  # a new round started since
            cur = None
        raw_gap = i - cur["last"] if cur is not None else 0
        gap = raw_gap - hidden_between(cur["last"], i) if cur is not None else 0
        if raw_gap > max_bridge_frames:
            gap = raw_gap
        last_count = cur["counts"][-1][1] if cur is not None else 0
        same = cur is not None and side == cur["side"]
        held_red = (
            same and gap > short_gap_frames and raw_gap <= max_bridge_frames
            and red_held(cur["last"], i, defender_of(side))
        )
        if same and count == 10 and last_count > 10 and (gap <= max_gap_frames or held_red):
            # "10" after a higher count is a covered 1x counter (lower bound)
            # of the same combo: it extends the combo but adds no count (so
            # it cannot confirm a one-off misread either)
            cur["last"] = i
            continue
        if hud.over[i]:
            # Round over: the KO combo's counter stays on screen through the
            # slash. It can still go up (the KO hit), it never starts a combo.
            if same and count >= last_count and (gap <= max_gap_frames or held_red):
                if count != last_count:
                    cur["hold_start"] = i
                cur["last"] = i
                cur["counts"].append((i, count))
            continue
        if cur is not None:
            held = cur["last"] - cur["hold_start"] + 1
            # a 10+ count above the last one continues the combo after a
            # longer gap (super cinematic, effects over the counter): a fresh
            # combo is first read at 2-3, never at 10+
            resumed = (
                count >= 10 and last_count < count
                and count - last_count <= max(3, RESUME_HITS_PER_S * gap / sample_fps)
                and gap <= long_gap_frames
            )
            # the defender's red never drained and the count did not fall
            # back: the same combo went on (2-3 is how a fresh combo is
            # first read, so those are not bridged)
            bridged = held_red and count >= max(last_count, 4)
            new = (
                not same
                or (gap > max_gap_frames and not resumed and not bridged)
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
                    and not bridged
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


def _bar_total(f: FrameData, side: Side) -> float | None:
    """Fill + red pending of a cleanly read bar (see ``_pending``): the HP the
    defender had before the hits whose red is still shown."""
    q = _pending(f, side)
    if q is None:
        return None
    raw = f.p1_health_raw if side == Side.P1 else f.p2_health_raw
    raw = raw or (f.p1_health if side == Side.P1 else f.p2_health)
    return (raw.raw_pct if raw.raw_pct is not None else raw.health_pct) + q


def _hp_before(
    frames: list[FrameData], defender: Side, first: int, last: int, floor: int, lookback: int, hidden: np.ndarray
) -> float | None:
    """Defender HP before hit 1.

    The counter appears on the 2nd hit and is sometimes read late, so the HP
    at the first counter frame is already short of hit 1 (and more). While a
    combo runs GGST keeps the HP it took on screen as the red pending
    segment and drains it only after the combo; so the bar total (fill + red)
    stays at the HP the defender had before the combo.

    The defender's HP (cleaned: it only falls within a round) is followed back
    from the first counter frame through the HP drops of the combo. The walk
    stops before a drop that cannot belong to this combo:

    * the red was off on RED_OFF_RUN cleanly read frames in a row (it drained:
      no combo was running there);
    * the red was shown with a steady bar total above the combo's own
      (median over the combo) on TOTAL_MISMATCH_RUN cleanly read frames in a
      row: an earlier combo's red (it started from more HP), still draining;
    * HIT_PAUSE frames passed without a drop (hits of a combo come faster);
    * the attacker's previous combo or the round start (``floor``), or
      ``lookback`` frames.

    The nearest drop before the first counter frame always counts (on some
    hits the red is not read at all).

    ``hp_before`` = the HP just before the earliest drop reached.
    """
    floor = max(0, min(floor, first), first - lookback)
    totals = [t for k in range(first, last + 1) if not hidden[k] and (t := _bar_total(frames[k], defender)) is not None]
    ref = float(np.median(totals)) if totals else None
    pre = first  # frame just before the earliest connected HP drop
    quiet = red_off = mismatch = 0
    prev_t = None
    later = None  # HP at the frame after k
    for k in range(first, floor - 1, -1):
        q = None if hidden[k] else _pending(frames[k], defender)
        # (only once a drop was reached: the nearest drop before the counter
        # is the combo's own, red read or not)
        if q is not None and k < pre < first:
            red_off = red_off + 1 if q < RED_CLEAR else 0
            if q >= RED_CLEAR and ref is not None:
                t = _bar_total(frames[k], defender)
                steady = prev_t is not None and abs(t - prev_t) <= TOTAL_STEADY
                mismatch = mismatch + 1 if t - ref > TOTAL_TOL and (steady or mismatch == 0) else 0
                prev_t = t
            if red_off >= RED_OFF_RUN or mismatch >= TOTAL_MISMATCH_RUN:
                break
        v = _hp(frames[k], defender)
        if v is not None and later is not None and v > later + 0.002:
            pre = k
            quiet = red_off = mismatch = 0
            prev_t = None
        else:
            quiet += 1
            if quiet > HIT_PAUSE:
                break
        if v is not None:
            later = v
    v = _hp(frames[pre], defender)
    if v is None:
        vals = [x for k in range(max(floor, first - lookback), first + 1) if (x := _hp(frames[k], defender)) is not None]
        v = max(vals) if vals else None
    return v


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
