"""Temporal cleaning of per-frame health readings.

The per-frame extractor (``replanal.extractors.health``) reports what each
frame shows, plus a confidence.  Occlusion, super flashes and VFX make single
frames wrong in both directions.  This module applies GGST's rules to the
whole series:

1. Within a round, health only goes DOWN.  The only way up is the reset to
   full at the start of the next round.  (Faust's healing items are the one
   exception; they heal a few percent and are flattened by the fit.)
2. Rounds are split where both bars are seen full again for a sustained run
   after real damage was taken.
3. Within a round, each side is fitted with a weighted L1 isotonic
   (non-increasing) regression.  L1 makes it a "monotone median": short runs
   of bad readings in either direction (occlusion spikes, false zeros from a
   flash, a sprite making the bar look full) are outvoted by the readings
   around them instead of being averaged in.
4. Frames without a usable reading inside a round (super cinematics, the bar
   hidden by a sprite during a combo) are filled by interpolating between the
   fitted values on either side, so the bar keeps going down through the gap
   and never comes back up.
5. The red pending segment is already-lost health: the extractor does not
   count it as fill, and the fit uses fill only.  ``pending`` is kept so damage
   events can tell when a combo is still running.

Everything works on plain numpy arrays so it can be tested without video.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Readings below this confidence are not used by the fit (they are still
# replaced by the fitted value in the output).
MIN_CONF = 0.35
# Empty-bar readings (KO, but also many occlusion/flash artifacts) get their
# weight scaled down: a real KO is a long sustained run and still wins.
EMPTY_PCT = 0.005
EMPTY_WEIGHT = 0.5
# An empty reading is credible when this much pending red is visible next to
# it, or when the side was already at or below LOW_HP.
PENDING_MIN = 0.004
LOW_HP = 0.20
# A new (lower) level must be supported by at least this much total
# confidence weight (about 4 good frames) to be accepted.
MIN_STEP_WEIGHT = 3.0
# Round reset: both sides' rolling median "full" for RESET_RUN frames, after
# either side went below "damaged" in this round.  Full = min(FULL_PCT, the
# side's observed full-bar reading - 0.05); damaged = full - 0.05.
FULL_PCT = 0.90
DAMAGED_PCT = 0.85
RESET_RUN = 15
ROLL = 9
# Alternative reset: both sides at least RISE above their lows in the round
# (and >= RISE_MIN), with one of them full.  Lows use a ROLL_LOW-frame median
# and skip the first LOW_SKIP frames of the segment.
RISE = 0.30
RISE_MIN = 0.60
ROLL_LOW = 31
LOW_SKIP = 90   # ignore the first 3 s of a segment (round-start screens) for lows
# Minimum round length between two resets (frames).
MIN_ROUND = 300
# Gaps in confident readings longer than this split the coverage (no fill).
MAX_FILL_GAP = 120
# A gap this long with no readings at all also starts a new segment, unless
# the round clock carries on across it (see _clock_continues).
SEGMENT_GAP = 150
# After one side is KO'd, the other side's readings are only used for this
# many more frames (allows a double KO); after that its HP is held.
DOUBLE_KO_FRAMES = 30
# A side counts as KO'd once its fit stays at/below this level.
KO_PCT = 0.005
# Raw KO evidence without the red segment: this many confident empty
# readings within KO_WIN frames (see _raw_ko).
KO_CONFIRM = 3
KO_WIN = 10
# ... and the side's readings over the next KO_AFTER frames must stay below
# min(KO_NEAR, half its level before the KO).  Readings at or below KO_FLOOR
# count as empty there: over an empty bar the KO cinematic reads a steady
# 1-11 px (<= 0.0153) of junk fill.
KO_AFTER = 45
KO_NEAR = 0.015
KO_FLOOR = 0.016
# The empty reading itself needs END_CONF, or only MIN_CONF when the red
# pending segment covers the side's whole remaining level (within END_TOL)
# and the frame before read the bar with fill at END_CONF: the green was
# visibly there and the red is drawn where it was, so the HUD is not hidden.
# (Without that confident frame before, a sliver under a busy super
# background reads empty with middling confidence while the combo goes on.)
OPP_OK_TOL = 0.10
# A bar "comes back" (see _comes_back) only with at least this much fill:
# static transition screens read a steady few percent.
BACK_MIN = 0.05
BACK_MAX = 0.80
# Round-end lock: the last run of >= END_RUN frames where both bars are read
# confidently marks the end of play in a segment.  After it (+ END_GRACE
# frames) readings may confirm a side's last level but not lower it, unless
# pending red is shown next to the reading (a real hit) or it is a credible
# empty-bar reading (the KO).  The KO cinematic and transition screens
# otherwise drag the survivor's HP down.
END_CONF = 0.6
END_RUN = 5
END_GRACE = 0
END_TOL = 0.02
# Play also ends at the last round-timer reading of a segment: the KO slash
# and transition screens hide the timer, so nothing after it is play.  A
# timer reading only counts with TIMER_RUN readings in the TIMER_WIN frames
# up to it (one-frame OCR misreads on transition screens do not).
TIMER_RUN = 3
TIMER_WIN = 15
# A segment is cleaned only up to KO_MARGIN frames after its last credible
# timer reading.  The KO can land under a super cinematic and the HUD come
# back without a readable timer for a few seconds (up to ~4.5 s seen), but
# later frames (win pose, transition, lobby, the next video) are never the
# round: without the cut, their junk readings changed the result depending
# on how much video followed.  The cut-off frames hold the last cleaned value.
KO_MARGIN = 150
# Frozen-clock KO: the HUD is up (timer read), the clock has stopped (the
# same value read over at least CLOCK_STOP frames, never lower afterwards)
# and one bar reads empty (<= HUD_EMPTY, robust to compression; confident)
# for HUD_KO_RUN frames in a row while the other bar shows fill, and is never
# read with fill again while the timer is shown.  A running clock ticks every
# second, so a frozen one with an empty bar is the KO even when the red
# pending segment is not seen (it went with the slash).  A live 1-px sliver
# reads 0.008 raw (Tappa #4, 14 R15), so empty stays below that, the same
# bound as the round detector's KO_HP (lib.rs).  Both were measured on one
# 1080p stream; other resolutions or sources may need a re-check.
CLOCK_STOP = 40
HUD_KO_RUN = 5
HUD_EMPTY = 0.005
# A round ends only by a KO or a time-out.  When no KO was seen but the next
# round starts (its timer intro within INTRO_BACK frames) and the clock last
# read above 0, the KO happened while the HUD was hidden (under a super or
# the slash): the side clearly behind (at or below LOW_HP, at least KO_LEAD
# below the other) is the loser, at 0 from just after the last timer reading.
# Not when that side's combo counter was up within HIDDEN_KO_COMBO frames of
# the last timer reading (the attacker essentially never loses HP during its
# own combo: it is the likely winner, e.g. a super landed from behind), nor
# when the clock last read below HIDDEN_KO_MIN_CLOCK (a time-out may have
# ended the round while the HUD was hidden; the loser keeps its HP).
KO_LEAD = 0.10
HIDDEN_KO_COMBO = 30
HIDDEN_KO_MIN_CLOCK = 4
# Attacker hold: while one side is comboing, its own HP stays where it was
# when the combo started (the player doing a combo essentially never loses HP
# during it).  A side is surely comboing while the combo counter is shown on
# its HUD side (COMBO_RUN+ consecutive frames), or while the other side's red
# pending segment keeps growing (2nd hit on, see _pending_growing) and this
# side shows no red itself: readings below its level from just before are
# occlusion (its own sprite, a super's VFX or a big object over its bar).
# While the other side merely shows red (one hit; its red lingers and this
# side can be hit back) only near-empty readings are dropped.
# OPP_PAD frames are added on both sides of each run; OPP_REF frames before a
# run give the reference level.
OPP_PAD = 6
OPP_REF = 15
COMBO_RUN = 3
# The reference needs OPP_REF_MIN confident readings; readings below it by
# more than OPP_TOL (reading noise), or below OPP_DROP x the reference (a tiny
# bar read empty), are dropped.  A real hit on the attacker (a trade, a
# reversal right after the combo) shows its own red pending, which ends the
# hold on those frames.
OPP_REF_MIN = 3
OPP_DROP = 0.5
OPP_TOL = 0.02
# An empty reading backs a KO with its red segment only when the red covers
# at least RED_COVER of the level read just before (see _clean_side).
RED_COVER = 0.5
# Pending red "growing": the run's highest pending reading rises by GROW_MIN.
GROW_MIN = 0.01
# Round intro: the timer already shows its start value (90 in GGST, or 99)
# while the bars are still empty / filling up.  A credible run of start
# readings within INTRO_BACK frames before a detected reset moves the reset
# back to it, so the empty intro bars do not end the previous round at 0.
TIMER_START_MIN = 89
INTRO_BACK = 150
# Timer-intro reset (fallback when the bars never read full, e.g. a bright
# stage washing out one bar or the replay input display covering the outer
# part of both): the round timer reappears after a gap of at least INTRO_GAP
# frames and reads start values for the next INTRO_SPAN frames (at least
# INTRO_MIN_READS readings, INTRO_TIMER_FRAC of them >= TIMER_START_MIN),
# after the segment's own timer had counted down below the start values; and
# the raw bars (any confidence: a washed-out bar reads with low confidence)
# rose: one side by at least RISE above its low in the segment, the other side
# not below its low - INTRO_TOL.  HP cannot rise within a round.
INTRO_GAP = 30
INTRO_SPAN = 90
INTRO_MIN_READS = 20
INTRO_TIMER_FRAC = 0.9
INTRO_BAR_READS = 20
INTRO_TOL = 0.05
# Near-double KO: both sides at or below NEAR_DKO when the KO lands.  A
# 1-px sliver flickers to "empty", so the first empty reading does not decide
# the loser: the bar still shown at its pre-KO level (within DKO_LEVEL_TOL)
# after the KO, while the other side reads empty, is the survivor.
NEAR_DKO = 0.03
DKO_LEVEL_TOL = 0.01
# Evidence is taken from the DKO_WIN frames after the KO only: later come the
# transition and the next round's intro screens, whose confident junk
# readings look like a sliver next to an empty bar.  A sliver reading only
# needs MIN_CONF (a 1-px bar is read with low confidence).
DKO_WIN = 210
# A KO ends the round: the clock stops and the HUD goes for the round-end
# slash.  A bar at or near 0 while the clock keeps counting down is a live
# sliver (the game keeps the last HP point visible as a 1-px sliver, which a
# 720p source cannot show at all).  Such a side is alive until the clock
# stops; its cleaned HP is kept at ALIVE_MIN or above until then (and to the
# end of the round if it survives), so "exactly 0" always means KO'd.
ALIVE_MIN = 0.008
# End-of-round evidence of who lost (see _end_loser), from the frames after
# the round-end slash in which the frozen round timer is read again (the HUD
# is back): a bar read empty (at most KO_FLOOR) next to one read at least
# HUD_SHOWN wider (a 720p sliver reads 2-3 px, an empty bar 0 px; at 1080p an
# empty bar can read a few px of junk, a sliver ~6 px) is the KO'd side.
HUD_SHOWN = 0.0025
# ... failing that, the side shown with the red pending segment (being hit)
# in the last END_HIT frames of play while the other side shows none, or the
# side whose combo counter (the attacker) is up in those last END_HIT frames
# of play, decides.  The counter lingers about a second after a combo ends,
# so an earlier counter says nothing about who landed the last hit (a
# one-hit reversal shows none).
END_HIT = 15
# A gap of this many frames without timer readings after the clock stopped
# is the HUD hidden by the round-end slash (or a KO cinematic).
HUD_GAP = 15


@dataclass
class CleanResult:
    p1: np.ndarray          # cleaned HP per frame (NaN = unknown)
    p2: np.ndarray
    segment: np.ndarray     # segment (round) id per frame, -1 = none
    resets: list[int]       # frame indices where a new segment starts


# ── Weighted L1 isotonic regression (non-increasing) ──────────────────────

def _wmedian(vals: np.ndarray, wts: np.ndarray) -> float:
    order = np.argsort(vals, kind="stable")
    v, w = vals[order], wts[order]
    cw = np.cumsum(w)
    k = int(np.searchsorted(cw, cw[-1] / 2.0))
    return float(v[min(k, len(v) - 1)])


def isotonic_l1_decreasing(y: np.ndarray, w: np.ndarray, min_block_weight: float = 0.0) -> np.ndarray:
    """Weighted L1 fit of a non-increasing sequence to ``y`` (pool-adjacent-violators).

    ``min_block_weight``: after fitting, a level supported by less total
    weight than this is merged into its closer-valued neighbour.  A real hit
    leaves the bar at its new level for many frames; a few stray low readings
    (often at the very end of a segment, where nothing later can outvote
    them) do not.
    """
    n = len(y)
    if n == 0:
        return np.zeros(0)
    # Each block: [start, end_exclusive, median]
    starts: list[int] = []
    ends: list[int] = []
    meds: list[float] = []
    for i in range(n):
        starts.append(i)
        ends.append(i + 1)
        meds.append(float(y[i]))
        # Non-increasing: a block's median must not exceed the previous one's.
        while len(meds) > 1 and meds[-2] < meds[-1]:
            s = starts[-2]
            e = ends[-1]
            starts.pop(); ends.pop(); meds.pop()
            starts[-1], ends[-1] = s, e
            meds[-1] = _wmedian(y[s:e], w[s:e])
    if min_block_weight > 0 and len(meds) > 1:
        # Merge the lightest under-weight block into its closer-valued
        # neighbour until every block is supported.  The weighted median of
        # two adjacent blocks lies between their medians, so the result stays
        # non-increasing.  Noisy readings on one level form many tiny blocks
        # that merge with each other; a few stray low readings at the end of
        # a segment merge back into the level before them.
        bw = [float(np.sum(w[a:b])) for a, b in zip(starts, ends)]
        while len(meds) > 1:
            k = int(np.argmin(bw))
            if bw[k] >= min_block_weight:
                break
            if k == 0:
                j = 1
            elif k == len(meds) - 1:
                j = k - 1
            else:
                j = k - 1 if abs(meds[k - 1] - meds[k]) <= abs(meds[k + 1] - meds[k]) else k + 1
            a, b = min(j, k), max(j, k)
            starts[a:b + 1] = [starts[a]]
            ends[a:b + 1] = [ends[b]]
            bw[a:b + 1] = [bw[a] + bw[b]]
            meds[a:b + 1] = [_wmedian(y[starts[a]:ends[a]], w[starts[a]:ends[a]])]
    out = np.empty(n)
    for s, e, m in zip(starts, ends, meds):
        out[s:e] = m
    return out


# ── Helpers ──────────────────────────────────────────────────────────────

def _rolling_nanmedian(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    half = win // 2
    for i in range(n):
        seg = x[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) * 2 >= min(win, n):
            out[i] = np.median(seg)
    return out


def full_level(m: np.ndarray) -> float:
    """Reading of a full bar for one side, from the rolling-median series.

    Usually ~1.0, but the game's own character name drawn on the bar (or a
    replay input display over it) can hide its far end (set 03: a full P2
    bar reads 0.81).  HP under the name
    is unobservable, so values are not rescaled; only the reset threshold
    adapts.
    """
    v = m[~np.isnan(m)]
    if len(v) == 0:
        return 1.0
    return float(min(1.0, np.percentile(v, 99.5)))


def _intro_starts(timer_val: np.ndarray) -> np.ndarray:
    """Frames that begin a credible run of round-start timer readings."""
    start = np.nan_to_num(timer_val, nan=-1) >= TIMER_START_MIN
    counts = np.convolve(start.astype(np.int32), np.ones(TIMER_WIN, dtype=np.int32))[TIMER_WIN - 1:]
    # counts[i] = start readings in [i, i + TIMER_WIN)
    return np.flatnonzero(start & (counts >= TIMER_RUN))


def _credible_low_timer(timer_val: np.ndarray) -> np.ndarray:
    """Prefix count of credible in-round timer readings (below the start values).

    ``out[i]`` = number of such readings in ``[0, i)``.
    """
    tv = np.nan_to_num(timer_val, nan=-1)
    low = (tv >= 0) & (tv < TIMER_START_MIN)
    counts = np.convolve(low.astype(np.int32), np.ones(TIMER_WIN, dtype=np.int32))[TIMER_WIN - 1:]
    cred = low & (counts >= TIMER_RUN)
    return np.concatenate([[0], np.cumsum(cred)])


def _timer_intros(timer_val: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> list[tuple[int, float, float]]:
    """Sustained round-start timer runs after a timer gap (see INTRO_GAP).

    Returns ``(frame, p1_level, p2_level)`` per candidate: the run's first
    frame and each side's median raw reading (any confidence) over the next
    INTRO_SPAN frames (NaN when too few readings).
    """
    n = len(timer_val)
    seen = ~np.isnan(timer_val)
    out = []
    last_seen = -10**9
    for a in range(n):
        if not seen[a]:
            continue
        gap = a - last_seen
        last_seen = a
        if gap <= INTRO_GAP or timer_val[a] < TIMER_START_MIN:
            continue
        tv = timer_val[a:a + INTRO_SPAN]
        tv = tv[~np.isnan(tv)]
        if len(tv) < INTRO_MIN_READS or np.mean(tv >= TIMER_START_MIN) < INTRO_TIMER_FRAC:
            continue
        levels = []
        for p in (p1, p2):
            v = p[a:a + INTRO_SPAN]
            v = v[~np.isnan(v)]
            levels.append(float(np.median(v)) if len(v) >= INTRO_BAR_READS else np.nan)
        out.append((a, levels[0], levels[1]))
    return out


def find_segments(p1: np.ndarray, p2: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                  timer_val: np.ndarray | None = None) -> list[int]:
    """Frame indices where a new round (segment) starts."""
    n = len(p1)
    g1 = np.where(c1 >= MIN_CONF, p1, np.nan)
    g2 = np.where(c2 >= MIN_CONF, p2, np.nan)
    m1 = _rolling_nanmedian(g1, ROLL)
    m2 = _rolling_nanmedian(g2, ROLL)
    # Heavier medians for the in-round lows used by the rise rule, so short
    # artifacts (an empty-looking bar under a sprite) do not count as a low.
    h1 = _rolling_nanmedian(g1, ROLL_LOW)
    h2 = _rolling_nanmedian(g2, ROLL_LOW)
    # Per-side thresholds relative to the observed full-bar reading.
    full1 = min(FULL_PCT, full_level(m1) - 0.05)
    full2 = min(FULL_PCT, full_level(m2) - 0.05)
    dmg1 = full1 - (FULL_PCT - DAMAGED_PCT)
    dmg2 = full2 - (FULL_PCT - DAMAGED_PCT)
    any_obs = ~np.isnan(p1) | ~np.isnan(p2)

    intro = _intro_starts(timer_val) if timer_val is not None else None
    # Timer-intro fallback resets, checked once the intro window has passed
    # (the regular rule gets the first chance).
    intro_at: dict[int, tuple[int, float, float]] = {}
    low_pref = None
    if timer_val is not None:
        low_pref = _credible_low_timer(timer_val)
        for a, l1, l2 in _timer_intros(timer_val, p1, p2):
            intro_at[min(n - 1, a + INTRO_SPAN)] = (a, l1, l2)
    starts: list[int] = []
    seg_start = None
    damaged = False
    full_run = 0
    last_obs = -10**9
    low1 = low2 = np.inf
    for i in range(n):
        if any_obs[i]:
            if seg_start is None or (i - last_obs > SEGMENT_GAP
                                     and not _clock_continues(timer_val, last_obs, i)):
                seg_start = i
                starts.append(i)
                damaged = False
                full_run = 0
                low1 = low2 = np.inf
            last_obs = i
        v1, v2 = m1[i], m2[i]
        ok1, ok2 = not np.isnan(v1), not np.isnan(v2)
        f1 = ok1 and v1 >= full1
        f2 = ok2 and v2 >= full2
        # Reset: both bars full, or both bars risen well above their lows in
        # this round with one of them full (HP cannot rise within a round;
        # covers a bar whose far end is hidden by an opaque name and reads
        # below "full").
        rose1 = ok1 and v1 >= max(RISE_MIN, low1 + RISE)
        rose2 = ok2 and v2 >= max(RISE_MIN, low2 + RISE)
        both_full = (f1 and f2) or (rose1 and rose2 and (f1 or f2))
        settled = seg_start is not None and i - seg_start >= LOW_SKIP
        if settled and full_run == 0 and not np.isnan(h1[i]):
            low1 = min(low1, h1[i])
        if settled and full_run == 0 and not np.isnan(h2[i]):
            low2 = min(low2, h2[i])
        if both_full:
            full_run += 1
        else:
            full_run = 0
            if seg_start is not None:
                if (not np.isnan(m1[i]) and m1[i] < dmg1) or (not np.isnan(m2[i]) and m2[i] < dmg2):
                    damaged = True
        if (damaged and full_run >= RESET_RUN and seg_start is not None
                and i - seg_start >= MIN_ROUND):
            new = i - full_run + 1
            if intro is not None:
                # Earliest intro run whose timer readings up to the reset are
                # all start values (a frozen in-round timer misread as 9x
                # interleaves with its true readings).
                for k in range(np.searchsorted(intro, new - INTRO_BACK), len(intro)):
                    a = int(intro[k])
                    if a >= new or a <= starts[-1]:
                        break
                    tv = timer_val[a:new]
                    tv = tv[~np.isnan(tv)]
                    if (tv >= TIMER_START_MIN).all():
                        new = a
                        break
            seg_start = new
            starts.append(new)
            damaged = False
            low1 = low2 = np.inf
        elif i in intro_at and seg_start is not None and damaged:
            a, l1, l2 = intro_at[i]
            new = _intro_reset(a, l1, l2, low1, low2, seg_start, g1, g2, low_pref)
            if new is not None:
                seg_start = new
                starts.append(new)
                damaged = False
                low1 = low2 = np.inf
    return sorted(set(starts))


def _clock_continues(timer_val: np.ndarray | None, a: int, b: int) -> bool:
    """Does the round clock carry on across a gap in the readings (frames a..b)?

    A long super cinematic hides the whole HUD mid-round for 5 s or more.
    The same round goes on when the clock after the gap shows an in-round
    value no higher than before it (a new round starts at the intro's start
    values).  Credible readings (TIMER_RUN equal within TIMER_WIN) only.
    """
    if timer_val is None:
        return False
    before = _credible_timer(timer_val[max(0, a - 4 * TIMER_WIN):a + 1])
    after = _credible_timer(timer_val[b:b + 4 * TIMER_WIN])
    if len(before) == 0 or len(after) == 0:
        return False
    tb = timer_val[max(0, a - 4 * TIMER_WIN) + before[-1]]
    ta = timer_val[b + after[0]]
    return bool(ta < TIMER_START_MIN and ta <= tb)


def _intro_reset(a: int, l1: float, l2: float, low1: float, low2: float, seg_start: int,
                 g1: np.ndarray, g2: np.ndarray, low_pref: np.ndarray) -> int | None:
    """Reset frame for a timer intro at ``a`` in the segment from ``seg_start``, or None.

    See INTRO_GAP.  The reset is moved back from ``a`` to just after the last
    confident reading of the old round's (low) bars: the bars fill up before
    the timer is shown.
    """
    if a - seg_start < MIN_ROUND or low_pref[a] - low_pref[seg_start] <= 0:
        return None
    if np.isnan(l1) or np.isnan(l2) or not (np.isfinite(low1) and np.isfinite(low2)):
        return None
    r1, r2 = l1 - low1, l2 - low2
    if max(r1, r2) < RISE or min(r1, r2) < -INTRO_TOL:
        return None
    thr1, thr2 = (low1 + l1) / 2, (low2 + l2) / 2
    floor = max(seg_start + MIN_ROUND, a - INTRO_BACK)
    j = a - 1
    while j >= floor and not (g1[j] < thr1 or g2[j] < thr2):
        j -= 1
    return j + 1


def _first_ko(fit: np.ndarray) -> int | None:
    """Index of the first frame where a (non-increasing) fit reaches KO level."""
    idx = np.flatnonzero(~np.isnan(fit) & (fit <= KO_PCT))
    return int(idx[0]) if len(idx) else None


def _comes_back(pv: np.ndarray, cv: np.ndarray, i: int, level: float,
                po: np.ndarray, opp_level: float) -> bool:
    """Does the bar show real fill again after span-relative frame ``i``?

    A KO'd bar never comes back, so a later run of END_RUN confident
    readings of at least BACK_MIN means an empty reading at ``i`` was the bar hidden
    (sprites can cover the whole HUD corner for seconds).  A frame counts
    when the other bar reads its normal level (``opp_level``; in the fade
    after a KO both bars read near 0), or when the reading is at most this
    side's ``level`` before ``i`` (HP cannot rise; the post-KO HUD shows the
    loser's true level).
    """
    tail = slice(i + 1, len(pv))
    v = pv[tail]
    # A (nearly) full bar is the next round's intro, not a comeback.
    good = (~np.isnan(v) & (cv[tail] >= END_CONF) & (np.nan_to_num(v) >= BACK_MIN)
            & (np.nan_to_num(v) < BACK_MAX))
    if not good.any():
        return False
    opp_ok = ~np.isnan(po[tail]) & (np.nan_to_num(po[tail]) >= opp_level - OPP_OK_TOL)
    not_above = np.nan_to_num(v, nan=np.inf) <= level + max(0.02, 0.25 * level)
    back = good & (opp_ok | not_above)
    run = np.convolve(back.astype(np.int32), np.ones(END_RUN, dtype=np.int32))[END_RUN - 1:]
    return bool((run >= END_RUN).any())


def _clock_runs_on(timer_val: np.ndarray | None, lo: int, hi: int, i: int) -> bool:
    """Does the round timer count down after span-relative frame ``i``?

    The clock freezes at a KO.  Compares credible readings (TIMER_RUN equal
    values within TIMER_WIN frames) before and after ``i``.
    """
    if timer_val is None:
        return False
    tv = timer_val[lo:hi]

    def credible(idx: np.ndarray) -> list[float]:
        out = []
        for k in idx:
            w = tv[k:k + TIMER_WIN]
            if np.sum(w == tv[k]) >= TIMER_RUN:
                out.append(float(tv[k]))
        return out

    seen = np.flatnonzero(~np.isnan(tv))
    before = credible(seen[seen <= i][-TIMER_WIN * 4:])
    after = credible(seen[seen > i])
    return bool(before) and bool(after) and min(after) < before[-1]


def _credible_timer(tv: np.ndarray) -> np.ndarray:
    """Indices of credible readings in a timer span (TIMER_RUN equal values within TIMER_WIN)."""
    seen = np.flatnonzero(~np.isnan(tv))
    return np.array([k for k in seen if np.sum(tv[k:k + TIMER_WIN] == tv[k]) >= TIMER_RUN], dtype=int)


def _supported_timer(tv: np.ndarray) -> np.ndarray:
    """Indices of timer readings with TIMER_RUN equal values (itself included)
    within TIMER_WIN frames on either side.

    Unlike _credible_timer this keeps the last readings of a run (before the
    HUD goes), and drops a lone misread (a covered tens digit reads "7" for
    97 on one frame), which would otherwise split a hidden-HUD gap.
    """
    seen = np.flatnonzero(~np.isnan(tv))
    return np.array([k for k in seen
                     if np.sum(tv[max(0, k - TIMER_WIN + 1):k + TIMER_WIN] == tv[k]) >= TIMER_RUN], dtype=int)


def _clock_freeze(timer_val: np.ndarray | None, lo: int, hi: int) -> int | None:
    """Span-relative first frame showing the clock's final value, or None.

    The clock stops at a KO (or at 0 on a time-out), so a KO is never before
    this frame: until it the clock was still counting down.
    """
    if timer_val is None:
        return None
    tv = timer_val[lo:hi]
    cred = _credible_timer(tv)
    if len(cred) == 0:
        return None
    final = tv[cred[-1]]
    i = len(cred) - 1
    while i > 0 and tv[cred[i - 1]] == final:
        i -= 1
    return int(cred[i])


def _hud_back(timer_val: np.ndarray | None, lo: int, hi: int, fz: int | None) -> tuple[int | None, np.ndarray]:
    """End of play and the HUD frames shown again after the round-end slash.

    Returns (span-relative last timer reading before the last gap of at
    least HUD_GAP frames since the clock's last tick (readings supported by
    their neighbours only, see _supported_timer), span-relative frames
    after that gap that read the same frozen clock).  The slash and KO cinematics
    hide the whole HUD; when it comes back with the clock still at its final
    value, the round is over and the bars show its result.
    """
    empty = np.zeros(0, dtype=int)
    if timer_val is None or fz is None:
        return None, empty
    tv = timer_val[lo:hi]
    final = tv[fz]
    seen = _supported_timer(tv)
    # From the last reading before the final value: the clock can tick for
    # the last time while the HUD is hidden (a KO under a super cinematic).
    before = seen[seen < fz]
    seen = seen[seen >= (before[-1] if len(before) else fz)]
    gaps = np.flatnonzero(np.diff(seen) >= HUD_GAP)
    if len(gaps) == 0:
        return int(seen[-1]) if len(seen) else None, empty
    # The last gap: the clock can stay on one value through a long combo and
    # a super cinematic (the HUD hidden mid-play) before the KO.
    g = gaps[-1]
    after = seen[g + 1:]
    return int(seen[g]), after[tv[after] == final]


def _end_loser(p1: np.ndarray, p2: np.ndarray, c1: np.ndarray, c2: np.ndarray,
               q1: np.ndarray | None, q2: np.ndarray | None,
               combo1: np.ndarray | None, combo2: np.ndarray | None,
               timer_val: np.ndarray | None, lo: int, hi: int, hud_only: bool = False) -> int | None:
    """The side KO'd at the end of a round in which both bars were low, or None.

    Decided in game terms, in this order:

    1. the HUD shown again after the round-end slash (same frozen clock):
       an empty bar next to a sliver (see KO_FLOOR / HUD_SHOWN);
    2. the red pending segment (a combo landing) on exactly one side in the
       last END_HIT frames of play (not with ``hud_only``);
    3. the combo counter (the attacker, who essentially never loses HP during
       its own combo) up on exactly one side in the last END_HIT frames of
       play: the other side lost.
    """
    fz = _clock_freeze(timer_val, lo, hi)
    play_end, back = _hud_back(timer_val, lo, hi, fz)
    if len(back):
        v1, v2 = p1[lo:hi][back], p2[lo:hi][back]
        k1, k2 = c1[lo:hi][back], c2[lo:hi][back]
        w1, w2 = np.nan_to_num(v1, nan=1.0), np.nan_to_num(v2, nan=1.0)
        # Empty: no more than the junk floor over an empty bar (KO_FLOOR);
        # the other bar must read clearly more (HUD_SHOWN above it).
        # One of the two readings must be confident (a 1-px bar, or an
        # empty one, is read with middling confidence).
        e1 = (w1 <= KO_FLOOR) & (k1 >= MIN_CONF)
        e2 = (w2 <= KO_FLOOR) & (k2 >= MIN_CONF)
        s1 = ~np.isnan(v1) & (k1 >= MIN_CONF) & (w1 >= w2 + HUD_SHOWN)
        s2 = ~np.isnan(v2) & (k2 >= MIN_CONF) & (w2 >= w1 + HUD_SHOWN)
        sure = np.maximum(k1, k2) >= END_CONF
        e1 &= sure
        e2 &= sure
        # A bar read empty while the other side shows red and it shows none
        # is the attacker's bar covered by its own combo's effects.
        r1, r2 = _shown(q1, lo, hi)[back], _shown(q2, lo, hi)[back]
        e1 &= ~(r2 & ~r1)
        e2 &= ~(r1 & ~r2)
        n1, n2 = int((e1 & s2).sum()), int((e2 & s1).sum())
        if n1 > 3 * n2:
            return 1
        if n2 > 3 * n1:
            return 2
    if play_end is None or hud_only:
        return None
    a = max(0, play_end + 1 - END_HIT)
    red = [_shown(q, lo, hi)[a:play_end + 1].sum() if q is not None else 0 for q in (q1, q2)]
    if (red[0] >= TIMER_RUN) != (red[1] >= TIMER_RUN):
        return 1 if red[0] >= TIMER_RUN else 2
    att = [_combo_shown(cb, lo, hi)[a:play_end + 1].any() for cb in (combo1, combo2)]
    if att[0] != att[1]:
        return 2 if att[0] else 1
    return None


def _raw_ko(p: np.ndarray, c: np.ndarray, q: np.ndarray | None, lo: int, hi: int,
            fit: np.ndarray, p_opp: np.ndarray, fit_opp: np.ndarray,
            timer_val: np.ndarray | None = None, q_opp: np.ndarray | None = None) -> int | None:
    """Span-relative frame of a KO seen in the raw readings, before the fit gets there.

    A confident empty-bar reading on a side whose fit is already low, backed by
    the red pending segment (the lethal combo) or by KO_CONFIRM more such
    readings within KO_WIN frames.  The KO slash hides the HUD right after, so
    the fit often never collects enough empty readings to reach 0 itself.
    """
    pv = p[lo:hi]
    cv = c[lo:hi]
    qv = np.zeros(len(pv)) if q is None else np.nan_to_num(q[lo:hi])
    red = qv >= PENDING_MIN
    # Fit level just before each frame.
    prev = np.concatenate([fit[:1], fit[:-1]])
    prev = np.where(np.isnan(prev), fit, prev)
    shown_before = np.zeros(len(pv), dtype=bool)
    shown_before[1:] = (cv[:-1] >= END_CONF) & (np.nan_to_num(pv[:-1]) > EMPTY_PCT)
    covers = (red & shown_before
              & (np.nan_to_num(pv) + qv >= np.nan_to_num(prev, nan=np.inf) - END_TOL))
    cand = (~np.isnan(pv) & (pv <= EMPTY_PCT) & ((cv >= END_CONF) | ((cv >= MIN_CONF) & covers))
            & ~np.isnan(fit) & (fit <= LOW_HP))
    ok = ~np.isnan(pv) & (cv >= MIN_CONF)
    # Empty readings while the other side shows its red pending segment and
    # this side shows none are this side attacking: its bar is covered by
    # the combo's effects (the attacker essentially never loses HP).
    attacking = _shown(q_opp, lo, hi) & ~_shown(q, lo, hi)
    frozen = _frozen_clock_ko(pv, cv, p_opp[lo:hi], fit, fit_opp,
                              None if timer_val is None else timer_val[lo:hi], attacking)
    for i in np.flatnonzero(cand & red):
        if frozen is not None and i >= frozen:
            break  # the frozen-clock KO is earlier
        level = prev[i]
        # The bar must stay (near) empty afterwards: a low bar hidden for a
        # frame mid-combo reads empty too, and then comes back.  Readings
        # above the level before the hit are not the bar (HP cannot rise):
        # the KO cinematic right after a lethal hit reads as junk fill.
        sl = slice(i + 1, i + 1 + KO_AFTER)
        nv = np.nan_to_num(pv[sl], nan=np.inf)
        nxt = pv[sl][ok[sl] & (nv <= level + max(0.02, 0.25 * level))]
        nxt = np.where(nxt <= KO_FLOOR, 0.0, nxt)
        if len(nxt) and float(np.median(nxt)) > min(KO_NEAR, fit[i] / 2):
            continue
        # The other side's level at the KO (its later fit may itself be
        # dragged down by KO-screen junk).
        pre = fit_opp[max(0, i - OPP_REF):i + 1]
        pre = pre[~np.isnan(pre)]
        opp_level = float(np.median(pre)) if len(pre) else np.inf
        if _comes_back(pv, cv, int(i), float(level), p_opp[lo:hi], opp_level):
            continue
        # A tiny bar flickers to empty while the round goes on.
        if _clock_runs_on(timer_val, lo, hi, int(i)):
            continue
        return int(i)
    return frozen


def _frozen_clock_ko(pv: np.ndarray, cv: np.ndarray, po: np.ndarray, fit: np.ndarray,
                     fit_opp: np.ndarray, tv: np.ndarray | None,
                     attacking: np.ndarray | None = None) -> int | None:
    """Span-relative KO frame from a frozen clock over an empty bar (see CLOCK_STOP), or None.

    The KO is placed just after the side's last confident reading with fill
    since the clock stopped (the lethal hit; the HUD is often hidden by the
    KO slash in between).
    """
    if tv is None:
        return None
    n = len(pv)
    seen = ~np.isnan(tv)
    opp_shown = ~np.isnan(po) & (np.nan_to_num(po) > HUD_EMPTY)
    hit = seen & ~np.isnan(pv) & (np.nan_to_num(pv) <= HUD_EMPTY) & (cv >= END_CONF) & opp_shown
    if attacking is not None:
        hit &= ~attacking[:n]
    ok = ~np.isnan(pv) & (cv >= MIN_CONF)
    i = 0
    while i < n:
        if not hit[i]:
            i += 1
            continue
        j = i
        while j < n and hit[j]:
            j += 1
        if j - i >= HUD_KO_RUN:
            v = tv[i:j]
            v = v[~np.isnan(v)]
            val = float(np.median(v))
            # First frame of the run of `val` readings before the empty run
            # (readings of other values end it).
            idx = np.flatnonzero(seen[:j])
            same = tv[idx] == val
            k = len(idx)
            while k > 0 and same[k - 1]:
                k -= 1
            first = int(idx[k]) if k < len(idx) else i
            later = tv[j:][seen[j:]]
            refilled = (seen[j:] & (cv[j:] >= END_CONF) & (np.nan_to_num(pv[j:]) > HUD_EMPTY)).any()
            if j - 1 - first >= CLOCK_STOP and not (later < val).any() and not refilled:
                level = fit[max(0, i - 1)]
                pre = fit_opp[max(0, i - OPP_REF):i + 1]
                pre = pre[~np.isnan(pre)]
                opp_level = float(np.median(pre)) if len(pre) else np.inf
                if not _comes_back(pv, cv, j - 1, float(level), po, opp_level):
                    filled = np.flatnonzero(ok[first:i] & (cv[first:i] >= END_CONF)
                                            & (np.nan_to_num(pv[first:i]) > HUD_EMPTY))
                    return first + int(filled[-1]) + 1 if len(filled) else i
        i = j
    return None


def _fit_points(pv: np.ndarray, cv: np.ndarray, use: np.ndarray,
                full_from: int | None = None) -> np.ndarray:
    """Isotonic fit on the ``use`` points of a span, interpolated across short gaps.

    ``full_from``: span-relative frame (the end of play) from which empty
    readings keep their full weight.  Only credible empties are in ``use``
    there, and the HUD is often up for just a few frames between the KO and
    the slash that hides it.
    """
    out = np.full(len(pv), np.nan)
    idx = np.flatnonzero(use)
    if len(idx) == 0:
        return out
    y = np.clip(pv[idx], 0.0, 1.0)
    w = cv[idx].astype(np.float64)
    scaled = y <= EMPTY_PCT
    if full_from is not None:
        scaled &= idx < full_from
    w = np.where(scaled, w * EMPTY_WEIGHT, w)
    out[idx] = isotonic_l1_decreasing(y, w, MIN_STEP_WEIGHT)
    # Fill frames between fitted points (low-confidence reads, missing reads).
    # Interpolation between two fitted values is itself non-increasing, so a
    # bar hidden during a combo "keeps going down" through the gap.
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= MAX_FILL_GAP:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [out[a], out[b]])
    return out


def _play_end(p1, p2, c1, c2, lo: int, hi: int) -> int | None:
    """Span-relative index of the last frame of the last both-sides-confident run."""
    both = ((c1[lo:hi] >= END_CONF) & (c2[lo:hi] >= END_CONF)
            & (np.nan_to_num(p1[lo:hi]) > EMPTY_PCT) & (np.nan_to_num(p2[lo:hi]) > EMPTY_PCT))
    run = 0
    last = None
    for i, b in enumerate(both):
        run = run + 1 if b else 0
        if run >= END_RUN:
            last = i
    return last


def _timer_end(timer: np.ndarray | None, lo: int, hi: int) -> int | None:
    """Span-relative index of the last credible round-timer reading."""
    if timer is None:
        return None
    seen = timer[lo:hi].astype(bool)
    counts = np.convolve(seen.astype(np.int32), np.ones(TIMER_WIN, dtype=np.int32))[:len(seen)]
    idx = np.flatnonzero(seen & (counts >= TIMER_RUN))
    return int(idx[-1]) if len(idx) else None


def _shown(q: np.ndarray | None, lo: int, hi: int) -> np.ndarray:
    """Pending red shown on a side (2 of 5 frames), span-relative."""
    if q is None:
        return np.zeros(hi - lo, dtype=bool)
    raw = (np.nan_to_num(q[lo:hi]) >= PENDING_MIN).astype(np.float64)
    return np.convolve(raw, np.ones(5), mode="same") >= 2


def _combo_shown(combo: np.ndarray | None, lo: int, hi: int) -> np.ndarray:
    """Combo counter shown on a side for COMBO_RUN+ consecutive frames, span-relative."""
    if combo is None:
        return np.zeros(hi - lo, dtype=bool)
    raw = np.asarray(combo[lo:hi], dtype=bool)
    out = np.zeros(len(raw), dtype=bool)
    i, n = 0, len(raw)
    while i < n:
        if not raw[i]:
            i += 1
            continue
        j = i
        while j < n and raw[j]:
            j += 1
        if j - i >= COMBO_RUN:
            out[i:j] = True
        i = j
    return out


def _pending_growing(qv: np.ndarray | None, theirs: np.ndarray) -> np.ndarray:
    """Frames of a run of the other side's pending red from its 2nd growth step on.

    The red segment grows with every hit of a combo; a single hit (whose red
    lingers ~1 s, during which this side can be hit back) grows it once.
    A step = the run's highest pending reading rising by GROW_MIN.
    """
    out = np.zeros(len(theirs), dtype=bool)
    if qv is None:
        return out
    q = np.nan_to_num(qv)
    i, n = 0, len(theirs)
    while i < n:
        if not theirs[i]:
            i += 1
            continue
        j = i
        while j < n and theirs[j]:
            j += 1
        top, steps = 0.0, 0
        for k in range(i, j):
            if q[k] >= top + GROW_MIN:
                top = q[k]
                steps += 1
                if steps == 2:
                    out[k:j] = True
        i = j
    return out


def _gate_runs(pv: np.ndarray, cv: np.ndarray, use: np.ndarray, empty: np.ndarray,
               mine: np.ndarray, gate: np.ndarray, strict: bool) -> np.ndarray:
    drop = np.zeros(len(pv), dtype=bool)
    i, n = 0, len(pv)
    while i < n:
        if not gate[i]:
            i += 1
            continue
        j = i
        while j < n and gate[j]:
            j += 1
        a, b = max(0, i - OPP_PAD), min(n, j + OPP_PAD)
        r0 = max(0, a - OPP_REF)
        ref_idx = np.flatnonzero(use[r0:a] & ~empty[r0:a] & (cv[r0:a] >= END_CONF))
        if len(ref_idx) >= OPP_REF_MIN:
            ref = float(np.median(pv[r0 + ref_idx]))
            seg = slice(a, b)
            thr = max(ref * OPP_DROP, ref - OPP_TOL) if strict else ref * OPP_DROP
            drop[seg] |= ~mine[seg] & (np.nan_to_num(pv[seg], nan=1.0) < thr)
        i = j
    return drop


def _opp_combo_gate(pv: np.ndarray, cv: np.ndarray, use: np.ndarray, empty: np.ndarray,
                    mine: np.ndarray, theirs: np.ndarray,
                    attacking: np.ndarray | None = None,
                    q_opp: np.ndarray | None = None) -> np.ndarray:
    """Readings to drop because this side is the one comboing (see OPP_PAD).

    ``mine`` / ``theirs``: red pending shown on this / the other side;
    ``attacking``: this side's combo counter shown; ``q_opp``: the other
    side's pending widths (span-relative).  While the other side merely shows
    red, only near-empty readings (below OPP_DROP x the level) are dropped;
    while this side is surely comboing (its counter is up, or the red grew
    with a 2nd hit) its HP is held at its level (within OPP_TOL)."""
    # np.convolve(mode="same") in _shown returns 5 values for a span shorter than 5.
    mine, theirs = mine[:len(pv)], theirs[:len(pv)]
    drop = _gate_runs(pv, cv, use, empty, mine, theirs & ~mine, strict=False)
    combo_on = _pending_growing(q_opp, theirs)
    if attacking is not None:
        combo_on |= attacking
    return drop | _gate_runs(pv, cv, use, empty, mine, combo_on & ~mine, strict=True)


def _clean_side(p: np.ndarray, c: np.ndarray, q: np.ndarray | None, lo: int, hi: int,
                end: int | None = None, q_opp: np.ndarray | None = None,
                strict_from: int | None = None, p_opp: np.ndarray | None = None,
                post_end_full: bool = False, combo: np.ndarray | None = None,
                back_hi: int | None = None) -> np.ndarray:
    """Fit one side within one segment [lo, hi).  Returns cleaned values for the span.

    ``back_hi``: the comeback check (see ``_comes_back``) looks at the frames
    up to ``back_hi`` (default ``hi``): a refit on the first frames after a KO
    must still see the survivor's bar shown again later.

    ``strict_from``: span-relative frame (the opponent's KO) after which no
    reading may lower this side.

    ``combo``: per-frame "this side's combo counter is shown" (full series).

    ``post_end_full``: credible empty readings after the end of play keep
    their full weight (see the KO fallback in ``clean_series``), if the other
    side was not low at the end of play.

    Empty-bar readings are the most common artifact (HUD hidden by a
    cinematic, a dark sprite or a transition screen still classifies as
    "no fill").  They are used only when credible: the red pending segment
    is visible next to the empty bar (a lethal combo), or the side was
    already low (<= LOW_HP) according to a first fit without them.
    """
    pv = p[lo:hi]
    cv = c[lo:hi]
    ok = ~np.isnan(pv) & (cv >= MIN_CONF)
    empty = ok & (pv <= EMPTY_PCT)
    if q is not None:
        red_here = np.nan_to_num(q[lo:hi]) >= PENDING_MIN
    else:
        red_here = np.zeros_like(empty)
    red_shown = empty & red_here
    if q is not None and red_shown.any():
        # A lethal combo's red segment covers the bar where the fill was:
        # red much narrower than the level just before is a red sprite or
        # effect over an emptied-looking bar, not the KO.
        qv = np.nan_to_num(q[lo:hi])
        filled = np.flatnonzero(ok & ~empty & (cv >= END_CONF))
        for i in np.flatnonzero(red_shown):
            k = np.searchsorted(filled, i)
            if k:
                level = float(np.median(pv[filled[max(0, k - END_RUN):k]]))
                if qv[i] < RED_COVER * level:
                    red_shown[i] = False
    use = ok & (~empty | red_shown)
    blocked = np.zeros(len(pv), dtype=bool)
    gated = np.zeros(len(pv), dtype=bool)
    if q_opp is not None or combo is not None:
        gated = _opp_combo_gate(pv, cv, use, empty, _shown(q, lo, hi), _shown(q_opp, lo, hi),
                                _combo_shown(combo, lo, hi),
                                None if q_opp is None else q_opp[lo:hi])
        use &= ~gated
    if end is not None and end + END_GRACE < len(pv):
        # After play ends a non-empty reading may confirm the side's last
        # in-play level (the HUD coming back after the KO cinematic) but not
        # lower it (cinematic / transition junk).  Credible empties still count.
        pre = np.flatnonzero(use[:end + 1] & ~empty[:end + 1])[-END_RUN:]
        if len(pre):
            ref = float(np.median(pv[pre]))
            after = np.zeros(len(pv), dtype=bool)
            after[end + END_GRACE:] = True
            # A real hit on the survivor at the very end (a trade) shows red.
            use &= ~(after & ~empty & (pv < ref - END_TOL) & ~red_here)
        else:
            use[end + END_GRACE:] &= empty[end + END_GRACE:]
    if strict_from is not None and strict_from < len(pv):
        pre = np.flatnonzero(use[:strict_from + 1] & ~empty[:strict_from + 1])[-END_RUN:]
        if len(pre):
            ref = float(np.median(pv[pre]))
            blocked[strict_from + 1:] = ~(pv[strict_from + 1:] >= ref - END_TOL)
            use &= ~blocked
    first = _fit_points(pv, cv, use)
    # Fitted value at or before each frame (forward-filled).
    idx = np.flatnonzero(~np.isnan(first))
    if len(idx):
        pos = np.searchsorted(idx, np.arange(len(pv)), side="right") - 1
        before = np.where(pos >= 0, first[idx[np.clip(pos, 0, None)]], np.nan)
    else:
        before = np.full(len(pv), np.nan)
    low_before = ~np.isnan(before) & (before <= LOW_HP)
    use2 = use | (empty & low_before & ~gated & ~blocked)
    if p_opp is not None:
        # Empty readings the bar later comes back from are occlusion, not a
        # KO (checked once per run of empty readings).
        po = p_opp[lo:hi]
        bh = hi if back_hi is None else max(hi, back_hi)
        pvb, cvb, pob = p[lo:bh], c[lo:bh], p_opp[lo:bh]
        cand = use2 & empty
        i = 0
        while i < len(pv):
            if not cand[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(pv) and (cand[j + 1] or np.isnan(pv[j + 1])):
                j += 1
            level = before[i - 1] if i > 0 and not np.isnan(before[i - 1]) else 1.0
            opp = po[max(0, i - OPP_REF):i + 1]
            opp = opp[~np.isnan(opp)]
            opp_level = float(np.median(opp)) if len(opp) else np.inf
            if _comes_back(pvb, cvb, j, float(level), pob, opp_level):
                use2[i:j + 1] &= ~cand[i:j + 1]
            i = j + 1
    # post_end_full: with the other side low too, slash-screen empties would
    # fake a double KO.
    full_from = None
    if post_end_full and end is not None and p_opp is not None:
        po_end = p_opp[lo:hi][max(0, end - OPP_REF):end + 1]
        po_end = po_end[~np.isnan(po_end)]
        if len(po_end) and float(np.median(po_end)) > LOW_HP:
            full_from = end + END_GRACE
    out = first if not (use2 & ~use).any() else _fit_points(pv, cv, use2, full_from)
    # Frames after the last usable reading where the bar was still read (but
    # the reading was rejected) keep the last fitted value: the HUD is up and
    # health cannot have gone up, and nothing credible says it went down.
    fitted = np.flatnonzero(~np.isnan(out))
    if len(fitted):
        last = fitted[-1]
        tail = ~np.isnan(pv[last + 1:])
        out[last + 1:][tail] = out[last]
    return out


def _hold_after(p: np.ndarray, c: np.ndarray, q: np.ndarray | None, lo: int, hi: int, cut: int,
                prev: np.ndarray, end: int | None = None, q_opp: np.ndarray | None = None,
                strict_from: int | None = None, p_opp: np.ndarray | None = None,
                combo: np.ndarray | None = None) -> np.ndarray:
    """Refit a side on [lo, lo+cut) only, then hold its last value where ``prev`` had coverage."""
    fit = _clean_side(p, c, q, lo, lo + cut, end, q_opp, strict_from, p_opp, combo=combo, back_hi=hi)
    out = prev.copy()
    out[:cut] = fit
    valid = fit[~np.isnan(fit)]
    if len(valid):
        tail = out[cut:]
        tail[~np.isnan(tail)] = valid[-1]
        out[cut:] = tail
    return out


def _near_double_ko_survivor(p1, p2, c1, c2, lo: int, hi: int, k_ko: int, loser: int,
                             f1: np.ndarray, f2: np.ndarray) -> int | None:
    """The side the HUD shows surviving a KO with both sides low, or None.

    Checked when the survivor is at or below LOW_HP just after the KO
    (span-relative ``k_ko``; ``loser`` is the side KO'd first by the fit), so
    its own later empty readings could make a double KO.  Naming the first
    loser as the survivor additionally needs a near-double KO: both sides at
    or below NEAR_DKO.  The last run of
    END_RUN frames within DKO_WIN after the KO in which one bar is shown at
    its level (between its fitted level just after the KO and its level
    before it, and at least 2 x EMPTY_PCT) while the other reads empty decides: the HUD after the KO
    cinematic shows the survivor's sliver and the loser's empty bar.
    """
    if k_ko < 1:
        return None
    pre, post = [], []
    for f in (f1, f2):
        v = f[max(0, k_ko - OPP_REF):k_ko]
        v = v[~np.isnan(v)]
        pre.append(float(np.median(v)) if len(v) else np.nan)
        # The survivor's fit may still drop with the KO exchange.
        v = f[k_ko:k_ko + DOUBLE_KO_FRAMES + 1]
        v = v[~np.isnan(v)]
        post.append(float(v[-1]) if len(v) else np.nan)
    surv = 2 - loser
    if np.isnan(pre).any() or np.isnan(post[surv]) or post[surv] > LOW_HP:
        return None
    near = post[surv] <= NEAR_DKO and pre[loser - 1] <= NEAR_DKO
    post[loser - 1] = 0.0
    a = slice(lo + k_ko + 1, min(hi, lo + k_ko + 1 + DKO_WIN))
    shown, empty = [], []
    for p, c, k in ((p1, c1, 0), (p2, c2, 1)):
        v = np.nan_to_num(p[a], nan=1.0)
        tol = max(DKO_LEVEL_TOL, 0.5 * pre[k])
        empty.append((c[a] >= END_CONF) & (v <= EMPTY_PCT))
        # A 1-px sliver is read with low confidence.
        shown.append((c[a] >= MIN_CONF) & (v >= 2 * EMPTY_PCT)
                     & (v >= min(pre[k], post[k]) - tol) & (v <= pre[k] + tol))
    last = None
    for side, pat in ((1, shown[0] & empty[1]), (2, shown[1] & empty[0])):
        run = np.convolve(pat.astype(np.int32), np.ones(END_RUN, dtype=np.int32))[END_RUN - 1:]
        idx = np.flatnonzero(run >= END_RUN)
        if len(idx) and (last is None or idx[-1] > last[0]):
            last = (int(idx[-1]), side)
    if last is None or (last[1] == loser and not near):
        return None
    return last[1]


def clean_series(
    p1: np.ndarray, p2: np.ndarray, c1: np.ndarray, c2: np.ndarray,
    q1: np.ndarray | None = None, q2: np.ndarray | None = None,
    timer: np.ndarray | None = None,
    combo1: np.ndarray | None = None, combo2: np.ndarray | None = None,
) -> CleanResult:
    """Clean per-frame fill readings (NaN = no reading).

    ``c1``/``c2`` are per-frame confidences in [0, 1]; ``q1``/``q2`` the
    pending (red) segment widths as a fraction of the bar, if known;
    ``timer`` the per-frame round timer reading (NaN = not read);
    ``combo1``/``combo2`` per-frame booleans, True where the combo counter is
    shown on that side's HUD (that side is doing a combo).
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    c1 = np.nan_to_num(np.asarray(c1, dtype=np.float64))
    c2 = np.nan_to_num(np.asarray(c2, dtype=np.float64))
    n = len(p1)
    timer_val = None if timer is None else np.asarray(timer, dtype=np.float64)
    starts = find_segments(p1, p2, c1, c2, timer_val)
    bounds = starts + [n]
    o1 = np.full(n, np.nan)
    o2 = np.full(n, np.nan)
    seg = np.full(n, -1, dtype=np.int32)
    for k, (lo, seg_hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg[lo:seg_hi] = k
        t_end = _timer_end(None if timer_val is None else ~np.isnan(timer_val), lo, seg_hi)
        # Clean only up to KO_MARGIN frames after the last timer reading.
        hi = seg_hi if t_end is None else min(seg_hi, lo + t_end + 1 + KO_MARGIN)
        end = _play_end(p1, p2, c1, c2, lo, hi)
        if t_end is not None and (end is None or t_end < end):
            end = t_end
        # Play goes on at least until the clock shows its final value: the
        # last run of both bars read with fill can end much earlier when one
        # side is down to a sliver the source cannot show.
        fz = _clock_freeze(timer_val, lo, hi)
        if fz is not None and end is not None and end < fz:
            end = fz
        f1 = _clean_side(p1, c1, q1, lo, hi, end, q2, p_opp=p2, combo=combo1)
        f2 = _clean_side(p2, c2, q2, lo, hi, end, q1, p_opp=p1, combo=combo2)
        # Frames after the cut that a fit of the whole segment covers (read,
        # or in a short gap between fitted points) hold the last value.
        shown1 = shown2 = None
        if hi < seg_hi:
            end_full = _play_end(p1, p2, c1, c2, lo, seg_hi)
            if end_full is None or t_end < end_full:
                end_full = t_end
            shown1 = ~np.isnan(_clean_side(p1, c1, q1, lo, seg_hi, end_full, q2, p_opp=p2, combo=combo1)[hi - lo:])
            shown2 = ~np.isnan(_clean_side(p2, c2, q2, lo, seg_hi, end_full, q1, p_opp=p1, combo=combo2)[hi - lo:])
        # KO lock: once one side is KO'd the round is over.  The survivor's
        # later readings (KO cinematic, slow-mo, transition screens) can only
        # be artifacts, so its HP is held from shortly after the KO.
        # KO candidates per side: the fit reaching 0 and the raw KO.  A KO
        # ends the round (the clock stops, the HUD goes for the slash): a
        # side read at KO level while the clock still counts down afterwards
        # is a live sliver, and that candidate is dropped.  A side with only
        # such candidates was KO'd, if at all, when play ended.
        alive = {1: None, 2: None}
        play_end, _ = _hud_back(timer_val, lo, hi, fz)
        k_end = None
        if fz is not None:
            k_end = fz if play_end is None else min(hi - lo - 1, play_end + 1)
        ks = []
        for side, f, r in ((1, f1, _raw_ko(p1, c1, q1, lo, hi, f1, p2, f2, timer_val, q2)),
                           (2, f2, _raw_ko(p2, c2, q2, lo, hi, f2, p1, f1, timer_val, q1))):
            cands = [k for k in (_first_ko(f), r) if k is not None]
            valid = [k for k in cands if fz is None or k >= fz or not _clock_runs_on(timer_val, lo, hi, k)]
            if len(valid) < len(cands):
                alive[side] = min(cands)
            if valid:
                ks.append(min(valid))
            elif cands:
                last = np.flatnonzero(~np.isnan(f))
                ks.append(min(k_end, int(last[-1])) if len(last) else k_end)
            else:
                ks.append(None)
        k1, k2 = ks
        if k1 is None and k2 is None and end is not None:
            # KO fallback: the HUD is often up for only a few frames between
            # the KO and the slash that hides it, too few for the down-weighted
            # empty readings to form a level.  Refit with full weight on
            # credible empties after the end of play; adopt it only if exactly
            # one side then reaches 0.
            g1 = _clean_side(p1, c1, q1, lo, hi, end, q2, p_opp=p2, post_end_full=True, combo=combo1)
            g2 = _clean_side(p2, c2, q2, lo, hi, end, q1, p_opp=p1, post_end_full=True, combo=combo2)
            j1, j2 = _first_ko(g1), _first_ko(g2)
            if (j1 is None) != (j2 is None):
                if j1 is not None:
                    f1, k1 = g1, j1
                else:
                    f2, k2 = g2, j2
        if k1 is None and k2 is None and t_end is not None:
            ev = (_end_loser(p1, p2, c1, c2, q1, q2, combo1, combo2, timer_val, lo, seg_hi, hud_only=True),
                  _end_loser(p1, p2, c1, c2, q1, q2, combo1, combo2, timer_val, lo, seg_hi))
            hidden = _hidden_ko(f1, f2, timer_val, lo, t_end, seg_hi, n, combo1, combo2, ev)
            if hidden is not None:
                # From just after play ended (the last timer reading before
                # the HUD went for the slash; the HUD can come back with the
                # clock frozen), or at least on the side's last cleaned frame
                # (its final HP).
                f_lose = f1 if hidden == 1 else f2
                k_h = t_end + 1 if play_end is None else min(t_end, play_end) + 1
                k_h = min(k_h, int(np.flatnonzero(~np.isnan(f_lose))[-1]))
                f_lose[k_h:][~np.isnan(f_lose[k_h:])] = 0.0
                _ko_shown(f_lose, _hud_back(timer_val, lo, hi, fz)[1], k_h)
                _alive_floor(f1, f2, hidden, k_h)
        kos = [(k, side) for k, side in ((k1, 1), (k2, 2)) if k is not None]
        if kos:
            k_ko, loser = min(kos)
            # A KO first seen on the HUD shown again after the round-end
            # slash (the clock frozen) happened while it was hidden: play
            # ended with the last timer reading before the slash.
            _, back = _hud_back(timer_val, lo, hi, fz)
            if len(back) and play_end is not None and k_ko >= back[0]:
                k_ko = play_end + 1
            k_other = k2 if loser == 1 else k1
            # The HUD after the KO cinematic may come back after the cut.
            dko_surv = _near_double_ko_survivor(p1, p2, c1, c2, lo, seg_hi, k_ko, loser, f1, f2)
            # Both sides at or near 0 when the round ended, or a KO that was
            # only placed at the clock's stop (the side was a live sliver
            # until then): who lost is decided by what the game shows.
            f_surv = f2 if loser == 1 else f1
            # Its fitted level around the KO (or its last one before it).
            v = f_surv[:k_ko + DOUBLE_KO_FRAMES + 1]
            v = v[~np.isnan(v)][-(OPP_REF + DOUBLE_KO_FRAMES):]
            if alive[loser] is not None or k_other is not None or (len(v) and v.min() <= NEAR_DKO):
                ev = _end_loser(p1, p2, c1, c2, q1, q2, combo1, combo2, timer_val, lo, seg_hi)
                if ev is not None:
                    dko_surv = 3 - ev
            if dko_surv is not None and dko_surv == loser:
                # The HUD shows who survived: the survivor keeps its level
                # from before the KO (its empty readings were a 1-px sliver
                # flickering, or KO-screen junk), the other side is KO'd.
                loser = 3 - dko_surv
                if dko_surv == 1:
                    f1 = _hold_after(p1, c1, q1, lo, hi, k_ko, f1, end, q2, None, p2, combo1)
                else:
                    f2 = _hold_after(p2, c2, q2, lo, hi, k_ko, f2, end, q1, None, p1, combo2)
                f_lose = f1 if loser == 1 else f2
                f_lose[k_ko:][~np.isnan(f_lose[k_ko:])] = 0.0
                _ko_shown(f_lose, back, k_ko)
                _alive_floor(f1, f2, loser, k_ko)
                _store(o1, f1, shown1, lo, hi)
                _store(o2, f2, shown2, lo, hi)
                continue
            # The loser is at 0 from its KO (a raw KO can precede the fit).
            f_lose = f1 if loser == 1 else f2
            f_lose[k_ko:][~np.isnan(f_lose[k_ko:])] = 0.0
            _ko_shown(f_lose, back, k_ko)
            # A double KO needs the survivor already low when the KO lands;
            # from high HP its later readings are KO-effect junk.
            surv = f2 if loser == 1 else f1
            pre = surv[max(0, k_ko - OPP_REF):k_ko]
            pre = pre[~np.isnan(pre)]
            low = len(pre) == 0 or float(np.median(pre)) <= LOW_HP
            cut = k_ko + DOUBLE_KO_FRAMES
            # From high HP, nothing after the KO may lower the survivor (KO
            # flash / slash junk, even with red shown).
            strict = None if low else k_ko
            if cut < hi - lo:
                ps1, ps2 = p1, p2
                if dko_surv is not None:
                    # Shown surviving after the KO cinematic: its empty
                    # readings around the KO are junk (its own super's
                    # sprites over its bar), not a double KO.
                    ps = (p2 if loser == 1 else p1).copy()
                    tail = ps[lo + max(0, k_ko - DOUBLE_KO_FRAMES):hi]
                    tail[tail <= EMPTY_PCT] = np.nan
                    ps1, ps2 = (p1, ps) if loser == 1 else (ps, p2)
                if loser == 1:
                    f2 = _hold_after(ps2, c2, q2, lo, hi, cut, f2, end, q1, strict, p1, combo2)
                else:
                    f1 = _hold_after(ps1, c1, q1, lo, hi, cut, f1, end, q2, strict, p2, combo1)
            # A survivor already low at the KO that empties within
            # DOUBLE_KO_FRAMES was KO'd too (double KO): it is only floored
            # until then.
            surv = f2 if loser == 1 else f1
            k_surv = _first_ko(surv[k_ko:k_ko + DOUBLE_KO_FRAMES + 1]) if low else None
            _alive_floor(f1, f2, loser, k_ko, None if k_surv is None else k_ko + k_surv)
        _store(o1, f1, shown1, lo, hi)
        _store(o2, f2, shown2, lo, hi)
    return CleanResult(p1=o1, p2=o2, segment=seg, resets=starts)


def _hidden_ko(f1: np.ndarray, f2: np.ndarray, timer_val: np.ndarray, lo: int, t_end: int,
               seg_hi: int, n: int, combo1: np.ndarray | None = None,
               combo2: np.ndarray | None = None,
               evidence: tuple[int | None, int | None] = (None, None)) -> int | None:
    """The side KO'd out of sight in a round that ended with the clock running (see KO_LEAD), or None.

    ``evidence``: (_end_loser with the HUD shown again after the slash only,
    _end_loser with all its rules).  With both sides low and close the
    evidence decides; the last round of a video (no next round's intro to
    prove the round ended) needs the HUD shown after the slash to agree.
    """
    if timer_val[lo + t_end] < HIDDEN_KO_MIN_CLOCK:
        return None
    intro = _intro_starts(timer_val)
    next_round = seg_hi < n and bool(((intro >= seg_hi - TIMER_WIN) & (intro <= seg_hi + INTRO_BACK)).any())
    ev_hud, ev_any = evidence
    if not next_round and ev_hud is None:
        return None
    last = []
    for f in (f1, f2):
        v = f[:t_end + 1]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return None
        last.append(float(v[-1]))
    lose = 1 if last[0] < last[1] else 2
    low, high = min(last), max(last)
    if low > LOW_HP:
        return None
    if high - low < KO_LEAD:
        # Both low: the game shows who lost (see _end_loser).
        return ev_any if next_round else ev_hud
    if not next_round and ev_hud != lose:
        return None
    a = max(0, t_end + 1 - HIDDEN_KO_COMBO)
    if _combo_shown(combo1 if lose == 1 else combo2, lo, lo + t_end + 1)[a:].any():
        return None
    return lose


def _ko_shown(f_lose: np.ndarray, back: np.ndarray, k_ko: int) -> None:
    """The KO'd side is 0 on the HUD frames shown again after the round-end
    slash (``back``, see _hud_back), in place.  Its fit can leave them empty
    (an empty bar read with middling confidence after a cinematic), and the
    round detector would then end the round at the last value before the KO."""
    f_lose[back[back >= k_ko]] = 0.0


def _alive_floor(f1: np.ndarray, f2: np.ndarray, loser: int, k_ko: int,
                 k_surv: int | None = None) -> None:
    """Keep a side that is alive at ALIVE_MIN or above (in place).

    The survivor is alive for the whole round, the loser until its KO frame:
    a bar fitted to (nearly) 0 there was a sliver the source could barely
    show, and "exactly 0" must mean KO'd (the round detector relies on it).
    ``k_surv``: the survivor's own KO frame in a double KO (it is only kept
    alive until then).
    """
    for side, f in ((1, f1), (2, f2)):
        if side == loser:
            span = f[:k_ko]
        else:
            span = f if k_surv is None else f[:k_surv]
        low = ~np.isnan(span) & (span < ALIVE_MIN)
        span[low] = ALIVE_MIN


def _store(out: np.ndarray, fit: np.ndarray, shown: np.ndarray | None, lo: int, hi: int) -> None:
    """Write a side's cleaned span; the frames after the cut (see KO_MARGIN)
    marked in ``shown`` hold the last cleaned value."""
    out[lo:hi] = fit
    valid = fit[~np.isnan(fit)]
    if shown is not None and len(valid):
        out[hi:hi + len(shown)][shown] = valid[-1]
