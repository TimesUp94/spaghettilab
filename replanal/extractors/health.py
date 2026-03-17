from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from replanal.config import HealthBarConfig
from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData, HealthReading, Side


class HealthBarExtractor(BaseExtractor):
    """Extracts health percentage from P1 and P2 health bars.

    Scans multiple Y bands to handle tournament overlays that shift the
    HUD vertically.  For each band, tries warm (red/orange), cool
    (blue/purple), and yellow/green detection.  Takes the best reading
    across all bands and color modes per player.

    P1 and P2 are processed independently — if one player's bar is
    unreadable (shifted, hidden, or too small), the other's reading is
    still emitted.

    A rolling median filter smooths out 1-3 frame brightness fluctuations
    from GGS screen dim/flash effects (supers, round transitions, hit
    animations) that would otherwise cause false 0% or inflated readings.
    """

    # x-ranges for health bar measurement.  The bars fill from the
    # portrait (edge) toward the center.  At low health the remaining
    # sliver sits near the center, so the range must extend close to
    # the timer area (~x=960) to catch it.
    P1_X1, P1_X2 = 200, 900
    P2_X1, P2_X2 = 1040, 1800

    # Primary Y band for health bars (confirmed from pixel inspection).
    _PRIMARY_BAND = (112, 142)
    # Alternate Y bands for shifted HUDs (tried only as fallback).
    _ALT_BANDS = [(88, 112), (142, 170)]
    # Pre-computed full Y range covering all bands (for single HSV conversion).
    _FULL_Y1 = 88
    _FULL_Y2 = 170

    # Minimum run length to consider a player's bar readable
    _MIN_RUN = 50

    # Anchor tolerance: the health bar fill must be anchored to the correct
    # edge of the region.  P1's bar depletes from the left (portrait side),
    # so the right end of the fill stays fixed near the right edge.  P2's
    # bar depletes from the right, so the left start stays near the left edge.
    # Readings that don't meet this constraint are spurious (e.g. round
    # transition UI elements that happen to have warm pixels).
    _P1_ANCHOR_END_MIN = 640   # P1 run must end at column >= this (out of 700)
    _P2_ANCHOR_START_MAX = 40  # P2 run must start at column <= this (out of 760)

    # Activation: either player must exceed this for N consecutive frames
    _ACTIVATE_THRESHOLD = 200
    _ACTIVATE_FRAMES = 10

    # Rolling median window.  Upward spikes (SLASH, LET'S ROCK text) are
    # already rejected by the spike-rejection cap, so this window only
    # needs to be large enough to smooth 1-3 frame screen-dim artifacts.
    # Kept small so real rapid health drops (combos/supers) shift the
    # median within ~4 frames.
    _MEDIAN_WINDOW = 9

    # Non-gameplay gap (in frame numbers) that triggers partial state reset.
    # Resets median buffers, but preserves max_w calibration.
    _RESET_GAP = 15

    # Max rate health can drop per frame (prevents instant 0% from noise).
    # 3% per frame ≈ 90% per second at 30fps — well above any real damage rate.
    _MAX_DROP_PER_FRAME = 0.03

    # Number of consecutive unreadable frames to consider a player "dead"
    # (bar gone after KO).  Must exceed any brief screen-flash duration.
    _DEATH_FRAMES = 10

    def __init__(self, config: HealthBarConfig | None = None):
        self.config = config or HealthBarConfig()
        self._last_frame: int = -9999
        self._max_w: tuple[int, int] = (0, 0)
        self._activated: bool = False
        self._activate_count: int = 0
        # Rolling median buffers for raw pixel measurements
        self._p1_buf: deque[int] = deque(maxlen=self._MEDIAN_WINDOW)
        self._p2_buf: deque[int] = deque(maxlen=self._MEDIAN_WINDOW)
        # Track last emitted percentages for monotonic constraint + hold
        self._last_p1_pct: float = 1.0
        self._last_p2_pct: float = 1.0
        # Death / post-round detection: when one player's bar disappears
        # for _DEATH_FRAMES consecutive frames while the other is still
        # readable, the round is over.  _last_frame stops advancing so
        # the gap-based reset fires naturally.
        self._post_round: bool = False
        self._p1_unreadable_count: int = 0
        self._p2_unreadable_count: int = 0
        # Track whether each player has been readable since the last
        # gap reset.  Prevents re-triggering death detection on a player
        # that was never visible in this round (e.g. P1 invisible in
        # round 3 of the Slycoops set).
        self._seen_p1: bool = False
        self._seen_p2: bool = False

    @property
    def required_rois(self) -> list[str]:
        return []

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        frame = ctx.frame_bgr

        # Reset after a non-gameplay gap (round/set transitions).
        # Also resets max_w so the new round recalibrates its 100%
        # reference from fresh observations.  This is critical when
        # transition VFX produce warm pixels at a different width than
        # the actual health bar (e.g. P2 warm at 414px during SLASH
        # while P2 at 100% health is 722px in a different round).
        if ctx.frame_number - self._last_frame > self._RESET_GAP:
            self._activated = False
            self._activate_count = 0
            self._p1_buf.clear()
            self._p2_buf.clear()
            self._last_p1_pct = 1.0
            self._last_p2_pct = 1.0
            self._max_w = (0, 0)
            self._post_round = False
            self._p1_unreadable_count = 0
            self._p2_unreadable_count = 0
            self._seen_p1 = False
            self._seen_p2 = False

        # ── Scan Y bands: primary first, alt bands only as fallback ─────
        # Convert the full potential range to HSV once, then slice per band.
        full_strip = frame[self._FULL_Y1:self._FULL_Y2]
        full_hsv = cv2.cvtColor(full_strip, cv2.COLOR_BGR2HSV)

        def _anchor_filter_p1(candidates: list[tuple[int, int]]) -> tuple[int, int]:
            """Pick best P1 fill that passes anchor check (end >= 640)."""
            best_w, best_s = 0, 0
            for w, s in candidates:
                if w >= self._MIN_RUN and (s + w) < self._P1_ANCHOR_END_MIN:
                    continue  # reject: not anchored to right edge
                if w > best_w:
                    best_w, best_s = w, s
            return best_w, best_s

        def _anchor_filter_p2(candidates: list[tuple[int, int]]) -> tuple[int, int]:
            """Pick best P2 fill that passes anchor check (start <= 40)."""
            best_w, best_s = 0, 0
            for w, s in candidates:
                if w >= self._MIN_RUN and s > self._P2_ANCHOR_START_MAX:
                    continue  # reject: not anchored to left edge
                if w > best_w:
                    best_w, best_s = w, s
            return best_w, best_s

        def _measure_band(y1: int, y2: int) -> tuple[int, int]:
            hsv = full_hsv[y1 - self._FULL_Y1 : y2 - self._FULL_Y1]
            # Each measure returns (run_length, start_index).
            # Apply anchor validation per-fill before selecting the best,
            # so an invalid fill can't beat a shorter but valid one.
            p1_w, p1_s = _anchor_filter_p1([
                self._measure_warm_fill(hsv, self.P1_X1, self.P1_X2),
                self._measure_yellow_green_fill(hsv, self.P1_X1, self.P1_X2),
            ])
            p2_w, p2_s = _anchor_filter_p2([
                self._measure_warm_fill(hsv, self.P2_X1, self.P2_X2),
                self._measure_cool_fill(hsv, self.P2_X1, self.P2_X2),
                self._measure_yellow_green_fill(hsv, self.P2_X1, self.P2_X2),
            ])
            return p1_w, p2_w

        # Try primary band first.
        p1_w, p2_w = _measure_band(*self._PRIMARY_BAND)

        # Only try alt bands for whichever player is below threshold,
        # AND only if the primary band showed at least a minimal signal.
        # Without this guard, transition UI elements in alt bands
        # produce spurious readings that prevent gap-based state resets.
        _ALT_MIN_PRIMARY = 20
        if p1_w < self._MIN_RUN or p2_w < self._MIN_RUN:
            for y1, y2 in self._ALT_BANDS:
                p1_alt, p2_alt = _measure_band(y1, y2)
                if p1_w < self._MIN_RUN and p1_w >= _ALT_MIN_PRIMARY:
                    p1_w = max(p1_w, p1_alt)
                if p2_w < self._MIN_RUN and p2_w >= _ALT_MIN_PRIMARY:
                    p2_w = max(p2_w, p2_alt)

        p1_readable = p1_w >= self._MIN_RUN
        p2_readable = p2_w >= self._MIN_RUN

        # Track readability for death detection.
        if p1_readable:
            self._seen_p1 = True
            self._p1_unreadable_count = 0
        else:
            self._p1_unreadable_count += 1
        if p2_readable:
            self._seen_p2 = True
            self._p2_unreadable_count = 0
        else:
            self._p2_unreadable_count += 1

        # Death detection: one player's bar disappeared (KO) while the
        # other is still readable → round is over.  Stop advancing
        # _last_frame so the gap-based reset fires naturally.
        if not self._post_round:
            if (self._p1_unreadable_count >= self._DEATH_FRAMES
                    and self._seen_p1 and p2_readable):
                self._post_round = True
            elif (self._p2_unreadable_count >= self._DEATH_FRAMES
                  and self._seen_p2 and p1_readable):
                self._post_round = True

        # At least one player's bar must be visible for this to be gameplay.
        if not p1_readable and not p2_readable:
            return

        # ── Activation (require N consecutive qualifying frames) ──────────
        if not self._activated:
            # Always advance _last_frame during warmup so the gap counter
            # doesn't fire while activation is building up.
            self._last_frame = ctx.frame_number
            if max(p1_w, p2_w) >= self._ACTIVATE_THRESHOLD:
                self._activate_count += 1
                if self._activate_count >= self._ACTIVATE_FRAMES:
                    self._activated = True
            else:
                self._activate_count = 0

        if not self._activated:
            return

        # ── Spike-rejection constraint (upward AND downward) ──────────
        # Skip spike rejection while buffer is warming up after a gap reset,
        # so the buffer can establish a true baseline from fresh readings.
        _FLOOR_FACTOR = 0.15
        _WARMUP_MIN = self._MEDIAN_WINDOW // 2
        # Stall threshold for _last_frame gating: if raw reading exceeds
        # the median by more than this factor, it's treated as a transition
        # signal and _last_frame is NOT advanced.  Higher than the 5% spike
        # cap to avoid stalling on normal frame-to-frame noise.
        _STALL_FACTOR = 1.10

        p1_was_clamped = False
        p2_was_clamped = False
        p1_significantly_above = False
        p2_significantly_above = False
        p1_raw, p2_raw = p1_w, p2_w  # save pre-clamped values

        if p1_readable and len(self._p1_buf) >= _WARMUP_MIN:
            prev_p1 = int(np.median(list(self._p1_buf)))
            cap_p1 = int(prev_p1 * 1.05) + 3
            floor_p1 = int(prev_p1 * _FLOOR_FACTOR)
            if p1_w > cap_p1:
                p1_w = cap_p1
                p1_was_clamped = True
            elif p1_w < floor_p1:
                p1_was_clamped = True
            if p1_raw > int(prev_p1 * _STALL_FACTOR) + 10:
                p1_significantly_above = True
        if p2_readable and len(self._p2_buf) >= _WARMUP_MIN:
            prev_p2 = int(np.median(list(self._p2_buf)))
            cap_p2 = int(prev_p2 * 1.05) + 3
            floor_p2 = int(prev_p2 * _FLOOR_FACTOR)
            if p2_w > cap_p2:
                p2_w = cap_p2
                p2_was_clamped = True
            elif p2_w < floor_p2:
                p2_was_clamped = True
            if p2_raw > int(prev_p2 * _STALL_FACTOR) + 10:
                p2_significantly_above = True

        # ── _last_frame gating ────────────────────────────────────────
        # Only advance _last_frame if at least one readable player has
        # a reading within 10% of its buffer median.  When ALL readable
        # bars are significantly above their median (transition screen
        # warm pixels or round-start refill), _last_frame stalls, and
        # the gap-based reset at the top of extract() eventually fires.
        #
        # Post-round override: after death detection triggers, _last_frame
        # is frozen regardless of readings, ensuring the gap reset fires.
        if not self._post_round:
            any_trustworthy = (
                (p1_readable and not p1_significantly_above)
                or (p2_readable and not p2_significantly_above)
            )
            if any_trustworthy:
                self._last_frame = ctx.frame_number

        # Update max widths (only from non-clamped, readable readings).
        p1_max, p2_max = self._max_w
        if p1_readable and p1_w > p1_max and not p1_was_clamped:
            p1_max = p1_w
        if p2_readable and p2_w > p2_max and not p2_was_clamped:
            p2_max = p2_w
        self._max_w = (p1_max, p2_max)

        # ── P1 processing ──────────────────────────────────────────────
        if p1_readable:
            if not p1_was_clamped:
                self._p1_buf.append(p1_w)
            p1_smooth = int(np.median(list(self._p1_buf))) if self._p1_buf else p1_w
            p1_pct = p1_smooth / p1_max if p1_max > 0 else 0.0

            _SLOW_DROP = 0.002
            p1_bar_gone = (p1_w == 0) or p1_was_clamped
            if self._last_p1_pct > 0:
                drop = _SLOW_DROP if p1_bar_gone else self._MAX_DROP_PER_FRAME
                if p1_pct < self._last_p1_pct - drop:
                    p1_pct = self._last_p1_pct - drop

            p1_pct = max(0.0, min(1.0, p1_pct))
            self._last_p1_pct = p1_pct
            data.p1_health = HealthReading(
                side=Side.P1,
                health_pct=p1_pct,
                bar_pixels_filled=p1_smooth,
                bar_pixels_total=p1_max,
            )

        # ── P2 processing ──────────────────────────────────────────────
        if p2_readable:
            if not p2_was_clamped:
                self._p2_buf.append(p2_w)
            p2_smooth = int(np.median(list(self._p2_buf))) if self._p2_buf else p2_w
            p2_pct = p2_smooth / p2_max if p2_max > 0 else 0.0

            _SLOW_DROP = 0.002
            p2_bar_gone = (p2_w == 0) or p2_was_clamped
            if self._last_p2_pct > 0:
                drop = _SLOW_DROP if p2_bar_gone else self._MAX_DROP_PER_FRAME
                if p2_pct < self._last_p2_pct - drop:
                    p2_pct = self._last_p2_pct - drop

            p2_pct = max(0.0, min(1.0, p2_pct))
            self._last_p2_pct = p2_pct
            data.p2_health = HealthReading(
                side=Side.P2,
                health_pct=p2_pct,
                bar_pixels_filled=p2_smooth,
                bar_pixels_total=p2_max,
            )

    # ── Fill width measurement ───────────────────────────────────────────

    def _measure_warm_fill(self, hsv: np.ndarray, x1: int, x2: int) -> tuple[int, int]:
        """Measure fill width using warm pixel detection (red/pink/orange bars).

        Warm pixels: H < 30 or H > 150, S > 30, V > 80.
        A column counts as "filled" if >35% of its vertical pixels are warm.
        Returns (run_length, start_index) of the longest continuous run.
        """
        h_ch = hsv[:, x1:x2, 0]
        s_ch = hsv[:, x1:x2, 1]
        v_ch = hsv[:, x1:x2, 2]

        warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)
        col_ratio = np.mean(warm, axis=0)
        is_filled = col_ratio > 0.35

        return self._longest_run(is_filled)

    def _measure_yellow_green_fill(self, hsv: np.ndarray, x1: int, x2: int) -> tuple[int, int]:
        """Measure fill width for yellow/green bars.

        Some tournament overlays render health bars in yellow or green
        (H approx 25-70).  Uses S > 40, V > 100 to distinguish from
        dark background.
        Returns (run_length, start_index).
        """
        h_ch = hsv[:, x1:x2, 0]
        s_ch = hsv[:, x1:x2, 1]
        v_ch = hsv[:, x1:x2, 2]

        filled = (h_ch >= 25) & (h_ch <= 70) & (s_ch > 40) & (v_ch > 100)
        col_ratio = np.mean(filled, axis=0)
        is_filled = col_ratio > 0.35

        return self._longest_run(is_filled)

    def _measure_cool_fill(self, hsv: np.ndarray, x1: int, x2: int) -> tuple[int, int]:
        """Measure fill width for blue/purple bars.

        In some tournament overlays, the P2 health bar is rendered in blue/purple
        (H approx 90-145). The filled portion has S > 40, V > 80 while the depleted
        portion is dark (V < 50).
        Returns (run_length, start_index).
        """
        h_ch = hsv[:, x1:x2, 0]
        s_ch = hsv[:, x1:x2, 1]
        v_ch = hsv[:, x1:x2, 2]

        is_blue = (h_ch >= 90) & (h_ch <= 145) & (s_ch > 40)
        filled = is_blue & (v_ch > 80)

        col_ratio = np.mean(filled, axis=0)
        is_filled = col_ratio > 0.35

        return self._longest_run(is_filled)

    @staticmethod
    def _longest_run(is_filled: np.ndarray) -> tuple[int, int]:
        """Find longest continuous run of True values.

        Returns (length, start_index) of the longest run.
        """
        best = best_start = 0
        current = start = 0
        for i, val in enumerate(is_filled):
            if val:
                if current == 0:
                    start = i
                current += 1
                if current > best:
                    best = current
                    best_start = start
            else:
                current = 0
        return best, best_start

    def reset_calibration(self) -> None:
        """Reset all state (call between replays)."""
        self._max_w = (0, 0)
        self._activated = False
        self._activate_count = 0
        self._p1_buf.clear()
        self._p2_buf.clear()
        self._last_frame = -9999
        self._last_p1_pct = 1.0
        self._last_p2_pct = 1.0
        self._post_round = False
        self._p1_unreadable_count = 0
        self._p2_unreadable_count = 0
        self._seen_p1 = False
        self._seen_p2 = False
