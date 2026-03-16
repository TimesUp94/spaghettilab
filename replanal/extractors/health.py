from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from replanal.config import HealthBarConfig
from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData, HealthReading, Side


class HealthBarExtractor(BaseExtractor):
    """Extracts health percentage from P1 and P2 health bars.

    Uses a single Y band (y=112-142) confirmed to work for both "sets/"
    tournament overlay and "Replay_*" overlay styles.

    For P2, both warm (red) and cool (blue) detection are tried.

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

    # Y band for health bars (confirmed from pixel inspection).
    BAND_Y1, BAND_Y2 = 112, 142

    # Minimum P1 run to produce any reading at all
    _MIN_P1_RUN = 50

    # Activation: P1 warm must exceed this for N consecutive frames
    _ACTIVATE_THRESHOLD = 200
    _ACTIVATE_FRAMES = 10

    # Rolling median window — must be large enough to outlast screen-wide
    # text effects (LET'S ROCK, SLASH, etc.) which last ~30 frames.  At 31
    # frames, even 15 corrupted values can't shift the median.
    _MEDIAN_WINDOW = 31

    # Non-gameplay gap (in frame numbers) that triggers partial state reset.
    # Resets median buffers, but preserves max_w calibration.
    _RESET_GAP = 15

    # Max rate health can drop per frame (prevents instant 0% from noise).
    # 3% per frame ≈ 90% per second at 30fps — well above any real damage rate.
    _MAX_DROP_PER_FRAME = 0.03

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

    @property
    def required_rois(self) -> list[str]:
        return []

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        frame = ctx.frame_bgr

        # Partial reset after a non-gameplay gap (round/set transitions).
        # Preserves max_w so calibration carries across rounds.
        if ctx.frame_number - self._last_frame > self._RESET_GAP:
            self._activated = False
            self._activate_count = 0
            self._p1_buf.clear()
            self._p2_buf.clear()
            self._last_p1_pct = 1.0
            self._last_p2_pct = 1.0

        # Measure health bar band
        strip = frame[self.BAND_Y1:self.BAND_Y2]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        p1_w = self._measure_warm_fill(hsv, self.P1_X1, self.P1_X2)
        if p1_w < self._MIN_P1_RUN:
            return

        # P2: try both warm and cool detection
        p2_w = max(
            self._measure_warm_fill(hsv, self.P2_X1, self.P2_X2),
            self._measure_cool_fill(hsv, self.P2_X1, self.P2_X2),
        )

        # Only update _last_frame when we have valid measurements.
        self._last_frame = ctx.frame_number

        # ── Activation (require N consecutive qualifying frames) ──────────
        if not self._activated:
            if p1_w >= self._ACTIVATE_THRESHOLD:
                self._activate_count += 1
                if self._activate_count >= self._ACTIVATE_FRAMES:
                    self._activated = True
            else:
                self._activate_count = 0

        if not self._activated:
            return

        # ── Spike-rejection constraint (upward AND downward) ──────────
        _FLOOR_FACTOR = 0.4

        p1_was_clamped = False
        p2_was_clamped = False
        if self._p1_buf:
            prev_p1 = int(np.median(list(self._p1_buf)))
            cap_p1 = int(prev_p1 * 1.05) + 3
            floor_p1 = int(prev_p1 * _FLOOR_FACTOR)
            if p1_w > cap_p1:
                p1_w = cap_p1
                p1_was_clamped = True
            elif p1_w < floor_p1:
                p1_was_clamped = True
        if self._p2_buf:
            prev_p2 = int(np.median(list(self._p2_buf)))
            cap_p2 = int(prev_p2 * 1.05) + 3
            floor_p2 = int(prev_p2 * _FLOOR_FACTOR)
            if p2_w > cap_p2:
                p2_w = cap_p2
                p2_was_clamped = True
            elif p2_w < floor_p2:
                p2_was_clamped = True

        # Update max widths (only from non-clamped readings).
        p1_max, p2_max = self._max_w
        if p1_w > p1_max and not p1_was_clamped:
            p1_max = p1_w
        if p2_w > p2_max and not p2_was_clamped:
            p2_max = p2_w
        self._max_w = (p1_max, p2_max)

        # Rolling median to smooth screen dim/flash artifacts.
        if not p1_was_clamped:
            self._p1_buf.append(p1_w)
        if not p2_was_clamped:
            self._p2_buf.append(p2_w)
        p1_smooth = int(np.median(list(self._p1_buf))) if self._p1_buf else p1_w
        p2_smooth = int(np.median(list(self._p2_buf))) if self._p2_buf else p2_w

        p1_pct = p1_smooth / p1_max if p1_max > 0 else 0.0
        p2_pct = p2_smooth / p2_max if p2_max > 0 else 0.0

        # ── Max drop rate + hold-on-zero ─────────────────────────────
        _SLOW_DROP = 0.002

        p2_bar_gone = (p2_w == 0) or p2_was_clamped
        p1_bar_gone = (p1_w == 0) or p1_was_clamped

        if self._last_p2_pct > 0:
            drop = _SLOW_DROP if p2_bar_gone else self._MAX_DROP_PER_FRAME
            if p2_pct < self._last_p2_pct - drop:
                p2_pct = self._last_p2_pct - drop

        if self._last_p1_pct > 0:
            drop = _SLOW_DROP if p1_bar_gone else self._MAX_DROP_PER_FRAME
            if p1_pct < self._last_p1_pct - drop:
                p1_pct = self._last_p1_pct - drop

        p1_pct = max(0.0, min(1.0, p1_pct))
        p2_pct = max(0.0, min(1.0, p2_pct))

        self._last_p1_pct = p1_pct
        self._last_p2_pct = p2_pct

        data.p1_health = HealthReading(
            side=Side.P1,
            health_pct=p1_pct,
            bar_pixels_filled=p1_smooth,
            bar_pixels_total=p1_max,
        )
        data.p2_health = HealthReading(
            side=Side.P2,
            health_pct=p2_pct,
            bar_pixels_filled=p2_smooth,
            bar_pixels_total=p2_max,
        )

    # ── Fill width measurement ───────────────────────────────────────────

    def _measure_warm_fill(self, hsv: np.ndarray, x1: int, x2: int) -> int:
        """Measure fill width using warm pixel detection (red/pink/orange bars).

        Warm pixels: H < 30 or H > 150, S > 30, V > 80.
        A column counts as "filled" if >35% of its vertical pixels are warm.
        Returns the longest continuous run of filled columns.
        """
        h_ch = hsv[:, x1:x2, 0]
        s_ch = hsv[:, x1:x2, 1]
        v_ch = hsv[:, x1:x2, 2]

        warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)
        col_ratio = np.mean(warm, axis=0)
        is_filled = col_ratio > 0.35

        return self._longest_run(is_filled)

    def _measure_cool_fill(self, hsv: np.ndarray, x1: int, x2: int) -> int:
        """Measure fill width for blue/purple bars.

        In some tournament overlays, the P2 health bar is rendered in blue/purple
        (H approx 90-145). The filled portion has S > 40, V > 80 while the depleted
        portion is dark (V < 50).
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
    def _longest_run(is_filled: np.ndarray) -> int:
        """Find longest continuous run of True values."""
        best = 0
        current = 0
        for val in is_filled:
            if val:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        return best

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
