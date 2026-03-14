from __future__ import annotations

import numpy as np

from replanal.config import HealthBarConfig
from replanal.models import DamageEvent, FrameData, Side


class HealthAggregator:
    """Detects damage events from sequential health readings."""

    # Frames to look ahead for continued decline after a brief pause.
    # At 30fps, 6 frames = 200ms — covers health bar animation easing.
    SETTLE_FRAMES = 6

    def __init__(self, config: HealthBarConfig | None = None):
        cfg = config or HealthBarConfig()
        self.min_delta = cfg.min_delta_pct
        self.window = cfg.smoothing_window

    # Minimum frames of stable readings required before a health drop
    # counts as a real damage event. Filters out transition animation noise.
    STABLE_FRAMES = 4

    def aggregate(self, frames: list[FrameData]) -> list[DamageEvent]:
        events: list[DamageEvent] = []
        for side in (Side.P1, Side.P2):
            raw = self._extract_series(frames, side)
            if len(raw) == 0:
                continue
            cleaned = self._mask_transitions(raw)
            smoothed = self._smooth(cleaned)
            events.extend(self._detect_events(smoothed, side, frames))
        events.sort(key=lambda e: e.timestamp_ms)
        return events

    # ------------------------------------------------------------------

    def _mask_transitions(self, series: np.ndarray) -> np.ndarray:
        """NaN-out frames around round transitions.

        A round transition is signaled by health jumping UP by > 8% between
        consecutive valid readings. We NaN-out a window around the jump to
        prevent the transition animation from generating false damage events.
        """
        out = series.copy()
        n = len(out)
        prev_valid = -1
        prev_val = np.nan
        for i in range(n):
            if np.isnan(out[i]):
                continue
            if not np.isnan(prev_val):
                delta = out[i] - prev_val
                if delta > 0.08:  # health jumped up > 8% → round reset
                    # NaN-out a window: 5 frames before to 15 frames after
                    lo = max(0, prev_valid - 5)
                    hi = min(n, i + 15)
                    out[lo:hi] = np.nan
            prev_valid = i
            prev_val = out[i]
        return out

    def _extract_series(self, frames: list[FrameData], side: Side) -> np.ndarray:
        values: list[float] = []
        for f in frames:
            if not f.is_gameplay:
                values.append(np.nan)
                continue
            reading = f.p1_health if side == Side.P1 else f.p2_health
            values.append(reading.health_pct if reading is not None else np.nan)
        return np.array(values, dtype=np.float64)

    def _smooth(self, series: np.ndarray) -> np.ndarray:
        """Moving median filter — resistant to single-frame misreads."""
        out = series.copy()
        half = self.window // 2
        for i in range(half, len(series) - half):
            window = series[i - half : i + half + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                out[i] = np.median(valid)
        return out

    def _detect_events(
        self, smoothed: np.ndarray, side: Side, frames: list[FrameData]
    ) -> list[DamageEvent]:
        events: list[DamageEvent] = []
        i = 0
        n = len(smoothed)
        while i < n - 1:
            if np.isnan(smoothed[i]):
                i += 1
                continue

            # Look for a health decrease
            if smoothed[i + 1] < smoothed[i] - self.min_delta:
                start = i
                pre_health = smoothed[i]
                j = i + 1
                # Walk forward while health is dropping or settling.
                # Allow brief pauses (up to SETTLE_FRAMES) where health
                # holds steady before resuming decline — this handles
                # the health bar animation easing between multi-hit combos.
                while j < n and not np.isnan(smoothed[j]):
                    # Check if decline continues within the settle window
                    found_more = False
                    lookahead = min(j + self.SETTLE_FRAMES + 1, n)
                    for k in range(j + 1, lookahead):
                        if np.isnan(smoothed[k]):
                            break
                        if smoothed[k] < smoothed[j] - 0.002:
                            j = k
                            found_more = True
                            break
                    if not found_more:
                        break

                post_health = smoothed[j]
                damage = pre_health - post_health

                if damage >= self.min_delta:
                    events.append(
                        DamageEvent(
                            timestamp_ms=frames[start].timestamp_ms,
                            frame_start=frames[start].frame_number,
                            frame_end=frames[j].frame_number,
                            target_side=side,
                            damage_pct=damage,
                            pre_health_pct=pre_health,
                            post_health_pct=post_health,
                        )
                    )
                i = j + 1
            else:
                i += 1

        return events
