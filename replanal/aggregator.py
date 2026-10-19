from __future__ import annotations

import numpy as np

from replanal.config import HealthBarConfig
from replanal.health_clean import CleanResult, clean_series
from replanal.models import DamageEvent, FrameData, HealthReading, Side

# Pending red at least this wide (fraction of the bar) counts as "shown".
_PENDING_SHOWN = 0.004


def clean_frames(frames: list[FrameData]) -> CleanResult:
    """Replace per-frame health readings with temporally cleaned values, in place.

    The extractor's per-frame readings are moved to ``p1_health_raw`` /
    ``p2_health_raw`` (kept for the parquet output and debugging);
    ``p1_health`` / ``p2_health`` become the cleaned readings, or None where
    the cleaner has no value (outside rounds, long HUD gaps).  Idempotent:
    a second call starts again from the raw readings.
    """
    n = len(frames)
    arrays = {}
    for side in (Side.P1, Side.P2):
        p = np.full(n, np.nan)
        c = np.zeros(n)
        q = np.zeros(n)
        for i, f in enumerate(frames):
            raw = _raw(f, side)
            if raw is None or not f.is_gameplay:
                continue
            p[i] = raw.raw_pct if raw.raw_pct is not None else raw.health_pct
            c[i] = raw.confidence
            q[i] = raw.pending_pct
        arrays[side] = (p, c, q)

    (p1, c1, q1), (p2, c2, q2) = arrays[Side.P1], arrays[Side.P2]
    timer = np.array([f.timer_value if f.is_gameplay and f.timer_value is not None else np.nan
                      for f in frames], dtype=np.float64)
    result = clean_series(p1, p2, c1, c2, q1, q2, timer)

    for side, cleaned in ((Side.P1, result.p1), (Side.P2, result.p2)):
        for i, f in enumerate(frames):
            raw = _raw(f, side)
            if raw is not None and _get_raw_slot(f, side) is None:
                _set_raw_slot(f, side, raw)
            v = cleaned[i]
            if np.isnan(v):
                new = None
            else:
                total = raw.bar_pixels_total if raw is not None else 719
                new = HealthReading(
                    side=side,
                    health_pct=float(v),
                    bar_pixels_filled=int(round(float(v) * total)),
                    bar_pixels_total=total,
                    raw_pct=(raw.raw_pct if raw.raw_pct is not None else raw.health_pct) if raw is not None else None,
                    pending_pct=raw.pending_pct if raw is not None else 0.0,
                    confidence=raw.confidence if raw is not None else 0.0,
                )
            if side == Side.P1:
                f.p1_health = new
            else:
                f.p2_health = new
    return result


def _get_raw_slot(f: FrameData, side: Side) -> HealthReading | None:
    return f.p1_health_raw if side == Side.P1 else f.p2_health_raw


def _set_raw_slot(f: FrameData, side: Side, r: HealthReading) -> None:
    if side == Side.P1:
        f.p1_health_raw = r
    else:
        f.p2_health_raw = r


def _raw(f: FrameData, side: Side) -> HealthReading | None:
    """The extractor's reading for this frame (before or after cleaning)."""
    r = _get_raw_slot(f, side)
    if r is not None:
        return r
    return f.p1_health if side == Side.P1 else f.p2_health


class HealthAggregator:
    """Cleans health readings (see ``replanal.health_clean``) and detects damage events."""

    # Frames to look ahead for continued decline after a brief pause.
    # At 30fps, 6 frames = 200ms — covers health bar animation easing.
    SETTLE_FRAMES = 6
    # While the red pending segment is shown the combo is still running, so
    # drops up to this many frames apart are one event.
    MAX_COMBO_GAP = 150

    def __init__(self, config: HealthBarConfig | None = None):
        cfg = config or HealthBarConfig()
        self.min_delta = cfg.min_delta_pct
        self.window = cfg.smoothing_window
        self.last_clean: CleanResult | None = None

    def aggregate(self, frames: list[FrameData], clean: bool = True) -> list[DamageEvent]:
        """Clean the health readings in place (unless ``clean=False``), then detect damage events."""
        if clean:
            self.last_clean = clean_frames(frames)
        events: list[DamageEvent] = []
        for side in (Side.P1, Side.P2):
            series = self._extract_series(frames, side)
            if len(series) == 0:
                continue
            pending = self._pending_shown(frames, side)
            cleaned = self._mask_transitions(series)
            smoothed = self._smooth(cleaned)
            events.extend(self._detect_events(smoothed, side, frames, pending))
        events.sort(key=lambda e: e.timestamp_ms)
        return events

    # ------------------------------------------------------------------

    def _mask_transitions(self, series: np.ndarray) -> np.ndarray:
        """NaN-out frames around round transitions.

        A round transition is signaled by health jumping UP by > 8% between
        consecutive valid readings (the cleaned series never rises inside a
        round).  We NaN-out a window around the jump to prevent the
        transition from generating false damage events.
        """
        out = series.copy()
        n = len(out)
        prev_valid = -1
        prev_val = np.nan
        for i in range(n):
            if np.isnan(series[i]):
                continue
            if not np.isnan(prev_val):
                delta = series[i] - prev_val
                if delta > 0.08:  # health jumped up > 8% → round reset
                    lo = max(0, prev_valid - 5)
                    hi = min(n, i + 15)
                    out[lo:hi] = np.nan
            prev_valid = i
            prev_val = series[i]
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

    @staticmethod
    def _pending_shown(frames: list[FrameData], side: Side) -> np.ndarray:
        """Per frame: is the red pending segment shown on this side (2 of 5 frames)?"""
        raw = np.array([
            (r.pending_pct if (r := _raw(f, side)) is not None else 0.0) >= _PENDING_SHOWN
            for f in frames
        ], dtype=np.float64)
        if len(raw) == 0:
            return raw.astype(bool)
        k = np.ones(5)
        return np.convolve(raw, k, mode="same") >= 2

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
        self,
        smoothed: np.ndarray,
        side: Side,
        frames: list[FrameData],
        pending: np.ndarray | None = None,
    ) -> list[DamageEvent]:
        events: list[DamageEvent] = []
        i = 0
        n = len(smoothed)
        if pending is None:
            pending = np.zeros(n, dtype=bool)
        while i < n - 1:
            if np.isnan(smoothed[i]):
                i += 1
                continue

            # Look for a health decrease
            if smoothed[i + 1] < smoothed[i] - self.min_delta:
                start = i
                pre_health = smoothed[i]
                j = i + 1
                # Walk forward while health is dropping or settling.  Allow
                # brief pauses (SETTLE_FRAMES) between hits, and longer ones
                # while the pending red segment is still shown (the combo is
                # still running; GGST drains it only when the combo ends).
                while j < n and not np.isnan(smoothed[j]):
                    found_more = False
                    limit = j + self.SETTLE_FRAMES
                    k = j + 1
                    while k < n and k <= limit:
                        if np.isnan(smoothed[k]):
                            break
                        if smoothed[k] < smoothed[j] - 0.002:
                            j = k
                            found_more = True
                            break
                        if pending[k]:
                            limit = max(limit, min(k + self.SETTLE_FRAMES, j + self.MAX_COMBO_GAP))
                        k += 1
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
