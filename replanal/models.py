from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Side(Enum):
    P1 = 1
    P2 = 2


@dataclass
class FrameContext:
    """Immutable context passed to every extractor for a single frame."""

    video_path: str
    frame_number: int
    timestamp_ms: float
    frame_bgr: np.ndarray
    rois: dict[str, np.ndarray]


@dataclass
class HealthReading:
    side: Side
    health_pct: float  # 0.0 to 1.0 (cleaned by HealthAggregator after extraction)
    bar_pixels_filled: int
    bar_pixels_total: int
    # Per-frame measurement before temporal cleaning (None = no reading this frame,
    # e.g. a gap filled in by the cleaner).
    raw_pct: Optional[float] = None
    # Red "pending" combo damage shown right after the fill (already-lost HP).
    pending_pct: float = 0.0
    # 0..1 per-frame measurement confidence (occlusion / fit quality).
    confidence: float = 1.0


@dataclass
class FrameData:
    """All extracted data for a single frame."""

    frame_number: int
    timestamp_ms: float
    is_gameplay: bool = True
    p1_health: Optional[HealthReading] = None
    p2_health: Optional[HealthReading] = None
    # Per-frame extractor readings, kept after HealthAggregator replaces
    # p1_health/p2_health with the temporally cleaned values.
    p1_health_raw: Optional[HealthReading] = None
    p2_health_raw: Optional[HealthReading] = None
    timer_value: Optional[int] = None
    p1_tension_pct: Optional[float] = None
    p2_tension_pct: Optional[float] = None
    # Per-frame readings before the replay-level clean-up (finalize_timer,
    # finalize_tension), kept so it can be re-run from the parquet.
    timer_raw: Optional[int] = None
    p1_tension_raw: Optional[float] = None
    p2_tension_raw: Optional[float] = None
    p1_burst_available: Optional[bool] = None
    p2_burst_available: Optional[bool] = None
    p1_rounds_won: Optional[int] = None
    p2_rounds_won: Optional[int] = None
    combo_count: Optional[int] = None
    combo_side: Optional[Side] = None
    text_popups: list[str] = field(default_factory=list)


@dataclass
class DamageEvent:
    """Aggregated from health deltas across frames."""

    timestamp_ms: float
    frame_start: int
    frame_end: int
    target_side: Side
    damage_pct: float  # positive = damage dealt
    pre_health_pct: float
    post_health_pct: float
