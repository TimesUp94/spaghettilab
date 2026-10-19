"""Per-frame health bar measurement.

This extractor only answers "what does THIS frame show?".  It does no
temporal smoothing and keeps no round state: the temporal rules (HP only goes
down within a round, it only resets to full between rounds, occlusion spikes
are ignored) live in ``replanal.health_clean`` and are applied by
``HealthAggregator`` once the whole video has been read.

Bar layout (GGST HUD at 1920x1080, same with and without the tournament
overlay): each bar is 719 px wide and is anchored at the centre of the
screen.  P1's fill ends at x=867 and shrinks towards the portrait on the left;
P2's fill starts at x=1052 and shrinks towards the right.  Each column of the
bar is classified as

* F  current health   (pastel pink / beige; orange at low HP)
* R  pending damage   (the dark red segment that shows damage from a combo
                       that is still running; it drains when the combo ends)
* W  neutral: bright colourless pixels (white sprites, super flashes, hit
     sparks) and pastel pixels too dark to be lit fill (fill dimmed by a
     flash or shadow).  These carry no information and are ignored.
* N  anything else: the (semi-transparent) empty part of the bar, or a
     coloured sprite drawn over the HUD.

The health edge is the split point that best explains the columns as
"F before, R/N after" (a 1-D step fit), so a sprite crossing part of the bar
lowers the confidence instead of cutting the reading in half the way the old
longest-run measurement did.  Pending red is NOT counted as health: in GGST
that damage is already lost, it is just shown until the combo ends.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from replanal.config import HealthBarConfig
from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData, HealthReading, Side


@dataclass
class BarMeasurement:
    """Result of measuring one bar in one frame (all widths in pixels)."""

    fill_px: int          # current health, measured from the anchored edge
    pending_px: int       # red pending-damage segment right after the fill
    total_px: int         # bar width (100 % health)
    consistency: float    # fraction of informative columns that agree with the fit
    occluded: float       # fraction of neutral columns (white / dimmed pastel: no info)
    ambiguity: float      # width of the best-fit plateau / total (edge uncertainty)
    mode: str             # colour mode that produced the fit

    @property
    def fill_pct(self) -> float:
        return self.fill_px / self.total_px

    @property
    def pending_pct(self) -> float:
        return self.pending_px / self.total_px

    @property
    def confidence(self) -> float:
        return max(0.0, self.consistency * (1.0 - self.occluded) * (1.0 - self.ambiguity))


class HealthBarExtractor(BaseExtractor):
    """Measures P1/P2 health (and pending combo damage) from the HUD bars."""

    # Bar spans [x1, x2) at 1920x1080.  Full health fills the whole span.
    P1_X1, P1_X2 = 149, 868     # anchored at the right end (x2)
    P2_X1, P2_X2 = 1052, 1771   # anchored at the left end (x1)
    P1_MAX_W = P1_X2 - P1_X1    # 719
    P2_MAX_W = P2_X2 - P2_X1    # 719

    # Rows inside the bar (border rows excluded) for the standard HUD position,
    # then fallback bands for overlays that shift the HUD vertically.
    _PRIMARY_BAND = (116, 139)
    _ALT_BANDS = [(88, 112), (142, 170)]
    _FULL_Y1 = 72
    _FULL_Y2 = 182
    # Rows checked above a band (relative to its top row) and below it
    # (relative to its bottom row).
    _ABOVE = (-12, -5)
    _BELOW = (5, 11)

    # A column belongs to a class when more than this fraction of its rows do.
    _COL_FRAC = 0.5

    # A side's reading is emitted when its fit explains at least this
    # fraction of the informative columns.
    _MIN_CONSISTENCY = 0.70
    # Fill widths below this are treated as "bar empty" (KO'd / 0 HP).
    _EMPTY_PX = 3
    # An alt band is only used when the primary band fit is this poor.
    _ALT_BAND_BELOW = 0.80

    def __init__(self, config: HealthBarConfig | None = None):
        self.config = config or HealthBarConfig()

    @property
    def required_rois(self) -> list[str]:
        return []

    # ── Pixel classification ───────────────────────────────────────────

    @staticmethod
    def _classify(hsv: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel masks (fill, pending_red, white) for one colour mode."""
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]
        red_hue = (h <= 7) | (h >= 168)
        # Pending damage: saturated, mid-brightness red (measured: H 0/179,
        # S 255, V 140-165).  Current-health orange at low HP is H 13-20,
        # S ~240, V ~230, and the pastel fill is S 40-150, V > 200.
        pending = red_hue & (s >= 180) & (v >= 90) & (v <= 205)
        warm_hue = (h <= 30) | (h >= 160)
        # Neutral (no information): white sprites / flashes, and pastel pixels
        # that are too dark to be lit fill -- the fill dimmed by a super flash
        # or a shadow looks like this, and so does the anchored end of the
        # bar under some VFX.  Counting them as "empty" produced false zeros.
        white = ((s < 30) & (v >= 170)) | (warm_hue & (s >= 30) & (s <= 170) & (v >= 80) & (v < 170))
        if mode == "warm":
            fill = warm_hue & (s >= 30) & (v >= 170) & ~pending
        elif mode == "yellow_green":
            fill = (h >= 25) & (h <= 70) & (s > 40) & (v > 100)
        elif mode == "cool":
            fill = (h >= 90) & (h <= 145) & (s > 40) & (v > 80)
        else:  # pragma: no cover
            raise ValueError(mode)
        return fill, pending, white

    def measure_bar(
        self,
        hsv_band: np.ndarray,
        side: Side,
        modes: tuple[str, ...] | None = None,
        hsv_above: np.ndarray | None = None,
        hsv_below: np.ndarray | None = None,
    ) -> BarMeasurement:
        """Fit the health edge for one side on one HSV band (rows x 1920 cols).

        ``hsv_above`` / ``hsv_below`` (a few rows just outside the bar) are used
        to reject fill-coloured columns that are not a bar at all: the bar has
        a finite height, so lit fill must not continue above or below it.
        Full-screen super cinematics in orange/yellow otherwise read as two
        full bars (they are letterboxed, so "above" alone is not enough).
        """
        if side == Side.P1:
            x1, x2 = self.P1_X1, self.P1_X2
        else:
            x1, x2 = self.P2_X1, self.P2_X2
        region = hsv_band[:, x1:x2]
        outside = [a[:, x1:x2] for a in (hsv_above, hsv_below) if a is not None]
        if side == Side.P1:
            region = region[:, ::-1]  # index 0 = anchored (centre) end
            outside = [a[:, ::-1] for a in outside]
        total = region.shape[1]
        if modes is None:
            modes = ("warm", "yellow_green", "cool") if side == Side.P2 else ("warm", "yellow_green")

        best: BarMeasurement | None = None
        best_key = None
        for mode in modes:
            fill, pending, white = self._classify(region, mode)
            F = fill.mean(axis=0) > self._COL_FRAC
            R = pending.mean(axis=0) > self._COL_FRAC
            W = (white.mean(axis=0) > self._COL_FRAC) & ~F & ~R
            if outside:
                not_bar = np.zeros_like(F)
                for a in outside:
                    a_fill, _, _ = self._classify(a, mode)
                    not_bar |= a_fill.mean(axis=0) > self._COL_FRAC
                W = W | ((F | R) & not_bar)
                F = F & ~not_bar
                R = R & ~not_bar
            m, fit_score = self._fit_columns(F, R, W, total, mode)
            # Prefer the colour mode with the most fill evidence, then the best fit.
            key = (fit_score, m.consistency)
            if best is None or key > best_key:
                best, best_key = m, key
        assert best is not None
        return best

    @staticmethod
    def _fit_columns(F: np.ndarray, R: np.ndarray, W: np.ndarray, total: int, mode: str) -> tuple[BarMeasurement, int]:
        """Step fit: choose e maximising  #F[:e] + #notF[e:]  (white columns are neutral)."""
        score = np.where(F, 1, np.where(W, 0, -1)).astype(np.int32)
        c = np.concatenate([[0], np.cumsum(score)])  # c[e] = sum(score[:e])
        best = c.max()
        plateau = np.flatnonzero(c == best)
        lo, hi = int(plateau[0]), int(plateau[-1])
        e = (lo + hi) // 2

        informative = int((~W).sum())
        n_F_before = int(F[:e].sum())
        n_notF_after = int((~F[e:] & ~W[e:]).sum())
        consistency = (n_F_before + n_notF_after) / informative if informative else 0.0
        occluded = float(W.mean())
        ambiguity = (hi - lo) / total

        # Pending red: contiguous-ish run of R right after the edge.
        pending_px = 0
        if e < total:
            tail = np.where(R[e:], 1, np.where(W[e:], 0, -1)).astype(np.int32)
            ct = np.concatenate([[0], np.cumsum(tail)])
            p = int(np.argmax(ct))
            if ct[p] >= 3:
                pending_px = p
        return BarMeasurement(
            fill_px=int(e),
            pending_px=pending_px,
            total_px=total,
            consistency=float(consistency),
            occluded=occluded,
            ambiguity=float(ambiguity),
            mode=mode,
        ), int(best)

    def measure_frame(self, frame_bgr: np.ndarray) -> tuple[BarMeasurement, BarMeasurement]:
        """Measure both bars, trying alt bands for a side whose primary fit is poor."""
        full = cv2.cvtColor(frame_bgr[self._FULL_Y1:self._FULL_Y2], cv2.COLOR_BGR2HSV)

        def band(y1: int, y2: int) -> np.ndarray:
            return full[y1 - self._FULL_Y1:y2 - self._FULL_Y1]

        def above(y1: int, _y2: int) -> np.ndarray:
            return band(y1 + self._ABOVE[0], y1 + self._ABOVE[1])

        def below(_y1: int, y2: int) -> np.ndarray:
            return band(y2 + self._BELOW[0], y2 + self._BELOW[1])

        primary = band(*self._PRIMARY_BAND)
        out = []
        for side in (Side.P1, Side.P2):
            m = self.measure_bar(primary, side, hsv_above=above(*self._PRIMARY_BAND),
                                 hsv_below=below(*self._PRIMARY_BAND))
            if m.consistency < self._ALT_BAND_BELOW or m.fill_px < self._EMPTY_PX:
                for y1, y2 in self._ALT_BANDS:
                    alt = self.measure_bar(band(y1, y2), side, hsv_above=above(y1, y2),
                                           hsv_below=below(y1, y2))
                    # Only switch to an alt band for a clearly better, non-empty fit.
                    if alt.fill_px >= 50 and alt.consistency > max(m.consistency, self._ALT_BAND_BELOW):
                        m = alt
            out.append(m)
        return out[0], out[1]

    # ── Extractor entry point ──────────────────────────────────────────

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        m1, m2 = self.measure_frame(ctx.frame_bgr)

        has1 = m1.fill_px >= self._EMPTY_PX or m1.pending_px >= self._EMPTY_PX
        has2 = m2.fill_px >= self._EMPTY_PX or m2.pending_px >= self._EMPTY_PX
        # No fill or pending red on either side: HUD absent (menus, cutscenes,
        # full-screen flashes).  Emit nothing.
        if not has1 and not has2:
            return

        for side, m, has, other_has in ((Side.P1, m1, has1, has2), (Side.P2, m2, has2, has1)):
            if m.consistency < self._MIN_CONSISTENCY:
                continue
            # An empty bar is only believable when the other bar shows the
            # HUD is on screen (a KO'd player's bar next to a live one).
            if not has and not other_has:
                continue
            reading = HealthReading(
                side=side,
                health_pct=m.fill_pct,
                bar_pixels_filled=m.fill_px,
                bar_pixels_total=m.total_px,
                raw_pct=m.fill_pct,
                pending_pct=m.pending_pct,
                confidence=m.confidence,
            )
            if side == Side.P1:
                data.p1_health = reading
            else:
                data.p2_health = reading

    def reset_calibration(self) -> None:
        """Kept for API compatibility; the extractor is stateless."""
