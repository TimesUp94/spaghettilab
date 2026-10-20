from replanal.extractors.health import HealthBarExtractor
from replanal.models import FrameContext, FrameData


class TestHealthBarExtractor:
    def test_reads_health_from_ggs_frame(self, sample_ggs_frame, default_config):
        """Goldlewis vs Potemkin at 10s — both should have significant health."""
        extractor = HealthBarExtractor(default_config.health_bar)
        rois = {}
        for name in extractor.required_rois:
            roi_def = default_config.rois[name]
            rois[name] = roi_def.crop(sample_ggs_frame)

        ctx = FrameContext(
            video_path="test",
            frame_number=300,
            timestamp_ms=10000,
            frame_bgr=sample_ggs_frame,
            rois=rois,
        )
        data = FrameData(frame_number=300, timestamp_ms=10000)
        extractor.extract(ctx, data)

        assert data.p1_health is not None
        assert data.p2_health is not None
        # Both should be between 0 and 1
        assert 0.0 <= data.p1_health.health_pct <= 1.0
        assert 0.0 <= data.p2_health.health_pct <= 1.0
        # At 10s into a match, both players should still have health
        assert data.p1_health.health_pct > 0.1
        assert data.p2_health.health_pct > 0.1

    def test_reads_health_different_characters(self, sample_ggs_frame_2, default_config):
        """Anji vs Baiken — different character colors should still work."""
        extractor = HealthBarExtractor(default_config.health_bar)
        rois = {}
        for name in extractor.required_rois:
            roi_def = default_config.rois[name]
            rois[name] = roi_def.crop(sample_ggs_frame_2)

        ctx = FrameContext(
            video_path="test",
            frame_number=300,
            timestamp_ms=10000,
            frame_bgr=sample_ggs_frame_2,
            rois=rois,
        )
        data = FrameData(frame_number=300, timestamp_ms=10000)
        extractor.extract(ctx, data)

        assert data.p1_health is not None
        assert data.p2_health is not None
        assert 0.0 <= data.p1_health.health_pct <= 1.0
        assert 0.0 <= data.p2_health.health_pct <= 1.0

    def test_roi_crop_dimensions(self, sample_ggs_frame, default_config):
        """Verify ROI crops produce expected dimensions."""
        for name in ("p1_health_bar", "p2_health_bar"):
            roi = default_config.rois[name]
            cropped = roi.crop(sample_ggs_frame)
            assert cropped.shape[1] == roi.width
            assert cropped.shape[0] == roi.height
            assert cropped.shape[2] == 3  # BGR


# ── Per-frame measurement on synthetic bars ──────────────────────────────

import numpy as np  # noqa: E402

from replanal.aggregator import HealthAggregator, clean_frames  # noqa: E402
from replanal.health_clean import clean_series, isotonic_l1_decreasing  # noqa: E402
from replanal.models import HealthReading, Side  # noqa: E402

PINK = (190, 180, 245)      # BGR pastel pink fill (H~0-10, S~70, V~245)
RED = (0, 0, 160)           # BGR pending-damage red (S 255, V 160)
DARK = (40, 35, 30)         # empty bar background


def _synthetic_frame(p1_fill: float, p2_fill: float, p1_pending: float = 0.0,
                     white_p1: tuple[int, int] | None = None) -> np.ndarray:
    ext = HealthBarExtractor
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:] = (20, 20, 20)
    y1, y2 = 113, 142
    for x1, x2, fill, pend, p1 in ((ext.P1_X1, ext.P1_X2, p1_fill, p1_pending, True),
                                   (ext.P2_X1, ext.P2_X2, p2_fill, 0.0, False)):
        w = x2 - x1
        img[y1:y2, x1:x2] = DARK
        fpx, ppx = int(round(fill * w)), int(round(pend * w))
        if p1:
            img[y1:y2, x2 - fpx:x2] = PINK
            if ppx:
                img[y1:y2, x2 - fpx - ppx:x2 - fpx] = RED
        else:
            img[y1:y2, x1:x1 + fpx] = PINK
    if white_p1 is not None:
        img[y1:y2, white_p1[0]:white_p1[1]] = (255, 255, 255)
    return img


def _extract(img):
    ext = HealthBarExtractor()
    data = FrameData(frame_number=0, timestamp_ms=0)
    ext.extract(FrameContext(video_path="t", frame_number=0, timestamp_ms=0, frame_bgr=img, rois={}), data)
    return data


class TestBarMeasurement:
    def test_fill_and_pending_are_separate(self):
        d = _extract(_synthetic_frame(0.40, 0.75, p1_pending=0.20))
        assert abs(d.p1_health.health_pct - 0.40) < 0.01
        assert abs(d.p1_health.pending_pct - 0.20) < 0.01  # red is NOT counted as health
        assert abs(d.p2_health.health_pct - 0.75) < 0.01

    def test_white_sprite_over_fill_does_not_cut_reading(self):
        # A sprite across the middle of P1's fill used to split the longest run.
        d = _extract(_synthetic_frame(0.90, 0.90, white_p1=(400, 460)))
        assert abs(d.p1_health.health_pct - 0.90) < 0.01
        assert d.p1_health.confidence < 1.0

    def test_low_hp_sliver_is_read(self):
        d = _extract(_synthetic_frame(0.03, 0.80))
        assert d.p1_health is not None
        assert abs(d.p1_health.health_pct - 0.03) < 0.01

    def test_no_hud_emits_nothing(self):
        img = np.zeros((1080, 1920, 3), np.uint8)
        d = _extract(img)
        assert d.p1_health is None and d.p2_health is None

    def test_fullscreen_orange_is_not_a_bar(self):
        # Super cinematics fill the HUD area with one colour (letterboxed top).
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[111:] = (40, 140, 240)  # orange
        d = _extract(img)
        for r in (d.p1_health, d.p2_health):
            assert r is None or r.confidence < 0.35


# ── Temporal cleaning ────────────────────────────────────────────────────

def test_isotonic_l1_is_monotone_and_robust():
    y = np.array([1.0, 1.0, 0.2, 1.0, 1.0, 0.8, 0.8, 0.95, 0.8, 0.8])
    fit = isotonic_l1_decreasing(y, np.ones_like(y))
    assert np.all(np.diff(fit) <= 1e-12)
    assert fit[2] == 1.0          # single low outlier ignored
    assert fit[-1] == 0.8


def _round(levels: list[tuple[int, float]]) -> np.ndarray:
    return np.concatenate([np.full(n, v) for n, v in levels])


def test_clean_series_rejects_spikes_and_resets_between_rounds():
    # Round 1: P1 1.0 -> 0.6 -> 0.0 (KO), P2 stays 0.7 after a hit.
    p1 = _round([(60, 1.0), (60, 0.6), (40, 0.0)])
    p2 = _round([(30, 1.0), (130, 0.7)])
    # Occlusion spikes in both directions.
    p1[70:73] = 1.0      # upward spike mid-round
    p2[90:94] = 0.0      # false zero (flash)
    # Gap, then round 2 at full HP.
    gap = np.full(60, np.nan)
    p1 = np.concatenate([p1, gap, _round([(100, 1.0), (100, 0.5)])])
    p2 = np.concatenate([p2, gap, _round([(200, 1.0)])])
    c = np.ones_like(p1)
    q1 = np.zeros_like(p1)
    q1[120:130] = 0.3    # lethal combo: pending red shown next to the emptied bar
    r = clean_series(p1, p2, c, c, q1, None)
    # Round 1
    assert r.p1[71] == 0.6
    assert abs(r.p2[92] - 0.7) < 1e-9
    assert r.p1[155] == 0.0 and abs(r.p2[155] - 0.7) < 1e-9
    # Never goes up within a segment
    for k in set(r.segment[r.segment >= 0]):
        for s in (r.p1, r.p2):
            v = s[(r.segment == k) & ~np.isnan(s)]
            assert np.all(np.diff(v) <= 1e-12)
    # Round 2 starts again at full
    assert len(r.resets) == 2
    assert r.p1[-150] == 1.0 and r.p1[-1] == 0.5


def test_occluded_frames_are_filled_without_going_up():
    p1 = _round([(50, 0.9), (20, np.nan), (50, 0.5)])
    p2 = np.full(120, 0.8)
    c1 = np.ones(120)
    c1[50:70] = 0.0
    r = clean_series(p1, p2, c1, np.ones(120))
    mid = r.p1[50:70]
    assert not np.isnan(mid).any()
    assert np.all(np.diff(r.p1) <= 1e-12)


def test_survivor_hp_not_lowered_after_ko():
    # P2 is KO'd (empty bar with pending red); afterwards junk low readings for P1.
    p1 = _round([(100, 0.8), (40, 0.1)])
    p2 = _round([(80, 0.5), (60, 0.0)])
    q2 = np.zeros(140)
    q2[80:90] = 0.2
    c = np.ones(140)
    c1 = c.copy()
    r = clean_series(p1, p2, c1, c, None, q2)
    assert r.p2[-1] == 0.0
    assert abs(r.p1[-1] - 0.8) < 1e-9


def test_aggregator_cleans_frames_in_place_and_keeps_raw():
    frames = []
    vals = [1.0] * 40 + [0.6] * 40
    vals[50] = 1.0  # spike
    for i, v in enumerate(vals):
        f = FrameData(frame_number=i, timestamp_ms=i * 33.3)
        f.p1_health = HealthReading(Side.P1, v, int(v * 719), 719, raw_pct=v, confidence=1.0)
        f.p2_health = HealthReading(Side.P2, 0.9, 647, 719, raw_pct=0.9, confidence=1.0)
        frames.append(f)
    events = HealthAggregator().aggregate(frames)
    assert frames[50].p1_health.health_pct == 0.6
    assert frames[50].p1_health_raw.raw_pct == 1.0
    assert len([e for e in events if e.target_side == Side.P1]) == 1
    # Idempotent
    clean_frames(frames)
    assert frames[50].p1_health.health_pct == 0.6


def test_attacker_bar_hidden_during_opponent_combo_is_not_damage():
    # Set 13 R3: P2 is being comboed (pending red shown) and P1's bar is
    # covered by its own super VFX, reading empty, then P2 is KO'd.
    n = 200
    p1 = np.full(n, 0.47)
    p2 = _round([(80, 0.30), (120, 0.03)])
    q1 = np.zeros(n)
    q2 = np.zeros(n)
    q2[80:152] = 0.17
    p1[125:150] = np.linspace(0.05, 0.09, 25)   # sliver of "fill" under the VFX
    p2[150:152] = 0.0
    p1[152:] = np.nan                           # KO slash hides the HUD
    p2[152:] = np.nan
    c = np.full(n, 0.85)
    r = clean_series(p1, p2, c, c, q1, q2)
    assert abs(np.nanmin(r.p1) - 0.47) < 1e-9


def test_ko_from_full_hp_locks_survivor():
    # Set 05 R9 (a Perfect): P2 is KO'd by a lethal combo; KO-screen junk
    # (with a little fake red) then reads P1's full bar as nearly empty.
    n = 160
    p1 = np.full(n, 1.0)
    p2 = _round([(60, 0.30), (60, 0.011), (2, 0.0), (38, np.nan)])
    q1 = np.zeros(n)
    q2 = np.zeros(n)
    q2[60:122] = 0.4
    c1 = np.full(n, 0.9)
    p1[125:142] = 0.2                           # KO-screen junk, fake red
    q1[125:142] = 0.05
    c1[125:142] = 0.55
    p1[142:] = np.nan
    r = clean_series(p1, p2, c1, np.full(n, 0.9), q1, q2)
    assert r.p2[121] == 0.0
    assert abs(np.nanmin(r.p1) - 1.0) < 1e-9


def test_bar_that_comes_back_was_not_koed():
    # Set 14 R1: a sprite covers P1's low bar for 2 s (reads empty, confident,
    # with its own red still shown), then the bar is visible again.
    n = 300
    p1 = _round([(60, 0.35), (240, 0.164)])
    p2 = np.full(n, 0.70)
    q1 = np.zeros(n)
    q1[60:104] = 0.3
    p1[100:160] = 0.0
    c = np.full(n, 0.85)
    r = clean_series(p1, p2, c, c, q1, np.zeros(n))
    assert abs(r.p1[-1] - 0.164) < 1e-9


def test_running_clock_vetoes_ko_of_tiny_bar():
    # A 3% bar flickers to empty (with a flash of red) while the round timer
    # keeps counting down: not a KO.
    n = 300
    p1 = np.full(n, 0.031)
    p1[:60] = 0.2
    p1[100:110] = 0.0
    p1[115:200] = 0.0
    q1 = np.zeros(n)
    q1[100:103] = 0.04
    p2 = np.full(n, 0.24)
    timer = np.full(n, np.nan)
    timer[60:100] = 84
    timer[150:190] = 83
    timer[220:260] = 82
    c = np.full(n, 0.85)
    r = clean_series(p1, p2, c, c, q1, np.zeros(n), timer)
    assert abs(r.p1[-1] - 0.031) < 1e-9
    assert abs(r.p2[-1] - 0.24) < 1e-9


def test_round_intro_belongs_to_the_next_round():
    # Set 01: the next round's timer already reads 90 while both bars are
    # still empty, before they fill up; the previous round's survivor must
    # not end at 0.
    p1 = _round([(400, 0.6), (60, np.nan), (12, 0.0), (300, 1.0)])
    p2 = _round([(300, 0.5), (100, 0.0), (60, np.nan), (12, 0.0), (300, 1.0)])
    q2 = np.zeros(len(p1))
    q2[300:310] = 0.3
    timer = np.full(len(p1), np.nan)
    timer[:390] = 60
    timer[460:772] = 90
    c = np.ones(len(p1))
    r = clean_series(p1, p2, c, c, None, q2, timer)
    assert r.resets[-1] <= 460
    assert abs(np.nanmin(r.p1[:460]) - 0.6) < 1e-9


def test_short_ko_before_slash_reaches_zero():
    # Set 12 R1: P2 (at 5 %) is KO'd during a super cinematic that hides the
    # HUD; the bar shows empty for only 6 frames before the slash.  Too few
    # down-weighted empty readings to form a level on their own.
    n = 260
    p1 = np.full(n, 0.873)
    p2 = np.full(n, 0.053)
    c1 = np.full(n, 0.95)
    c2 = np.full(n, 0.8)
    timer = np.full(n, np.nan)
    timer[:100] = 72
    p1[100:] = np.nan                           # cinematic: HUD hidden
    p2[100:] = np.nan
    p2[150:160] = 0.06                          # junk under the cinematic
    c2[150:160] = 0.65
    p1[200:206] = 0.03                          # the HUD is back for 6 frames
    c1[200:206] = 0.43
    p2[200:206] = 0.0
    c2[200:206] = 0.95
    r = clean_series(p1, p2, c1, c2, np.zeros(n), np.zeros(n), timer)
    assert r.p2[205] == 0.0
    assert abs(np.nanmin(r.p1) - 0.873) < 1e-9


def test_timer_intro_starts_round_when_a_bar_never_reads_full():
    # yt_KWXz7dmt2Pk ~490 s: on a bright stage a full P1 bar reads 0.62 with
    # confidence 0.05, so the bars are never both seen full.  The round
    # timer reappears after the KO and reads its start value for seconds.
    n1, gap, n2 = 400, 60, 400
    p1 = np.concatenate([np.full(n1, 0.3), np.full(gap, np.nan), np.full(n2, 0.62)])
    p2 = np.concatenate([np.linspace(1.0, 0.05, n1), np.full(gap, np.nan), np.full(n2, 1.0)])
    c1 = np.concatenate([np.full(n1, 0.8), np.zeros(gap), np.full(n2, 0.05)])
    c2 = np.concatenate([np.full(n1, 0.8), np.zeros(gap), np.full(n2, 0.95)])
    timer = np.concatenate([np.full(n1, 70.0), np.full(gap, np.nan), np.full(120, 99.0),
                            np.full(n2 - 120, 90.0)])
    r = clean_series(p1, p2, c1, c2, None, None, timer)
    assert any(n1 <= s <= n1 + gap for s in r.resets)
    # Round 2 is not fitted together with round 1: P2 is back at full.
    assert abs(r.p2[-1] - 1.0) < 1e-9
    assert np.nanmin(r.p2[:n1]) < 0.1


def test_timer_intro_needs_a_countdown_before_it():
    # The same bars, but the segment's timer never left its start values
    # (still the first seconds of a round): no reset from the intro rule.
    n1, gap, n2 = 400, 60, 400
    p1 = np.concatenate([np.full(n1, 0.3), np.full(gap, np.nan), np.full(n2, 0.62)])
    p2 = np.concatenate([np.linspace(1.0, 0.05, n1), np.full(gap, np.nan), np.full(n2, 1.0)])
    c1 = np.concatenate([np.full(n1, 0.8), np.zeros(gap), np.full(n2, 0.05)])
    c2 = np.concatenate([np.full(n1, 0.8), np.zeros(gap), np.full(n2, 0.95)])
    timer = np.concatenate([np.full(n1, 95.0), np.full(gap, np.nan), np.full(n2, 99.0)])
    r = clean_series(p1, p2, c1, c2, None, None, timer)
    assert not any(n1 <= s <= n1 + gap for s in r.resets)


def test_near_double_ko_survivor_is_the_bar_shown_after_the_ko():
    # yt_PbsXLfP7C1M R0: P1 at 1.7 %, P2 at 12.7 %.  In the last exchange
    # P1's sliver reads empty (with red) and P2 is hit to 1.7 %; after the KO
    # cinematic the HUD shows P1's sliver and P2's empty bar: P2 lost.
    n = 400
    p1 = np.full(n, 0.017)
    p2 = np.full(n, 0.127)
    c1 = np.full(n, 0.9)
    c2 = np.full(n, 0.9)
    q1 = np.zeros(n)
    q2 = np.zeros(n)
    p1[200:230] = 0.0
    q1[200:215] = 0.8
    p2[200:230] = 0.017
    q2[200:215] = 0.7
    p1[230:330] = np.nan                        # KO cinematic and slash
    p2[230:330] = np.nan
    c1[230:330] = 0.0
    c2[230:330] = 0.0
    p2[330:345] = 0.0                           # HUD back: P1 shown, P2 empty
    p1[345:] = np.nan
    p2[345:] = np.nan
    timer = np.full(n, np.nan)
    timer[:200] = 80
    r = clean_series(p1, p2, c1, c2, q1, q2, timer)
    assert abs(r.p1[344] - 0.017) < 1e-9
    assert r.p2[344] == 0.0


def test_survivor_shown_after_ko_is_not_a_double_ko():
    # 05_UF R6: P1 (2.5 %) is KO'd; P2's 5 % bar reads empty under the KO
    # cinematic, then the HUD shows it again next to P1's empty bar.
    n = 400
    p1 = np.full(n, 0.025)
    p2 = np.full(n, 0.05)
    c1 = np.full(n, 0.9)
    c2 = np.full(n, 0.8)
    q1 = np.zeros(n)
    p1[200:] = 0.0
    q1[200:205] = 0.3
    p2[210:225] = 0.0                           # junk under the cinematic
    p1[225:300] = np.nan
    p2[225:300] = np.nan
    c1[225:300] = 0.0
    c2[225:300] = 0.0
    p1[300:320] = 0.0                           # HUD back: P2 shown, P1 empty
    p2[300:320] = 0.05
    p1[320:] = np.nan
    p2[320:] = np.nan
    timer = np.full(n, np.nan)
    timer[:200] = 80
    r = clean_series(p1, p2, c1, c2, q1, np.zeros(n), timer)
    assert r.p1[319] == 0.0
    assert abs(r.p2[319] - 0.05) < 1e-9
