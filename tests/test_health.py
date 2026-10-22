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

    def test_warm_scenery_above_the_bar_does_not_hide_fill(self):
        # New VOD set 01 at 140 s: a cream/pink stage right above P2's pink
        # bar marked those columns "not a bar" and cut the reading at the
        # start of the scenery (0.37 instead of 0.41).
        img = _synthetic_frame(0.80, 0.41)
        img[100:110, 1180:1400] = (170, 200, 235)   # pale warm scenery, not the bar's colour
        d = _extract(img)
        assert abs(d.p2_health.health_pct - 0.41) < 0.01

    def test_dim_blue_stage_through_an_empty_bar_is_not_fill(self):
        # New VOD set 01 at 411 s: P2 is KO'd; a blue stage seen through the
        # semi-transparent empty bar (H ~95, S ~70, V ~80) read as 7 % fill.
        img = _synthetic_frame(0.25, 0.0)
        img[116:139, 1052:1100] = (90, 72, 45)
        d = _extract(img)
        assert d.p2_health is not None and d.p2_health.health_pct < 0.005

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
    q2[80:90] = 0.5    # the lethal combo's red covers the bar where the fill was
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


def test_lethal_hit_before_ko_cinematic_reaches_zero():
    # 11_WFinal R12 (~670-677 s): P1 (Goldlewis, a 1 % sliver) combos P2
    # (1.8 %, red pending shown, P1's combo counter up).  P1's own sprite
    # covers its bar (reads empty).  The lethal hit shows P2 empty for one
    # frame, then the KO cinematic hides the HUD and reads as junk fill above
    # P2's level, with low confidence.  P2 must end at 0 and P1 keep 1 %.
    n = 260
    p1 = np.full(n, 0.010)
    p2 = _round([(60, 0.06), (140, 0.018), (60, np.nan)])
    c1 = np.full(n, 0.78)
    c2 = np.full(n, 0.95)
    q1 = np.zeros(n)
    q2 = np.zeros(n)
    q2[40:200] = 0.39
    combo1 = np.zeros(n, dtype=bool)
    combo1[30:201] = True
    timer = np.full(n, np.nan)
    timer[:200] = 62
    p1[100:115] = 0.0                       # P1's bar under its own sprite
    p1[140:150] = 0.0
    p2[199] = 0.0                           # the lethal hit
    q2[199] = 0.41
    p1[201:] = np.nan
    p1[204] = 0.0                           # KO cinematic junk
    c1[204] = 0.85
    p1[217:220] = (0.063, 0.045, 0.0)
    c1[217:220] = (0.77, 0.83, 0.99)
    p2[204] = 0.032
    c2[204] = 0.29
    p2[217:220] = (0.093, 0.106, 0.110)
    c2[217:220] = (0.45, 0.40, 0.36)
    p2[226:229] = (0.103, 0.30, 0.0)
    c2[226:229] = (0.30, 0.04, 0.37)
    r = clean_series(p1, p2, c1, c2, q1, q2, timer, combo1, np.zeros(n, dtype=bool))
    assert r.p2[199] == 0.0
    assert np.nanmax(r.p2[199:]) == 0.0
    assert abs(np.nanmin(r.p1) - 0.010) < 1e-9


def test_attacker_hp_held_while_its_combo_counter_is_shown():
    # The defender's red pending segment is not seen (bright stage), but the
    # combo counter is on P1's side: P1 is the one comboing, so a run of low
    # readings on P1's bar (its own sprite over the bar's far end) is
    # occlusion, not damage.  The HUD is then hidden (a cinematic).
    n = 220
    p1 = np.full(n, 0.50)
    p2 = _round([(80, 0.40), (40, 0.25), (40, 0.12), (60, np.nan)])
    p1[120:160] = 0.31
    p1[160:] = np.nan
    c = np.full(n, 0.9)
    combo1 = np.zeros(n, dtype=bool)
    combo1[82:160] = True
    timer = np.full(n, np.nan)
    timer[:158] = 50
    zeros = np.zeros(n)
    r = clean_series(p1, p2, c, c, zeros, zeros, timer, combo1, np.zeros(n, dtype=bool))
    assert abs(np.nanmin(r.p1) - 0.50) < 1e-9
    assert abs(r.p2[159] - 0.12) < 1e-9


def test_attacker_hit_with_its_own_red_is_damage():
    # Same as above, but P1 really is hit (a trade): its own red pending is
    # shown, so the lower level is accepted.
    n = 220
    p1 = np.full(n, 0.50)
    p1[120:] = 0.31
    p2 = _round([(80, 0.40), (40, 0.25), (100, 0.12)])
    q1 = np.zeros(n)
    q1[120:135] = 0.19
    c = np.full(n, 0.9)
    combo1 = np.zeros(n, dtype=bool)
    combo1[82:160] = True
    r = clean_series(p1, p2, c, c, q1, np.zeros(n), None, combo1, np.zeros(n, dtype=bool))
    assert abs(r.p1[-1] - 0.31) < 1e-9


def test_attacker_hold_on_a_segment_shorter_than_the_pending_window():
    # A 4-frame segment (resplit 03_UF): the 5-frame pending smoothing must
    # not break the attacker hold.
    p1 = _round([(400, 1.0), (200, np.nan), (4, 0.9)])
    p2 = _round([(400, 0.5), (200, np.nan), (4, 0.9)])
    c = np.full(len(p1), 0.9)
    q = np.zeros(len(p1))
    q[-4:] = 0.1
    combo = np.zeros(len(p1), dtype=bool)
    combo[-4:] = True
    r = clean_series(p1, p2, c, c, q, q, None, combo, combo)
    assert len(r.p1) == len(p1)


def test_lethal_hit_read_with_low_confidence_but_red_covering_the_bar_is_a_ko():
    # 07_UF R5 (~275.3 s): P2 sits at 5 % with red shown, the lethal hit reads
    # P2 empty for one frame with confidence 0.57 (< END_CONF), but the red
    # pending segment covers P2's whole remaining level, so the HUD is visible.
    # The KO cinematic then reads junk 0-2 % at low/middling confidence.
    n = 330
    p1 = np.full(n, 0.713)
    p2 = np.full(n, 0.054)
    p2[:60] = 0.30
    c1 = np.full(n, 0.85)
    c2 = np.full(n, 0.73)
    q1 = np.zeros(n)
    q2 = np.zeros(n)
    q2[60:200] = 0.06
    timer = np.full(n, np.nan)
    timer[:200] = 77
    p2[199], c2[199], q2[199] = 0.014, 0.70, 0.10
    p2[200], c2[200], q2[200] = 0.0, 0.57, 0.106      # the lethal hit
    p1[201:], p2[201:] = np.nan, np.nan
    junk = [0.0, 0.019, 0.015, 0.011, 0.008, 0.008, 0.013, 0.011, 0.0, 0.004, 0.01, 0.07, 0.024, 0.0]
    conf = [0.24, 0.52, 0.53, 0.52, 0.52, 0.48, 0.45, 0.29, 0.22, 0.20, 0.21, 0.36, 0.49, 0.55]
    for k, (v, cc) in enumerate(zip(junk, conf)):
        p2[205 + 2 * k], c2[205 + 2 * k] = v, cc
        p1[205 + 2 * k], c1[205 + 2 * k] = 0.0, 0.95   # P1's HUD is junk too
    r = clean_series(p1, p2, c1, c2, q1, q2, timer)
    assert r.p2[200] == 0.0
    assert np.nanmax(r.p2[200:]) == 0.0
    assert abs(r.p1[200] - 0.713) < 1e-9


def test_ko_from_a_sliver_with_cinematic_junk_at_the_bar_floor():
    # 10_UF R1 (~67.8 s): P1 (3 %, red shown) is hit down to a 5-px sliver for
    # half a second, then reads empty (confident, red shown).  The KO
    # cinematic reads junk: a burst of high fill, then a steady 11 px
    # (0.0153) over the empty bar.  That steady floor junk is not the bar.
    n = 400
    p1 = np.full(n, 0.051)
    p1[:60] = 0.30
    p1[100:] = 0.031
    p1[150:165] = 0.006
    q1 = np.zeros(n)
    q1[60:100] = 0.031
    q1[100:150] = 0.051
    q1[150:166] = 0.076
    p2 = np.full(n, 0.654)
    c1 = np.full(n, 0.95)
    c2 = np.full(n, 0.89)
    timer = np.full(n, np.nan)
    timer[:166] = 76
    p1[165], c1[165], q1[165] = 0.0, 0.98, 0.082      # the lethal hit
    p1[166:], p2[166:] = np.nan, np.nan
    burst = [0.064, 0.344, 0.092, 0.314, 0.051, 0.316, 0.317, 0.090, 0.061, 0.061, 0.134, 0.149,
             0.072, 0.070, 0.108, 0.071, 0.025, 0.071, 0.090, 0.007, 0.0, 0.135, 0.106, 0.088,
             0.079, 0.072, 0.086, 0.058, 0.093, 0.014, 0.070, 0.060, 0.070, 0.115, 0.057]
    floor = [0.0153] * 10 + [0.057] * 9 + [0.092] * 6
    for k, v in enumerate(burst + floor):
        p1[166 + k], c1[166 + k] = v, 0.5
        p2[166 + k], c2[166 + k] = 0.0, 0.5
    r = clean_series(p1, p2, c1, c2, q1, np.zeros(n), timer)
    assert r.p1[165] == 0.0
    assert np.nanmax(r.p1[165:]) == 0.0
    assert abs(r.p2[165] - 0.654) < 1e-9


def test_low_bar_hidden_mid_combo_with_red_shown_is_not_a_ko():
    # Guard for the two rules above: a 3 % bar (red covering it) reads empty
    # for a few frames with middling confidence, then is shown at its level
    # again.  No round timer to veto it: the bar coming back alone must.
    n = 300
    p1 = np.full(n, 0.03)
    p1[:60] = 0.25
    q1 = np.zeros(n)
    q1[60:140] = 0.05
    p1[120:125] = 0.0
    c1 = np.full(n, 0.9)
    c1[120:125] = 0.45
    p2 = np.full(n, 0.60)
    c2 = np.full(n, 0.9)
    r = clean_series(p1, p2, c1, c2, q1, np.zeros(n))
    assert abs(r.p1[-1] - 0.03) < 1e-9
    assert np.nanmin(r.p1) > 0.02


def test_sliver_read_empty_under_a_super_background_is_not_the_ko():
    # yt_KWXz7dmt2Pk R0 (~45.8 s): P2 is a 1 % sliver (red shown) under a long
    # super; its bar reads empty at low confidence for seconds, once at 0.59
    # with the red covering it.  The combo goes on (the KO is ~1.3 s later),
    # so that frame is not the KO: no confident reading of fill just before
    # it shows the HUD was visible.
    n = 300
    p1 = np.full(n, 0.13)
    p2 = np.full(n, 0.008)
    p2[:60] = 0.20
    c1 = np.full(n, 0.85)
    c2 = np.full(n, 0.85)
    q2 = np.zeros(n)
    q2[60:] = 0.13
    c2[100:240] = 0.2                       # busy super background
    p2[100:240] = 0.0
    p2[150], c2[150], q2[150] = 0.0, 0.59, 0.086
    p2[240], c2[240], q2[240] = 0.0, 0.65, 0.23   # the real KO
    timer = np.full(n, np.nan)
    timer[:100] = 67
    p1[241:], p2[241:] = np.nan, np.nan
    r = clean_series(p1, p2, c1, c2, np.zeros(n), q2, timer)
    assert r.p2[200] > 0.0
    assert r.p2[240] == 0.0


def _countdown(n: int, start: int, every: int = 30) -> np.ndarray:
    """A round timer counting down one per second (``every`` samples)."""
    return np.array([start - k // every for k in range(n)], dtype=float)


def test_frozen_clock_over_an_empty_bar_is_a_ko():
    # 02_UF R1 (e2e copy): P2 (1 %) is KO'd under the slash with no red seen;
    # the HUD comes back for 1.5 s with the clock stopped at 57 and P2's bar
    # empty.  The win pose then reads P1 empty and P2 at its old sliver
    # (no timer): that is not P1's KO.
    n = 420
    p1 = np.full(n, 0.197)
    p2 = np.full(n, 0.011)
    p2[:60] = 0.30
    c1 = np.full(n, 0.96)
    c2 = np.full(n, 0.86)
    timer = np.full(n, np.nan)
    timer[:200] = _countdown(200, 63)                 # 63 .. 57
    p1[202:290] = np.nan                              # the slash hides the HUD
    p2[202:290] = np.nan
    p1[205:260] = 0.0                                 # slash junk on P1's bar
    c1[205:260] = 0.5
    timer[290:336] = 57                               # HUD back, clock stopped
    p2[290:336] = 0.0
    c2[290:336] = 0.95
    p1[336:345] = np.nan
    p2[336:345] = np.nan
    p1[345:] = 0.0                                    # win pose: P1 read empty
    c1[345:] = 0.5
    c2[345:] = 0.65                                   # P2 read at its sliver
    r = clean_series(p1, p2, c1, c2, np.zeros(n), np.zeros(n), timer)
    assert r.p2[335] == 0.0
    assert abs(np.nanmin(r.p1) - 0.197) < 1e-9


def test_round_result_does_not_depend_on_the_video_after_it():
    # 01_WRI R5: after the last round the stream shows its lobby for minutes
    # (no timer); its junk "bars" must not pull the round's values (P1 held
    # 0.287 from the lobby instead of its 0.164).  The KO is not seen.
    n = 700
    p1 = np.full(n, 0.164)
    p1[:200] = 0.5
    p2 = np.full(n, 0.08)
    p2[:100] = 0.6
    c = np.full(n, 0.9)
    q2 = np.zeros(n)
    timer = _countdown(n, 80)
    p1[660:], p2[660:], timer[660:] = np.nan, np.nan, np.nan
    alone = clean_series(p1, p2, c, c, np.zeros(n), q2, timer)
    lobby = 3000
    pl1 = np.concatenate([p1, np.full(60, np.nan), np.full(lobby, 0.287)])
    pl2 = np.concatenate([p2, np.full(60, np.nan), np.full(lobby, 0.287)])
    cl = np.concatenate([c, np.zeros(60), np.full(lobby, 0.8)])
    ql = np.concatenate([q2, np.zeros(60 + lobby)])
    tl = np.concatenate([timer, np.full(60 + lobby, np.nan)])
    followed = clean_series(pl1, pl2, cl, cl, np.zeros(len(pl1)), ql, tl)
    for a, b in ((alone.p1, followed.p1), (alone.p2, followed.p2)):
        both = ~np.isnan(a[:n]) & ~np.isnan(b[:n])
        assert both[:660].all()
        assert np.allclose(a[:n][both], b[:n][both])
    assert abs(followed.p1[655] - 0.164) < 1e-9 and abs(followed.p2[655] - 0.08) < 1e-9


def test_round_that_ends_with_the_clock_running_ended_in_a_ko():
    # 12_LSemi R1: P2 (5 %) is KO'd under a super with the clock at 77; the
    # HUD never shows the empty bar, but the next round starts.  A round
    # only ends by a KO or a time-out (clock at 0), so P2 is at 0.  Ending
    # on a time-out instead keeps P2's level.
    for last, expect in ((77, 0.0), (0, 0.053)):
        n1, gap, n2 = 400, 90, 200
        p1 = np.concatenate([np.full(n1, 0.873), np.full(gap, np.nan), np.full(n2, 1.0)])
        p2 = np.concatenate([np.full(n1, 0.053), np.full(gap, np.nan), np.full(n2, 1.0)])
        p2[:100] = 0.4
        c = np.where(np.isnan(p1), 0.0, 0.9)
        timer = np.concatenate([_countdown(n1, last + 13), np.full(gap, np.nan), np.full(n2, 99.0)])
        r = clean_series(p1, p2, c, c, None, None, timer)
        assert len(r.resets) == 2
        assert abs(r.p2[n1 - 1] - expect) < 1e-9, last
        assert abs(r.p1[n1 - 1] - 0.873) < 1e-9


def test_hidden_ko_is_not_given_to_the_side_doing_a_combo_or_on_a_low_clock():
    # As above, P2 is behind (5 %) when the HUD hides with the clock running.
    # - P2's combo counter is up just before: P2 is attacking (a super landed
    #   from behind under the cinematic), so P2 is not declared KO'd.
    # - The clock last read 2: the round may have ended on time while the
    #   HUD was hidden, so P2 keeps its level too.
    # Without either, P2 is at 0 (the rule above).
    n1, gap, n2 = 400, 90, 200
    for combo_up, last, expect in ((False, 77, 0.0), (True, 77, 0.053), (False, 2, 0.053)):
        p1 = np.concatenate([np.full(n1, 0.873), np.full(gap, np.nan), np.full(n2, 1.0)])
        p2 = np.concatenate([np.full(n1, 0.053), np.full(gap, np.nan), np.full(n2, 1.0)])
        p2[:100] = 0.4
        c = np.where(np.isnan(p1), 0.0, 0.9)
        timer = np.concatenate([_countdown(n1, last + 13), np.full(gap, np.nan), np.full(n2, 99.0)])
        combo2 = np.zeros(n1 + gap + n2, dtype=bool)
        if combo_up:
            combo2[n1 - 20:n1] = True
        r = clean_series(p1, p2, c, c, None, None, timer, None, combo2)
        assert abs(r.p2[n1 - 1] - expect) < 1e-9, (combo_up, last)
        assert abs(r.p1[n1 - 1] - 0.873) < 1e-9


def test_survivor_shown_again_after_the_ko_keeps_its_level():
    # 11_WFinal R4: P1 (2 %) is KO'd; one junk frame then reads P2's 7 % bar
    # nearly empty (within the double-KO window), but the HUD later shows P2
    # at its level with the clock stopped: no double KO.
    n = 400
    p1 = np.full(n, 0.021)
    p2 = np.full(n, 0.071)
    p1[:60] = 0.2
    c1 = np.full(n, 0.9)
    c2 = np.full(n, 0.95)
    q1 = np.zeros(n)
    timer = np.full(n, np.nan)
    timer[:200] = _countdown(200, 72)
    p1[200:] = 0.0                                    # the lethal hit
    q1[195:203] = 0.03
    p1[203:280], p2[203:280] = np.nan, np.nan         # KO cinematic
    p2[208:220] = 0.004                               # junk on the survivor
    c2[208:220] = 0.75
    timer[280:320] = 66                               # HUD back, clock stopped
    p1[280:320] = 0.008
    c1[280:320] = 0.55
    p2[280:320] = 0.089
    c2[280:320] = 1.0
    p1[320:], p2[320:] = np.nan, np.nan
    r = clean_series(p1, p2, c1, c2, q1, np.zeros(n), timer)
    assert r.p1[319] == 0.0
    assert abs(r.p2[319] - 0.071) < 1e-9


# ── A KO ends the round (clock stops, HUD goes for the slash) ────────────

def _clock(n: int, start: int, stop: int, every: int = 30) -> np.ndarray:
    """Timer counting down one per ``every`` samples until frame ``stop``, then frozen."""
    t = np.array([start - min(k, stop) // every for k in range(n)], dtype=float)
    return t


def test_sliver_while_the_clock_runs_is_not_a_ko():
    # New VOD set 04 R3 / set 10 R0: P1 is down to the last HP point (a 1-px
    # sliver read 0.003) from frame 100, but the clock keeps counting down and
    # P2 keeps taking damage.  P2 is KO'd at the end by P1's super (HUD hidden,
    # then shown again with the clock stopped: P2 empty, P1's sliver).
    n = 560
    timer = _clock(n, 70, 420)
    p1 = np.full(n, 0.30)
    p1[100:] = 0.003
    p2 = np.full(n, 0.60)
    q2 = np.zeros(n)
    p2[200:], q2[200:230] = 0.40, 0.20
    p2[350:], q2[350:380] = 0.20, 0.20
    p2[410:], q2[410:430] = 0.01, 0.19
    c1, c2 = np.full(n, 0.8), np.full(n, 0.9)
    combo1 = np.zeros(n, dtype=bool)
    combo1[400:430] = True
    hidden = slice(430, 480)
    for a in (p1, p2, timer):
        a[hidden] = np.nan
    p2[480:] = 0.0
    p1[480:] = 0.004
    r = clean_series(p1, p2, c1, c2, np.zeros(n), q2, timer, combo1, None)
    assert r.p2[-1] == 0.0
    assert r.p1[-1] > 0.0 and np.nanmin(r.p1) > 0.0      # alive the whole round
    assert abs(r.p2[300] - 0.40) < 0.01                  # damage after the "KO" is kept
    assert abs(r.p2[400] - 0.20) < 0.01


def test_winner_is_frozen_at_the_ko_through_the_slash_wipe():
    # New VOD set 06 R8 / set 09 R4: P1 is KO'd while the HUD is hidden; when
    # the HUD comes back (clock stopped) the round-end wipe covers part of
    # P2's bar, which reads 0.75 instead of 0.95.
    n = 420
    timer = _clock(n, 70, 290)
    p1 = np.full(n, 0.30)
    q1 = np.zeros(n)
    p1[250:], q1[250:300] = 0.05, 0.25
    p2 = np.full(n, 0.95)
    c = np.full(n, 0.9)
    hidden = slice(300, 360)
    for a in (p1, p2, timer):
        a[hidden] = np.nan
    p1[360:] = 0.0
    p2[360:] = 0.75
    r = clean_series(p1, p2, c, c, q1, np.zeros(n), timer)
    assert r.p1[-1] == 0.0
    assert abs(r.p2[-1] - 0.95) < 1e-9


def test_red_sprite_over_the_winner_bar_is_not_its_ko():
    # yt_KWXz7dmt2Pk R2: the attacker's red sprite covers its own bar right
    # before it KOs P2: P1 reads empty with "pending red" 0.14 next to a 0.49
    # bar.  A lethal combo's red covers the bar where the fill was.
    n = 300
    timer = _clock(n, 80, 200)
    p1 = np.full(n, 0.49)
    q1 = np.zeros(n)
    p1[190:200], q1[190:200] = 0.0, 0.14
    p2 = np.full(n, 0.04)
    q2 = np.zeros(n)
    p2[200:], q2[200:215] = 0.0, 0.04
    c = np.full(n, 0.85)
    for a in (p1, p2, timer):
        a[215:250] = np.nan
    r = clean_series(p1, p2, c, c, q1, q2, timer)
    assert r.p2[-1] == 0.0
    assert abs(r.p1[-1] - 0.49) < 1e-9


def test_hidden_ko_with_both_sides_low_goes_to_the_side_being_hit():
    # New VOD set 07 R1: both bars low and close (0.035 / 0.047) when the HUD
    # goes; P2 shows the red of P1's combo (P1's counter is up).  The next
    # round starts without the HUD coming back.
    n = 700
    timer = _clock(n, 60, 330)
    p1 = np.full(n, 0.035)
    p1[:100] = 0.4
    p2 = np.full(n, 0.047)
    p2[:200] = 0.5
    q2 = np.zeros(n)
    q2[200:330] = 0.45
    combo1 = np.zeros(n, dtype=bool)
    combo1[260:325] = True
    c = np.full(n, 0.9)
    for a in (p1, p2, timer):
        a[330:450] = np.nan
    # Next round: intro timer and full bars.
    timer[450:] = 99
    p1[450:], p2[450:], q2[450:] = 1.0, 1.0, 0.0
    r = clean_series(p1, p2, c, c, np.zeros(n), q2, timer, combo1, None)
    assert r.resets[-1] >= 440
    assert r.p2[320] == 0.0 or r.p2[329] == 0.0
    assert r.p1[329] > 0.0


def test_double_ko_both_sides_reach_zero():
    # Both sides low (0.15 / 0.12) and emptied by simultaneous hits with the
    # red shown; the clock stops.  A double KO: both end at exactly 0 (the
    # alive floor only applies to a side that survived).
    n = 400
    timer = _clock(n, 60, 300)
    p1, p2 = np.full(n, 0.15), np.full(n, 0.12)
    q1, q2 = np.zeros(n), np.zeros(n)
    p1[300:], q1[300:330] = 0.0, 0.15
    p2[300:], q2[300:330] = 0.0, 0.12
    c = np.full(n, 0.9)
    r = clean_series(p1, p2, c, c, q1, q2, timer)
    assert r.p1[-1] == 0.0 and r.p2[-1] == 0.0
    assert r.p1[299] > 0.0 and r.p2[299] > 0.0


def test_clock_carrying_on_across_a_long_hud_gap_keeps_one_round():
    # Tappa #4 14_Grands R6: a super hides the whole HUD for 5.6 s (more than
    # SEGMENT_GAP) with the clock at 84 on both sides: still one segment.
    from replanal.health_clean import SEGMENT_GAP, _clock_continues
    n = 900
    timer = _clock(n, 90, 900)
    gap = slice(300, 300 + SEGMENT_GAP + 20)
    timer[gap] = np.nan
    assert _clock_continues(timer, gap.start - 1, gap.stop)
    p1, p2 = np.full(n, 0.8), np.full(n, 0.6)
    p1[gap] = p2[gap] = np.nan
    c = np.full(n, 0.9)
    r = clean_series(p1, p2, c, c, None, None, timer)
    assert len(r.resets) == 1
    # A new round's intro value after the gap is no continuation.
    t2 = timer.copy()
    t2[gap.stop:] = 99
    assert not _clock_continues(t2, gap.start - 1, gap.stop)


def test_end_loser_needs_the_attacker_counter_at_the_end_of_play():
    # Both low when the HUD goes for good (no HUD shown after the slash, no
    # red on either side).  P1's counter from a combo that ended 2 s before
    # the end of play says nothing about the last hit (P2's one-hit reversal
    # shows no counter); a counter up in the last frames of play does.
    from replanal.health_clean import _end_loser
    n = 400
    timer = _clock(n, 40, 300)
    timer[300:] = np.nan
    p1, p2 = np.full(n, 0.05), np.full(n, 0.04)
    c = np.full(n, 0.9)
    q = np.zeros(n)
    combo1 = np.zeros(n, dtype=bool)
    combo1[200:250] = True
    assert _end_loser(p1, p2, c, c, q, q, combo1, None, timer, 0, n) is None
    combo1[285:300] = True
    assert _end_loser(p1, p2, c, c, q, q, combo1, None, timer, 0, n) == 2


def test_kod_side_is_empty_on_the_hud_shown_after_the_slash():
    # New VOD set 08 R10 (590-690 s of the raw readings): both on slivers,
    # P2 KO'd by P1's last hit, the HUD comes back at 674.3 s with the clock
    # frozen at 42 and P2's bar empty (read with confidence 0.64-0.89).  The
    # fit left P2 empty (None) there, so the round ended at P2's 0.021.
    from pathlib import Path
    d = np.load(Path(__file__).parent / "data" / "health" / "nv08_r10_end.npz")
    r = clean_series(d["p1"], d["p2"], d["c1"], d["c2"], d["q1"], d["q2"], d["timer"], d["combo1"], d["combo2"])
    back = (d["t"] > 674.2) & (d["t"] < 676.7) & ~np.isnan(d["timer"])
    assert back.sum() > 50
    assert (r.p2[back] == 0.0).all()
    assert (r.p1[back] > 0.02).all()
