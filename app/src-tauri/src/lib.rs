use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Read as _, Write as _};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::Emitter;

// ── Data models ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct Replay {
    pub replay_id: String,
    pub video_path: String,
    pub duration_ms: f64,
    pub frame_count: i64,
}

#[derive(Debug, Serialize, Clone)]
pub struct FrameDataPoint {
    pub timestamp_ms: f64,
    pub p1_health_pct: Option<f64>,
    pub p2_health_pct: Option<f64>,
    pub p1_tension_pct: Option<f64>,
    pub p2_tension_pct: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct DamageEvent {
    pub event_id: i64,
    pub replay_id: String,
    pub timestamp_ms: f64,
    pub frame_start: i64,
    pub frame_end: i64,
    pub target_side: i64,
    pub damage_pct: f64,
    pub pre_health_pct: f64,
    pub post_health_pct: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct RoundResult {
    pub replay_id: String,
    pub round_index: usize,
    pub round_start_ms: f64,
    pub round_end_ms: f64,
    pub winner: String,
    pub winner_final_hp: f64,
    pub loser_final_hp: f64,
    pub winner_min_hp: f64,
    pub max_deficit: f64,
    pub deficit_timestamp_ms: f64,
    pub is_comeback: bool,
    pub is_match_start: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct MatchStats {
    pub replay_id: String,
    pub total_rounds: usize,
    pub p1_round_wins: usize,
    pub p2_round_wins: usize,
    pub total_damage_events: usize,
    pub p1_damage_taken: f64,
    pub p2_damage_taken: f64,
    pub p1_biggest_hit: f64,
    pub p2_biggest_hit: f64,
    pub avg_round_duration_s: f64,
    pub longest_round_s: f64,
    pub shortest_round_s: f64,
    pub comeback_count: usize,
    pub close_rounds: usize,
    pub avg_winner_final_hp: f64,
    pub duration_s: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct Highlight {
    pub kind: String,         // "comeback", "close_round", "big_damage", "perfect"
    pub label: String,
    pub timestamp_ms: f64,
    pub end_ms: f64,
    pub details: String,
    pub severity: f64,        // 0-1, for sorting
}

#[derive(Debug, Serialize, Clone)]
pub struct Note {
    pub note_id: i64,
    pub replay_id: String,
    pub timestamp_ms: f64,
    pub text: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct AnalysisStatus {
    pub running: bool,
    pub progress_lines: Vec<String>,
    pub error: Option<String>,
    pub db_path: Option<String>,
}

// ── Constants for round detection ────────────────────────────────────────────

const HP_CEILING: f64 = 0.875;
const MIN_ROUND_FRAMES: usize = 450;  // ~15 seconds at 30fps
const COMEBACK_DEFICIT: f64 = 0.35;
const DEFICIT_SMOOTH_WINDOW: usize = 91;  // ~3s heavy smoothing for deficit tracking
const TIMER_SMOOTH_WINDOW: usize = 61;  // ~2s window for timer noise filtering
const HP_RESET_WINDOW: usize = 90;  // ~3s window for HP reset detection

// ── Signal processing helpers ────────────────────────────────────────────────

fn rolling_median(arr: &[f64], window: usize) -> Vec<f64> {
    let half = window / 2;
    let min_valid = (window / 3).max(3); // require at least 1/3 of window to be valid data
    let n = arr.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = if i + half + 1 <= n { i + half + 1 } else { n };
        let mut valid: Vec<f64> = arr[lo..hi].iter().copied().filter(|v| !v.is_nan()).collect();
        if valid.len() >= min_valid {
            valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[i] = valid[valid.len() / 2];
        }
    }
    out
}

fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Forward-fill None values (carry last known value forward).
fn forward_fill_i32(arr: &[Option<i32>]) -> Vec<Option<i32>> {
    let mut out = arr.to_vec();
    let mut last: Option<i32> = None;
    for v in out.iter_mut() {
        if v.is_some() {
            last = *v;
        } else {
            *v = last;
        }
    }
    out
}

/// Rolling median for optional i32 values.
fn rolling_median_i32(arr: &[Option<i32>], window: usize) -> Vec<Option<i32>> {
    let half = window / 2;
    let n = arr.len();
    let mut out = vec![None; n];
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = if i + half + 1 <= n { i + half + 1 } else { n };
        let mut vals: Vec<i32> = arr[lo..hi].iter().filter_map(|v| *v).collect();
        if !vals.is_empty() {
            vals.sort();
            out[i] = Some(vals[vals.len() / 2]);
        }
    }
    out
}

// ── Round detection using timer + HP resets + gaps ──────────────────────────
//
// GGS round detection strategy (three complementary signals):
// 1. TIMER: Timer counts down from ~99 each round. Smoothed with 61-frame
//    window to filter OCR noise. Jump from <=70 to >=88 = new round.
// 2. HP RESETS: Both players' rolling median HP (90-frame window) goes
//    from <0.50 to >0.85 = new round boundary.
// 3. GAME BREAKS: Data gaps >5s = character select / game boundary.
//
// A typical GGS round lasts ~30 seconds. Rounds >45s are recursively
// split at internal gaps or HP resets.

fn detect_rounds_for_replay(
    conn: &Connection,
    replay_id: &str,
) -> Result<Vec<RoundResult>, String> {
    // Read ALL frame data (including NULL HP for gap detection)
    let has_timer_col = conn
        .prepare("SELECT timer_value FROM frame_data LIMIT 1")
        .is_ok();
    let has_rounds_won_col = conn
        .prepare("SELECT p1_rounds_won FROM frame_data LIMIT 1")
        .is_ok();

    // (timestamp_ms, p1_hp, p2_hp, timer, p1_rounds_won, p2_rounds_won)
    let rows: Vec<(f64, Option<f64>, Option<f64>, Option<i32>, Option<i32>, Option<i32>)> = if has_timer_col && has_rounds_won_col {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct, timer_value, p1_rounds_won, p2_rounds_won
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let result: Vec<_> = stmt
            .query_map([replay_id], |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, Option<f64>>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                    row.get::<_, Option<i32>>(3)?,
                    row.get::<_, Option<i32>>(4)?,
                    row.get::<_, Option<i32>>(5)?,
                ))
            })
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    } else if has_timer_col {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct, timer_value
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let result: Vec<_> = stmt
            .query_map([replay_id], |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, Option<f64>>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                    row.get::<_, Option<i32>>(3)?,
                    None::<i32>,
                    None::<i32>,
                ))
            })
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let result: Vec<_> = stmt
            .query_map([replay_id], |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, Option<f64>>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                    None::<i32>,
                    None::<i32>,
                    None::<i32>,
                ))
            })
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    };

    let n = rows.len();
    if n < MIN_ROUND_FRAMES {
        return Ok(vec![]);
    }

    let ts: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let p1_raw: Vec<f64> = rows
        .iter()
        .map(|r| r.1.unwrap_or(f64::NAN))
        .collect();
    let p2_raw: Vec<f64> = rows
        .iter()
        .map(|r| r.2.unwrap_or(f64::NAN))
        .collect();
    let timer_raw: Vec<Option<i32>> = rows.iter().map(|r| r.3).collect();
    let p1_rounds_won: Vec<Option<i32>> = rows.iter().map(|r| r.4).collect();
    let p2_rounds_won: Vec<Option<i32>> = rows.iter().map(|r| r.5).collect();

    // Normalize HP by ceiling
    let mut p1_norm: Vec<f64> = p1_raw
        .iter()
        .map(|v| if v.is_nan() { f64::NAN } else { clip(v / HP_CEILING, 0.0, 1.0) })
        .collect();
    let mut p2_norm: Vec<f64> = p2_raw
        .iter()
        .map(|v| if v.is_nan() { f64::NAN } else { clip(v / HP_CEILING, 0.0, 1.0) })
        .collect();

    // ── Data cleaning: filter junk HP from non-gameplay screens ──
    // If both players' HP stays below 0.15 for > 5 seconds (150 frames),
    // it's a lobby/menu screen misidentified as gameplay. Null out those frames.
    // Threshold is intentionally low (0.15) because both players CAN legitimately
    // be at low health in close GGS rounds. Only catch clear non-gameplay.
    let mut junk_count = 0usize;
    {
        let junk_thresh = 0.15;
        let junk_window = 150usize; // ~5 seconds at 30fps
        let p1_junk = rolling_median(&p1_norm, junk_window);
        let p2_junk = rolling_median(&p2_norm, junk_window);
        for i in 0..n {
            if !p1_junk[i].is_nan() && !p2_junk[i].is_nan()
                && p1_junk[i] < junk_thresh && p2_junk[i] < junk_thresh
            {
                // Debug: log first few junk regions
                if junk_count < 5 || (junk_count % 200 == 0) {
                    eprintln!("[JUNK-DBG] {} NaN-ing i={} t={:.1}s p1_junk={:.3} p2_junk={:.3} p1_raw={:.3} p2_raw={:.3}",
                        replay_id, i, ts[i]/1000.0, p1_junk[i], p2_junk[i], p1_norm[i], p2_norm[i]);
                }
                junk_count += 1;
                p1_norm[i] = f64::NAN;
                p2_norm[i] = f64::NAN;
            }
        }
    }
    if junk_count > 0 {
        eprintln!("[JUNK-DBG] {} total junk frames NaN-ed: {}", replay_id, junk_count);
    }

    // Smooth HP for round metrics and winner detection (window=31 ≈ 1s)
    // Must be wide enough to filter OCR/scene-detection noise spikes
    let p1_smooth = rolling_median(&p1_norm, 31);
    let p2_smooth = rolling_median(&p2_norm, 31);

    // Heavy smoothing for deficit/comeback tracking (window=91 ≈ 3s)
    // HP extraction noise can create false deficits with lighter smoothing
    let p1_heavy = rolling_median(&p1_norm, DEFICIT_SMOOTH_WINDOW);
    let p2_heavy = rolling_median(&p2_norm, DEFICIT_SMOOTH_WINDOW);

    // ── Signal 1: Timer-based round starts ──
    // Forward-fill + wide-window smooth to filter OCR noise, then detect jumps.
    let timer_ff = forward_fill_i32(&timer_raw);
    let timer_smooth = rolling_median_i32(&timer_ff, TIMER_SMOOTH_WINDOW);

    let mut timer_starts: Vec<usize> = Vec::new();
    let mut lowest_since_start: i32 = 99;
    for i in 0..n {
        if let Some(t) = timer_smooth[i] {
            lowest_since_start = lowest_since_start.min(t);
            // Timer jumped from low (<=70) to high (>=88) = potential new round
            if t >= 88 && lowest_since_start <= 70 {
                if timer_starts.last().map_or(true, |&last| i - last > MIN_ROUND_FRAMES) {
                    // Validate: smoothed timer must reach ≥90 within 90 frames.
                    // Filters wallbreak/super flash OCR noise (peaks briefly at ~88).
                    // Real round starts show timer ≥90 consistently (OCR reads 90-96
                    // for the "99" display depending on overlay font).
                    let check_end = (i + 90).min(n);
                    let confirmed = (i..check_end)
                        .any(|j| timer_smooth[j].map_or(false, |tv| tv >= 90));
                    if confirmed {
                        eprintln!("[ROUND-DBG] timer_start at i={} t={:.1}s (smooth_timer={}, lowest={})",
                            i, ts[i]/1000.0, t, lowest_since_start);
                        timer_starts.push(i);
                        lowest_since_start = t;
                    } else {
                        eprintln!("[ROUND-DBG] timer_start REJECTED (no >=90 confirm) at i={} t={:.1}s (smooth_timer={}, lowest={})",
                            i, ts[i]/1000.0, t, lowest_since_start);
                    }
                    // else: false timer jump — don't push, don't reset lowest
                }
            }
        }
    }

    // ── Signal 2: HP reset detection ──
    // When both players' rolling median HP goes from low (<0.50) to high (>0.85),
    // a new round has started.
    let p1_rm = rolling_median(&p1_norm, HP_RESET_WINDOW);
    let p2_rm = rolling_median(&p2_norm, HP_RESET_WINDOW);
    let mut hp_resets: Vec<usize> = Vec::new();
    let mut min_hp_since_reset: f64 = 1.0;
    for i in 0..n {
        if !p1_rm[i].is_nan() && !p2_rm[i].is_nan() {
            let current_min = p1_rm[i].min(p2_rm[i]);
            min_hp_since_reset = min_hp_since_reset.min(current_min);
            if p1_rm[i] > 0.85 && p2_rm[i] > 0.85 && min_hp_since_reset < 0.50 {
                if hp_resets.last().map_or(true, |&last| i - last > MIN_ROUND_FRAMES) {
                    eprintln!("[ROUND-DBG] hp_reset at i={} t={:.1}s (p1_rm={:.3}, p2_rm={:.3}, min_since={:.3})",
                        i, ts[i]/1000.0, p1_rm[i], p2_rm[i], min_hp_since_reset);
                    hp_resets.push(i);
                    min_hp_since_reset = current_min;
                }
            }
        }
    }

    // ── Signal 2b: Gap-based HP reset detection ──
    // Detects round resets across NaN gaps that the rolling-median HP reset
    // misses.  When both players' health drops to low values, then a NaN gap
    // occurs (KO animation / round transition), and health returns to high
    // (both > 0.85), that's unambiguously a new round.
    let mut gap_hp_resets: Vec<usize> = Vec::new();
    {
        let mut last_both_valid_idx: Option<usize> = None;
        let mut last_p1_val: f64 = f64::NAN;
        let mut last_p2_val: f64 = f64::NAN;
        for i in 0..n {
            if !p1_norm[i].is_nan() && !p2_norm[i].is_nan() {
                if let Some(prev) = last_both_valid_idx {
                    let gap_frames = i - prev;
                    // Gap >= 30 frames (~1s), at least one player was low before,
                    // and both are high now → round reset
                    if gap_frames >= 30
                        && (last_p1_val < 0.50 || last_p2_val < 0.50)
                        && p1_norm[i] > 0.85 && p2_norm[i] > 0.85
                    {
                        if gap_hp_resets.last().map_or(true, |&last| i - last > MIN_ROUND_FRAMES) {
                            eprintln!("[ROUND-DBG] gap_hp_reset at i={} t={:.1}s (gap={}f, pre_p1={:.3}, pre_p2={:.3})",
                                i, ts[i]/1000.0, gap_frames, last_p1_val, last_p2_val);
                            gap_hp_resets.push(i);
                        }
                    }
                }
                last_both_valid_idx = Some(i);
                last_p1_val = p1_norm[i];
                last_p2_val = p2_norm[i];
            }
        }
    }

    // ── Signal 3: Game break detection (gaps > 5s) ──
    // Track gaps in health data for EITHER player.  When one player is at
    // very low health (bar too small to read), the other player's bar is
    // still valid and prevents false game breaks mid-round.
    let mut game_breaks: Vec<usize> = Vec::new();
    let mut prev_valid_idx: Option<usize> = None;
    for i in 0..n {
        if !p1_norm[i].is_nan() || !p2_norm[i].is_nan() {
            if let Some(pv) = prev_valid_idx {
                let gap_ms = ts[i] - ts[pv];
                if gap_ms > 5000.0 {
                    eprintln!("[ROUND-DBG] game_break at i={} t={:.1}s (gap={:.1}s)",
                        i, ts[i]/1000.0, gap_ms/1000.0);
                    game_breaks.push(i);
                }
            }
            prev_valid_idx = Some(i);
        }
    }

    // ── Signal 4: Match boundary (timer=99 + both players 0 rounds won) ──
    // When the round counter hearts reset to 2-2, a new match has started.
    // Use forward-filled rounds_won and smoothed timer to detect this reliably.
    let p1_rw_ff = forward_fill_i32(&p1_rounds_won);
    let p2_rw_ff = forward_fill_i32(&p2_rounds_won);
    let mut match_boundaries: Vec<usize> = Vec::new();
    {
        let mut prev_had_wins = false; // true if either player had round wins before
        for i in 0..n {
            let t = timer_smooth[i].unwrap_or(0);
            let p1w = p1_rw_ff[i].unwrap_or(-1);
            let p2w = p2_rw_ff[i].unwrap_or(-1);

            if p1w > 0 || p2w > 0 {
                prev_had_wins = true;
            }

            // Timer at 99, both players at 0 wins, and we previously saw wins
            // = definitive new match start
            if t >= 97 && p1w == 0 && p2w == 0 && prev_had_wins {
                if match_boundaries.last().map_or(true, |&last| i - last > MIN_ROUND_FRAMES) {
                    match_boundaries.push(i);
                    prev_had_wins = false;
                }
            }
        }
    }

    // ── Merge all boundary signals ──
    // Game breaks (large data gaps) are exempt from the wallbreak timer filter
    // because a multi-second gap is unambiguously a round/game boundary regardless
    // of timer state (post-KO transition screens don't show readable timers).
    let game_break_set: std::collections::HashSet<usize> = game_breaks.iter()
        .chain(gap_hp_resets.iter())
        .copied().collect();

    let mut all_indices: Vec<usize> = Vec::new();
    all_indices.extend_from_slice(&timer_starts);
    all_indices.extend_from_slice(&hp_resets);
    all_indices.extend_from_slice(&gap_hp_resets);
    all_indices.extend_from_slice(&game_breaks);
    all_indices.extend_from_slice(&match_boundaries);
    all_indices.sort();
    all_indices.dedup();

    eprintln!("[ROUND-DBG] {} — pre-filter: {} timer_starts, {} hp_resets, {} gap_hp_resets, {} game_breaks, {} match_bounds",
        replay_id, timer_starts.len(), hp_resets.len(), gap_hp_resets.len(), game_breaks.len(), match_boundaries.len());
    eprintln!("[ROUND-DBG] all_indices before wallbreak filter: {:?}",
        all_indices.iter().map(|&i| format!("{}({:.1}s)", i, ts[i]/1000.0)).collect::<Vec<_>>());

    // ── Wallbreak filter: real round starts always have timer ≈90+ ──
    // Apply BEFORE dedup so that false boundaries (wallbreak/super flash) don't
    // shadow valid nearby boundaries during the 10-second dedup window.
    // Game breaks are exempt — large data gaps are unambiguous boundaries.
    all_indices.retain(|&b| {
        if game_break_set.contains(&b) {
            return true; // game breaks always pass
        }
        let check_end = (b + 90).min(n);
        let pass = (b..check_end).any(|j| timer_smooth[j].map_or(false, |t| t >= 90));
        if !pass {
            eprintln!("[ROUND-DBG] wallbreak filter REMOVED i={} t={:.1}s (max smooth timer in 90 frames: {:?})",
                b, ts[b]/1000.0,
                (b..check_end).filter_map(|j| timer_smooth[j]).max());
        }
        pass
    });

    // Deduplicate within 10 seconds
    let mut deduped: Vec<usize> = Vec::new();
    for &idx in &all_indices {
        if deduped.last().map_or(true, |&last| ts[idx] - ts[last] > 10000.0) {
            deduped.push(idx);
        }
    }

    // Add first valid data frame as start if needed
    // (first-valid doesn't need timer validation — could be mid-round)
    let first_valid = (0..n).find(|&i| !p1_norm[i].is_nan() || !p2_norm[i].is_nan());
    if let Some(fv) = first_valid {
        if deduped.is_empty() || (deduped[0] > fv && deduped[0] - fv > MIN_ROUND_FRAMES) {
            deduped.insert(0, fv);
        }
    }

    eprintln!("[ROUND-DBG] {} — final boundaries: {:?}",
        replay_id, deduped.iter().map(|&i| format!("{}({:.1}s)", i, ts[i]/1000.0)).collect::<Vec<_>>());

    // Track which boundary indices are match starts
    let match_boundary_set: std::collections::HashSet<usize> = match_boundaries.iter().copied().collect();
    // Map deduped boundaries to whether they correspond to a match boundary
    // (a deduped boundary is a match start if any match_boundary is within 10s)
    let is_match_start_boundary: Vec<bool> = deduped.iter().map(|&idx| {
        match_boundary_set.iter().any(|&mb| {
            let diff = if idx > mb { idx - mb } else { mb - idx };
            (diff as f64) < 300.0 // within ~10 seconds in frame indices
        })
    }).collect();

    // ── Build rounds from boundaries ──
    let mut round_starts: Vec<(usize, f64, bool)> = Vec::new();
    for (i, &idx) in deduped.iter().enumerate() {
        round_starts.push((idx, ts[idx], is_match_start_boundary[i]));
    }

    // Helper: find KO moment and determine winner for a span.
    // Uses smoothed HP. Handles two GGS edge cases:
    //   1. Last round of match: bars disappear (HP goes NaN before KO visible)
    //   2. Wallbreak: HP shows combo damage but doesn't visually reach zero
    // Strategy: KO moment detection → fallback to average HP in last 25% if ambiguous
    let find_winner_ko = |s: usize, e: usize| -> Option<(String, f64, f64)> {
        let span = e - s;
        let search_start = s + (span as f64 * 0.4) as usize;

        // Early check: count raw frames with HP < 0.15 in last 60%.
        // Filter post-KO garbage: only count if the OTHER player is NOT at full HP.
        // After a KO, the winning player's bar resets to ~1.0 while the loser's
        // shows residual low values — these aren't real combat frames.
        let p1_ko_count = (search_start..e)
            .filter(|&j| !p1_norm[j].is_nan() && p1_norm[j] < 0.15
                      && !p2_norm[j].is_nan() && p2_norm[j] < 0.90)
            .count();
        let p2_ko_count = (search_start..e)
            .filter(|&j| !p2_norm[j].is_nan() && p2_norm[j] < 0.15
                      && !p1_norm[j].is_nan() && p1_norm[j] < 0.90)
            .count();
        if p1_ko_count >= 3 && p1_ko_count >= p2_ko_count * 3 + 1 {
            return Some(("P2".to_string(), 0.0, 1.0));
        }
        if p2_ko_count >= 3 && p2_ko_count >= p1_ko_count * 3 + 1 {
            return Some(("P1".to_string(), 1.0, 0.0));
        }
        // Both players have KO frames but neither dominates — could be KO
        // animation artifact OR wallbreak/comeback. When the two KO moments
        // are close together (≤90 frames / 3s), it's a KO animation and the
        // first player to reach zero is the real loser. When far apart (>90),
        // it's a wallbreak/comeback — let the KO clustering handle it.
        if p1_ko_count >= 3 && p2_ko_count >= 3 {
            let first_p1 = (search_start..e).find(|&j| !p1_norm[j].is_nan() && p1_norm[j] < 0.05);
            let first_p2 = (search_start..e).find(|&j| !p2_norm[j].is_nan() && p2_norm[j] < 0.05);
            if let (Some(fp1), Some(fp2)) = (first_p1, first_p2) {
                let gap = if fp1 > fp2 { fp1 - fp2 } else { fp2 - fp1 };
                if gap <= 90 {
                    // Only use first-to-zero when it's a one-sided KO
                    // (other player > 0.15 HP). If both are near KO at the
                    // earlier moment, it's ambiguous — check the later zero.
                    let (earlier, later, earlier_is_p1) = if fp1 < fp2 {
                        (fp1, fp2, true)
                    } else {
                        (fp2, fp1, false)
                    };
                    let other_at_earlier = if earlier_is_p1 { p2_norm[earlier] } else { p1_norm[earlier] };
                    let other_at_later = if earlier_is_p1 { p1_norm[later] } else { p2_norm[later] };
                    if other_at_earlier > 0.15 {
                        // Earlier zero is one-sided — earlier player died
                        if earlier_is_p1 {
                            return Some(("P2".to_string(), 0.0, 1.0));
                        } else {
                            return Some(("P1".to_string(), 1.0, 0.0));
                        }
                    } else if other_at_later > 0.15 {
                        // Later zero is one-sided — later player died
                        if earlier_is_p1 {
                            return Some(("P1".to_string(), 1.0, 0.0));
                        } else {
                            return Some(("P2".to_string(), 0.0, 1.0));
                        }
                    }
                }
            }
        }

        // Search last 60% of the round for the frame where min(p1, p2) is lowest
        let mut best_min_hp: f64 = 2.0;
        let mut best_p1: f64 = 0.5;
        let mut best_p2: f64 = 0.5;
        let mut valid_frames: Vec<(usize, f64, f64)> = Vec::new();  // (frame_idx, p1, p2)

        for j in search_start..e {
            if !p1_smooth[j].is_nan() && !p2_smooth[j].is_nan() {
                valid_frames.push((j, p1_smooth[j], p2_smooth[j]));
                let min_hp = p1_smooth[j].min(p2_smooth[j]);
                if min_hp < best_min_hp {
                    best_min_hp = min_hp;
                    best_p1 = p1_smooth[j];
                    best_p2 = p2_smooth[j];
                }
            }
        }

        if valid_frames.is_empty() {
            return None;
        }

        // Collect frames near the KO threshold, then split into contiguous
        // temporal clusters. Use the LAST cluster since KO happens at round end.
        // This avoids mixing disconnected mid-round dips with the actual KO.
        let ko_frames: Vec<(usize, f64, f64)> = valid_frames
            .iter()
            .copied()
            .filter(|&(_, p1, p2)| p1.min(p2) < best_min_hp + 0.10)
            .collect();

        let (ep1, ep2) = if !ko_frames.is_empty() {
            // Split into clusters separated by > 60 frames (~2s)
            let mut clusters: Vec<Vec<(usize, f64, f64)>> = vec![vec![ko_frames[0]]];
            for &item in &ko_frames[1..] {
                let last_cluster = clusters.last().unwrap();
                if item.0 - last_cluster.last().unwrap().0 > 60 {
                    clusters.push(vec![item]);
                } else {
                    clusters.last_mut().unwrap().push(item);
                }
            }

            // Use the last cluster (closest to round end = actual KO)
            let last_cluster = clusters.last().unwrap();
            let take_count = last_cluster.len().min(30);
            let ko_use = &last_cluster[last_cluster.len() - take_count..];
            let mut p1_vals: Vec<f64> = ko_use.iter().map(|f| f.1).collect();
            let mut p2_vals: Vec<f64> = ko_use.iter().map(|f| f.2).collect();
            p1_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p2_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (p1_vals[p1_vals.len() / 2], p2_vals[p2_vals.len() / 2])
        } else {
            (best_p1, best_p2)
        };

        let hp_diff = (ep1 - ep2).abs();

        // If KO moment is clear (large HP gap), use it directly
        if hp_diff >= 0.15 {
            let winner = if ep1 >= ep2 { "P1".to_string() } else { "P2".to_string() };
            return Some((winner, ep1, ep2));
        }

        // Ambiguous KO moment — fallback strategies:

        // Strategy A: Find last valid HP readings before data gaps out
        // (handles last-round-of-match where bars disappear)
        let last_quarter_start = s + (span as f64 * 0.75) as usize;
        let mut last_valid_p1: Option<f64> = None;
        let mut last_valid_p2: Option<f64> = None;
        for j in (last_quarter_start..e).rev() {
            if last_valid_p1.is_none() && !p1_smooth[j].is_nan() {
                last_valid_p1 = Some(p1_smooth[j]);
            }
            if last_valid_p2.is_none() && !p2_smooth[j].is_nan() {
                last_valid_p2 = Some(p2_smooth[j]);
            }
            if last_valid_p1.is_some() && last_valid_p2.is_some() {
                break;
            }
        }

        // Find minimum HP each player reaches in the last 40% (smoothed)
        let mut p1_min_last = 2.0f64;
        let mut p2_min_last = 2.0f64;
        for j in search_start..e {
            if !p1_smooth[j].is_nan() { p1_min_last = p1_min_last.min(p1_smooth[j]); }
            if !p2_smooth[j].is_nan() { p2_min_last = p2_min_last.min(p2_smooth[j]); }
        }

        // Combine signals: whoever reached lower minimum HP is the loser
        let mut min_signal = p2_min_last - p1_min_last;  // positive = P1 went lower = P2 wins

        // Detect transition artifact: both min values in [0.40, 0.55] with
        // tiny difference = "loading screen" where both bars show ~50%.
        if p1_min_last >= 0.40 && p1_min_last <= 0.55
            && p2_min_last >= 0.40 && p2_min_last <= 0.55
            && min_signal.abs() < 0.05
        {
            min_signal = 0.0;
        }

        // Strategy B: Average HP in last 25% of the round.
        // Exclude transition artifacts where both bars drop simultaneously
        // below 0.60 (loading screen pattern).
        let mut p1_avg_vals: Vec<f64> = Vec::new();
        let mut p2_avg_vals: Vec<f64> = Vec::new();
        for j in last_quarter_start..e {
            if !p1_smooth[j].is_nan() && !p2_smooth[j].is_nan() {
                if p1_smooth[j] < 0.60 && p2_smooth[j] < 0.60 {
                    continue;
                }
                p1_avg_vals.push(p1_smooth[j]);
                p2_avg_vals.push(p2_smooth[j]);
            }
        }

        let avg_signal = if p1_avg_vals.len() >= 10 && p2_avg_vals.len() >= 10 {
            let p1_avg: f64 = p1_avg_vals.iter().sum::<f64>() / p1_avg_vals.len() as f64;
            let p2_avg: f64 = p2_avg_vals.iter().sum::<f64>() / p2_avg_vals.len() as f64;
            p2_avg - p1_avg  // positive = P1 lower avg = P2 wins
        } else if let (Some(lp1), Some(lp2)) = (last_valid_p1, last_valid_p2) {
            lp2 - lp1  // positive = P1 lower = P2 wins
        } else {
            ep2 - ep1  // fall back to KO moment
        };

        // Dynamic weighting: when both players' min HP is similar, the
        // difference is noise (transition artifacts, OCR jitter). Use
        // avg_signal which captures the post-round result screen HP.
        let hp_range = (p1_min_last - p2_min_last).abs();
        let min_weight = (hp_range / 0.20).min(0.8);
        let avg_weight = 1.0 - min_weight;
        let combined = min_signal * min_weight + avg_signal * avg_weight;
        let (winner, _fp1, _fp2) = if combined > 0.0 {
            ("P2".to_string(), p1_min_last.min(2.0), p2_min_last.min(2.0))
        } else {
            ("P1".to_string(), p1_min_last.min(2.0), p2_min_last.min(2.0))
        };

        // Use the better HP values for reporting
        let report_p1 = if p1_min_last < 2.0 { p1_min_last } else { ep1 };
        let report_p2 = if p2_min_last < 2.0 { p2_min_last } else { ep2 };
        Some((winner, report_p1, report_p2))
    };

    // Helper: recursively split long rounds at gaps or HP resets.
    // All split candidates must be confirmed by timer or HP reset evidence.
    fn split_long_round(
        s_idx: usize,
        e_idx: usize,
        ts: &[f64],
        p1_norm: &[f64],
        p2_norm: &[f64],
        p1_smooth: &[f64],
        p2_smooth: &[f64],
        timer_smooth: &[Option<i32>],
        depth: usize,
    ) -> Vec<(usize, usize)> {
        let n = timer_smooth.len();
        let dur = (ts[e_idx.min(ts.len() - 1)] - ts[s_idx]) / 1000.0;
        if dur <= 45.0 || depth > 3 {
            return vec![(s_idx, e_idx)];
        }

        // Timer confirmation: smoothed timer must reach ≥90 within 210 frames (7s).
        let timer_ok = |idx: usize| -> bool {
            let check_end = (idx + 210).min(n);
            (idx..check_end).any(|j| timer_smooth[j].map_or(false, |t| t >= 90))
        };

        // Alternative confirmation for wallbreak transitions: KO evidence before
        // the gap + timer ≥90 nearby + both HP reset to >0.90 after.
        let round_reset_ok = |idx: usize| -> bool {
            // Check for KO evidence before the gap (within preceding 450 frames / 15s)
            let ko_start = if idx >= 450 { idx - 450 } else { 0 };
            let p1_ko = (ko_start..idx)
                .filter(|&j| !p1_norm[j].is_nan() && p1_norm[j] < 0.10)
                .count();
            let p2_ko = (ko_start..idx)
                .filter(|&j| !p2_norm[j].is_nan() && p2_norm[j] < 0.10)
                .count();
            if p1_ko < 3 && p2_ko < 3 {
                return false;
            }
            // Timer must be ≥90 nearby
            let timer_near = (idx + 120).min(n);
            let has_timer = (idx..timer_near)
                .any(|j| timer_smooth[j].map_or(false, |t| t >= 90));
            if !has_timer {
                return false;
            }
            // Both smoothed HP must reset to >0.90 after the gap
            let hp_check = (idx + 300).min(n);
            (idx..hp_check).any(|j| {
                !p1_smooth[j].is_nan() && !p2_smooth[j].is_nan()
                    && p1_smooth[j] > 0.90 && p2_smooth[j] > 0.90
            })
        };

        let split_confirmed = |idx: usize| -> bool {
            timer_ok(idx) || round_reset_ok(idx)
        };

        let mut best_split: Option<usize> = None;
        let mut best_score: f64 = 0.0;

        // Strategy 1: largest data gap > 1.5s (timer or round reset must confirm)
        let mut last_v: Option<usize> = None;
        for i in s_idx..e_idx {
            if !p1_norm[i].is_nan() {
                if let Some(lv) = last_v {
                    let g = (ts[i] - ts[lv]) / 1000.0;
                    if g > 1.5 && g > best_score {
                        let mid_t = ts[i] / 1000.0;
                        let start_t = ts[s_idx] / 1000.0;
                        let end_t = ts[e_idx.min(ts.len() - 1)] / 1000.0;
                        let dur1 = mid_t - start_t;
                        let dur2 = end_t - mid_t;
                        if dur1 >= 12.0 && dur2 >= 12.0 && split_confirmed(i) {
                            best_split = Some(i);
                            best_score = g;
                        }
                    }
                }
                last_v = Some(i);
            }
        }

        // Strategy 2: HP reset within the long round (timer must confirm)
        let local_len = e_idx - s_idx;
        if local_len > 60 {
            let half = 30usize;
            let mut min_hp_local: f64 = 1.0;
            for j in 0..local_len {
                let gi = s_idx + j;
                let lo = if j >= half { gi - half } else { s_idx };
                let hi = (gi + half + 1).min(e_idx);
                let mut p1_chunk: Vec<f64> = (lo..hi)
                    .filter(|&k| !p1_norm[k].is_nan())
                    .map(|k| p1_norm[k])
                    .collect();
                let mut p2_chunk: Vec<f64> = (lo..hi)
                    .filter(|&k| !p2_norm[k].is_nan())
                    .map(|k| p2_norm[k])
                    .collect();
                if p1_chunk.len() >= 10 && p2_chunk.len() >= 10 {
                    p1_chunk.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    p2_chunk.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p1_med = p1_chunk[p1_chunk.len() / 2];
                    let p2_med = p2_chunk[p2_chunk.len() / 2];
                    let current_min = p1_med.min(p2_med);
                    min_hp_local = min_hp_local.min(current_min);

                    if p1_med > 0.75 && p2_med > 0.75 && min_hp_local < 0.55 {
                        let mid_t = ts[gi] / 1000.0;
                        let start_t = ts[s_idx] / 1000.0;
                        let end_t = ts[e_idx.min(ts.len() - 1)] / 1000.0;
                        let dur1 = mid_t - start_t;
                        let dur2 = end_t - mid_t;
                        if dur1 >= 12.0 && dur2 >= 12.0 && timer_ok(gi) {
                            let balance = 1.0 - (dur1 - dur2).abs() / dur;
                            let hp_drop = 1.0 - min_hp_local;
                            let score = balance * 0.3 + hp_drop * 0.7;
                            if score > best_score {
                                best_split = Some(gi);
                                best_score = score;
                            }
                        }
                        min_hp_local = current_min;
                    }
                }
            }
        }

        if let Some(split_idx) = best_split {
            let mut result = split_long_round(s_idx, split_idx, ts, p1_norm, p2_norm, p1_smooth, p2_smooth, timer_smooth, depth + 1);
            result.extend(split_long_round(split_idx, e_idx, ts, p1_norm, p2_norm, p1_smooth, p2_smooth, timer_smooth, depth + 1));
            return result;
        }

        vec![(s_idx, e_idx)]
    }

    // ── Build initial round spans ──
    struct RoundSpan {
        s_idx: usize,
        e_idx: usize,
        is_match_start: bool,
    }

    let mut initial_spans: Vec<RoundSpan> = Vec::new();
    for ri in 0..round_starts.len() {
        let (s_idx, _, is_ms) = round_starts[ri];
        let e_idx = if ri + 1 < round_starts.len() {
            round_starts[ri + 1].0
        } else {
            n - 1
        };

        let total_frames = e_idx - s_idx;
        if total_frames < MIN_ROUND_FRAMES {
            eprintln!("[ROUND-DBG] span {:.1}s-{:.1}s DROPPED: too short ({} < {})",
                ts[s_idx]/1000.0, ts[e_idx.min(n-1)]/1000.0, total_frames, MIN_ROUND_FRAMES);
            continue;
        }

        // Valid data ratio check — accept frames where at least one player
        // has health data.  When one player is at very low health (<50 pixels),
        // their bar is too small to read but the other player's bar is still valid.
        let valid_count = (s_idx..e_idx)
            .filter(|&j| !p1_norm[j].is_nan() || !p2_norm[j].is_nan())
            .count();
        // Diagnostic: count NaN breakdown for dropped spans
        if total_frames > 0 && (valid_count as f64) / (total_frames as f64) < 0.30 {
            let p1_nan = (s_idx..e_idx).filter(|&j| p1_norm[j].is_nan()).count();
            let p2_nan = (s_idx..e_idx).filter(|&j| p2_norm[j].is_nan()).count();
            let both_nan = (s_idx..e_idx).filter(|&j| p1_norm[j].is_nan() && p2_norm[j].is_nan()).count();
            let p1_raw_nan = (s_idx..e_idx).filter(|&j| p1_raw[j].is_nan()).count();
            let p2_raw_nan = (s_idx..e_idx).filter(|&j| p2_raw[j].is_nan()).count();
            // Sample values at 25%, 50%, 75% through span
            let samples: Vec<String> = [0.25, 0.50, 0.75].iter().map(|&frac| {
                let si = s_idx + (total_frames as f64 * frac) as usize;
                format!("{:.1}s:p1={:.3}/p2={:.3}", ts[si]/1000.0,
                    p1_norm[si], p2_norm[si])
            }).collect();
            eprintln!("[ROUND-DBG] span {:.1}s-{:.1}s DROPPED: low valid ratio ({}/{} = {:.2}) \
                p1_nan={} p2_nan={} both_nan={} p1_raw_nan={} p2_raw_nan={} samples=[{}]",
                ts[s_idx]/1000.0, ts[e_idx.min(n-1)]/1000.0, valid_count, total_frames,
                valid_count as f64 / total_frames as f64,
                p1_nan, p2_nan, both_nan, p1_raw_nan, p2_raw_nan,
                samples.join(", "));
            continue;
        }

        eprintln!("[ROUND-DBG] span KEPT: {:.1}s-{:.1}s ({} frames, {}/{} valid, match_start={})",
            ts[s_idx]/1000.0, ts[e_idx.min(n-1)]/1000.0, total_frames, valid_count, total_frames, is_ms);
        initial_spans.push(RoundSpan { s_idx, e_idx, is_match_start: is_ms });
    }

    // ── Split long rounds and build results ──
    let mut results: Vec<RoundResult> = Vec::new();
    let mut round_counter: usize = 0;

    for span in &initial_spans {
        let dur_s = (ts[span.e_idx.min(n - 1)] - ts[span.s_idx]) / 1000.0;
        let sub_spans = if dur_s > 45.0 {
            split_long_round(span.s_idx, span.e_idx, &ts, &p1_norm, &p2_norm, &p1_smooth, &p2_smooth, &timer_smooth, 0)
        } else {
            vec![(span.s_idx, span.e_idx)]
        };

        for (sub_i, (s_idx, e_idx)) in sub_spans.iter().enumerate() {
            let (s_idx, e_idx) = (*s_idx, *e_idx);
            // Only the first sub-span inherits the match_start flag
            let is_match_start = sub_i == 0 && span.is_match_start;

            // Trim span to first/last frame with valid HP data.
            // This excludes dead time (character select, menus, transitions)
            // that appears as NULL HP gaps at the start/end of a round boundary.
            let trimmed_start = (s_idx..e_idx)
                .find(|&j| !p1_norm[j].is_nan() && !p2_norm[j].is_nan());
            let trimmed_end = (s_idx..e_idx).rev()
                .find(|&j| !p1_norm[j].is_nan() && !p2_norm[j].is_nan());
            let (s_idx, e_idx) = match (trimmed_start, trimmed_end) {
                (Some(s), Some(e)) => (s, e + 1),
                _ => continue, // no valid data in this span
            };

            let s_ms = ts[s_idx];
            let e_ms = ts[e_idx.min(n - 1)];
            let sub_dur = (e_ms - s_ms) / 1000.0;

            if sub_dur < 12.0 {
                continue;
            }

            // Valid data check for sub-span
            let valid_count = (s_idx..e_idx)
                .filter(|&j| !p1_norm[j].is_nan() && !p2_norm[j].is_nan())
                .count();
            if valid_count == 0 {
                continue;
            }

            // Determine winner using KO-moment approach
            let (winner, ep1, ep2) = match find_winner_ko(s_idx, e_idx) {
                Some(v) => v,
                None => continue,
            };

            // ── Round metrics ──
            let (w_smooth, l_smooth) = if winner == "P2" {
                (&p2_smooth, &p1_smooth)
            } else {
                (&p1_smooth, &p2_smooth)
            };

            let mut w_valid: Vec<f64> = Vec::new();
            let mut l_valid: Vec<f64> = Vec::new();
            let mut ts_valid: Vec<f64> = Vec::new();

            for i in s_idx..e_idx {
                if !w_smooth[i].is_nan() && !l_smooth[i].is_nan() {
                    w_valid.push(w_smooth[i]);
                    l_valid.push(l_smooth[i]);
                    ts_valid.push(ts[i]);
                }
            }

            if w_valid.is_empty() {
                continue;
            }

            let winner_final_hp = if winner == "P2" { ep2 } else { ep1 };
            let loser_final_hp = if winner == "P2" { ep1 } else { ep2 };
            let winner_min_hp = w_valid.iter().cloned().fold(f64::INFINITY, f64::min);

            // Use heavy smoothing for deficit tracking to filter HP noise
            let (w_heavy, l_heavy) = if winner == "P2" {
                (&p2_heavy, &p1_heavy)
            } else {
                (&p1_heavy, &p2_heavy)
            };

            // Skip first 15% of round to avoid HP reset transition artifacts
            let deficit_start = s_idx + (e_idx - s_idx) * 15 / 100;
            let mut max_deficit: f64 = 0.0;
            let mut deficit_ts: f64 = ts[s_idx];
            for i in deficit_start..e_idx {
                if !w_heavy[i].is_nan() && !l_heavy[i].is_nan() {
                    let deficit = l_heavy[i] - w_heavy[i];
                    if deficit > max_deficit {
                        max_deficit = deficit;
                        deficit_ts = ts[i];
                    }
                }
            }

            let is_comeback = max_deficit >= COMEBACK_DEFICIT;

            results.push(RoundResult {
                replay_id: replay_id.to_string(),
                round_index: round_counter,
                round_start_ms: s_ms,
                round_end_ms: e_ms,
                winner,
                winner_final_hp,
                loser_final_hp,
                winner_min_hp,
                max_deficit,
                deficit_timestamp_ms: deficit_ts,
                is_comeback,
                is_match_start,
            });
            round_counter += 1;
        }
    }

    // ── Hearts-based winner override ──────────────────────────────────────
    // GGS pending combo damage appears as a lighter bar segment that our warm
    // pixel detection still counts as health. This causes the health-based
    // winner detection to misidentify the winner when a lethal combo kills a
    // player who appeared to have more health. The round counter hearts are
    // ground truth — use them to override when they disagree.
    let p1_rw_smooth = rolling_median_i32(&p1_rw_ff, 121);
    let p2_rw_smooth = rolling_median_i32(&p2_rw_ff, 121);

    // Helper: find the MODE (most common value) of hearts in a frame window.
    let hearts_mode = |arr: &[Option<i32>], start: usize, end: usize| -> Option<i32> {
        let clamped_end = end.min(arr.len());
        if start >= clamped_end { return None; }
        let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for j in start..clamped_end {
            if let Some(v) = arr[j] {
                *counts.entry(v).or_insert(0) += 1;
            }
        }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(v, _)| v)
    };

    // For each round, check hearts after it ends to determine the true winner.
    // Compare with hearts at the start of this round to see who gained a point.
    for ri in 0..results.len() {
        // Find frame indices for this round's start and the next round's start
        let round_end_idx = ts.iter().position(|&t| t >= results[ri].round_end_ms).unwrap_or(0);
        let next_round_start_idx = if ri + 1 < results.len() {
            ts.iter().position(|&t| t >= results[ri + 1].round_start_ms).unwrap_or(n)
        } else {
            n
        };

        // Hearts at round start (look back 300 frames / 10s before this round)
        let round_start_idx = ts.iter().position(|&t| t >= results[ri].round_start_ms).unwrap_or(0);

        // Hearts BEFORE this round (from previous round end to this round start)
        let pre_p1 = hearts_mode(&p1_rw_smooth, if round_start_idx >= 150 { round_start_idx - 150 } else { 0 }, round_start_idx);
        let pre_p2 = hearts_mode(&p2_rw_smooth, if round_start_idx >= 150 { round_start_idx - 150 } else { 0 }, round_start_idx);

        // Hearts AFTER this round ends (between round end and next round start)
        // Use a generous window but don't go past next round start
        let post_window_end = if next_round_start_idx > round_end_idx {
            next_round_start_idx
        } else {
            (round_end_idx + 600).min(n)
        };
        let post_p1 = hearts_mode(&p1_rw_smooth, round_end_idx, post_window_end);
        let post_p2 = hearts_mode(&p2_rw_smooth, round_end_idx, post_window_end);

        // If we have hearts before and after, check who gained a point
        if let (Some(pre1), Some(pre2), Some(post1), Some(post2)) = (pre_p1, pre_p2, post_p1, post_p2) {
            let p1_gained = post1 > pre1;
            let p2_gained = post2 > pre2;

            // Match start: hearts reset to 0/0, so compare with 0
            if results[ri].is_match_start {
                let p1_gained = post1 > 0;
                let p2_gained = post2 > 0;
                if p1_gained && !p2_gained && results[ri].winner != "P1" {
                    results[ri].winner = "P1".to_string();
                    let tmp = results[ri].winner_final_hp;
                    results[ri].winner_final_hp = results[ri].loser_final_hp;
                    results[ri].loser_final_hp = tmp;
                } else if p2_gained && !p1_gained && results[ri].winner != "P2" {
                    results[ri].winner = "P2".to_string();
                    let tmp = results[ri].winner_final_hp;
                    results[ri].winner_final_hp = results[ri].loser_final_hp;
                    results[ri].loser_final_hp = tmp;
                }
            } else if p1_gained && !p2_gained && results[ri].winner != "P1" {
                results[ri].winner = "P1".to_string();
                let tmp = results[ri].winner_final_hp;
                results[ri].winner_final_hp = results[ri].loser_final_hp;
                results[ri].loser_final_hp = tmp;
            } else if p2_gained && !p1_gained && results[ri].winner != "P2" {
                results[ri].winner = "P2".to_string();
                let tmp = results[ri].winner_final_hp;
                results[ri].winner_final_hp = results[ri].loser_final_hp;
                results[ri].loser_final_hp = tmp;
            }
        }
    }

    Ok(results)
}

fn compute_match_stats(
    replay: &Replay,
    rounds: &[RoundResult],
    damage_events: &[DamageEvent],
) -> MatchStats {
    let p1_wins = rounds.iter().filter(|r| r.winner == "P1").count();
    let p2_wins = rounds.iter().filter(|r| r.winner == "P2").count();

    let mut p1_dmg_taken = 0.0;
    let mut p2_dmg_taken = 0.0;
    let mut p1_biggest = 0.0f64;
    let mut p2_biggest = 0.0f64;

    for e in damage_events {
        if e.target_side == 1 {
            p1_dmg_taken += e.damage_pct;
            p1_biggest = p1_biggest.max(e.damage_pct);
        } else {
            p2_dmg_taken += e.damage_pct;
            p2_biggest = p2_biggest.max(e.damage_pct);
        }
    }

    let durations: Vec<f64> = rounds
        .iter()
        .map(|r| (r.round_end_ms - r.round_start_ms) / 1000.0)
        .collect();
    let avg_dur = if durations.is_empty() {
        0.0
    } else {
        durations.iter().sum::<f64>() / durations.len() as f64
    };
    let longest = durations.iter().cloned().fold(0.0f64, f64::max);
    let shortest = durations
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let shortest = if shortest.is_infinite() {
        0.0
    } else {
        shortest
    };

    let comeback_count = rounds.iter().filter(|r| r.is_comeback).count();
    let close_rounds = rounds.iter().filter(|r| r.winner_final_hp < 0.20).count();
    let avg_winner_hp = if rounds.is_empty() {
        0.0
    } else {
        rounds.iter().map(|r| r.winner_final_hp).sum::<f64>() / rounds.len() as f64
    };

    MatchStats {
        replay_id: replay.replay_id.clone(),
        total_rounds: rounds.len(),
        p1_round_wins: p1_wins,
        p2_round_wins: p2_wins,
        total_damage_events: damage_events.len(),
        p1_damage_taken: p1_dmg_taken,
        p2_damage_taken: p2_dmg_taken,
        p1_biggest_hit: p1_biggest,
        p2_biggest_hit: p2_biggest,
        avg_round_duration_s: avg_dur,
        longest_round_s: longest,
        shortest_round_s: shortest,
        comeback_count,
        close_rounds,
        avg_winner_final_hp: avg_winner_hp,
        duration_s: replay.duration_ms / 1000.0,
    }
}

fn generate_highlights(
    rounds: &[RoundResult],
    damage_events: &[DamageEvent],
) -> Vec<Highlight> {
    let mut highlights: Vec<Highlight> = Vec::new();

    // Comeback highlights
    for r in rounds {
        if r.is_comeback {
            highlights.push(Highlight {
                kind: "comeback".to_string(),
                label: format!(
                    "{} comeback ({:.0}% deficit)",
                    r.winner,
                    r.max_deficit * 100.0
                ),
                timestamp_ms: r.round_start_ms,
                end_ms: r.round_end_ms,
                details: format!(
                    "{} was down {:.0}% HP but won with {:.0}% remaining. Min HP: {:.0}%",
                    r.winner,
                    r.max_deficit * 100.0,
                    r.winner_final_hp * 100.0,
                    r.winner_min_hp * 100.0
                ),
                severity: r.max_deficit,
            });
        }
    }

    // Close rounds (winner < 20% HP)
    for r in rounds {
        if r.winner_final_hp < 0.20 && !r.is_comeback {
            highlights.push(Highlight {
                kind: "close_round".to_string(),
                label: format!(
                    "Close round - {} wins at {:.0}% HP",
                    r.winner,
                    r.winner_final_hp * 100.0
                ),
                timestamp_ms: r.round_start_ms,
                end_ms: r.round_end_ms,
                details: format!(
                    "{} barely won with only {:.0}% HP remaining",
                    r.winner,
                    r.winner_final_hp * 100.0
                ),
                severity: 1.0 - r.winner_final_hp,
            });
        }
    }

    // Big damage events (> 30%)
    for e in damage_events {
        if e.damage_pct > 0.30 {
            let target = if e.target_side == 1 { "P1" } else { "P2" };
            highlights.push(Highlight {
                kind: "big_damage".to_string(),
                label: format!("{:.0}% damage to {}", e.damage_pct * 100.0, target),
                timestamp_ms: e.timestamp_ms,
                end_ms: e.timestamp_ms + 2000.0,
                details: format!(
                    "{} took {:.0}% damage ({:.0}% -> {:.0}%)",
                    target,
                    e.damage_pct * 100.0,
                    e.pre_health_pct * 100.0,
                    e.post_health_pct * 100.0
                ),
                severity: e.damage_pct,
            });
        }
    }

    // Perfect rounds (winner ends > 90%)
    for r in rounds {
        if r.winner_final_hp > 0.90 {
            highlights.push(Highlight {
                kind: "perfect".to_string(),
                label: format!("Near-perfect by {}", r.winner),
                timestamp_ms: r.round_start_ms,
                end_ms: r.round_end_ms,
                details: format!(
                    "{} won with {:.0}% HP remaining (min: {:.0}%)",
                    r.winner,
                    r.winner_final_hp * 100.0,
                    r.winner_min_hp * 100.0
                ),
                severity: r.winner_final_hp,
            });
        }
    }

    highlights.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    highlights
}

// ── Tauri commands ───────────────────────────────────────────────────────────

#[tauri::command]
fn get_replays(db_path: String) -> Result<Vec<Replay>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT replay_id, video_path, duration_ms, frame_count FROM replays ORDER BY replay_id")
        .map_err(|e| e.to_string())?;

    let replays: Vec<Replay> = stmt
        .query_map([], |row| {
            Ok(Replay {
                replay_id: row.get(0)?,
                video_path: row.get(1)?,
                duration_ms: row.get(2)?,
                frame_count: row.get(3)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(replays)
}

#[tauri::command]
fn get_frame_data(db_path: String, replay_id: String) -> Result<Vec<FrameDataPoint>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Check if tension columns exist
    let has_tension = conn
        .prepare("SELECT p1_tension_pct FROM frame_data LIMIT 1")
        .is_ok();

    let points: Vec<FrameDataPoint> = if has_tension {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct, p1_tension_pct, p2_tension_pct
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt.query_map([&replay_id], |row| {
                Ok(FrameDataPoint {
                    timestamp_ms: row.get(0)?,
                    p1_health_pct: row.get(1)?,
                    p2_health_pct: row.get(2)?,
                    p1_tension_pct: row.get(3)?,
                    p2_tension_pct: row.get(4)?,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.filter_map(|r| r.ok()).collect()
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT timestamp_ms, p1_health_pct, p2_health_pct
                 FROM frame_data
                 WHERE replay_id = ?
                 ORDER BY timestamp_ms",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt.query_map([&replay_id], |row| {
                Ok(FrameDataPoint {
                    timestamp_ms: row.get(0)?,
                    p1_health_pct: row.get(1)?,
                    p2_health_pct: row.get(2)?,
                    p1_tension_pct: None,
                    p2_tension_pct: None,
                })
            })
            .map_err(|e| e.to_string())?;
        rows.filter_map(|r| r.ok()).collect()
    };

    Ok(points)
}

#[tauri::command]
fn get_damage_events(db_path: String, replay_id: String) -> Result<Vec<DamageEvent>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT event_id, replay_id, timestamp_ms, frame_start, frame_end,
                    target_side, damage_pct, pre_health_pct, post_health_pct
             FROM damage_events
             WHERE replay_id = ?
             ORDER BY timestamp_ms",
        )
        .map_err(|e| e.to_string())?;

    let events: Vec<DamageEvent> = stmt
        .query_map([&replay_id], |row| {
            Ok(DamageEvent {
                event_id: row.get(0)?,
                replay_id: row.get(1)?,
                timestamp_ms: row.get(2)?,
                frame_start: row.get(3)?,
                frame_end: row.get(4)?,
                target_side: row.get(5)?,
                damage_pct: row.get(6)?,
                pre_health_pct: row.get(7)?,
                post_health_pct: row.get(8)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(events)
}

#[tauri::command]
fn get_rounds(db_path: String, replay_id: String) -> Result<Vec<RoundResult>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    detect_rounds_for_replay(&conn, &replay_id)
}

#[tauri::command]
fn get_match_stats(db_path: String, replay_id: String) -> Result<MatchStats, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Get replay info
    let replay: Replay = conn
        .query_row(
            "SELECT replay_id, video_path, duration_ms, frame_count FROM replays WHERE replay_id = ?",
            [&replay_id],
            |row| {
                Ok(Replay {
                    replay_id: row.get(0)?,
                    video_path: row.get(1)?,
                    duration_ms: row.get(2)?,
                    frame_count: row.get(3)?,
                })
            },
        )
        .map_err(|e| e.to_string())?;

    let rounds = detect_rounds_for_replay(&conn, &replay_id)?;

    // Get damage events
    let mut stmt = conn
        .prepare(
            "SELECT event_id, replay_id, timestamp_ms, frame_start, frame_end,
                    target_side, damage_pct, pre_health_pct, post_health_pct
             FROM damage_events WHERE replay_id = ? ORDER BY timestamp_ms",
        )
        .map_err(|e| e.to_string())?;

    let events: Vec<DamageEvent> = stmt
        .query_map([&replay_id], |row| {
            Ok(DamageEvent {
                event_id: row.get(0)?,
                replay_id: row.get(1)?,
                timestamp_ms: row.get(2)?,
                frame_start: row.get(3)?,
                frame_end: row.get(4)?,
                target_side: row.get(5)?,
                damage_pct: row.get(6)?,
                pre_health_pct: row.get(7)?,
                post_health_pct: row.get(8)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(compute_match_stats(&replay, &rounds, &events))
}

#[tauri::command]
fn get_highlights(db_path: String, replay_id: String) -> Result<Vec<Highlight>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let rounds = detect_rounds_for_replay(&conn, &replay_id)?;

    let mut stmt = conn
        .prepare(
            "SELECT event_id, replay_id, timestamp_ms, frame_start, frame_end,
                    target_side, damage_pct, pre_health_pct, post_health_pct
             FROM damage_events WHERE replay_id = ? ORDER BY timestamp_ms",
        )
        .map_err(|e| e.to_string())?;

    let events: Vec<DamageEvent> = stmt
        .query_map([&replay_id], |row| {
            Ok(DamageEvent {
                event_id: row.get(0)?,
                replay_id: row.get(1)?,
                timestamp_ms: row.get(2)?,
                frame_start: row.get(3)?,
                frame_end: row.get(4)?,
                target_side: row.get(5)?,
                damage_pct: row.get(6)?,
                pre_health_pct: row.get(7)?,
                post_health_pct: row.get(8)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(generate_highlights(&rounds, &events))
}

/// Find project root: try CARGO_MANIFEST_DIR (dev), then walk up from exe.
/// Find the root directory containing scripts/analyze_replay.py.
///
/// Search order:
/// 1. CARGO_MANIFEST_DIR parent (dev builds)
/// 2. Walk up from exe (dev / portable)
/// 3. Resource directory next to exe (installed via NSIS)
fn find_project_root() -> Result<PathBuf, String> {
    let marker = |d: &PathBuf| d.join("scripts").join("analyze_replay.py").exists();

    // 1. Dev: walk up from Cargo manifest dir (app/src-tauri -> app -> repo root)
    let mut manifest_dir: Option<PathBuf> = Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    while let Some(d) = manifest_dir {
        if marker(&d) {
            return Ok(d);
        }
        manifest_dir = d.parent().map(|p| p.to_path_buf());
    }

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let exe_dir = exe.parent().map(|p| p.to_path_buf())
        .ok_or_else(|| "Cannot determine exe directory".to_string())?;

    // 2. Walk up from exe (dev / portable)
    let mut dir = Some(exe_dir.clone());
    while let Some(d) = dir {
        if marker(&d) {
            return Ok(d);
        }
        dir = d.parent().map(|p| p.to_path_buf());
    }

    // 3. Installed: resources are bundled next to the exe in a resources/ subdir,
    //    or directly next to the exe (NSIS places them in the install dir).
    //    Tauri NSIS puts resources at <install_dir>/resources/
    let resource_dir = exe_dir.join("resources");
    if marker(&resource_dir) {
        return Ok(resource_dir);
    }

    // 4. Also check exe_dir itself (some bundle layouts)
    if marker(&exe_dir) {
        return Ok(exe_dir);
    }

    Err(format!(
        "Cannot find project root with scripts/analyze_replay.py. Searched near: {}",
        exe_dir.display()
    ))
}

/// Return the default output directory and DB path.
///
/// For installed apps, output goes to AppData/Local/SpaghettiLab/output.
/// For dev, output goes to <project_root>/output.
fn default_output_paths() -> Result<(PathBuf, PathBuf), String> {
    // Try dev location first
    if let Ok(root) = find_project_root() {
        // Check if this looks like a dev environment (has pyproject.toml or .git)
        if root.join("pyproject.toml").exists() || root.join(".git").exists() {
            let output_dir = root.join("output");
            let db_path = output_dir.join("analysis.db");
            return Ok((output_dir, db_path));
        }
    }

    // Installed: use AppData/Local
    let app_data = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_fallback()
        });
    let output_dir = app_data.join("SpaghettiLab").join("output");
    let db_path = output_dir.join("analysis.db");
    Ok((output_dir, db_path))
}

fn dirs_fallback() -> PathBuf {
    // Fallback: put data next to exe
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

#[tauri::command]
fn get_default_db_path() -> Result<String, String> {
    let (_, db_path) = default_output_paths()?;
    Ok(db_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn analyze_video(
    video_path: String,
    _sample_every: Option<u32>,
) -> Result<String, String> {
    let project_root = find_project_root()?;
    let (output_dir, db_path) = default_output_paths()?;

    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

    let script = project_root.join("scripts").join("analyze_replay.py");
    let config = project_root.join("config").join("default.yaml");

    let mut cmd = Command::new("python");
    // Set PYTHONPATH so `import replanal` works from installed location
    cmd.env("PYTHONPATH", project_root.to_str().unwrap())
        .arg(script.to_str().unwrap())
        .arg(&video_path)
        .arg("--config")
        .arg(config.to_str().unwrap())
        .arg("--output")
        .arg(output_dir.to_str().unwrap());

    let output = cmd.output().map_err(|e| format!("Failed to run Python: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        return Err(format!("Analysis failed:\n{}\n{}", stdout, stderr));
    }

    Ok(db_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn reanalyze_replay(db_path: String, replay_id: String) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let video_path = resolve_video_for_replay(&conn, &replay_id)?;
    drop(conn);

    let project_root = find_project_root()?;
    let (output_dir, _) = default_output_paths()?;

    let script = project_root.join("scripts").join("analyze_replay.py");
    let config = project_root.join("config").join("default.yaml");

    let mut cmd = Command::new("python");
    cmd.env("PYTHONPATH", project_root.to_str().unwrap())
        .arg(script.to_str().unwrap())
        .arg(video_path.to_str().unwrap())
        .arg("--config")
        .arg(config.to_str().unwrap())
        .arg("--output")
        .arg(output_dir.to_str().unwrap());

    let output = cmd.output().map_err(|e| format!("Failed to run Python: {}", e))?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Reanalysis failed:\n{}\n{}", stdout, stderr));
    }

    Ok(())
}

#[tauri::command]
async fn export_clip(
    video_path: String,
    start_ms: f64,
    end_ms: f64,
    output_path: String,
) -> Result<(), String> {
    let start_s = start_ms / 1000.0;
    let duration_s = (end_ms - start_ms) / 1000.0;

    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-ss",
            &format!("{:.3}", start_s),
            "-i",
            &video_path,
            "-t",
            &format!("{:.3}", duration_s),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            &output_path,
        ])
        .output()
        .map_err(|e| format!("Failed to run ffmpeg: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg error: {}", stderr));
    }
    Ok(())
}

#[tauri::command]
fn resolve_video_path(db_path: String, replay_id: String) -> Result<String, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    let path: String = conn
        .query_row(
            "SELECT video_path FROM replays WHERE replay_id = ?",
            [&replay_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;

    let p = PathBuf::from(&path);

    // If already absolute and exists, return as-is
    if p.is_absolute() && p.exists() {
        return Ok(path);
    }

    // Try resolving relative to project root
    if let Ok(root) = find_project_root() {
        let resolved = root.join(&p);
        if resolved.exists() {
            return Ok(resolved.to_string_lossy().to_string());
        }
    }

    // Return the raw path — frontend will handle "not found"
    Ok(path)
}

// ── Notes ────────────────────────────────────────────────────────────────────

fn ensure_notes_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS notes (
            note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id  TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            text       TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );"
    ).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_notes(db_path: String, replay_id: String) -> Result<Vec<Note>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    let mut stmt = conn.prepare(
        "SELECT note_id, replay_id, timestamp_ms, text, created_at
         FROM notes WHERE replay_id = ? ORDER BY timestamp_ms"
    ).map_err(|e| e.to_string())?;
    let notes = stmt.query_map([&replay_id], |row| {
        Ok(Note {
            note_id: row.get(0)?,
            replay_id: row.get(1)?,
            timestamp_ms: row.get(2)?,
            text: row.get(3)?,
            created_at: row.get(4)?,
        })
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();
    Ok(notes)
}

#[tauri::command]
fn add_note(db_path: String, replay_id: String, timestamp_ms: f64, text: String) -> Result<Note, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    conn.execute(
        "INSERT INTO notes (replay_id, timestamp_ms, text) VALUES (?, ?, ?)",
        rusqlite::params![&replay_id, timestamp_ms, &text],
    ).map_err(|e| e.to_string())?;
    let note_id = conn.last_insert_rowid();
    let note = conn.query_row(
        "SELECT note_id, replay_id, timestamp_ms, text, created_at FROM notes WHERE note_id = ?",
        [note_id],
        |row| Ok(Note {
            note_id: row.get(0)?,
            replay_id: row.get(1)?,
            timestamp_ms: row.get(2)?,
            text: row.get(3)?,
            created_at: row.get(4)?,
        })
    ).map_err(|e| e.to_string())?;
    Ok(note)
}

#[tauri::command]
fn update_note(db_path: String, note_id: i64, text: String, timestamp_ms: Option<f64>) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    match timestamp_ms {
        Some(ts) => conn.execute(
            "UPDATE notes SET text = ?, timestamp_ms = ? WHERE note_id = ?",
            rusqlite::params![&text, ts, note_id],
        ),
        None => conn.execute(
            "UPDATE notes SET text = ? WHERE note_id = ?",
            rusqlite::params![&text, note_id],
        ),
    }.map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn delete_note(db_path: String, note_id: i64) -> Result<(), String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    ensure_notes_table(&conn)?;
    conn.execute("DELETE FROM notes WHERE note_id = ?", [note_id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ── VOD Splitter ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RoiRect {
    pub y1: u32,
    pub y2: u32,
    pub x1: u32,
    pub x2: u32,
}

impl RoiRect {
    fn to_arg(&self) -> String {
        format!("{},{},{},{}", self.y1, self.y2, self.x1, self.x2)
    }
}

#[derive(Debug, Deserialize)]
pub struct VodRoiConfig {
    pub p1_tension: RoiRect,
    pub p2_tension: RoiRect,
    pub timer: RoiRect,
    pub banner: RoiRect,
}

#[derive(Debug, Serialize, Clone)]
pub struct DetectedSetInfo {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
    pub gameplay_duration_secs: f64,
    pub game_count: u32,
}

#[tauri::command]
async fn extract_preview_frame(
    video_path: String,
    timestamp_secs: f64,
) -> Result<String, String> {
    let (output_dir, _) = default_output_paths()?;
    let preview_path = output_dir.join(".vod_preview.png");
    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-ss", &format!("{:.3}", timestamp_secs),
            "-i", &video_path,
            "-frames:v", "1",
            "-q:v", "2",
            "-v", "quiet",
            preview_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("Failed to run ffmpeg: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg error: {}", stderr));
    }

    Ok(preview_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn scan_vod(
    app_handle: tauri::AppHandle,
    video_path: String,
    roi_config: VodRoiConfig,
) -> Result<Vec<DetectedSetInfo>, String> {
    let project_root = find_project_root()?;
    let script = project_root.join("scripts").join("split_vod.py");

    let mut cmd = Command::new("python");
    cmd.env("PYTHONPATH", project_root.to_str().unwrap())
        .arg(script.to_str().unwrap())
        .arg(&video_path)
        .arg("--preview")
        .arg("--json-output")
        .arg("--p1-tension").arg(roi_config.p1_tension.to_arg())
        .arg("--p2-tension").arg(roi_config.p2_tension.to_arg())
        .arg("--timer-roi").arg(roi_config.timer.to_arg())
        .arg("--banner-roi").arg(roi_config.banner.to_arg())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn: {}", e))?;

    let stdout = child.stdout.take().unwrap();
    let mut json_result: Option<String> = None;

    for line in std::io::BufReader::new(stdout).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.starts_with("PROGRESS:") {
            let _ = app_handle.emit("vod-scan-progress", &line);
        } else if line.starts_with("JSON_RESULT:") {
            json_result = Some(line["JSON_RESULT:".len()..].to_string());
        }
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("VOD scan failed".into());
    }

    let json_str = json_result.ok_or("No JSON result from split_vod.py")?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let sets = parsed["sets"]
        .as_array()
        .ok_or("Invalid JSON: missing sets array")?
        .iter()
        .map(|s| DetectedSetInfo {
            index: s["index"].as_u64().unwrap_or(0) as u32,
            start_secs: s["start_secs"].as_f64().unwrap_or(0.0),
            end_secs: s["end_secs"].as_f64().unwrap_or(0.0),
            gameplay_duration_secs: s["gameplay_duration_secs"].as_f64().unwrap_or(0.0),
            game_count: s["game_count"].as_u64().unwrap_or(0) as u32,
        })
        .collect();

    Ok(sets)
}

#[derive(Debug, Deserialize)]
pub struct CutSetRequest {
    pub index: u32,
    pub start_secs: f64,
    pub end_secs: f64,
}

#[tauri::command]
async fn cut_vod_sets(
    app_handle: tauri::AppHandle,
    video_path: String,
    sets: Vec<CutSetRequest>,
    output_dir: String,
) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;
    let mut paths: Vec<String> = Vec::new();
    let padding_before = 10.0;
    let padding_after = 5.0;

    for (i, set) in sets.iter().enumerate() {
        let start = (set.start_secs - padding_before).max(0.0);
        let duration = (set.end_secs + padding_after) - start;
        let out_path = PathBuf::from(&output_dir)
            .join(format!("set_{:02}.mp4", set.index));

        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-ss", &format!("{:.3}", start),
                "-i", &video_path,
                "-t", &format!("{:.3}", duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-v", "warning",
                out_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("ffmpeg error: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to cut set {}: {}", set.index, stderr));
        }

        paths.push(out_path.to_string_lossy().to_string());
        let _ = app_handle.emit("vod-cut-progress", format!("{}/{}", i + 1, sets.len()));
    }

    Ok(paths)
}

// ── .spag file format ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct SpagSession {
    pub db_path: String,
    pub video_path: String,
    pub spag_path: String,
    pub replay_id: String,
}

/// Resolve the video path for a replay (reusable helper).
fn resolve_video_for_replay(conn: &Connection, replay_id: &str) -> Result<PathBuf, String> {
    let raw_path: String = conn
        .query_row(
            "SELECT video_path FROM replays WHERE replay_id = ?",
            [replay_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;

    let p = PathBuf::from(&raw_path);
    if p.is_absolute() && p.exists() {
        return Ok(p);
    }
    if let Ok(root) = find_project_root() {
        let resolved = root.join(&p);
        if resolved.exists() {
            return Ok(resolved);
        }
    }
    Err(format!("Video file not found: {}", raw_path))
}

#[tauri::command]
async fn export_spag(
    db_path: String,
    replay_id: String,
    output_path: String,
) -> Result<(), String> {
    let src_conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // Resolve the video file
    let video_path = resolve_video_for_replay(&src_conn, &replay_id)?;

    // Create temp DB with just this replay's data using ATTACH DATABASE
    let tmp_db = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    let tmp_db_path = tmp_db.path().to_path_buf();
    {
        // Use the source connection and attach the destination
        let tmp_str = tmp_db_path.to_string_lossy().replace('\\', "/");
        src_conn.execute(&format!("ATTACH DATABASE '{}' AS dst", tmp_str), [])
            .map_err(|e| e.to_string())?;

        // Create tables in destination
        src_conn.execute_batch(
            "CREATE TABLE dst.replays (
                replay_id TEXT PRIMARY KEY,
                video_path TEXT,
                duration_ms REAL,
                frame_count INTEGER
            );
            CREATE TABLE dst.frame_data (
                replay_id TEXT,
                frame_number INTEGER,
                timestamp_ms REAL,
                p1_health_pct REAL,
                p2_health_pct REAL,
                timer_value INTEGER,
                p1_rounds_won INTEGER,
                p2_rounds_won INTEGER
            );
            CREATE TABLE dst.damage_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                replay_id TEXT,
                timestamp_ms REAL,
                frame_start INTEGER,
                frame_end INTEGER,
                target_side INTEGER,
                damage_pct REAL,
                pre_health_pct REAL,
                post_health_pct REAL
            );
            CREATE TABLE dst.notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                replay_id TEXT NOT NULL,
                timestamp_ms REAL NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );"
        ).map_err(|e| e.to_string())?;

        // Copy data with video_path rewritten
        src_conn.execute(
            "INSERT INTO dst.replays SELECT replay_id, 'video.mp4', duration_ms, frame_count FROM main.replays WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        src_conn.execute(
            "INSERT INTO dst.frame_data SELECT * FROM main.frame_data WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        src_conn.execute(
            "INSERT INTO dst.damage_events SELECT * FROM main.damage_events WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        ensure_notes_table(&src_conn)?;
        src_conn.execute(
            "INSERT INTO dst.notes SELECT * FROM main.notes WHERE replay_id = ?",
            [&replay_id],
        ).map_err(|e| e.to_string())?;

        src_conn.execute("DETACH DATABASE dst", []).map_err(|e| e.to_string())?;
    }

    // Build the ZIP
    let out_file = std::fs::File::create(&output_path)
        .map_err(|e| format!("Cannot create {}: {}", output_path, e))?;
    let mut zip = zip::ZipWriter::new(out_file);

    // Add replay.db (compressed)
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("replay.db", options).map_err(|e| e.to_string())?;
    let db_bytes = std::fs::read(&tmp_db_path).map_err(|e| e.to_string())?;
    zip.write_all(&db_bytes).map_err(|e| e.to_string())?;

    // Add video.mp4 (stored, no re-compression)
    let stored = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
    zip.start_file("video.mp4", stored).map_err(|e| e.to_string())?;
    let mut vf = std::fs::File::open(&video_path)
        .map_err(|e| format!("Cannot open video: {}", e))?;
    let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8MB buffer
    loop {
        let n = vf.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        zip.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn open_spag(spag_path: String) -> Result<SpagSession, String> {
    let spag = PathBuf::from(&spag_path);
    if !spag.exists() {
        return Err(format!("File not found: {}", spag_path));
    }

    // Deterministic extraction dir based on file path
    let app_data = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_fallback());
    let hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        spag_path.hash(&mut h);
        h.finish()
    };
    let session_dir = app_data.join("SpaghettiLab").join("spag_sessions").join(format!("{:016x}", hash));
    std::fs::create_dir_all(&session_dir).map_err(|e| e.to_string())?;

    let db_dest = session_dir.join("replay.db");
    let video_dest = session_dir.join("video.mp4");

    // Extract (always re-extract to get fresh data)
    let file = std::fs::File::open(&spag).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;

    // Extract replay.db
    {
        let mut entry = archive.by_name("replay.db").map_err(|e| e.to_string())?;
        let mut out = std::fs::File::create(&db_dest).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
    }

    // Extract video.mp4
    {
        let mut entry = archive.by_name("video.mp4").map_err(|e| e.to_string())?;
        let mut out = std::fs::File::create(&video_dest).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
    }

    // Get replay_id from extracted DB
    let conn = Connection::open(&db_dest).map_err(|e| e.to_string())?;
    let replay_id: String = conn
        .query_row("SELECT replay_id FROM replays LIMIT 1", [], |row| row.get(0))
        .map_err(|e| e.to_string())?;

    Ok(SpagSession {
        db_path: db_dest.to_string_lossy().to_string(),
        video_path: video_dest.to_string_lossy().to_string(),
        spag_path,
        replay_id,
    })
}

#[tauri::command]
async fn save_spag(spag_path: String, db_path: String) -> Result<(), String> {
    let spag = PathBuf::from(&spag_path);

    // Read the original .spag to get the video
    let orig_file = std::fs::File::open(&spag).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(orig_file).map_err(|e| e.to_string())?;

    // Write to a temp file next to the original, then rename
    let tmp_path = spag.with_extension("spag.tmp");
    {
        let out_file = std::fs::File::create(&tmp_path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipWriter::new(out_file);

        // Write updated DB
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("replay.db", options).map_err(|e| e.to_string())?;
        let db_bytes = std::fs::read(&db_path).map_err(|e| e.to_string())?;
        zip.write_all(&db_bytes).map_err(|e| e.to_string())?;

        // Copy video from original archive
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file("video.mp4", stored).map_err(|e| e.to_string())?;
        let mut video_entry = archive.by_name("video.mp4").map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        loop {
            let n = video_entry.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 { break; }
            zip.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        }

        zip.finish().map_err(|e| e.to_string())?;
    }

    // Atomic rename
    std::fs::rename(&tmp_path, &spag).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn reanalyze_all(db_path: String) -> Result<(), String> {
    let ids: Vec<String> = {
        let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT replay_id FROM replays ORDER BY replay_id")
            .map_err(|e| e.to_string())?;
        let result: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        result
    };

    for id in &ids {
        reanalyze_replay(db_path.clone(), id.clone()).await?;
    }
    Ok(())
}

// ── App setup ────────────────────────────────────────────────────────────────

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Check if launched with a .spag file argument (file association)
            let args: Vec<String> = std::env::args().collect();
            if let Some(path) = args.get(1) {
                if path.ends_with(".spag") {
                    let path = path.clone();
                    let handle = app.handle().clone();
                    // Emit after a short delay so frontend is ready
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        let _ = handle.emit("open-spag-file", &path);
                    });
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_replays,
            get_frame_data,
            get_damage_events,
            get_rounds,
            get_match_stats,
            get_highlights,
            get_default_db_path,
            analyze_video,
            reanalyze_replay,
            reanalyze_all,
            export_clip,
            resolve_video_path,
            extract_preview_frame,
            scan_vod,
            cut_vod_sets,
            get_notes,
            add_note,
            update_note,
            delete_note,
            export_spag,
            open_spag,
            save_spag,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
