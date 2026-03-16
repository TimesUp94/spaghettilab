"""Check round detection on the Ricbionicle test DB by replicating Rust logic.

Faithfully mirrors detect_rounds_for_replay() from app/src-tauri/src/lib.rs
including timer-based round starts, HP resets, game breaks, wallbreak filter,
deduplication, long-round splitting, and KO detection.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "output" / "sets_test" / "analysis.db"

HP_CEILING = 0.875
MIN_ROUND_FRAMES = 450  # ~15 seconds
TIMER_SMOOTH_WINDOW = 61
HP_RESET_WINDOW = 90


def rolling_median(data: list[float], window: int) -> list[float]:
    """Replicate the Rust rolling_median: centered, ignoring NaN, min_valid = max(window//3, 3)."""
    n = len(data)
    result = [float('nan')] * n
    half = window // 2
    min_valid = max(window // 3, 3)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        vals = sorted(v for v in data[start:end] if not np.isnan(v))
        if len(vals) >= min_valid:
            result[i] = vals[len(vals) // 2]
    return result


def rolling_median_i32(data: list, window: int) -> list:
    """Rolling median for optional int values (None = missing)."""
    n = len(data)
    result = [None] * n
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        vals = sorted(v for v in data[start:end] if v is not None)
        if vals:
            result[i] = vals[len(vals) // 2]
    return result


def forward_fill(data: list) -> list:
    """Forward-fill None values."""
    out = list(data)
    last = None
    for i, v in enumerate(out):
        if v is not None:
            last = v
        else:
            out[i] = last
    return out


def main():
    conn = sqlite3.connect(str(DB))
    replay_id = "04_UF_GuessingGame_vs_Ricbionicle"

    rows = conn.execute(
        "SELECT timestamp_ms, p1_health_pct, p2_health_pct, timer_value FROM frame_data "
        "WHERE replay_id = ? ORDER BY timestamp_ms", (replay_id,)
    ).fetchall()
    conn.close()

    if not rows:
        print("No data found!")
        return

    n = len(rows)
    print(f"Loaded {n} frames")

    ts = [r[0] for r in rows]
    p1_raw = [r[1] if r[1] is not None else float('nan') for r in rows]
    p2_raw = [r[2] if r[2] is not None else float('nan') for r in rows]
    timer_raw = [r[3] for r in rows]  # None or int

    # Normalize by ceiling
    p1_norm = [min(1.0, max(0.0, v / HP_CEILING)) if not np.isnan(v) else float('nan') for v in p1_raw]
    p2_norm = [min(1.0, max(0.0, v / HP_CEILING)) if not np.isnan(v) else float('nan') for v in p2_raw]

    # Junk data filter
    junk_thresh = 0.35
    junk_window = 150
    p1_junk = rolling_median(p1_norm, junk_window)
    p2_junk = rolling_median(p2_norm, junk_window)
    for i in range(n):
        if (not np.isnan(p1_junk[i]) and not np.isnan(p2_junk[i])
                and p1_junk[i] < junk_thresh and p2_junk[i] < junk_thresh):
            p1_norm[i] = float('nan')
            p2_norm[i] = float('nan')

    # Smooth HP (31-frame window)
    p1_smooth = rolling_median(p1_norm, 31)
    p2_smooth = rolling_median(p2_norm, 31)

    # ── Signal 1: Timer-based round starts ──
    timer_ff = forward_fill(timer_raw)
    timer_smooth = rolling_median_i32(timer_ff, TIMER_SMOOTH_WINDOW)

    timer_starts = []
    lowest_since_start = 99
    for i in range(n):
        t = timer_smooth[i]
        if t is not None:
            lowest_since_start = min(lowest_since_start, t)
            if t >= 88 and lowest_since_start <= 70:
                if not timer_starts or i - timer_starts[-1] > MIN_ROUND_FRAMES:
                    # Validate: timer must reach >=93 within 90 frames
                    check_end = min(i + 90, n)
                    confirmed = any(
                        timer_smooth[j] is not None and timer_smooth[j] >= 93
                        for j in range(i, check_end)
                    )
                    if confirmed:
                        timer_starts.append(i)
                        lowest_since_start = t

    print(f"\nTimer starts: {len(timer_starts)}")
    for idx in timer_starts:
        print(f"  idx={idx}, t={ts[idx]/1000:.1f}s, timer={timer_smooth[idx]}")

    # ── Signal 2: HP reset detection (90-frame window) ──
    p1_rm = rolling_median(p1_norm, HP_RESET_WINDOW)
    p2_rm = rolling_median(p2_norm, HP_RESET_WINDOW)
    hp_resets = []
    min_hp_since_reset = 1.0
    for i in range(n):
        if not np.isnan(p1_rm[i]) and not np.isnan(p2_rm[i]):
            current_min = min(p1_rm[i], p2_rm[i])
            min_hp_since_reset = min(min_hp_since_reset, current_min)
            if p1_rm[i] > 0.85 and p2_rm[i] > 0.85 and min_hp_since_reset < 0.50:
                if not hp_resets or i - hp_resets[-1] > MIN_ROUND_FRAMES:
                    hp_resets.append(i)
                    min_hp_since_reset = current_min

    print(f"\nHP resets: {len(hp_resets)}")
    for idx in hp_resets:
        print(f"  idx={idx}, t={ts[idx]/1000:.1f}s")

    # ── Signal 3: Game break detection (5s gaps) ──
    game_breaks = []
    prev_valid = None
    for i in range(n):
        if not np.isnan(p1_norm[i]):
            if prev_valid is not None:
                gap_ms = ts[i] - ts[prev_valid]
                if gap_ms > 5000.0:
                    game_breaks.append(i)
            prev_valid = i

    print(f"\nGame breaks: {len(game_breaks)}")
    for idx in game_breaks:
        print(f"  idx={idx}, t={ts[idx]/1000:.1f}s")

    # ── Merge all boundary signals ──
    all_boundaries = sorted(set(timer_starts + hp_resets + game_breaks))

    # ── Wallbreak filter: require timer >=93 within 90 frames ──
    def timer_confirmed(b):
        check_end = min(b + 90, n)
        return any(
            timer_smooth[j] is not None and timer_smooth[j] >= 93
            for j in range(b, check_end)
        )

    pre_filter = len(all_boundaries)
    all_boundaries = [b for b in all_boundaries if timer_confirmed(b)]
    print(f"\nWallbreak filter: {pre_filter} -> {len(all_boundaries)} boundaries")

    # Deduplicate within 10 seconds
    deduped = []
    for idx in all_boundaries:
        if not deduped or ts[idx] - ts[deduped[-1]] > 10000.0:
            deduped.append(idx)

    # Add first valid data frame if needed
    first_valid = next((i for i in range(n) if not np.isnan(p1_norm[i])), None)
    if first_valid is not None:
        if not deduped or deduped[0] - first_valid > MIN_ROUND_FRAMES:
            deduped.insert(0, first_valid)

    print(f"\nRound boundaries: {len(deduped)}")
    for idx in deduped:
        timer_val = timer_smooth[idx] if timer_smooth[idx] is not None else "N/A"
        print(f"  idx={idx}, t={ts[idx]/1000:.1f}s, timer={timer_val}")

    # ── KO detection + winner determination (matching Rust) ──
    print(f"\n{'='*60}")
    print("ROUND RESULTS:")
    print(f"{'='*60}")

    round_num = 0
    for r_idx in range(len(deduped)):
        s = deduped[r_idx]
        e = deduped[r_idx + 1] if r_idx + 1 < len(deduped) else n - 1
        span = e - s
        if span < MIN_ROUND_FRAMES:
            continue

        # Valid data ratio check
        valid_count = sum(1 for j in range(s, e)
                         if not np.isnan(p1_norm[j]) and not np.isnan(p2_norm[j]))
        if valid_count / span < 0.30:
            continue

        search_start = s + int(span * 0.4)

        # Count KO frames (HP < 0.15 with other player NOT at full)
        p1_ko = sum(1 for j in range(search_start, e)
                     if not np.isnan(p1_norm[j]) and p1_norm[j] < 0.15
                     and not np.isnan(p2_norm[j]) and p2_norm[j] < 0.90)
        p2_ko = sum(1 for j in range(search_start, e)
                     if not np.isnan(p2_norm[j]) and p2_norm[j] < 0.15
                     and not np.isnan(p1_norm[j]) and p1_norm[j] < 0.90)

        winner = None
        method = ""

        if p1_ko >= 3 and p1_ko >= p2_ko * 3 + 1:
            winner = "P2"
            method = "KO(P1 died)"
        elif p2_ko >= 3 and p2_ko >= p1_ko * 3 + 1:
            winner = "P1"
            method = "KO(P2 died)"
        else:
            # Both have KO frames — check first-to-zero
            if p1_ko >= 3 and p2_ko >= 3:
                first_p1 = next((j for j in range(search_start, e)
                                 if not np.isnan(p1_norm[j]) and p1_norm[j] < 0.05), None)
                first_p2 = next((j for j in range(search_start, e)
                                 if not np.isnan(p2_norm[j]) and p2_norm[j] < 0.05), None)
                if first_p1 is not None and first_p2 is not None:
                    gap = abs(first_p1 - first_p2)
                    if gap <= 90:
                        earlier, later, earlier_is_p1 = (first_p1, first_p2, True) if first_p1 < first_p2 else (first_p2, first_p1, False)
                        other_at_earlier = p2_norm[earlier] if earlier_is_p1 else p1_norm[earlier]
                        if not np.isnan(other_at_earlier) and other_at_earlier > 0.15:
                            winner = "P2" if earlier_is_p1 else "P1"
                            method = "first-to-zero"

            if winner is None:
                # KO moment: find frame where min(p1, p2) is lowest in last 60%
                best_min_hp = 2.0
                best_p1 = best_p2 = 0.5
                valid_frames = []
                for j in range(search_start, e):
                    if not np.isnan(p1_smooth[j]) and not np.isnan(p2_smooth[j]):
                        valid_frames.append((j, p1_smooth[j], p2_smooth[j]))
                        min_hp = min(p1_smooth[j], p2_smooth[j])
                        if min_hp < best_min_hp:
                            best_min_hp = min_hp
                            best_p1 = p1_smooth[j]
                            best_p2 = p2_smooth[j]

                if not valid_frames:
                    continue

                # Cluster KO frames
                ko_frames = [(j, p1, p2) for j, p1, p2 in valid_frames
                             if min(p1, p2) < best_min_hp + 0.10]

                if ko_frames:
                    clusters = [[ko_frames[0]]]
                    for item in ko_frames[1:]:
                        if item[0] - clusters[-1][-1][0] > 60:
                            clusters.append([item])
                        else:
                            clusters[-1].append(item)

                    last_cluster = clusters[-1]
                    take_count = min(len(last_cluster), 30)
                    ko_use = last_cluster[-take_count:]
                    p1_vals = sorted(f[1] for f in ko_use)
                    p2_vals = sorted(f[2] for f in ko_use)
                    ep1 = p1_vals[len(p1_vals) // 2]
                    ep2 = p2_vals[len(p2_vals) // 2]
                else:
                    ep1 = best_p1
                    ep2 = best_p2

                hp_diff = abs(ep1 - ep2)
                if hp_diff >= 0.15:
                    winner = "P1" if ep1 >= ep2 else "P2"
                    method = f"KO-moment(diff={hp_diff:.2f})"
                else:
                    # Fallback: min HP + avg HP weighted combination
                    p1_min_last = 2.0
                    p2_min_last = 2.0
                    for j in range(search_start, e):
                        if not np.isnan(p1_smooth[j]):
                            p1_min_last = min(p1_min_last, p1_smooth[j])
                        if not np.isnan(p2_smooth[j]):
                            p2_min_last = min(p2_min_last, p2_smooth[j])

                    min_signal = p2_min_last - p1_min_last

                    # Transition artifact detection
                    if (0.40 <= p1_min_last <= 0.55 and 0.40 <= p2_min_last <= 0.55
                            and abs(min_signal) < 0.05):
                        min_signal = 0.0

                    # Avg HP in last 25%
                    last_quarter = s + int(span * 0.75)
                    p1_avg_vals = []
                    p2_avg_vals = []
                    for j in range(last_quarter, e):
                        if not np.isnan(p1_smooth[j]) and not np.isnan(p2_smooth[j]):
                            if p1_smooth[j] < 0.60 and p2_smooth[j] < 0.60:
                                continue
                            p1_avg_vals.append(p1_smooth[j])
                            p2_avg_vals.append(p2_smooth[j])

                    if len(p1_avg_vals) >= 10:
                        avg_signal = np.mean(p2_avg_vals) - np.mean(p1_avg_vals)
                    else:
                        avg_signal = ep2 - ep1

                    hp_range = abs(p1_min_last - p2_min_last)
                    min_weight = min(hp_range / 0.20, 0.8)
                    avg_weight = 1.0 - min_weight
                    combined = min_signal * min_weight + avg_signal * avg_weight
                    winner = "P2" if combined > 0 else "P1"
                    method = f"fallback(min_sig={min_signal:.3f},avg_sig={avg_signal:.3f},combined={combined:.3f})"

        round_num += 1
        dur_s = (ts[e] - ts[s]) / 1000.0 if e < n else (ts[-1] - ts[s]) / 1000.0
        print(f"  Round {round_num}: t={ts[s]/1000:.1f}-{ts[min(e,n-1)]/1000:.1f}s  "
              f"dur={dur_s:.1f}s  Winner={winner}  P1_ko={p1_ko} P2_ko={p2_ko}  "
              f"method={method}")


if __name__ == "__main__":
    main()
