"""Validate round detection across all set videos."""
import sqlite3
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HP_CEILING = 0.875
MIN_ROUND_FRAMES = 450  # 15s at 30fps


def rolling_median(vals, window):
    n = len(vals)
    result = [float("nan")] * n
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = [v for v in vals[lo:hi] if not np.isnan(v)]
        if len(chunk) >= window // 3:
            result[i] = float(np.median(chunk))
    return result


def forward_fill(vals):
    result = list(vals)
    last = None
    for i in range(len(result)):
        if result[i] is not None:
            last = result[i]
        else:
            result[i] = last
    return result


def find_winner(p1_smooth, p2_smooth, s_idx, e_idx):
    """Determine round winner using KO moment + fallback strategies.

    Handles GGS edge cases:
      1. Last round of match: bars disappear (HP goes NaN)
      2. Wallbreak: HP shows combo damage but doesn't visually reach zero
    """
    span = e_idx - s_idx
    search_start = s_idx + int(span * 0.4)

    best_min_hp = 2.0
    best_p1 = None
    best_p2 = None

    valid_frames = []
    for j in range(search_start, e_idx):
        if not np.isnan(p1_smooth[j]) and not np.isnan(p2_smooth[j]):
            valid_frames.append((p1_smooth[j], p2_smooth[j]))
            min_hp = min(p1_smooth[j], p2_smooth[j])
            if min_hp < best_min_hp:
                best_min_hp = min_hp
                best_p1 = p1_smooth[j]
                best_p2 = p2_smooth[j]

    if best_p1 is not None and best_p2 is not None:
        ko_frames = [(p1, p2) for p1, p2 in valid_frames
                     if min(p1, p2) < best_min_hp + 0.10]
        if ko_frames:
            p1_med = float(np.median([f[0] for f in ko_frames[-30:]]))
            p2_med = float(np.median([f[1] for f in ko_frames[-30:]]))
        else:
            p1_med, p2_med = best_p1, best_p2

        hp_diff = abs(p1_med - p2_med)

        # Clear KO moment — use it directly
        if hp_diff >= 0.15:
            winner = "P1" if p1_med > p2_med else "P2"
            return winner, f"[{p1_med:.2f}, {p2_med:.2f}]"

        # Ambiguous — use fallback strategies
        last_q_start = s_idx + int(span * 0.75)

        # Min HP each player reaches in last 40% (catches wallbreak combo)
        p1_min_last = 2.0
        p2_min_last = 2.0
        for j in range(search_start, e_idx):
            if not np.isnan(p1_smooth[j]):
                p1_min_last = min(p1_min_last, p1_smooth[j])
            if not np.isnan(p2_smooth[j]):
                p2_min_last = min(p2_min_last, p2_smooth[j])

        min_signal = p2_min_last - p1_min_last  # positive = P1 went lower = P2 wins

        # Average HP in last 25%
        p1_vals = [p1_smooth[j] for j in range(last_q_start, e_idx) if not np.isnan(p1_smooth[j])]
        p2_vals = [p2_smooth[j] for j in range(last_q_start, e_idx) if not np.isnan(p2_smooth[j])]
        if len(p1_vals) >= 10 and len(p2_vals) >= 10:
            avg_signal = np.mean(p2_vals) - np.mean(p1_vals)
        else:
            avg_signal = p2_med - p1_med

        combined = min_signal * 0.6 + avg_signal * 0.4
        winner = "P2" if combined > 0 else "P1"
        rp1 = p1_min_last if p1_min_last < 2.0 else p1_med
        rp2 = p2_min_last if p2_min_last < 2.0 else p2_med
        return winner, f"[{rp1:.2f}, {rp2:.2f}]"
    return "??", "[no data]"


def detect_rounds(rows):
    """Detect rounds from frame data rows."""
    n = len(rows)
    timestamps = [r[1] for r in rows]
    p1_norm = [min(r[2] / HP_CEILING, 1.0) if r[2] is not None else float("nan") for r in rows]
    p2_norm = [min(r[3] / HP_CEILING, 1.0) if r[3] is not None else float("nan") for r in rows]
    timer_raw = [r[4] for r in rows]

    # Forward-fill + smooth timer with wider window (61 frames = ~2s)
    timer_ff = forward_fill(timer_raw)
    timer_smooth = []
    for i in range(n):
        if timer_ff[i] is None:
            timer_smooth.append(None)
            continue
        lo = max(0, i - 30)
        hi = min(n, i + 31)
        chunk = [timer_ff[j] for j in range(lo, hi) if timer_ff[j] is not None]
        timer_smooth.append(int(np.median(chunk)) if chunk else None)

    # Junk data filter
    p1_junk = rolling_median(p1_norm, 150)
    p2_junk = rolling_median(p2_norm, 150)
    for i in range(n):
        if not np.isnan(p1_junk[i]) and not np.isnan(p2_junk[i]):
            if p1_junk[i] < 0.35 and p2_junk[i] < 0.35:
                p1_norm[i] = float("nan")
                p2_norm[i] = float("nan")

    # Smooth HP for winner detection and metrics (window=31 ≈ 1s)
    p1_smooth = rolling_median(p1_norm, 31)
    p2_smooth = rolling_median(p2_norm, 31)

    # Timer-based round starts (primary signal)
    # Track the lowest smoothed timer since last detected round start.
    # When timer jumps from low (<= 70) to high (>= 88), it's a new round.
    # The wide smoothing window (61 frames) filters OCR noise.
    timer_starts = []
    lowest_since_start = 99
    for i in range(n):
        t = timer_smooth[i]
        if t is not None:
            lowest_since_start = min(lowest_since_start, t)
            if t >= 88 and lowest_since_start <= 70:
                if not timer_starts or (i - timer_starts[-1]) > MIN_ROUND_FRAMES:
                    # Validate: smoothed timer must reach ≥93 within 90 frames.
                    # Filters wallbreak/super flash noise without blocking real boundaries.
                    check_end = min(i + 90, n)
                    confirmed = any(
                        timer_smooth[j] is not None and timer_smooth[j] >= 93
                        for j in range(i, check_end)
                    )
                    if confirmed:
                        timer_starts.append(i)
                        lowest_since_start = t

    # HP reset detection: both HP go from low to high, indicating new round
    # Use rolling median HP over 90 frames (3s). When BOTH medians are > 0.85
    # after a point where at least one was < 0.50, that's a round boundary.
    hp_resets = []
    p1_rm = rolling_median(p1_norm, 90)
    p2_rm = rolling_median(p2_norm, 90)
    min_hp_since_reset = 1.0  # track min HP of either player
    for i in range(n):
        if not np.isnan(p1_rm[i]) and not np.isnan(p2_rm[i]):
            current_min = min(p1_rm[i], p2_rm[i])
            min_hp_since_reset = min(min_hp_since_reset, current_min)
            if p1_rm[i] > 0.85 and p2_rm[i] > 0.85 and min_hp_since_reset < 0.50:
                if not hp_resets or (i - hp_resets[-1]) > MIN_ROUND_FRAMES:
                    hp_resets.append(i)
                    min_hp_since_reset = current_min

    # Game break detection: gaps > 5s in valid HP data
    game_breaks = []
    last_valid = None
    for i in range(n):
        if not np.isnan(p1_norm[i]):
            if last_valid is not None:
                gap_s = (timestamps[i] - timestamps[last_valid]) / 1000.0
                if gap_s > 5.0:
                    game_breaks.append(i)
            last_valid = i

    # Merge all boundaries (timer + HP resets + game breaks)
    all_bounds = sorted(set(timer_starts + hp_resets + game_breaks))

    # Deduplicate within 10 seconds
    merged = []
    for b in all_bounds:
        if not merged or (timestamps[b] - timestamps[merged[-1]]) > 10000:
            merged.append(b)

    # Wallbreak filter: real round starts always have timer ≈99.
    # During wallbreak, bars disappear briefly but timer stays mid-round.
    # Check that smoothed timer reaches ≥93 within 90 frames (~3s) after boundary.
    merged = [b for b in merged
              if any(timer_smooth[j] is not None and timer_smooth[j] >= 93
                     for j in range(b, min(b + 90, n)))]

    # Add first valid data frame as start if needed
    # (first-valid doesn't need timer validation — could be mid-round)
    first_valid = next((i for i in range(n) if not np.isnan(p1_norm[i])), None)
    if first_valid is not None and (not merged or merged[0] - first_valid > MIN_ROUND_FRAMES):
        merged.insert(0, first_valid)

    # Build round spans
    rounds = []
    for k in range(len(merged)):
        s_idx = merged[k]
        e_idx = merged[k + 1] if k + 1 < len(merged) else n - 1

        # Valid data ratio
        valid_count = sum(1 for j in range(s_idx, e_idx)
                         if not np.isnan(p1_norm[j]) and not np.isnan(p2_norm[j]))
        total = e_idx - s_idx
        if total > 0 and valid_count / total < 0.30:
            continue
        if total < MIN_ROUND_FRAMES:
            continue

        dur = (timestamps[e_idx] - timestamps[s_idx]) / 1000.0
        winner, hp_info = find_winner(p1_smooth, p2_smooth, s_idx, e_idx)
        rounds.append((dur, winner, hp_info, timestamps[s_idx] / 1000.0, s_idx, e_idx))

    # Post-process: recursively split rounds > 45s
    # Timer confirmation for split candidates (wider window for KO animation lag)
    def timer_ok(idx):
        check_end = min(idx + 180, n)
        return any(timer_smooth[j] is not None and timer_smooth[j] >= 90
                   for j in range(idx, check_end))

    def split_long_round(s_idx, e_idx, depth=0):
        """Try to split a long round at gaps or HP resets. Timer must confirm."""
        dur = (timestamps[e_idx] - timestamps[s_idx]) / 1000.0
        start_t = timestamps[s_idx] / 1000.0

        if dur <= 45 or depth > 3:
            winner, hp_info = find_winner(p1_smooth, p2_smooth, s_idx, e_idx)
            return [(dur, winner, hp_info, start_t)]

        # Strategy 1: Split at largest data gap > 1.5s (timer must confirm)
        best_split = None
        best_score = 0
        last_v = None
        for i in range(s_idx, e_idx):
            if not np.isnan(p1_norm[i]):
                if last_v is not None:
                    g = (timestamps[i] - timestamps[last_v]) / 1000.0
                    if g > 1.5 and g > best_score:
                        mid_t = timestamps[i] / 1000.0
                        dur1 = mid_t - start_t
                        dur2 = start_t + dur - mid_t
                        if dur1 >= 12 and dur2 >= 12 and timer_ok(i):
                            best_split = i
                            best_score = g
                last_v = i

        # Strategy 2: Split at HP reset (timer must confirm)
        p1_local = rolling_median(p1_norm[s_idx:e_idx], 60)
        p2_local = rolling_median(p2_norm[s_idx:e_idx], 60)
        min_hp_local = 1.0
        for j in range(len(p1_local)):
            if not np.isnan(p1_local[j]) and not np.isnan(p2_local[j]):
                current_min = min(p1_local[j], p2_local[j])
                min_hp_local = min(min_hp_local, current_min)
                if p1_local[j] > 0.75 and p2_local[j] > 0.75 and min_hp_local < 0.55:
                    real_idx = s_idx + j
                    mid_t = timestamps[real_idx] / 1000.0
                    dur1 = mid_t - start_t
                    dur2 = start_t + dur - mid_t
                    if dur1 >= 12 and dur2 >= 12 and timer_ok(real_idx):
                        balance = 1.0 - abs(dur1 - dur2) / dur
                        hp_drop = 1.0 - min_hp_local
                        score = balance * 0.3 + hp_drop * 0.7
                        if score > best_score:
                            best_split = real_idx
                            best_score = score
                    min_hp_local = current_min

        if best_split is not None:
            left = split_long_round(s_idx, best_split, depth + 1)
            right = split_long_round(best_split, e_idx, depth + 1)
            return left + right

        # Can't split — return as-is
        winner, hp_info = find_winner(p1_smooth, p2_smooth, s_idx, e_idx)
        return [(dur, winner, hp_info, start_t)]

    final_rounds = []
    for dur, winner, hp, start_t, s_idx, e_idx in rounds:
        if dur > 45:
            final_rounds.extend(split_long_round(s_idx, e_idx))
        else:
            final_rounds.append((dur, winner, hp, start_t))

    return final_rounds


def main():
    db_path = ROOT / "output" / "analysis.db"
    conn = sqlite3.connect(str(db_path))

    replays = conn.execute("""
        SELECT DISTINCT replay_id FROM frame_data
        WHERE replay_id NOT LIKE 'Replay_%'
        ORDER BY replay_id
    """).fetchall()

    for (replay_id,) in replays:
        rows = conn.execute("""
            SELECT frame_number, timestamp_ms, p1_health_pct, p2_health_pct, timer_value
            FROM frame_data WHERE replay_id = ? ORDER BY frame_number
        """, (replay_id,)).fetchall()

        rounds = detect_rounds(rows)
        total_dur = rows[-1][1] / 1000.0

        print(f"\n=== {replay_id} ({total_dur:.0f}s) === {len(rounds)} rounds")
        for i, (dur, winner, hp, start_t) in enumerate(rounds):
            flag = ""
            if dur < 15:
                flag = " [SHORT]"
            elif dur > 45:
                flag = " [LONG]"
            print(f"  R{i+1:2d}: {start_t:6.1f}s  {dur:5.1f}s  winner={winner}  HP={hp}{flag}")

    conn.close()


if __name__ == "__main__":
    main()
