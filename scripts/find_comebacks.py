"""Detect comeback rounds from frame-level health data.

A "comeback" = a player was at a large HP disadvantage during the round
but still won (opponent's health reached 0 first).

Usage:
    python scripts/find_comebacks.py output/sets/sets_analysis.db
    python scripts/find_comebacks.py output/sets/sets_analysis.db --min-deficit 30
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass

import numpy as np


# -- Health signal constants --------------------------------------------------
HP_CEILING = 0.875
SMOOTH_WINDOW = 31        # ~2s at 15fps effective rate

# -- Round detection ----------------------------------------------------------
MIN_ROUND_FRAMES = 30     # round must have at least 30 combat frames (~2s)
MAX_ROUND_FRAMES = 1600   # ~107s at 15fps


@dataclass
class RoundResult:
    replay_id: str
    round_start_ms: float
    round_end_ms: float
    winner: str
    winner_final_hp: float
    loser_final_hp: float
    winner_min_hp: float
    max_deficit: float
    deficit_timestamp_ms: float


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    half = window // 2
    n = len(arr)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        valid = arr[lo:hi][~np.isnan(arr[lo:hi])]
        if len(valid) >= 3:
            out[i] = np.median(valid)
    return out


def _find_round_boundaries(
    ts: np.ndarray,
    p1_raw_norm: np.ndarray,
    p2_raw_norm: np.ndarray,
) -> list[int]:
    """Find frame indices where round transitions occur.

    A round transition = SLASH!/DESTROYED animation where both health bars
    go near zero, followed by both bars recovering to near-full for the
    next round.  Works on raw (un-smoothed) normalized health values.

    Returns the index of the first both-near-zero frame for each transition.
    """
    BOTH_ZERO_THRESH = 0.08  # raw health below this = "near zero"
    RECOVERY_THRESH = 0.55   # both bars must recover above this
    RECOVERY_WINDOW = 120    # ~8s at 15fps — allow for long transitions
    MIN_ZERO_HOLD = 1        # at least 1 frame of both-near-zero

    boundaries: list[int] = []
    n = len(p1_raw_norm)
    i = 0

    while i < n:
        # Skip NaN frames
        if np.isnan(p1_raw_norm[i]) or np.isnan(p2_raw_norm[i]):
            i += 1
            continue

        # Look for both bars near zero
        if p1_raw_norm[i] < BOTH_ZERO_THRESH and p2_raw_norm[i] < BOTH_ZERO_THRESH:
            zero_start = i
            zero_count = 1

            # Count consecutive near-zero frames (tolerating NaN gaps)
            j = i + 1
            nan_gap = 0
            while j < n:
                if np.isnan(p1_raw_norm[j]) or np.isnan(p2_raw_norm[j]):
                    nan_gap += 1
                    if nan_gap > 30:
                        break
                    j += 1
                    continue
                nan_gap = 0
                if p1_raw_norm[j] < BOTH_ZERO_THRESH and p2_raw_norm[j] < BOTH_ZERO_THRESH:
                    zero_count += 1
                    j += 1
                else:
                    break

            if zero_count >= MIN_ZERO_HOLD:
                # Check for recovery: both bars > RECOVERY_THRESH within window
                look_end = min(n, j + RECOVERY_WINDOW)
                recovered = False
                for k in range(j, look_end):
                    if np.isnan(p1_raw_norm[k]) or np.isnan(p2_raw_norm[k]):
                        continue
                    if p1_raw_norm[k] > RECOVERY_THRESH and p2_raw_norm[k] > RECOVERY_THRESH:
                        recovered = True
                        break
                if recovered:
                    boundaries.append(zero_start)
                    i = k  # skip past recovery point
                    continue

            i = j
        else:
            i += 1

    return boundaries


def _pre_boundary_health(raw_norm: np.ndarray, boundary_idx: int) -> float:
    """Get raw health from the last few gameplay frames before a boundary.

    Walks backward from the boundary, skipping the initial SLASH! animation
    frames, and collects the last 5 valid readings.  Uses median of these.
    The narrow window avoids visual-effect corruption from earlier in the
    round while capturing the true end-of-round health state.
    """
    skip = 5    # skip frames closest to boundary (SLASH! start)
    want = 5    # collect this many valid frames
    collected: list[float] = []
    for i in range(boundary_idx - skip, max(0, boundary_idx - 60), -1):
        if np.isnan(raw_norm[i]):
            continue
        collected.append(raw_norm[i])
        if len(collected) >= want:
            break
    if len(collected) == 0:
        return 0.5
    return float(np.median(collected))


def _find_combat_start(
    p1_smooth: np.ndarray, p2_smooth: np.ndarray,
    boundary_idx: int,
) -> int:
    """Walk backward from a round boundary to find where combat started.

    Combat start = the frame where at least one player drops below 85%.
    Stops at max lookback of MAX_ROUND_FRAMES or if another "both full"
    region is found (= previous round boundary).
    """
    COMBAT_HP = 0.85
    FULL_HP = 0.90
    FULL_RUN_NEEDED = 5  # both above 90% for 5 frames = "both full"

    start = max(0, boundary_idx - MAX_ROUND_FRAMES)
    full_run = 0

    for i in range(boundary_idx - 1, start - 1, -1):
        if np.isnan(p1_smooth[i]) or np.isnan(p2_smooth[i]):
            full_run = 0
            continue
        if p1_smooth[i] > FULL_HP and p2_smooth[i] > FULL_HP:
            full_run += 1
            if full_run >= FULL_RUN_NEEDED:
                return i + full_run  # combat starts after the "both full" region
        else:
            full_run = 0

    return start


def _stable_hp(series: np.ndarray, idx: int, lookback: int = 15) -> float:
    """Get median of `lookback` valid values before `idx`."""
    lo = max(0, idx - lookback)
    chunk = series[lo:idx]
    valid = chunk[~np.isnan(chunk)]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def detect_rounds(conn: sqlite3.Connection, replay_id: str) -> list[RoundResult]:
    rows = conn.execute("""
        SELECT timestamp_ms, p1_health_pct, p2_health_pct
        FROM frame_data
        WHERE replay_id = ? AND p1_health_pct IS NOT NULL AND p2_health_pct IS NOT NULL
        ORDER BY timestamp_ms
    """, (replay_id,)).fetchall()

    if len(rows) < SMOOTH_WINDOW:
        return []

    ts = np.array([r[0] for r in rows])
    p1_raw = np.array([r[1] for r in rows])
    p2_raw = np.array([r[2] for r in rows])

    # Smooth + normalise
    p1 = np.clip(_rolling_median(p1_raw, SMOOTH_WINDOW) / HP_CEILING, 0.0, 1.0)
    p2 = np.clip(_rolling_median(p2_raw, SMOOTH_WINDOW) / HP_CEILING, 0.0, 1.0)
    p1_raw_norm = np.clip(p1_raw / HP_CEILING, 0.0, 1.0)
    p2_raw_norm = np.clip(p2_raw / HP_CEILING, 0.0, 1.0)

    # Find round boundaries (SLASH!/DESTROYED transitions)
    boundaries = _find_round_boundaries(ts, p1_raw_norm, p2_raw_norm)

    results: list[RoundResult] = []
    prev_boundary = 0  # floor for combat_start to prevent overlap

    for boundary_idx in boundaries:
        # Determine winner: player with higher raw health before boundary
        p1_pre = _pre_boundary_health(p1_raw_norm, boundary_idx)
        p2_pre = _pre_boundary_health(p2_raw_norm, boundary_idx)
        winner = "P1" if p1_pre > p2_pre else "P2"

        # Find combat start by walking backward, floored at previous boundary
        combat_start = max(_find_combat_start(p1, p2, boundary_idx), prev_boundary)
        prev_boundary = boundary_idx + 1
        round_len = boundary_idx - combat_start
        if round_len < MIN_ROUND_FRAMES or round_len > MAX_ROUND_FRAMES:
            continue

        # Compute round timestamps
        round_start_ms = float(ts[combat_start])
        round_end_ms = float(ts[boundary_idx])

        # Compute comeback metrics over the combat region
        body = slice(combat_start, boundary_idx)
        w_series = p2[body] if winner == "P2" else p1[body]
        l_series = p1[body] if winner == "P2" else p2[body]
        body_ts = ts[body]

        valid = ~np.isnan(w_series) & ~np.isnan(l_series)
        if not np.any(valid):
            continue

        w_valid = w_series[valid]
        l_valid = l_series[valid]
        ts_valid = body_ts[valid]

        winner_min_hp = float(np.min(w_valid))
        deficits = l_valid - w_valid
        max_def_idx = int(np.argmax(deficits))
        max_deficit = float(max(deficits[max_def_idx], 0.0))
        deficit_ts = float(ts_valid[max_def_idx])

        # Winner's HP just before the round ends
        w_all = p2 if winner == "P2" else p1
        winner_final = _stable_hp(w_all, boundary_idx)

        results.append(RoundResult(
            replay_id=replay_id,
            round_start_ms=round_start_ms,
            round_end_ms=round_end_ms,
            winner=winner,
            winner_final_hp=winner_final,
            loser_final_hp=0.0,
            winner_min_hp=winner_min_hp,
            max_deficit=max_deficit,
            deficit_timestamp_ms=deficit_ts,
        ))

    return results


def fmt_ts(ms: float) -> str:
    s = ms / 1000
    return f"{int(s//60)}:{s%60:04.1f}"


def main():
    parser = argparse.ArgumentParser(description="Find comeback rounds")
    parser.add_argument("db", help="Path to analysis SQLite database")
    parser.add_argument("--min-deficit", type=float, default=20.0,
                        help="Minimum HP deficit %% the winner overcame (default: 20)")
    parser.add_argument("--top", type=int, default=15,
                        help="Show top N comebacks (default: 15)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    replays = [r[0] for r in conn.execute("SELECT replay_id FROM replays ORDER BY replay_id").fetchall()]

    all_rounds: list[RoundResult] = []
    for replay_id in replays:
        rounds = detect_rounds(conn, replay_id)
        all_rounds.extend(rounds)

    conn.close()

    print(f"Detected {len(all_rounds)} rounds across {len(replays)} sets\n")

    # Filter to comebacks
    comebacks = [r for r in all_rounds if r.max_deficit >= args.min_deficit / 100]
    comebacks.sort(key=lambda r: r.max_deficit, reverse=True)

    if not comebacks:
        print(f"No comebacks found with deficit >= {args.min_deficit}%")
        return

    print(f"Found {len(comebacks)} comeback rounds (deficit >= {args.min_deficit}%)\n")
    print(f"{'#':<3} {'Set':<40} {'Time':>10} {'Winner':>6} {'Deficit':>8} {'Win HP':>7} {'Min HP':>7} {'Dur':>5}")
    print("-" * 90)

    for i, r in enumerate(comebacks[:args.top], 1):
        set_name = r.replay_id[:37]
        round_time = fmt_ts(r.round_end_ms)
        duration_s = (r.round_end_ms - r.round_start_ms) / 1000

        print(f"{i:<3} {set_name:<40} {round_time:>10} {r.winner:>6} {r.max_deficit:>7.1%} {r.winner_final_hp:>6.1%} {r.winner_min_hp:>6.1%} {duration_s:>4.0f}s")
        print(f"    Round: {fmt_ts(r.round_start_ms)}-{round_time}, worst deficit at {fmt_ts(r.deficit_timestamp_ms)}")
        print()

    # Overall stats
    print("=" * 90)
    p1_wins = sum(1 for r in all_rounds if r.winner == "P1")
    p2_wins = sum(1 for r in all_rounds if r.winner == "P2")
    avg_hp = sum(r.winner_final_hp for r in all_rounds) / len(all_rounds) if all_rounds else 0
    close = sum(1 for r in all_rounds if r.winner_final_hp < 0.20)
    durations = [(r.round_end_ms - r.round_start_ms) / 1000 for r in all_rounds]
    print(f"Round statistics across all sets:")
    print(f"  Total rounds: {len(all_rounds)} (P1 wins: {p1_wins}, P2 wins: {p2_wins})")
    print(f"  Avg winner final HP: {avg_hp:.1%}")
    print(f"  Avg round duration: {np.mean(durations):.0f}s (min {np.min(durations):.0f}s, max {np.max(durations):.0f}s)")
    print(f"  Close rounds (winner < 20% HP): {close}")
    print(f"  Comeback rounds (deficit >= {args.min_deficit}%): {len(comebacks)}")


if __name__ == "__main__":
    main()
