# Rust Round Detection & Winner Algorithm

Round detection and winner determination runs entirely in the Rust/Tauri backend (`app/src-tauri/src/lib.rs`), operating on the per-frame data stored by the Python CV pipeline. The algorithm is implemented in `detect_rounds_for_replay()` and called on-demand by the `get_rounds`, `get_match_stats`, and `get_highlights` Tauri commands.

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `HP_CEILING` | 0.875 | Health normalization ceiling. Raw health values from the CV pipeline are divided by this to get 0.0-1.0 range. Accounts for the fact that health bars in GGS top out at ~87.5% of the pixel range. |
| `MIN_ROUND_FRAMES` | 450 | ~15 seconds at 30fps. Minimum frame count for a valid round. Also used as minimum gap between detected boundaries. |
| `COMEBACK_DEFICIT` | 0.35 | Max deficit threshold for comeback detection. A round where the winner was behind by >= 35% at any point is flagged as a comeback. |
| `DEFICIT_SMOOTH_WINDOW` | 91 | ~3 seconds. Heavy smoothing window for deficit/comeback tracking. Wider than the standard smoothing to suppress HP extraction noise that creates false deficits. |
| `TIMER_SMOOTH_WINDOW` | 61 | ~2 seconds. Rolling median window for timer noise filtering. Applied to forward-filled timer values. |
| `HP_RESET_WINDOW` | 90 | ~3 seconds. Rolling median window for HP reset detection. |

---

## Signal Processing Helpers

### rolling_median(arr, window)

Rolling median filter for f64 arrays. Handles NaN values by filtering them out of each window. Requires at least 1/3 of the window to contain valid data, otherwise outputs NaN.

### forward_fill_i32(arr)

Carries the last known `Some(value)` forward through `None` gaps. Used for timer and round counter data where OCR produces sparse readings.

### rolling_median_i32(arr, window)

Rolling median for `Option<i32>` values. Outputs `None` when no values exist in the window.

---

## Data Loading & Normalization

1. Read all `frame_data` rows for the replay, ordered by `timestamp_ms`
2. Handle schema variations (with/without timer column, with/without round counter columns)
3. Normalize HP by dividing by `HP_CEILING` and clamping to [0.0, 1.0]

---

## Data Cleaning Passes

### Pass 1: Junk Frame Detection

**Purpose:** Remove lobby/menu screens misidentified as gameplay.

- Compute rolling median of both players' HP with a 150-frame window (~5s)
- If both players' rolling median HP stays below 0.15 for sustained periods, set both to NaN
- Threshold is intentionally low (0.15) to avoid false positives during legitimate close rounds

### Pass 2: Non-Gameplay Detection (No Timer)

**Purpose:** Remove character select, lobby, replay browser, and loading screens.

- The round timer is the most reliable gameplay indicator
- Scan for runs of frames with no timer reading (`timer_raw[i] == None`)
- Runs of 600+ frames (~20 seconds) are classified as non-gameplay
- All HP data in these regions is set to NaN
- These regions become gaps that the game-break detector handles as boundaries

---

## HP Smoothing

Three smoothing passes are computed for different purposes:

| Variable | Window | Purpose |
|----------|--------|---------|
| `p1_smooth / p2_smooth` | 31 (~1s) | Round metrics and winner detection |
| `p1_heavy / p2_heavy` | 91 (~3s) | Deficit/comeback tracking |
| `p1_rm / p2_rm` | 90 (~3s) | HP reset detection |

---

## Four Boundary Signals

### Signal 1: Timer Jumps

Detects new rounds by watching for timer resets.

**Preprocessing:**
1. Forward-fill timer values (carry last known value through gaps)
2. Apply 61-frame rolling median to filter OCR noise

**Detection:**
- Track `lowest_since_start` (lowest smoothed timer seen since last boundary)
- When smoothed timer reaches >= 88 AND `lowest_since_start` was <= 70, a potential boundary is detected
- Must be >= `MIN_ROUND_FRAMES` from the last detected boundary
- **Confirmation required:** Smoothed timer must reach >= 90 within the next 90 frames. This filters false positives from wallbreak/super flash OCR noise that briefly reads ~88 but never reaches 90.

### Signal 2: HP Resets

Detects rounds where both players' health returns to full after being low.

**Detection:**
- Use 90-frame rolling median HP for both players
- Track `min_hp_since_reset` (minimum of both players' rolling HP since last reset)
- When both `p1_rm > 0.85` AND `p2_rm > 0.85` AND `min_hp_since_reset < 0.50`, a new round has started
- Must be >= `MIN_ROUND_FRAMES` from the last HP reset

### Signal 2b: Gap-based HP Resets

Detects round resets across NaN gaps that Signal 2 misses (the rolling median can't bridge gaps).

**Detection:**
- Track last frame where both players had valid HP data and their values
- When a gap of >= 30 frames occurs AND at least one player was low (< 0.50) before the gap AND both players are high (> 0.85) after the gap, it's a new round
- For short gaps (< 60 frames / ~2s), additionally require at least one player to have been near-KO (< 0.25) before the gap. This filters wallbreaks where moderate-HP players have a brief NaN gap from stage transition VFX.

### Signal 3: Game Breaks (Gaps > 5s)

Detects character select / game boundaries from large data gaps.

- Track the previous frame index where either player had valid HP data
- When the gap between consecutive valid frames exceeds 5000ms (5 seconds), a game break is recorded
- Game breaks are unambiguous boundaries and are exempt from the wallbreak timer filter

### Signal 4: Match Boundaries (Timer >= 97 + Both 0 Wins)

Detects the start of a new match (game) within a set.

**Detection:**
- Forward-fill round counter data for both players
- Use smoothed timer values
- When `timer >= 97` AND `p1_rounds_won == 0` AND `p2_rounds_won == 0` AND either player previously had wins, a new match has started
- Must be >= `MIN_ROUND_FRAMES` from the last match boundary

---

## Boundary Merging Pipeline

### 1. Collect All Boundaries

All four signal types are merged into a single sorted, deduplicated list of frame indices.

### 2. Wallbreak Filter

Applied BEFORE deduplication to prevent false boundaries from shadowing valid ones.

**Rule:** A boundary must have smoothed timer >= 90 within 90 frames of the boundary index.

**Exemption:** Game breaks (Signal 3) and gap HP resets (Signal 2b) are always kept, since large data gaps are unambiguous boundaries regardless of timer state.

### 3. Deduplication (10-Second Window)

Boundaries within 10 seconds of each other (by timestamp) are merged, keeping the earliest.

### 4. First-Valid Insertion

If the first valid data frame is more than `MIN_ROUND_FRAMES` before the first boundary, it is inserted as a boundary (the replay may start mid-round).

### 5. Match Start Mapping

Each deduplicated boundary is checked against the match boundary set: a boundary is a match start if any Signal 4 boundary is within ~10 seconds (300 frame indices).

---

## Round Span Construction

### Initial Spans

From consecutive boundary pairs, create spans with:
- `s_idx`: start frame index
- `e_idx`: end frame index (next boundary or end of data)
- `is_match_start`: inherited from match boundary mapping

**Filtering:**
- Spans shorter than `MIN_ROUND_FRAMES` (450 frames, ~15s) are dropped
- Spans with less than 30% valid data ratio (frames where at least one player has HP) are dropped

### Long Round Splitting

Rounds longer than 45 seconds are recursively split using `split_long_round()`:

**Strategy 1: Data Gap Split**
- Find the largest internal gap > 1.5s
- Both sub-spans must be >= 12s
- Split point must be confirmed by timer (>= 90 within 210 frames) or round reset evidence

**Strategy 2: HP Reset Split**
- Compute local rolling median HP within the long round
- Find where both players' HP rises above 0.75 after being below 0.55
- Score candidates by balance (even sub-span durations) and HP drop magnitude
- Timer confirmation required (>= 90 nearby)

**Round reset confirmation:**
- KO evidence before the gap: >= 3 frames where either player's HP < 0.10 in the preceding 15s
- Timer >= 90 within 120 frames after the split
- Both smoothed HP > 0.90 within 300 frames after the split

Recursion depth limit: 3

### Trimming

Each sub-span is trimmed to the first/last frame where both players have valid HP data. Sub-spans shorter than 12 seconds after trimming are discarded.

---

## find_winner_ko Closure

The winner determination algorithm for each round span. Returns `(winner: String, p1_hp: f64, p2_hp: f64, confident: bool)`.

### Step 1: Wallbreak Detection Within Rounds

Scans the post-40% portion of the round for mid-round wallbreaks:
- Looks for frames where both players' HP resets to > 0.85 after one was < 0.50
- **KO rejection:** If either player was near-dead (< 0.02) in the preceding 180 frames, the "reset" is a post-KO animation artifact, not a wallbreak
- **Post-valid requirement:** At least 20 valid frames must exist after the reset (real wallbreaks have continued gameplay; KO animations go to NaN quickly)
- If a valid wallbreak is found, `effective_search_start` advances past it, excluding pre-wallbreak damage from KO detection

### Step 2: KO Counting

Count frames where each player's HP < 0.15 while the other player's HP < 0.90 (filtering post-KO garbage where the winner's bar has reset to ~1.0):

- If `p1_ko_count >= 3` AND `p1_ko_count >= p2_ko_count * 3 + 1` -> P2 wins (P1 was KO'd)
- If `p2_ko_count >= 3` AND `p2_ko_count >= p1_ko_count * 3 + 1` -> P1 wins (P2 was KO'd)

### Step 3: Dual KO Handling

When both players have >= 3 KO frames but neither dominates:
- Find the first frame where each player drops below 0.05
- If they're within 90 frames (~3s) of each other, use first-to-zero logic:
  - The earlier zero is checked: if the other player has > 0.15 HP at that moment, the earlier player died
  - If both are near-KO at the earlier moment, check the later zero instead

### Step 4: Cluster Analysis

Collect all frames where `min(p1, p2)` is within 0.10 of the best (lowest) minimum HP. Split into temporal clusters separated by > 60 frames (~2s). Use the **last cluster** (closest to round end = actual KO).

From the last cluster, take up to 30 frames from the end. Compute median P1 and P2 HP values.

If HP difference >= 0.15 -> clear winner (higher HP wins). **Confident.**

### Step 5: Min-HP Fallback with Weighted Average

When the cluster analysis is ambiguous (HP diff < 0.15):

**Signal A: Minimum HP**
- Find each player's minimum smoothed HP post-wallbreak
- `min_signal = p2_min - p1_min` (positive = P1 went lower = P2 wins)
- Transition artifact filter: if both min values are in [0.40, 0.55] with < 0.05 difference, zero out the signal

**Signal B: Average HP**
- Compute average smoothed HP in the last 25% of the round
- Exclude frames where both players' HP < 0.60 (loading screen pattern)
- Fall back to last valid HP readings, then KO moment values

**Combined signal:**
- Dynamic weighting based on HP range: `min_weight = min(hp_range / 0.20, 0.8)`, `avg_weight = 1.0 - min_weight`
- `combined = min_signal * min_weight + avg_signal * avg_weight`
- **Confident** if `|combined| >= 0.10`

### Step 6: Post-Round Tiebreaker

When `|combined| < 0.10` (very ambiguous):
- Scan 300 frames (10s) after the round end
- Count visible (non-NaN) frames for each player
- In GGS, the winner's HP bar stays visible on the result screen while the loser's disappears
- If one player has 10+ more visible frames -> that player is likely the winner
- **NOT marked as confident** since next-round starts can pollute this signal

### Confidence Tracking

`winner_confident` is `true` when:
- KO counting produced a clear winner
- Cluster HP difference >= 0.15
- Combined fallback signal >= 0.10

`winner_confident` is `false` when:
- The winner was determined by the post-round tiebreaker
- The combined signal was too weak

---

## Hearts-Based Winner Override

After all rounds are determined by HP analysis, round counter hearts provide a ground-truth correction.

### Why This is Needed

GGS pending combo damage appears as a lighter bar segment that warm pixel detection still counts as health. This causes HP-based winner detection to misidentify winners when a lethal combo kills a player who appeared to have more health.

### Algorithm

Uses **RAW** round counter data (not forward-filled or smoothed) to avoid propagating sparse noisy spikes (e.g., lobby UI reading as `p1rw=1`).

For each round:
1. Compute hearts **before** the round: mode of raw `p1_rounds_won` / `p2_rounds_won` in 150 frames preceding round start
2. Compute hearts **after** the round: mode in the window between round end and next round start
3. **Confidence threshold:** The mode value must account for >= 40% of valid frames in the window (filters noisy OCR from lobby/transition screens)
4. Compare pre vs. post:
   - If P1's wins increased but P2's didn't -> P1 won this round
   - If P2's wins increased but P1's didn't -> P2 won this round
   - If this disagrees with the HP-based winner, **override** the winner and swap `winner_final_hp` / `loser_final_hp`
5. For match-start rounds (both pre-hearts are 0), compare post-hearts against 0 instead of pre-values

---

## Uncertain Winner System

### winner_confident Field

Each `RoundResult` has a `winner_confident: bool` field. Set to `false` when the winner was determined by the post-round tiebreaker or when the combined HP signal was too weak.

### winner_overrides SQLite Table

```sql
CREATE TABLE IF NOT EXISTS winner_overrides (
    replay_id   TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    winner      TEXT NOT NULL,
    PRIMARY KEY (replay_id, round_index)
);
```

The `set_round_winner` Tauri command inserts/replaces entries in this table.

### Applied at Read-Time

In the `get_rounds` command:
1. Detect rounds via `detect_rounds_for_replay()`
2. Query `winner_overrides` for the replay
3. For each override, find the matching round by `round_index`, swap winner/loser HP if needed, set `winner_confident = true`

This means overrides are non-destructive and survive reanalysis.

---

## Comeback Detection

For each round, the maximum deficit is computed:
- Use heavy-smoothed HP (91-frame window) for both winner and loser
- Skip the first 15% of the round (avoids HP reset transition artifacts)
- `deficit = loser_heavy_hp - winner_heavy_hp` at each frame
- `max_deficit` is the maximum deficit observed
- `is_comeback = max_deficit >= COMEBACK_DEFICIT (0.35)`

The `deficit_timestamp_ms` records when the maximum deficit occurred.

---

## Highlight Detection

Generated by `generate_highlights()` from rounds and damage events. Sorted by severity (descending).

| Kind | Condition | Severity |
|------|-----------|----------|
| `comeback` | `is_comeback == true` | `max_deficit` (0.35-1.0) |
| `close_round` | `winner_final_hp < 0.20` AND not a comeback | `1.0 - winner_final_hp` |
| `big_damage` | `damage_pct > 0.30` | `damage_pct` |
| `perfect` | `winner_final_hp > 0.90` | `winner_final_hp` |

---

## Match Stats Computation

`compute_match_stats()` aggregates across all rounds and damage events:
- Round win counts per player
- Damage taken per player (sum of all `damage_pct` targeting that player)
- Biggest single hit per player
- Round duration statistics (average, longest, shortest)
- Comeback count, close round count
- Average winner final HP
- Total set duration
