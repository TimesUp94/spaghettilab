# Rust Round Detection & Winner Algorithm

Round detection and winner determination runs entirely in the Rust/Tauri backend (`app/src-tauri/src/lib.rs`), operating on the per-frame data stored by the Python CV pipeline. The algorithm is implemented in `detect_rounds_for_replay()` and called on-demand by the `get_rounds`, `get_match_stats`, and `get_highlights` Tauri commands.

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `KO_HP` | 0.005 | HP at/below this is a KO (see Step 1b). |
| `NEAR_KO_HP` | 0.13125 | A side at/below this is near a KO; a KO with the other side above it is one-sided. |
| `UNCERTAIN_HP` | 0.0875 | Below this HP margin an HP-based winner is uncertain. |
| `MIN_ROUND_FRAMES` | 450 | ~15 seconds at 30fps. Minimum frame count for a valid round. Also used as minimum gap between detected boundaries. |
| `LEAD_PARTIAL_MIN_FRAMES` | 240 | ~8 seconds. Minimum length of a leading partial round that ends in a KO (see below). |
| `COMEBACK_DEFICIT` | 0.30 | Max deficit threshold for comeback detection. A round where the winner was behind by >= 30% of the bar at any point is flagged as a comeback. |
| `DEFICIT_SMOOTH_WINDOW` | 91 | ~3 seconds. Heavy smoothing window for deficit/comeback tracking. Wider than the standard smoothing to suppress HP extraction noise that creates false deficits. |
| `TIMER_SMOOTH_WINDOW` | 61 | ~2 seconds. Rolling median window for timer noise filtering. Applied to forward-filled timer values. |
| `HP_RESET_WINDOW` | 90 | ~3 seconds. Rolling median window for HP reset detection. |

**Sample rate:** all sample-count windows (the constants above and the literal windows below)
are written for 30 samples per second, which is what the pipeline produces (it reads every
`round(fps/30)`-th video frame). `sample_rate_scale(ts)` measures the rate from the median
timestamp step and `scale_frames` converts each window, so data sampled at another rate keeps
the same windows in seconds. The scale is exactly 1.0 within 5 % of 30/s, so 30/s data gives
the same result as before. (Checked by duplicating every row of the harness data to 60/s and
dropping every other row to 15/s: 159 rounds -> 160 / 158, against 162 / 155 without scaling.)

---

## Signal Processing Helpers

### rolling_median(arr, window)

Rolling median filter for f64 arrays. Handles NaN values by filtering them out of each window. Requires at least 1/3 of the window to contain valid data, otherwise outputs NaN.

### forward_fill_i32(arr)

Carries the last known `Some(value)` forward through `None` gaps. Used for timer and round counter data where OCR produces sparse readings.

### rolling_median_i32(arr, window)

Rolling median for `Option<i32>` values. Outputs `None` when no values exist in the window.

---

## Data Loading & HP Scale

1. Read all `frame_data` rows for the replay, ordered by `timestamp_ms`
2. Handle schema variations (with/without timer column, with/without round counter columns)
3. Clamp HP to [0.0, 1.0]

### HP scale

`p1_health_pct` / `p2_health_pct` are the filled fraction of the bar: `health.py` measures
fill/total, so a full bar is 1.0 and a KO is 0. Every HP value and threshold in this document, in
`RoundResult` (`winner_final_hp`, `loser_final_hp`, `winner_min_hp`, `max_deficit`) and in the UI
is on that scale.

Until 2026-10-22 the Rust code (and `HealthTimeline.tsx`) divided HP by `HP_CEILING = 0.875` a
second time, a leftover from an older reader whose full bar read 0.875. Every HP below the top
showed x1.143 too high and anything >= 0.875 showed as 100 %. When that was removed, each
detection threshold that had been tuned in the inflated space was converted to the same HP on
the bar (old value x 0.875: 0.15 -> 0.13125, 0.10 -> 0.0875, 0.85 -> 0.74375, 0.50 -> 0.4375,
0.90 -> 0.7875, ...), so the detection behaves exactly as before on the same data (checked on
Tappa #4, its blacked-out copy, the 720p test VOD and the three YouTube replays: no boundary,
winner, confidence or match-start change; HP values are x0.875 except values that used to clip
at 1.0). The very small KO bounds (`KO_HP` 0.005, the 0.02 / 0.01 "at zero" checks) were already
meant as raw HP and did not change. The user-facing labels keep their stated values, now on the
real scale: close round < 0.20 and near-perfect / DOMINANT > 0.90 of the bar, CLOSE badge < 0.10.

**Older data** (the March 2026 builds, and `.spag`/`.spagz` files made with them) really reads a
full bar at 0.875. It is recognised per replay (`replay_hp_full`) from where the full bars pile
up, since every round starts full: readings within 0.01 of 0.875 outnumbering readings at
>= 0.985 by more than 2x (and at least 30 of them) mean the old scale, and HP read from
`frame_data` and `damage_events` is divided by 0.875 (round detection, `get_frame_data`, the
damage events of `get_damage_events` / `get_match_stats` / `get_highlights`). Measured ratio:
6.3-10.6 on the old-scale databases in `data/`, at most 0.63 on all current data (Tappa #4, the
720p test VOD, the YouTube replays). No schema change, so files keep round-tripping with the
web app; whether the web app (FGCScoreboard) still divides by 0.875 has not been checked.
`COMEBACK_DEFICIT` became 0.30 (0.35 x 0.875); the clipping at 1.0 used to under-report a deficit
while the leading side was above 0.875, so 5 more rounds on Tappa #4 (3 on the test VOD, 1 on
YouTube) are now comebacks with real deficits of 0.31-0.41 (one more on Tappa #4, 12_LSemi R2 at
0.33, after the round-end change below moved where the first 15 % of the round ends).

All current data (the live DB, `.spag`/`.spagz` files made after the re-analysis of 2026-10-20)
comes from the fill/total reader, so there is no migration or version flag: older files open
unchanged and are shown on the real scale.

---

## Data Cleaning Passes

### Pass 1: Junk Frame Detection

**Purpose:** Remove lobby/menu screens misidentified as gameplay.

- Compute rolling median of both players' HP with a 150-frame window (~5s)
- If both players' rolling median HP stays below 0.13125 for sustained periods, set both to NaN
- Threshold is intentionally low (0.13125) to avoid false positives during legitimate close rounds
- Frames with a round-timer reading from 150 frames before to 75 frames after are kept: both players can sit near KO for seconds, and the last combo and KO often play with the timer hidden by a sprite or super (set 11 R13)

### Pass 2: Non-Gameplay Detection (No Timer)

**Purpose:** Remove character select, lobby, replay browser, and loading screens.

- The round timer is the most reliable gameplay indicator
- Scan for runs of frames with no timer reading (`timer_raw[i] == None`)
- Runs of 600+ frames (~20 seconds) are classified as non-gameplay
- All HP data in these regions is set to NaN
- These regions become gaps that the game-break detector handles as boundaries
- A shorter stretch without a timer (e.g. the ~20 s lobby after a file's last round) is not
  NaN-ed, but no round runs into it: each round ends at most `ROUND_KO_MARGIN` after its last
  timer reading (see Initial Spans)
- A KO in the first `ROUND_KO_MARGIN` samples of a region is kept (the samples up to it), when
  the round before it was not decided yet: the last round of a set can end in a KO under the
  slash after its last timer reading with the HUD never shown again (720p test VOD set 10 R9:
  P1 at a 0.015 sliver, KO'd at 531.2 s, the last reading at 528.3 s, then 43 s without a timer;
  the round used to end at 528.4 s with the loser at 0.015).

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
1. Drop unsupported start readings (`supported_timer`): a reading >= 88 needs at least 2 other
   readings within ±1 s of the same or an adjacent value. The timer shows each value for about
   a second; a single misread "97" on a super flash, forward-filled across the HUD gap after it,
   made a fake round start (yt_KWXz7dmt2Pk 575 s). Lower readings are kept as they are (an
   isolated low reading cannot start a round). Junk and non-gameplay detection still use the
   raw readings.
2. Forward-fill timer values (carry last known value through gaps)
3. Apply 61-frame rolling median to filter OCR noise

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
- When both `p1_rm > 0.74375` AND `p2_rm > 0.74375` AND `min_hp_since_reset < 0.4375`, a new round has started
- Must be >= `MIN_ROUND_FRAMES` from the last HP reset

### Signal 2b: Gap-based HP Resets

Detects round resets across NaN gaps that Signal 2 misses (the rolling median can't bridge gaps).

**Detection:**
- Track last frame where both players had valid HP data and their values
- When a gap of >= 30 frames occurs AND at least one player was low (< 0.4375) before the gap AND both players are high (> 0.74375) after the gap, it's a new round
- For short gaps (< 60 frames / ~2s), additionally require at least one player to have been near-KO (< 0.21875) before the gap. This filters wallbreaks where moderate-HP players have a brief NaN gap from stage transition VFX.

### Signal 3: Game Breaks (Gaps > 5s)

Detects character select / game boundaries from large data gaps.

- Track the previous frame index where either player had valid HP data
- When the gap between consecutive valid frames exceeds 5000ms (5 seconds), a game break is recorded
- Unless the round clock carries on across it (`clock_continues`): a super cinematic can hide
  the whole HUD mid-round for more than 5 s (Tappa #4 14_Grands R6: 5.6 s with the clock at 84
  on both sides). The nearest supported readings (2 more of the same value within +-1 s) within
  3 s before and after the gap must show an in-round value after it (<= 88) no higher than
  before it; a new round starts at its intro's 99. Mirrors the cleaner's `_clock_continues`.
- Game breaks are unambiguous boundaries and are exempt from the wallbreak timer filter

### (Former Signal 4: match boundaries)

Match starts no longer create boundaries: a hearts reading alone must never split a round. They
are set per round from the hearts once the rounds are known (see Round-Counter Hearts below).

---

## Boundary Merging Pipeline

### 1. Collect All Boundaries

The timer, HP-reset, gap-reset and game-break signals are merged into a single sorted, deduplicated list of frame indices.

### 2. Wallbreak Filter

Applied BEFORE deduplication to prevent false boundaries from shadowing valid ones.

**Rule:** A boundary must have smoothed timer >= 90 within 90 frames of the boundary index.

**Exemption:** Game breaks (Signal 3) and gap HP resets (Signal 2b) are always kept, since large data gaps are unambiguous boundaries regardless of timer state.

### 3. Deduplication (10-Second Window)

Boundaries within 10 seconds of each other (by timestamp) are merged, keeping the earliest.

### 4. First-Valid Insertion

If the first valid data frame is more than `MIN_ROUND_FRAMES` before the first boundary, it is inserted as a boundary (the replay may start mid-round).

**Leading partial round:** a video can open mid-round, or with its first round start unreadable (e.g. the stream shows the game scaled down for the first seconds, so no timer is read and that stretch counts as non-gameplay). When the span from the first valid frame to the first boundary is shorter than `MIN_ROUND_FRAMES` but at least `LEAD_PARTIAL_MIN_FRAMES` (240, ~8 s), it is still kept as round 0 if it ends in a KO: the *first* frame where one side's cleaned HP is <= `KO_HP` (0.005) while the other is > 0.02 must lie in the last 40% of the span and at least ~8 s after its start. Such a span also bypasses the 15 s span minimum and the 12 s sub-span minimum (8 s instead). A video that opens on the last seconds of a round (KO and win pose only, e.g. `03_UF` at 0-5 s) is not kept.

---

## Round Span Construction

### Initial Spans

From consecutive boundary pairs, create spans with:
- `s_idx`: start frame index
- `e_idx`: end frame index (next boundary or end of data)
- `is_match_start`: always set later, per round, from the hearts (see Round-Counter Hearts)

Each span ends at most `ROUND_KO_MARGIN` (150 frames, 5 s) after its last timer reading: the HUD
(timer and bars) goes away at the slash after the KO and stays hidden through the win pose,
loading screen and lobby, but the cleaned HP can be carried into them, and the lobby after a
file's last round is often just under the 20 s non-gameplay threshold (the last round of 720p
test VOD sets 03, 04, 06 and 10 ran 15-20 s into the lobby, as did the last round of 6 files of the
blacked-out Tappa #4 copy; on Tappa #4 itself 33 round ends moved 1-14 s earlier, none past its
KO, with no winner change). The margin covers a KO
that lands after the last reading because a super cinematic hid the timer (measured up to 3.3 s;
a hidden-HUD super can last 5.6 s, but the cleaner places a KO it did not see right after the
last timer reading before the HUD went, so the loser is at 0 inside the margin);
the stopped clock after a KO is still read, so a normal KO is inside it. Within that, the span's
end is trimmed to 30 frames after its last HP reading: the loading screen, or the rest of a stream after the last set, is not part of the round (set 07's final round was dropped by the valid-ratio check below because 255 s of stream followed it). A short trailing block of readings (under 2 s, no timer reading) after a HUD gap of 3 s or more is ignored when finding that last reading: it is the next round's intro, whose bars read before the timer appears (set 09_UF R11 ran 6.6 s into the "Duel 1" screen).

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
- Find where both players' HP rises above 0.65625 after being below 0.48125
- Score candidates by balance (even sub-span durations) and HP drop magnitude
- Timer confirmation required (>= 90 nearby)

**Round reset confirmation:**
- KO evidence before the gap: >= 3 frames where either player's HP < 0.0875 in the preceding 15s
- Timer >= 90 within 120 frames after the split
- Both smoothed HP > 0.7875 within 300 frames after the split

Recursion depth limit: 3

### Trimming

Each sub-span is trimmed to the first/last frame where both players have valid HP data. Sub-spans shorter than 12 seconds after trimming are discarded.

---

## find_winner_ko Closure

The winner determination algorithm for each round span. Returns `(winner: String, p1_hp: f64, p2_hp: f64, confident: bool)`.

### Step 1: Wallbreak Detection Within Rounds

Scans the post-40% portion of the round for mid-round wallbreaks:
- Looks for frames where both players' HP resets to > 0.74375 after one was < 0.4375
- **KO rejection:** If either player was near-dead (< 0.02) in the preceding 180 frames, the "reset" is a post-KO animation artifact, not a wallbreak
- **Post-valid requirement:** At least 20 valid frames must exist after the reset (real wallbreaks have continued gameplay; KO animations go to NaN quickly)
- If a valid wallbreak is found, `effective_search_start` advances past it, excluding pre-wallbreak damage from KO detection

### Step 1b: Terminal State

The cleaned series (see `docs/cv-pipeline.md`) is non-increasing within a round and reads a KO as exactly 0, so each side's minimum over the search window is its final HP:
- Exactly one side at <= `KO_HP` (0.005) -> the other side wins (confident)
- Neither at <= 0.005 and the final HPs differ by >= 0.02625 -> the lower side loses (confident if the difference is >= `UNCERTAIN_HP`, 0.0875)

The KO bound is 0.005 because a live 1-px sliver reads 0.008 on a 1080p source: with 0.01,
14_Grands R15 (P1 survives on a sliver, P2's bar empty with the clock stopped at 39, the KO
slash over Leo) counted as a double KO and the KO-count fallback gave it to P2.
- Otherwise (both at 0, or too close) fall through to the steps below

### Step 2: KO Counting

Count frames where each player's HP < `NEAR_KO_HP` (0.13125) while the other player's HP < 0.7875 (filtering post-KO garbage where the winner's bar has reset to ~1.0):

- If `p1_ko_count >= 3` AND `p1_ko_count >= p2_ko_count * 3 + 1` -> P2 wins (P1 was KO'd)
- If `p2_ko_count >= 3` AND `p2_ko_count >= p1_ko_count * 3 + 1` -> P1 wins (P2 was KO'd)

Steps 2 and 3 report the winner's HP at the loser's first frame at <= 0.02 and the loser's minimum (they used to report a hard-coded 1.0 / 0.0).

### Step 3: Dual KO Handling

When both players have >= 3 KO frames but neither dominates:
- Find the first frame where each player drops below 0.04375
- If they're within 90 frames (~3s) of each other, use first-to-zero logic:
  - The earlier zero is checked: if the other player has > `NEAR_KO_HP` at that moment, the earlier player died
  - If both are near-KO at the earlier moment, check the later zero instead

### Step 4: Cluster Analysis

Collect all frames where `min(p1, p2)` is within 0.0875 of the best (lowest) minimum HP. Split into temporal clusters separated by > 60 frames (~2s). Use the **last cluster** (closest to round end = actual KO).

From the last cluster, take up to 30 frames from the end. Compute median P1 and P2 HP values.

If HP difference >= 0.13125 -> clear winner (higher HP wins). **Confident.** The medians are
smoothed and can pick up the next round's full bars, so when the loser has a frame at <= 0.02
the reported HP is the cleaned HP at that KO (winner) and the loser's minimum, as in the
KO-count rules (`ko_hp`); otherwise the medians are reported.

### Step 5: Min-HP Fallback with Weighted Average

When the cluster analysis is ambiguous (HP diff < 0.13125):

**Signal A: Minimum HP**
- Find each player's minimum smoothed HP post-wallbreak
- `min_signal = p2_min - p1_min` (positive = P1 went lower = P2 wins)
- Transition artifact filter: if both min values are in [0.35, 0.48125] with < 0.04375 difference, zero out the signal

**Signal B: Average HP**
- Compute average smoothed HP in the last 25% of the round
- Exclude frames where both players' HP < 0.525 (loading screen pattern)
- Fall back to last valid HP readings, then KO moment values

**Combined signal:**
- Dynamic weighting based on HP range: `min_weight = min(hp_range / 0.175, 0.8)`, `avg_weight = 1.0 - min_weight`
- `combined = min_signal * min_weight + avg_signal * avg_weight`
- **Confident** if `|combined| >= UNCERTAIN_HP` (0.0875)

### Step 6: Post-Round Tiebreaker

When `|combined| < UNCERTAIN_HP` (very ambiguous):
- Scan 300 frames (10s) after the round end
- Count visible (non-NaN) frames for each player
- In GGS, the winner's HP bar stays visible on the result screen while the loser's disappears
- If one player has 10+ more visible frames -> that player is likely the winner
- **NOT marked as confident** since next-round starts can pollute this signal

### Confidence Tracking

`winner_confident` is `true` when:
- KO counting produced a clear winner
- Cluster HP difference >= 0.13125
- Combined fallback signal >= `UNCERTAIN_HP` (0.0875)

`winner_confident` is `false` when:
- The winner was determined by the post-round tiebreaker
- The combined signal was too weak
- The round is cut off by the end of the video: it ends within 2 s of the last frame and the loser never reaches 0.01, or only within the last 2 s (the KO and slash are not visible). Not when the KO is complete: after it the HUD shows the clock stopped (one value, read >= 5 times, at most one below the last value before the KO; a round in play counts down) and the loser stays at 0 (12_LSemi R10 in a cut ending 1.9 s after the KO)

---

## Round-Counter Hearts: Match Starts and Winner Check

The two hearts on each side of the timer are the game's own HUD (same pixels on any 1080p GGST
video, verified on non-overlay YouTube replays); only the name/score banner above them belongs
to the Spaghetti Showdown stream overlay and must never be read. A side loses a heart for each
round the opponent wins in the current match (`p1_rounds_won` = hearts P2 lost).

**Timing:** the HUD updates the hearts at the start of the next round. The state read early in
round k is the result of the rounds before k; between rounds (and in the seconds before a round
starts) the previous state is still shown. The old windows (150 frames before the round start
vs. between the round end and the next start) therefore compared two readings of the same state
and, with the rewritten hearts reader, credited the previous round's result to the next one.

**State of a round:** the mode of the raw (not forward-filled) readings with both sides read, from
0.5 s to 10 s after the round start (clipped to the round end); the mode needs >= 1 s of readings
(30 at 30/s) and >= 80 % of all readings, else unknown.

**HUD gap + timer intro:** a match (like every round) starts after the HUD has been off screen:
>= 1 s with no timer reading, looked for from 15 s before to 5 s after the round's first HP
reading (the start of the video counts as a gap). After at least one such gap, a supported timer
reading >= 97 must follow within 5 s (the timer's intro value is 99). Any gap in the window may
supply it: an effect can hide the timer for > 1 s just after the intro (14_Grands R10).

**Match start:** HUD gap + timer intro, and
- hearts state known: 0/0 (both sides full for >= 1 s), and the previous round's state is not 0/0
  (or this is the first round);
- hearts state unknown (unreadable): the previous round ended a first-to-2 match, i.e. the latest
  known hearts state since the last match start (or 0/0 at that start) plus the HP winners of the
  rounds since gives a side 2 wins.

The old hearts reader read 0/0 nearly everywhere and the old timer reader never read 97-99, so data
from the old readers still flags no match starts. The hearts never add, move or split round
boundaries; those come only from HP/timer resets and data gaps.

**Winner check:** compare the state of round k with that of round k+1:
- one side up by exactly one, the other unchanged: that side won round k;
- round k+1 back at 0/0 (a new match): round k ended the match, won by the side that already had
  a round (from 1-0 or 0-1; from 1-1 either side could have won, no check);
- anything else (unknown states, missed or split rounds): no check.

The hearts are a secondary signal; HP and timer stay primary:
- hearts agree with an uncertain HP winner: the round becomes confident;
- hearts disagree with an uncertain HP winner: tie-break, the hearts' side wins
  (`winner_final_hp` / `loser_final_hp` swapped) but the round stays uncertain;
- hearts disagree with a confident HP winner: the HP winner is kept and the round is marked
  uncertain, until the match rules below settle it.
The last round of a video has no next state and is never checked.

**Match rules** (`match_sequence_possible`): a match is first to 2 rounds. The winners of the
rounds from one match start to the next must be possible: nobody reaches 2 before the match's last
round, and a match followed by another one ended with 2 wins (a new match after 1-1, or a third
win, is impossible). The last match may be cut off by the end of the video; rounds before the
first match start (the video opens mid-match) are not checked. This only resolves
contradictions:
- an impossible match is fixed by flipping one round the hearts disagree with, or an uncertain
  one, when exactly one such flip makes it possible; the round is confident when the hearts agree
  with the result (e2e 02_UF R1: HP said P2 after a missed KO, the next round started a new match
  at what would be 1-1, the hearts said P1). A single KO on HP (loser <= `KO_HP`, winner above
  it, confident on HP) is only a candidate when no other round is: the hearts must disagree with
  it (so it is uncertain), it must be the only flip that makes the match possible, and it must
  have been a near double KO (the HP winner at <= `NEAR_DKO_HP`, 0.03), where which bar is empty
  comes down to a few pixels and the cleaner can take a 1-px sliver for a KO (720p test VOD set
  04 R3 before the cleaner learned that a sliver with the clock running is alive). Such a flip
  rests on the hearts alone, which misread lost hearts, so the round stays uncertain for the
  user. A KO with the winner's bar still clearly drawn (a 0.13 bar is ~90 px at 1080p) and a
  manually overridden round are never flipped: a misread heart would otherwise make the KO'd
  side the winner (with a winner HP of 0);
- a confident HP winner the hearts disagree with stays confident when the hearts' version would
  make the match impossible (the hearts reader misreads lost hearts);
- an uncertain winner the hearts confirmed or gave becomes confident when the other winner would
  make the match impossible.
Otherwise (no or several candidate flips) nothing changes. On the live, e2e and YouTube data the
resolver changes nothing after the cleaner fixes of 2026-10-21; on the old e2e cleaning it fixes
02_UF R1.

**Results** (re-extracted data, `output/reanalysis2/merged.db` and `yt/`; dumps in
`output/round_fix/v65`, `v65_yt`): match starts are identical to v62 on all 18 videos (64 on our
15 replays, 0 first-to-2 violations; KWX 6 matches, Pbs 3, gbt 2, checked against the in-game
hearts and the YouTube score strip). The only change: 05_UF R3 (double KO, P1?) becomes confident
P1, which the next round's hearts and the stream show. 11_WFinal R12 stays P1? via the tie-break.

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

### Applied Before the Match Rules

In `detect_rounds_for_replay()` (so `get_rounds`, `get_match_stats` and every other caller see
them), after each round's HP winner and before the hearts and match rules:
1. Query `winner_overrides` for the replay (`load_winner_overrides`; none when the table doesn't exist)
2. For each override, find the matching round by `round_index`, swap winner/loser HP if needed, set `winner_confident = true`
3. Mark the round fixed: the hearts check, the match rules and the end-of-video rule never change it

The match rules then work from the corrected winners: an automatic flip that rested on a wrong
neighbouring round of the same match is undone when the user corrects that round, so a
correction can't leave a match at 1-1 or with a third win. Overrides are non-destructive and
survive reanalysis.

---

## Comeback Detection

For each round, the maximum deficit is computed:
- Use heavy-smoothed HP (91-frame window) for both winner and loser
- Skip the first 15% of the round (avoids HP reset transition artifacts)
- `deficit = loser_heavy_hp - winner_heavy_hp` at each frame
- `max_deficit` is the maximum deficit observed
- `is_comeback = max_deficit >= COMEBACK_DEFICIT (0.30)`

The `deficit_timestamp_ms` records when the maximum deficit occurred.

---

## Highlight Detection

Generated by `generate_highlights()` from rounds and damage events. Sorted by severity (descending).

| Kind | Condition | Severity |
|------|-----------|----------|
| `comeback` | `is_comeback == true` | `max_deficit` (0.30-1.0) |
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
