# lab_result.json: schema `spaghetti-lab-result/1`

The single result document of a Spaghetti Lab analysis of one Strive match video. It is written by
`tools/ggst_logger/model/lab_analyze.py` (`<out>/lab_result.json`, next to the chain's `openers.json`).
The opener service returns it from `GET /jobs/{id}/result`, and FGCBackend passes it through unchanged.
The desktop app (app/) and the web UI (FGCScoreboard) both read it.

Sample: `tools/ggst_logger/model/fixtures/lab_result_sample.json` (148 KB, also at
`G:\Projects\replanal-data\lab_fixture\lab_result_sample.json`). It is a real test-fold replay: Asuka (P1) vs
Elphelt (P2), 2:30 at 1080p60, P2 wins 2-0, 21 openers, 149 transcript events, 1,125 state tubelets. The video
is `G:\Projects\replanal-data\ggst_rec\inputs_20261022_181805.mp4`. `lab_fixture.py` built it from that replay's
saved model outputs with lab_analyze's own mapping code. It has the transcript and the states; `timings_s` has
only `facts`, since nothing else ran.

Versioning: fields may be **added** within `/1`; nothing is removed or changes meaning. Readers ignore
unknown fields. A breaking change becomes `spaghetti-lab-result/2`.

## Top level

| field | type | meaning |
|---|---|---|
| `schema` | `"spaghetti-lab-result/1"` | |
| `video` | str | the **file name** (no directory) of the video lab_analyze was given. On the service it is the upload's temporary name (`video.mp4`): show the user's own file name instead |
| `duration_s` | float | video length in seconds |
| `fps` | float | the video's frame rate (30 or 60 usually). All times below are seconds of the video; `frame` fields are always on a **60 fps** timeline (`frame / 60 = t`), whatever the video's fps |
| `characters` | `{"P1": str, "P2": str}` | the opener chain's character vote (names as in `tools/ggst_logger/dataset/moves.json`, e.g. `"Sol"`, `"Asuka"`, `"Elphelt"`) |
| `capabilities` | `{"openers": true, "transcript": bool, "facts": true}` | what **this run** produced. `transcript` false means `transcript` and `states` are `null` |
| `openers` | list | the opener chain's combo openers, exactly as `recognize.py` writes them in `openers.json` (below) |
| `transcript` | `null` or `{"events": [event...]}` | every move start of both players, sorted by `frame` (below) |
| `states` | `null` or `{"hz": 7.5, "P1": [class...], "P2": [class...]}` | per-tubelet state class of each side, present when the transcript ran |
| `facts` | object | the language layer's facts document (`ggst-match-facts/1`, `lang_summary.py`) plus `md` |
| `prompts` | `{"it": str, "en": str}` | the LLM prompt text (system + user) per language. Nothing is ever sent to an LLM |
| `timings_s` | object | seconds per stage: `hud`, `detect`, `moves`, `transcript` (if run), `facts`, `total` |

### `openers[]` (unchanged recognize.py objects)

`time_s` (float, seconds), `video_frame` (int, frame of the video at its own fps), `attacker` (`"P1"`/`"P2"`),
`attacker_char`, `defender_char`, `detector_score`, `category` (normal / special / super / throw / projectile / other),
`category_p`, `move` (notation such as `"5H"`, `"236K"`, `"j.S"`, or `"(rare)"` = a move the model does not name),
`move_p`, `move_top3` (`[[move, p], ...]`), `normal` (`"H/stand"` etc. for normals, else null),
`hit_kind` (`"counter"` / `"punish"` / `"none"`, from the game's banner), `hit_kind_p`.

### `transcript.events[]`

| field | type | meaning |
|---|---|---|
| `t` | float | seconds into the video of the move's start (= `frame / 60`, 3 decimals) |
| `frame` | int | start frame on the 60 fps timeline |
| `side` | `"P1"` / `"P2"` | who does the move |
| `char` | str | that side's character (the transcript's own character vote; it agreed with the truth on 439 of 440 test sides) |
| `category` | str | `normal` / `special` / `super` / `throw` / `other` (burst, Roman Cancel, misc). There is no `projectile` here, unlike openers |
| `move` | str | notation (`"5H"`, `"236S"`, `"Burst"`, `"RC"`, ...). `"(rare)"`: a move the model does not name |
| `move_p` | float | probability of `move` (< 0.5: show as uncertain) |
| `normal` | str or null | button/stance for normals (`"H/stand"`, `"S/crouch"`, `"K/air"`, ...), null otherwise |
| `outcome` | str | `counter` / `hit` / `block` / `whiff` / `interrupted` / `teched` / `none` (burst, RC, movement, ...) |
| `outcome_p` | float or null | probability of `outcome`. null only in files made from older saved predictions that lack it; the sample has it |
| `opener` | bool | a combo opener of the opener chain starts at this event (rule below) |
| `hit_kind` | str or null | for opener events, the chain's banner reading (`counter` / `punish` / `none`); null for the others |

How an event becomes an `opener`: take each `openers[]` entry (side = `attacker`, frame = round(`time_s` × 60)).
Among the same side's events with `-4 <= opener_frame - event.frame <= 40`, mark the one nearest to the opener and
at or before it (if all candidates come after it, the nearest one). Each opener marks at most one event. The chain
anchors on the hit, the transcript on the move start, so the times differ by a few frames. The opener list stays
the better source for openers (move right 0.887 vs 0.834 for the transcript on the test fold).

Accuracy (test fold, ±4 frames): moves found 0.78 recall at 0.85 precision. Whiffs are found only 0.57 of the time
and interrupted moves 0.64. Moves starting under 8 frames after the previous one of the same side are often merged
into one event. Present it as "what the model saw", not a complete record.

### `states`

`hz` is 7.5. `states.P1[i]` and `states.P2[i]` are the class of the tubelet `i`, which covers 60 fps frames
`8i .. 8i+7`, i.e. the time `i / 7.5` s. The classes are
`neutral`, `attacking`, `blockstun`, `hitstun`, `knockdown`, `airborne`. Both lists have the same length. They are
already smoothed (3-tap mean of the class probabilities, then argmax, as `search_index.state_streams`). The per-side
class is right about 0.85 of the time on the test fold.

### `facts`

The `lang_summary.match_facts` document, unchanged (`schema: "ggst-match-facts/1"`). Its main parts:
`result` (`winner`, `score` e.g. `"2-1"`, `uncertain`), `rounds[]` (`n`, `start_s`, `end_s`, `winner`, `uncertain`,
`t_start`/`t_end` m:ss.s), `match_id` (the match title given with `--title`, e.g. the user's original file name without
extension, or null; the markdown heading is `Match facts: P1 <char> vs P2 <char>` plus ` (<title>)` when there is
one. No server path or job id is ever in the facts or prompts), `players.P1/P2` (opener counts and mix, top opening moves, biggest combos, damage), and
`openers[]` (`id` "O1"..., `t_s`, `t` m:ss.s, `attacker`, `char`, `move`, `move_name`, `hit_kind` neither/counter/punish/unclear,
`damage_pct`, `ko`, `uncertain`). With the transcript there is also `transcript.P1/P2` (attack outcome rates, most
used attacks, what counter-hit them, habits). `sources` and `caveats` say what was used. Added here: `md`
(`{"it": str, "en": str}`), the markdown rendering of the facts per language, ready to display.

### `prompts`

`"[system]\n<system prompt>\n\n[user]\n<facts + task>"` per language, the same text `lang_summary.py facts`
writes to `prompt_<lang>.txt`. It is for copy/paste or a future, opt-in provider. The UIs must not send it anywhere
by themselves.

## Search presets (client side)

The UIs filter the event list on the client. The list is `transcript.events` when `transcript` is not null.
Otherwise it is the openers, seen as events by this mapping:

```
opener o -> { t: o.time_s, frame: round(o.time_s * 60), side: o.attacker, char: o.attacker_char,
              category: o.category, move: o.move, move_p: o.move_p, normal: o.normal,
              outcome: o.hit_kind == "counter" ? "counter" : "hit",
              outcome_p: o.hit_kind == "counter" ? o.hit_kind_p : null,
              opener: true, hit_kind: o.hit_kind }
```

Every preset returns events (moments to jump to: seek the video to `t`, ideally a second earlier). Sort results by
`frame`. These definitions mirror `search_index.py`'s `query()` and its evaluation families, so the numbers in the
README hold for them. Implement them exactly so the two UIs return the same moments. The Python reference is in
`lab_analyze.py` (`opener_events`, `preset_move_outcome`, `preset_counter_by`, `preset_after_block`, `wakeups`,
`preset_after_kd`), and `tests/test_lab_analyze.py` checks it against `search_index.query` on the sample. The
sample has hits for every preset, so a UI can be checked against those functions.

Common helper:

```
outcomeMatches(e, want):           // want: "any" | "connected" | one outcome
  want == "any"       -> true
  want == "connected" -> e.outcome in {"hit", "counter"}
  else                -> e.outcome == want
own(events, side) = events.filter(e => e.side == side) sorted by frame   // the side's own events
```

1. **Move + outcome** (`move_outcome` / `move` families). Parameters: `side` (P1 / P2 / any), `move`
   (exact string compare on `move`; empty = any), `outcome` (any / connected / counter / hit / block / whiff /
   interrupted / teched / none). Optional `openersOnly` (`e.opener`). Result: events with
   `(side == any || e.side == side) && (move empty || e.move == move) && outcomeMatches(e, outcome)`.
   Example: "every whiffed 5H of P1" = side P1, move 5H, outcome whiff.
   Offer the move list from the events of the chosen side (distinct `move`, by count; leave out `"(rare)"`).
2. **Counter-hit by** (`counter_by`). Parameter: `victim` (P1 / P2), optional `move`. Result: events with
   `e.side != victim && e.outcome == "counter" && (move empty || e.move == move)`, i.e. the opponent's moves that
   counter-hit the victim. Example: "when did P1 get counter-hit, and by what".
3. **After blocked** (`after_block`). Parameter: `side`. What the side does next after one of its own moves was
   blocked. For each pair of consecutive own events `p, r` of `own(events, side)` (the opponent's events in between
   do not matter): return `r` when `p.outcome == "block" && r.frame - p.frame <= 40`. Show `p.move` -> `r.move`.
   With the openers fallback, openers are never blocked, so this preset returns nothing: disable it.
4. **After knockdown** (`after_kd`). Parameter: `side`. The side's first own event after waking up. This needs
   `states`, so disable it when `states` is null. Rule (`search_index._state_runs` and `_link`):
   ```
   cls = states[side]
   wakes = []                                   // 60 fps frames at which the side gets up
   i = 0
   while i < cls.length:
     if cls[i] == "knockdown":
       j = i; while j < cls.length && cls[j] == "knockdown": j++
       if j - i >= 2 && j < cls.length && cls[j] != "hitstun": wakes.push(8 * j)
       i = j
     else i++
   ev = own(events, side); w = 0
   for k in 0 .. ev.length-1:
     r = ev[k]; p = k > 0 ? ev[k-1] : null
     while w < wakes.length && wakes[w] < r.frame - 60: w++
     if w < wakes.length && wakes[w] - 8 <= r.frame && r.frame <= wakes[w] + 60
        && (p == null || p.frame < wakes[w] - 8):
       result.push(r)  (with frames since wake-up = r.frame - wakes[w]); w++
   ```
   The wake-up moments themselves (`wakes[i] / 60` s) are also worth listing, including the ones without an
   event ("waited / blocked").

Measured quality of these presets over the test fold (`search_index.py eval`, F1 at ±8 frames): move + outcome 0.79
(whiff only 0.54), counter-hit by 0.84, opener with move M 0.78, after blocked 0.84, after knockdown 0.73.

## Producing it

```
.venv/Scripts/python.exe tools/ggst_logger/model/lab_analyze.py <video> --out <dir> [--device cuda|cpu] [--light]
       [--transcript auto|on|off] [--analysis <analyze_replay output dir>] [--threads N] [--title <match title>]
```

`--title` is the match title in the facts and prompts. The service passes the user's original file name (job
`meta.filename`). Without `--title`, the heading names only the characters. The desktop app should pass the
replay's name.

It prints exactly these lines, in this order: `stage: hud`, `stage: detect`, `stage: moves`, `stage: transcript` (only
when it runs), `stage: facts`, `stage: done`. The opener service turns them into the job's `stage`. It adds its own
`starting` and, for `/jobs/url`, `download` before `hud`. `--transcript auto`
runs the transcript only with a CUDA GPU (it needs about 77 s of V-JEPA 2 per match). `off` gives `transcript: null`,
`states: null` and `capabilities.transcript: false`. Openers and facts always run, on the CPU too.
