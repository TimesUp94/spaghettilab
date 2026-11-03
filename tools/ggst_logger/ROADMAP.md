# Roadmap: modern video models for Strive analysis

Status 2026-11-04 (points 1 and 2 done; 3, 4 and 5 have first versions measured). Today's chain (`model/`, see README): HUD + 64×36 frame detector, ResNet player
locator, ResNet18 + ResNet50 move ensemble on attacker crops and whole frames, banner reader.
Whole chain 0.888 at 720p30 (0.878 light, the CPU server's setting). It learns only from the
**combo openers**; the game log has every frame (moves, inputs, positions, HP, meter) of 1,090+
replays, which is most of the value left on the table.

Rules that hold for every step: models see only the video (the log makes labels and scores);
game-native HUD only, never the stream overlay; the frozen test fold (`model/folds.json`) is the
yardstick and every step reports old / new / 720p30; the reverse engineering behind the log is
never released; weights stay private until decided otherwise.

## 1. Pretrained video backbones (done)

- **Goal:** better features than ImageNet ResNets, above all for rare moves (accuracy 0.46 with <5
  training examples vs 0.85 for common ones) and projectile vs special (0.69–0.72 right).
- **How:** frozen V-JEPA 2 (video), DINOv2 / SigLIP 2 (per frame) on the same opener clips; small
  temporal head; same split and metrics as `video_moves.py`; then LoRA fine-tuning of the winner
  if frozen features look promising.
- **Done when:** a table backbone × {move, category, rare-move buckets, projectile/special,
  chain} on the old and new test replays, and a decision: adopt, fine-tune, or stay on ResNets.
- **Status 2026-11-02:** table in the README ("Pretrained backbones (2026-11)"). Frozen, whole
  chain old / new: V-JEPA 2 ViT-L 0.844 / 0.792, DINOv2 base 0.811 / 0.764, SigLIP 2 base
  0.805 / 0.760, ImageNet ResNet18 control 0.780 / 0.729, against v2's 0.887 / 0.855. The 50/50
  averages with v2 are within noise. V-JEPA 2 with its last 8 blocks fine-tuned (`exp_bb_ft.py`,
  4 epochs, ~4.5 h): 0.869 / 0.835, still below v2, not adopted. Decision: **stay on the ResNets**
  for openers.
- **Backbone for 2:** frozen **V-JEPA 2 ViT-L** (`facebook/vjepa2-vitl-fpc64-256`), the best frozen
  features by +0.03 chain over DINOv2 and +0.06 over ImageNet, strongest where motion matters
  (normals, 5–19 bucket). Measured on the 3090: ~290 frames/s at 256×256 through the encoder
  (151 ms per 2 × 22-frame clip, 3 loader workers, pre-cut JPEG tiles, 1.2 GB GPU). Output: one
  1024-d token per 2-frame tubelet after the mean over space, i.e. 512-d-equivalent per frame,
  1 KB fp16 per tubelet.
  - Affordable setting: **15 fps, 256×256**, one square whole-frame view (letterbox or two
    square halves rather than the squash used for openers). A 3-minute match is 2,700 frames,
    ~9 s of GPU; all 1,090 replays ~3 h plus video decode; ~3 MB per match, ~3 GB for all
    mean-pooled (a 2×2 spatial grid: ~12 GB, near the budget). 30 fps doubles time and storage
    (~6 h, ~6 GB).
  - DINOv2 base per frame (~550 tiles/s, 768-d) is the fallback if the transcript needs per-frame
    rather than per-tubelet resolution.

## 2. Full match transcript (done, first version)

- **Goal:** every move of both players, frame by frame: whiffs, blocked moves, combo routes,
  throws/techs, bursts, supers. Openers become a special case.
- **Unlocks in the Lab:** wakeup habits, neutral whiffs, how often you get counter-hit by what,
  punish rates, combo route stats.
- **Status 2026-11-03:** done; details and the full table in the README ("Match transcript
  (2026-11)").
  - Labels: 169,383 move starts from the logs (`dataset/transcript_labels.py`), with outcomes.
  - Features: frozen V-JEPA 2 ViT-L at 15 fps, 7.5 Hz tubelets, scene + two mirrored player
    squares with 2×2 quadrant tokens, plus the HUD.
  - Model: per-side local-window transformer spotting head, 5 seeds (`model/video_transcript.py`,
    `ens5grid`).
  - Test, k=4 frames, micro F1, old / new: category 0.815 / 0.798, button/stance 0.712 / 0.685,
    exact move 0.735 / 0.716, outcome 0.770 / 0.760.
  - Like for like with the opener chain: openers recovered with the right move 0.834 / 0.823
    against the chain's 0.887 / 0.855. The chain stays for openers.
  - Costs: features 6.7 h + 4.7 h (grid pass) for 1,090 replays, 10.4 GB fp16; about 1 min of
    GPU-box time per match plus a 3 s head; a net trains in 12.5 min.
- **Still open:**
  - whiffs (found 0.57) and interrupted moves (0.64);
  - starts < 8 frames apart (0.47): one start per tubelet;
  - throws (0.68 F1);
  - not wired into `recognize.py` or the app.
  - Next candidates: 15 Hz tokens or two slots per tubelet; a finer grid or a crop around the
    attacker; one combined feature pass instead of two.

## 3. Inputs from video (inverse dynamics) (first version)

- **Goal:** directions and buttons per frame from the video (as VPT did for Minecraft), so any VOD
  gets notation, execution errors (dropped links, late inputs) and reaction times.
- **How:** the log's inputs per frame as labels; a model over the transcript (2) plus frames;
  scored on input-string accuracy and timing error in frames.
- **Depends on:** 2 (shares the embeddings and the dense labels).
- **Next steps, given 2:**
  - Labels: per-frame direction (numpad 1–9, mirrored to the player's facing) and button presses
    from the logs, on transcript_splits.json. Score the direction per frame and the button presses
    as spotting at ±k, like `transcript_eval`.
  - Start with a head on the same `ens5grid` inputs plus the transcript's move posteriors:
    the move fixes most of the motion input (236 for a quarter-circle special), so input
    accuracy given a right move is the first number to get.
  - Timing needs more than 7.5 Hz: the transcript's starts are good to ±2–4 frames, but links and
    reactions need ±1. Re-extract 30 fps tokens (or per-frame DINOv2 on the player squares) only
    around the transcript's events rather than for the whole match, to stay inside the disk budget.
  - Inputs without a visible move (walks, dashes, guard, held directions) are where it can fail;
    measure them separately.
- **Status 2026-11-03:** first version measured; details in the README ("Inputs from video
  (2026-11)"). Test old / new:
  - Direction per frame 0.597 / 0.572 (5 alone 0.443 / 0.403); balanced 0.37. Against each
    state's own most common direction (verification): idle 0.77 / 0.70 vs 0.72 / 0.64, moves
    0.58 / 0.56 vs 0.56 / 0.52, guard 0.75 / 0.73 vs 0.63 / 0.60, dash 0.60 / 0.57 vs 0.37 / 0.40,
    jump 0.44 vs 0.25, hit 0.51 / 0.50 vs 0.36 / 0.32, but walks 0.54-0.74 vs 0.99: the gains are in
    jump, hit, dash and guard, and walking is not recognised.
  - Button presses F1 at ±2 frames 0.276 / 0.276 (±1 0.19, ±4 0.36). Presses that start a
    visible move: recall 0.44; presses without one (60% of all presses: mashing, buffered,
    kara, cancels into nothing): 0.09.
  - Given a move the transcript spotted right, the move's most common command from training
    is the best rule: command 0.593 / 0.576, direction + button 0.653 / 0.650. Reading the
    model's per-frame streams gets 0.42 / 0.41; an event head trained for it 0.578 / 0.564.
    The per-move mode fitted on the test fold itself, on the same matched events, reaches 0.633 /
    0.637 (command) and 0.687 / 0.706 (direction + button), so the prior is 0.04-0.06 below its
    ceiling; the rest is which variant of the motion the player typed (41236 or 236, a charge read
    from a held back), not visible. Over all test events the prior gets 0.558 / 0.540.
  - 60 fps DINOv2-small of the player squares (563 replays, 2.4 h, 7.1 GB) did not improve the
    dense model. A window refiner around the transcript's starts (`inputs_refine.py`) moves
    the start exactly right from 0.336 / 0.346 to 0.376 / 0.378 and within ±1 from 0.792 / 0.776 to
    0.828 / 0.806 (±2 slightly lower), but not the press (buffered presses are invisible).
  - Labels are good to about ±1 frame (60 fps pixel check on 6 test replays: the HUD HP change at
    label frame -1, 0 or +1 depending on the replay), which bounds every exact-frame number.
  - About 3.7 GPU-hours; 8.1 GB under `ggst_dataset/inputs/`.
  - Decision: the useful product is transcript move + its usual command (notation), held
    directions in neutral from the dense model, and frame-exact timing from the move start, not
    the press.
- **Still open:** presses that make no visible move (recall 0.09); which motion variant was typed;
  directions in moves, hitstun and jumps; walks not recognised; per-replay ±1-frame label
  alignment; train-time posteriors are OOF single nets while test uses ens5grid; single seed;
  no 720p30 check; not in `recognize.py`.
- **Next steps:**
  - Ship the parts that work: transcript move + its usual command as notation, a refined move
    start, held direction only in guard / dash / jump / hit where it beats the state's majority.
  - Per-replay alignment: fit a -1 / 0 / +1 offset per replay from the HUD HP changes before using
    labels for frame-exact training or scoring.
  - Walks: feed the state stream from point 4 or the player track's velocity; score per state
    against the state majority, not against always-5.
  - Seeds / an ensemble and early stopping for the dense model; ens5grid-like posteriors at train
    time (cross-fitted 5-seed members) to close the train / test posterior gap.
  - 720p30 copies before any of it goes into `recognize.py`.

## 4. Game-state embeddings and search (first version)

- **Goal:** "find every situation like this" across all VODs; player habit clusters.
- **How:** contrastive alignment (CLIP-style) between video clips and the logged state vector
  (positions, HP, meter, current moves, distance); the video tower then reads state without the log.
- **Depends on:** 1 (backbone), helps 2 and 3 as an input.
- **Next steps, given 2:**
  - The whole-match V-JEPA 2 tokens (scene, both player squares, grid) and the HUD of all 1,090
    replays already exist at 7.5 Hz, aligned to the log; no new extraction is needed.
  - State vector per tubelet from the log: both positions, distance, HP, tension, burst, the
    current move / state of each side (idle, blockstun, hitstun, knockdown), facing, corner.
  - Contrastive (CLIP-style) training of a small tower over a few seconds of tokens against the
    state vector; check it with state regression on the test fold (distance, HP, who is in
    hitstun) and with "situation retrieval" against the logged state.
  - The transcript gives the first searchable keys right away (move, outcome, counter, punish at
    a time): an index over its events answers "every whiffed 5H" or "every time X got
    counter-hit by Y" before any embedding training.
  - Habit clusters: per player, sequences of transcript events after knockdowns and blocked
    pressure.
- **Status 2026-11-04:** first version measured; details in the README ("Search and state
  embeddings (2026-11)"). Test old / new:
  - Event index (`model/search_index.py`): queries over the predicted events against the same
    queries over the logged ones, moments matched within ±8 frames.
    - Move + outcome F1 0.787 / 0.788 over keys with ≥ 5 logged results; over all keys
      (verification, `(rare)` excluded) 0.756 / 0.748. "X counter-hit by Y" 0.836 / 0.850; openers 0.784 / 0.791;
      the move after a blocked one 0.840 / 0.842; the first move after a knockdown 0.744 / 0.721.
    - Counts are off by a median 11–20% per query.
    - Whiff queries: recall 0.46 / 0.48.
  - State labels from the log per tubelet: positions relative to the stage section (sections
    move by multiples of 1.6 after wallbreaks), distance, HP, facing, corner, and a class per side
    (neutral / attacking / blockstun / hitstun / knockdown / airborne). The log has no tension or
    burst.
  - State embedding (`model/state_embed.py`, `clip_sup`): CLIP + supervised heads over 16
    tubelets of the existing tokens + HUD. MLP probe:
    - class per side 0.899 / 0.885 (raw tokens 0.757 / 0.740);
    - who is hit 0.97, corner 0.94;
    - distance R² 0.94 / 0.92.
  - The class stream the index uses (the sup heads, not the MLP probe) is right 0.855 / 0.848 per
    side on old, 0.845 / 0.836 on new.
  - Situation retrieval top-10 from other replays: the same joint class 0.66 / 0.64. Random picks
    get 0.08, but random picks with the same predicted class already get 0.60, so the embedding
    adds about 0.05 over its own heads. The supervised-only net gets 0.75 but loses the
    distance (Δ 0.16 vs 0.09).
  - Habits per character: wake-ups (found P 0.89 / R 0.85 from the state stream) and blocked
    strings (P 0.76 / R 0.83).
    - Predicted vs logged option shares: TV 0.22 (wake-up) and 0.17 (blocked string) by move. The
      split-half comparison (0.33 / 0.44) is not a valid noise floor (smaller halves, same
      situations); the option agreement on matched pairs (0.67 / 0.86) is the honest number.
    - The model reports "no move" after 0.63 of wake-ups where the log has 0.56.
  - About 1 GPU-hour; 104 MB of new data.
- **Still open:** whiffs and wake-up moves are limited by the transcript; per player rather
  than per character (no player ids); no punish key; walking and dashing are not split from
  neutral; the index covers the test fold only and is not in the app; the habit groups are option
  shares, not learned clusters; single seed.
- **Next steps:**
  - Report the all-keys F1 and the same-predicted-class baseline as the headline from now on.
  - A pipeline for unlabeled VODs: transcript + state stream + index from `recognize.py`'s outputs,
    so the Lab can query a user's own matches (needs point 6 to be cheap).
  - A punish key from the chain's banner reader; walk / dash classes from point 3's model.
  - Habits per player once replays carry player ids (the search batch's manifest has names).

## 5. Language layer (local part done; provider not decided)

- **Goal:** coaching notes and per-set summaries in plain language (it/en).
- **How:** an LLM over the structured output (transcript, HP, openers, habits), never over raw
  frames (VLMs can't count frames or tell 2K from 2S). Prompted, cited back to timestamps.
- **Depends on:** 2 for anything beyond openers; a first version can run on openers today.
- **Status 2026-11-04:** details in the README ("Language layer (2026-11)").
  - `model/lang_summary.py`: facts document per match (rounds, openers with damage, habits,
    m:ss.s timestamps, uncertain marks), prompts in it / en, a grounding checker. No network, no
    GPU; the only provider that runs is `none` (writes the prompt); `ClaudeProvider` is a gated
    stub that was never called. 14 tests.
  - 6 test-fold replays: round winners right on all 6; unmarked openers match the log 86%
    (146 / 170), marked ones 10 / 15, so the p < 0.5 mark does not separate well yet.
  - Round winner rules on train + val (870, never looked at): 831 replays fully right, unmarked
    rounds 99.6% right.
  - Prompts are about 2.0-2.6k input tokens per match (estimate, no tokenizer offline).
  - Checker gaps found by verification: case-varied or slang names ("gun flame", "DP"), other
    notations ("5HS", "cS"), times in seconds, "N percent", numbers in words, frame-data claims
    without numbers, and a wrong match result in the summary.
- **Open decision (user):** may the facts leave the machine, and to which provider? Only the facts
  document would be sent (characters, timestamps, move names, damage, round results, habit
  counts; no video, frames, player names or log data). Per match, about 2-2.6k tokens in plus
  1.5-3k out (thinking billed as output): Claude Opus 5.5 ($4 / $20 per M) about $0.04-0.07, Claude
  Sonnet 5.5 ($2 / $10) $0.02-0.035, Claude Haiku 5.5 ($0.10 / $0.50) about $0.001-0.002; the Batch
  API halves that. Or a local model: the checker works with any provider.
- **Next steps:**
  - Close the checker gaps above (case-insensitive and alias matching, seconds and word numbers,
    qualitative frame-data claims, a check of the match result / score).
  - Once a provider is decided: real answers on val replays through `check_answer`, regenerate or
    reject failures; measure the failure rate and what slips past the checker by hand.
  - Calibrate the uncertain thresholds on val; one prompt per set with [M2 1:23.4] citations.
  - Wire it after `recognize.py` (facts are cheap; the expensive part is the transcript, see 6).

## 6. Distillation for the CPU server

- **Goal:** keep serving cheap whatever the research models are: today 4 min 15 s and 4.4 GB per
  3-minute match on 4 CPU threads.
- **How:** a small student (MobileNet-class / X3D) trained on the big model's outputs over all
  replays and unlabeled VODs; ONNX int8; frame caching across overlapping opener windows.
- **Depends on:** whatever 1–4 settle on; also applies to the current chain now.
- **For the transcript:** the cost is the frozen ViT-L pass (about 300M parameters, 3 views + the
  grid pass), not the head (3 s per match on the GPU). A CPU version needs:
  - a student that maps frames straight to the head's inputs, or a student of the whole
    transcript, trained on the ens5grid outputs over all replays and unlabeled VODs;
  - views cut by the player locator the server already runs, at 256×256, 15 fps;
  - measuring it on the 720p30 copies (`degrade_eval.py`), which the transcript has not been yet;
  - a target time next to today's 4 min 15 s per match;
  - one combined pass for the mean and quadrant tokens.
- **What 3-5 need from it now (2026-11-04):**
  - Everything new runs on top of the transcript: the inputs model and refiner (3), the search
    index and state stream (4) and the richer facts (5) all take the ens5grid posteriors plus the
    V-JEPA 2 tokens. So the one student that matters is the **transcript student**: frames to
    the transcript's events and posteriors, plus the state-stream class (4) as a second head, at
    256×256 / 15 fps on the player locator's views.
  - The refiner (3) needs 60 fps DINOv2-small crops only in 32-frame windows around events; on CPU
    that is a few hundred crops per match, and a small CNN student of it is likely enough.
  - Language (5) costs nothing on the server beyond the facts (pure Python); the LLM call, if any,
    is per match and outside the GPU budget.
  - Scoring: the student must be measured with the same `transcript_eval`, `inputs_eval` and
    `search_index eval` on the frozen test fold, old / new and on the 720p30 copies, next to the
    ens5grid numbers (move F1@4 0.735 / 0.716, index all-keys F1 0.756 / 0.748), plus wall time and
    memory per match on 4 CPU threads.
  - Training data: ens5grid outputs exist as final predictions only for val / test; the student
    needs cross-fitted teacher outputs on the training replays too (the 4 OOF nets from 3 are a
    start) and, later, unlabeled VODs.

## Order

1 → 2 → (3, 4 in parallel) → 6 → 5 in the app. 5 can prototype early on openers; 6 can start on
the current chain any time the server needs it. As of 2026-11-04: 1-4 have first versions and 5
its local part; next is 6 (the transcript student), with 5's provider decision and checker gaps
in parallel.
