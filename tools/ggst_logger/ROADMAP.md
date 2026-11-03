# Roadmap: modern video models for Strive analysis

Status 2026-11-03 (points 1 and 2 done). Today's chain (`model/`, see README): HUD + 64×36 frame detector, ResNet player
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

## 3. Inputs from video (inverse dynamics)

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

## 4. Game-state embeddings and search

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

## 5. Language layer

- **Goal:** coaching notes and per-set summaries in plain language (it/en).
- **How:** an LLM over the structured output (transcript, HP, openers, habits), never over raw
  frames (VLMs can't count frames or tell 2K from 2S). Prompted, cited back to timestamps.
- **Depends on:** 2 for anything beyond openers; a first version can run on openers today.

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

## Order

1 → 2 → (3, 4 in parallel) → 6 → 5 in the app. 5 can prototype early on openers; 6 can start on
the current chain any time the server needs it.
