# Roadmap: modern video models for Strive analysis

Status 2026-11-02. Today's chain (`model/`, see README): HUD + 64×36 frame detector, ResNet player
locator, ResNet18 + ResNet50 move ensemble on attacker crops and whole frames, banner reader.
Whole chain 0.888 at 720p30 (0.878 light, the CPU server's setting). It learns only from the
**combo openers**; the game log has every frame (moves, inputs, positions, HP, meter) of 1,090+
replays, which is most of the value left on the table.

Rules that hold for every step: models see only the video (the log makes labels and scores);
game-native HUD only, never the stream overlay; the frozen test fold (`model/folds.json`) is the
yardstick and every step reports old / new / 720p30; the reverse engineering behind the log is
never released; weights stay private until decided otherwise.

## 1. Pretrained video backbones (in progress)

- **Goal:** better features than ImageNet ResNets, above all for rare moves (accuracy 0.46 with <5
  training examples vs 0.85 for common ones) and projectile vs special (0.69–0.72 right).
- **How:** frozen V-JEPA 2 (video), DINOv2 / SigLIP 2 (per frame) on the same opener clips; small
  temporal head; same split and metrics as `video_moves.py`; then LoRA fine-tuning of the winner
  if frozen features look promising.
- **Done when:** a table backbone × {move, category, rare-move buckets, projectile/special,
  chain} on the old and new test replays, and a decision: adopt, fine-tune, or stay on ResNets.
- **Cost:** days. GPU 3090 (24 GB), PC near its commit limit, ~30 GB free on G:.

## 2. Full match transcript (next)

- **Goal:** every move of both players, frame by frame: whiffs, blocked moves, combo routes,
  throws/techs, bursts, supers. Openers become a special case.
- **How:** labels from the log for every frame; frame embeddings of the whole match (winner of 1,
  at 15–30 fps); temporal action localisation (ActionFormer / TriDet-style, or a transformer over
  the embeddings) per player; evaluation as action spotting (precision/recall at ±k frames, per
  category and move).
- **Unlocks in the Lab:** wakeup habits, neutral whiffs, how often you get counter-hit by what,
  punish rates, combo route stats.
- **Cost:** weeks. Storage: whole-match embeddings ~10–15 GB fp16.

## 3. Inputs from video (inverse dynamics)

- **Goal:** directions and buttons per frame from the video (as VPT did for Minecraft), so any VOD
  gets notation, execution errors (dropped links, late inputs) and reaction times.
- **How:** the log's inputs per frame as labels; a model over the transcript (2) plus frames;
  scored on input-string accuracy and timing error in frames.
- **Depends on:** 2 (shares the embeddings and the dense labels).

## 4. Game-state embeddings and search

- **Goal:** "find every situation like this" across all VODs; player habit clusters.
- **How:** contrastive alignment (CLIP-style) between video clips and the logged state vector
  (positions, HP, meter, current moves, distance); the video tower then reads state without the log.
- **Depends on:** 1 (backbone), helps 2 and 3 as an input.

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

## Order

1 → 2 → (3, 4 in parallel) → 6 → 5 in the app. 5 can prototype early on openers; 6 can start on
the current chain any time the server needs it.
