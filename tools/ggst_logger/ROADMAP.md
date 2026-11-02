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

## 1. Pretrained video backbones (frozen comparison done; fine-tune running)

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
  averages with v2 are within noise. Decision: **stay on the ResNets** for openers. Running:
  V-JEPA 2 with its last 8 blocks fine-tuned (`exp_bb_ft.py`, ~4.5 h); adopt it only if it beats
  v2's chain on both subsets beyond seed noise (about ±0.004 old, ±0.01 new).
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

## 2. Full match transcript (next)

- **Goal:** every move of both players, frame by frame: whiffs, blocked moves, combo routes,
  throws/techs, bursts, supers. Openers become a special case.
- **How:** labels from the log for every frame; frame embeddings of the whole match (frozen V-JEPA 2
  ViT-L from 1, 15 fps, 256×256); temporal action localisation (ActionFormer / TriDet-style, or a transformer over
  the embeddings) per player; evaluation as action spotting (precision/recall at ±k frames, per
  category and move).
- **Unlocks in the Lab:** wakeup habits, neutral whiffs, how often you get counter-hit by what,
  punish rates, combo route stats.
- **Cost:** weeks. Storage: whole-match embeddings ~3 GB fp16 mean-pooled at 15 fps, ~12 GB with a
  2×2 spatial grid.

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
