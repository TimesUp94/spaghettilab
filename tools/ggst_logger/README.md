# Strive replay logger and recorder

Research tooling for a labelled combo-opener dataset. It plays Guilty Gear -Strive- replays, records clean video of them, and logs the game's own per-frame state from memory. The state covers inputs, each player's current move name, facing, position and HP. The visual models trained on this data must work from video alone; this is only how the training labels are made.

Windows only. **Never go online with the mod installed.** To play online, remove `dwmapi.dll` (the UE4SS loader) from the game folder.

## Parts

- **`Main.cs` / `GGSTInputLogger.csproj`:** a UE4SS .NET plugin.
  - It hooks the battle update and input functions, using the signatures from [WistfulHopes/GGSTReplayTakeover](https://github.com/WistfulHopes/GGSTReplayTakeover) (MIT).
  - In replay and training modes it writes one CSV row per game frame to `$GGST_INPUT_LOG_DIR` (default `G:\Projects\replanal-data\ggst_logs`).
  - It starts a new file whenever the game frame counter restarts.
  - It also logs the game's pool of 100 objects (projectiles, summons, spells, detached hitboxes) as `objs` (`index/player/action/x/y/parent`, joined by `|`). Each player's hit-by field goes in `p1_hitby` / `p2_hitby`: the attacking player for a direct hit, the pool object (`o5`) for an object's hit. `dataset/build.py` uses these to give an object's hit to the move that made the object. With `GGST_LOG_REFS=1` set before starting the game, `p*_refs` and per-object refs list every pointer field, for finding new fields.
  - F6 and F7 write memory snapshots (`dump_*.bin`) that are used to find new fields.
  - The offsets are for game version 5.03 and are listed at the top of `Main.cs`. A game patch can move them.
- **`drive/`:** drives the game.
  - `ggst.py` sends keyboard input and takes window screenshots. Screenshots use FFmpeg 9's `gfxcapture`, so they work while the window is covered.
  - `nav.py` is the menu navigator. It checks every screen against `ref/*.jpg` before it presses a key.
  - `batch.py` plays saved replays one by one, records each one, and appends to `ggst_rec/manifest.jsonl`.
    A recording stops when the end-of-replay menu is recognised, 60 s after a player's `CmnActMatchWin` in the log, or after 10 minutes; the manifest's `stop` field says which (`menu`, `match over`, `max duration`). Only the recording stops early: the next key still waits for the recognised menu.
  - `dumps.py` and `act.py` read the memory snapshots.
- **`dataset/build.py`:** builds the dataset from the recordings.
  - It runs the CV pipeline on each recording.
  - It aligns game frames to video frames, using the qpc clock plus HP-drop matching.
  - It extracts every combo opener with the game's own move label and writes `ggst_dataset/openers.parquet` plus a frame strip for each opener.

## Setup

1. Install the .NET 8 SDK.
2. Install UE4SS together with its .NET plugin host into `GUILTY GEAR STRIVE\RED\Binaries\Win64`. The GGSTReplayTakeover v0.1.2 release bundles a compatible pair; leave its own plugin DLL out.
3. Build the plugin: `dotnet build -c Release -p:GGSTWin64="<that folder>"`.
4. Copy `bin/Release/net8.0/GGSTInputLogger.dll` into `DotNetPlugins/`.
5. Get FFmpeg 9 or newer, which has `gfxcapture`. Put it where `GGST_FFMPEG` points, or under `$GGST_DATA/ggst_mod/dl/`.

## Use

```bash
# Game running at the title screen, keyboard controls (U confirm, WASD, K/L tabs):
python tools/ggst_logger/drive/batch.py --start 0 --count 100
python tools/ggst_logger/dataset/build.py --idle
```

`GGST_DATA` sets the data root; the default is `G:\Projects\replanal-data`.

## More

- **`drive/search_batch.py`:** records other players' replays from Replay → Search, character by character and rank tier by rank tier (Iron … Vanquisher). It records `--per-rank` replays each and saves the character and rank range in the manifest.
- **`dataset/moves.py`:** names characters from their action prefixes or signature actions, and maps action names to notation, move name and category. The logger's `p*_sprite` column (e.g. `sol000_11`) starts with the character code.
- **`model/`:** visual models of combo openers. Each script states how to run it.
  - `extract_crops.py`, `finetune.py`, `train_heads.py`: the first models, on attacker crops placed by the game log (5-fold CV, no saved weights).
  - The video-only chain, which needs nothing from the log at inference:
    1. `video_openers.py`: combo starts and the attacker's side, from the CV pipeline's HUD readings plus the whole frame at 64×36 at every step (`video_tiny.py`; a small CNN for super flashes, cinematics with the HUD hidden and hitsparks), temporal conv + BiGRU. The image stream cut the false alarms by three quarters, so the chain detects at 0.1 instead of 0.3.
    2. `video_players.py`: both players' positions and both characters on 256×144 frames (ResNet18, CenterNet-style heatmap).
    3. `video_crops.py`: cuts the attacker crop strip around each detected start; `video_scene.py` adds the whole frame at 384×216 at the same moments (mirrored like the crop), for projectiles, summons and hits away from the attacker.
    4. `video_moves.py`: category, button/stance, exact move (restricted to the attacker's character), and "none" for a false start, from the crop and the whole frame (ResNet crop and ResNet18 whole-frame backbones, temporal fusion). The weights are an ensemble of a ResNet18 (20 epochs) and a ResNet50 (10 epochs) crop net, averaged over both and over the training frame shifts (-1..+2).
    5. `banner_reader.py`: counter / punish / neither from the game's own banner around each start: the yellow COUNTER / PUNISH word in the attacker's HUD box (the dataset build's templates), and a large counter hit's letterbox or big white COUNTER! (random forest on the scores of every 2nd frame from -60 to +60). It replaces `video_moves`' counter/punish head, which guessed from the attacker crop (0.65).
  - `recognize.py <video>`: runs the whole chain with the `*_final.pt` weights and writes `openers.json`. It runs on the GPU when there is one, else on the CPU (`--device`, `GGST_DEVICE`); `--light` keeps only the move ensemble's ResNet18 and drops the frame-shift averaging (about a third of the CPU time; on the 720p30 test copies the whole chain gets 0.878 instead of 0.888, exact move 0.932 instead of 0.945: `degrade_eval.py eval --light`). Weights come from `GGST_MODEL_DIR` when set. `server/opener_service` serves it to the website. All three finals (detector, `banner_reader`, `video_moves_final.pt`) are trained on all 1,090 replays.
  - `degrade_eval.py`: the same chain on 720p30 3 Mbit/s copies of the test replays (a typical VOD).
  - `chain_eval.py`: the whole chain on the test fold with another detector (cuts crops only for new detections); `--rebuild` cuts the chain's own crops and whole frames at `detections.json`, reusing the tiles of an earlier cut.
  - `error_report.py`: where the chain loses the test fold's openers (missed, dropped as "none", wrong move) by category, character and move. `video_moves.py` and `video_openers.py` take `--frac` for a learning curve.
  - Weights and results are in `$GGST_DATA/ggst_dataset/model/` and are not in the repo.
  - Folds: `model/folds.json` freezes the split of the first 722 replays, and later replays are split among themselves (`common.replay_fold_map`). This keeps the test fold comparable as the dataset grows. Results are reported for the whole test fold, its `old` replays (the frozen 144) and its `new` ones. `old_unleaked` is the old ones minus 5 replays that had been recorded twice with the other copy in training.
  - Duplicates: `dataset/build.py` drops replays recorded twice (same characters and logged inputs: `ggst_dataset/duplicates.json`), keeping the frozen test fold's copy.
  - `video_moves.py --init <weights>` warm-starts from an earlier net (heads mapped by class name).
    - Measured once: 8 epochs from the previous ResNet18 against 20 from ImageNet, on the same split. It saves 61% of the time (57 vs 146 min).
    - Whole chain: 0.878 vs 0.883 on the old test replays (within run-to-run noise), but 0.835 vs 0.846 on the new ones.
    - The weights here are trained from scratch.
    - `--accum 2` splits a batch into two micro-batches (needed for the ResNet50 on this machine's memory).
  - Test fold, from the 1080p60 recordings: 220 held-out replays of 1,090, of which the old 144 have 4,715 logged openers (bursts apart) and the new 76 have 2,612. Labels have counter / punish fixed since 2026-10-29 (`build.mark_counter_hits`). Numbers are for the old 144; new 76 in brackets.
    - Starts found: 0.955 at 0.943 precision (0.951 at 0.933).
      - Before the dataset grew: 0.95 at 0.92.
      - HUD only: 0.94 at 0.83 at threshold 0.3; supers 0.84, HUD only 0.72.
      - The app's damage events found 0.78 at 0.50.
    - Found and kept: 0.94, with 0.98 of the kept ones real (0.93, 0.98).
    - Characters right on every old replay (0.99 of the new ones' openers).
    - Given found: category 0.93 (0.95), button/stance 0.96 (0.95), exact move 0.94 (0.91), counter/punish/neither 0.98 (0.99).
    - The whole chain gets 0.887 of all openers found with the right move (0.855 on the new replays).
      - Before the dataset grew it was 0.865 (0.862 on the 139 old replays without a duplicate in training; the new chain gets 0.887 there).
      - Earlier: 0.84 with one ResNet18 move net and no frame-shift averaging, 0.82 with the HUD-only detector, 0.79 with the attacker crop alone.
  - On the 720p30 copies of the old test replays (`degrade_eval.py`):
    - the new chain loses nothing: starts 0.947 at 0.943 precision, found and kept 0.94 with 0.98 of them real, characters right on every replay, category 0.93, exact move 0.945, counter/punish/neither 0.98, whole chain 0.888;
    - the previous chain lost almost nothing too: starts 0.95 at 0.93, category 0.93, button/stance 0.94, exact move 0.93, counter/punish/neither 0.96, whole chain 0.86.

### Pretrained backbones (2026-11)

ROADMAP point 1: frozen pretrained backbones against the v2 ResNet move ensemble, like for like.

- Scripts: `model/backbone_feats.py` (extract pooled features per frame to `ggst_dataset/feats/<name>/`, fp16; train a small temporal head on them; `baseline`; `table`), `model/exp_bb_ensemble.py` (fixed 50/50 average with v2's test probabilities), `model/exp_bb_ft.py` (V-JEPA 2 fine-tune of the last blocks). Outputs are `exp_`-prefixed in `model/`.
- Same items as `video_moves.py`: the chain's crops and whole frames (`vcrops_test`, `vscene_test`), 29,090 training items from the non-test folds, 7,354 test items. Same vocabulary (904 moves), labels, losses and character masks; scored through `video_moves.test_results` / `end_to_end` with the `banner_reader` hit kind. The v2 baseline reproduces `results_v2_chain.json` to every digit.
- One fixed head setup for every backbone (30 epochs, AdamW 1e-3, wd 0.05, OneCycle, last epoch); 3 seeds, seed 0 reported, chain in brackets = 3-seed mean. Nothing selected on the test fold.
- Frozen means: no image augmentation; V-JEPA 2 gets one clip (no frame-shift averaging) and its whole frame squashed to 256×256. This tilts the comparison against the backbones, not for them.

Old 144 test replays (chain = all openers found with the right move; move = exact move given found; <5 = moves with <5 training examples; projectile / special = move given found, chain right):

| | chain | move | category | <5 | projectile | special |
|---|---|---|---|---|---|---|
| v2 ResNet18+50 ensemble (fine-tuned) | 0.887 | 0.944 | 0.935 | 0.57 | 0.77 / 0.69 | 0.93 / 0.88 |
| ResNet18 ImageNet, frozen (control) | 0.780 (0.780) | 0.837 | 0.879 | 0.26 | 0.66 / 0.59 | 0.86 / 0.80 |
| SigLIP 2 base, frozen | 0.805 (0.805) | 0.863 | 0.893 | 0.30 | 0.73 / 0.65 | 0.87 / 0.81 |
| DINOv2 base, frozen | 0.811 (0.812) | 0.867 | 0.899 | 0.37 | 0.73 / 0.65 | 0.88 / 0.82 |
| V-JEPA 2 ViT-L, frozen | 0.844 (0.841) | 0.900 | 0.905 | 0.40 | 0.74 / 0.66 | 0.92 / 0.86 |
| V-JEPA 2, last 8 blocks fine-tuned | 0.869 | 0.925 | 0.923 | 0.41 | 0.77 / 0.69 | 0.92 / 0.86 |
| v2 + DINOv2, 50/50 | 0.893 | 0.949 | 0.934 | 0.55 | 0.80 / 0.72 | |
| v2 + SigLIP 2, 50/50 | 0.888 | 0.945 | 0.936 | 0.50 | 0.79 / 0.72 | |
| v2 + V-JEPA 2, 50/50 | 0.889 | 0.945 | 0.937 | 0.53 | 0.78 / 0.71 | |

New 76 test replays:

| | chain | move | category | <5 | projectile | special |
|---|---|---|---|---|---|---|
| v2 ResNet18+50 ensemble (fine-tuned) | 0.855 | 0.915 | 0.949 | 0.59 | 0.75 / 0.72 | 0.91 / 0.82 |
| ResNet18 ImageNet, frozen (control) | 0.729 (0.734) | 0.784 | 0.876 | 0.30 | 0.65 / 0.62 | 0.83 / 0.74 |
| SigLIP 2 base, frozen | 0.760 (0.760) | 0.821 | 0.915 | 0.40 | 0.71 / 0.67 | 0.84 / 0.74 |
| DINOv2 base, frozen | 0.764 (0.762) | 0.826 | 0.903 | 0.30 | 0.65 / 0.62 | 0.85 / 0.75 |
| V-JEPA 2 ViT-L, frozen | 0.792 (0.791) | 0.854 | 0.923 | 0.45 | 0.71 / 0.67 | 0.88 / 0.78 |
| V-JEPA 2, last 8 blocks fine-tuned | 0.835 | 0.896 | 0.938 | 0.55 | 0.73 / 0.68 | 0.92 / 0.83 |
| v2 + DINOv2, 50/50 | 0.853 | 0.914 | 0.951 | 0.49 | 0.75 / 0.72 | |
| v2 + SigLIP 2, 50/50 | 0.847 | 0.907 | 0.957 | 0.52 | 0.75 / 0.72 | |
| v2 + V-JEPA 2, 50/50 | 0.855 | 0.916 | 0.954 | 0.55 | 0.76 / 0.72 | |

Costs (RTX 3090, 3 loader workers; one item = 21 frames × crop + whole frame):

| | extract (1,090 replays) | ms/item | throughput | per frame | features | head |
|---|---|---|---|---|---|---|
| ResNet18 | 10 min | 16 | ~2,600 tiles/s | 512-d | 1.5 GB | 5 min |
| DINOv2 base | 46 min | 76 | ~550 tiles/s | 768-d | 2.2 GB | 2 min |
| SigLIP 2 base | 35 min | 58 | ~720 tiles/s | 768-d | 2.2 GB | 3 min |
| V-JEPA 2 ViT-L | 92 min | 151 | ~290 frames/s at 256×256 | 1024-d per 2-frame tubelet | 1.5 GB | 3 min |
| V-JEPA 2 fine-tune | – | ~130 (train) | – | – | – | ~64 min/epoch × 4, 9.4 GB GPU |

- Ranking of the frozen features: V-JEPA 2 > DINOv2 ≈ SigLIP 2 > ImageNet ResNet18. V-JEPA 2 is +0.06 chain over the control and +0.03 over DINOv2 on both subsets (seed spread about ±0.004), closing a bit more than half the gap from frozen ImageNet to the fine-tuned v2. Its gain is largest on normals and the 5–19 bucket, where motion matters.
- None is close to v2 frozen: V-JEPA 2 is 0.043 chain below on old, 0.063 on new.
- The 50/50 averages with v2 are within noise on chain and move (old +0.001 to +0.006, new −0.008 to 0.000) and lower the <5 bucket; V-JEPA 2's lifts new projectile category recall from 0.74 to 0.81. Not worth a ViT pass at inference.
- Caveats from the independent check (all metrics recomputed from the saved probabilities with a separate loop, every value matched):
  - The <5 bucket has no class of its own for any of its moves (`collapse_rare` at 5): it scores picking the character's `(rare)` catch-all class, not the exact move. About 115 old / 87 new found openers, SE about 0.045, so the differences between frozen backbones there are noise; only the gap to v2 is real.
  - Seed 0 happens to be V-JEPA 2's best seed on old (0.844 vs 0.841 mean).
  - The head setup was fixed while the control's test numbers were visible; it is the same for every backbone and was run once, so the ranking holds.
  - On `old_unleaked` (139 replays) nothing moves: v2 0.887, V-JEPA 2 0.843 chain.
- Decision: **stay on the ResNets** for the opener chain (`*_final.pt` unchanged). The V-JEPA 2 fine-tune (`exp_bb_ft.py`, 4 epochs, `model/results_exp_bb_vjepa2_large_ft.json`, frame-shift averaged) gets chain 0.869 / 0.835: +0.025 / +0.043 over frozen but still 0.018 / 0.020 below v2, so it is not adopted (its 50/50 with v2 was not run). Its loss went NaN through most of epoch 2 and recovered in epoch 3. V-JEPA 2 ViT-L, frozen, is the backbone for the whole-match embeddings (ROADMAP point 2, below).

### Match transcript (2026-11)

ROADMAP point 2: every move start of both players from the video alone, not only combo openers.

- **Predicts**, per side, a list of events: start frame (60 fps), category (normal / special / super / throw / other = burst, RC, misc), button/stance for normals (15 classes), exact move (1,215 classes: `char:move`, moves with <5 training starts folded into `char:(rare)`), outcome (counter / hit / block / whiff / interrupted / teched / none), with scores.
- **Labels** (`dataset/transcript_labels.py`, CPU, ~50 s): 169,383 move starts in the 1,090 v2 replays, from the game logs, mapped to video frames with build.py's alignment. Outcomes are derived (the log has no hit/block flag): damage state = hit; guard state plus hit-by change, chip or blockstop freeze = block; G2's hit-by field credits contacts to the move (summons, projectiles). On G2 the fallback without hit-by agrees on 98.7% of outcomes. 99.9% of `openers.parquet` rows link to a dense event, 99.3% with the same move. Limits: in G0 a repeated 5P 5P is one event; without hit-by (G0/G1) about 15% of object blocks are missed.
- **Split and metric** (`model/transcript_eval.py stats | selftest | score preds.parquet --subset all|old|new`): test = frozen fold 0 (old 144 replays / 21,889 events, new 76 / 12,092); validation = 10% of the rest by name hash (93 replays), train 777 (`transcript_splits.json`). Action spotting per level: a prediction matches an unmatched true event of the same side and class within ±k frames (k = 2, 4, 8), greedy by |dt|; micro / macro P, R, F1 and mAP. `openers_recovered@k`: share of test openers whose dense event is matched at move level.
- **Features** (`model/transcript_feats.py`, `model/transcript_feats_grid.py`): frozen V-JEPA 2 ViT-L at 15 fps, 16-frame clips, one token per 8-video-frame tubelet (7.5 Hz). Three 256×256 views: the whole frame squashed, and a 640 px square on each player (positions from `video_players_test.pt`, mirrored so the player faces the opponent). Each player square also keeps 2×2 quadrant tokens, PCA 1024 → 192 (94% of the variance, fitted on 24 training replays). Plus the CV pipeline's 23 HUD features at 30 Hz and the voted characters.
- **Model** (`model/video_transcript.py train | eval | ensemble | predict`, final `ens5grid`): per side, conv stem + 4-layer local-window transformer (d 384, 6 heads, ±32 tubelets = ±4.3 s, relative-position bias), heads per tubelet for background + category, 8-way offset of the start inside the tubelet, move (restricted to the voted character), button/stance, outcome. 40 epochs, background weight 0.15, input/view dropout. 5 seeds averaged; epoch, threshold 0.65 and merge distance 8 frames chosen on validation.

Test fold, micro F1 (old 144 / new 76):

| level | k=2 | k=4 | k=8 | macro F1 k=4 | mAP k=4 |
|---|---|---|---|---|---|
| category | 0.749 / 0.722 | 0.815 / 0.798 | 0.840 / 0.825 | 0.812 / 0.792 | 0.795 / 0.759 |
| button/stance | 0.656 / 0.619 | 0.712 / 0.685 | 0.729 / 0.704 | 0.642 / 0.605 | 0.530 / 0.491 |
| exact move | 0.685 / 0.658 | 0.735 / 0.716 | 0.750 / 0.733 | 0.615 / 0.612 | 0.558 / 0.562 |
| outcome | 0.705 / 0.686 | 0.770 / 0.760 | 0.795 / 0.790 | 0.693 / 0.713 | 0.599 / 0.648 |

- At k=4: precision 0.855 / 0.845, recall 0.779 / 0.756 (category). Given a category match the outcome is right 0.925 / 0.931. The character vote is right on 439 of 440 test sides.
- Per category (k=4, whole test fold): other 0.94, special 0.82, normal 0.80, super 0.79, throw 0.68 F1.
- Found (any class, k=4) by true outcome: counter 0.93, hit 0.91, block 0.90, RC 0.94, interrupted 0.64, whiff 0.57. Move right when found: hit 0.93, block 0.94, whiff 0.82, interrupted 0.60.
- Starts less than 8 frames after the previous one of the same side are found 0.47 of the time (one start per tubelet, merge 8).
- The 2×2 grid tokens were the one kept improvement over the mean-pooled first model (`ens5`): move F1@4 0.697 → 0.735 old, 0.678 → 0.716 new; bootstrap over replays +0.038 [0.033, 0.042] old, +0.039 [0.033, 0.045] new. 95% interval on the absolute move F1@4 about ±0.014. One threshold per category (`ensemble --percat`) was within noise and is not used.
- Versus the opener chain, like for like (the chain's 4,715 / 2,612 openers without bursts, the chain's ±6-frame tolerance): the transcript recovers 0.834 / 0.823 with the right move, the chain 0.887 / 0.855. The anchors differ (move start here, hit frame for the chain). Keep the chain for openers; the transcript adds every other move. Its own `openers_recovered@4` (dense openers incl. bursts) is 0.845 / 0.835.
- By log generation (move F1@4): G0 old 0.807, G1 old 0.726, G2 old 0.716, G2 new 0.716, so the old/new gap is mostly composition.

Costs (RTX 3090):
- Features: 6.7 h for the three views of all 1,090 replays (98 sampled frames/s wall, GPU 99.8% busy, 1.6 GB GPU; median match 140 s → 62 s wall, 21 s of it V-JEPA 2 GPU time), plus 4.7 h for the grid pass (a second run of the two player views; one combined pass would not need it). Disk: 7.0 GB + 3.4 GB fp16 under `ggst_dataset/transcript/` (about 10 MB per match).
- Training: about 12.5 min per net (15–19 s per epoch, all features held on the GPU, ~15 GB peak), 1.05 h for the 5 members. All training experiments: about 3.8 h for the first model's sweep, 2.2 h for the grid round (pilot, 5 nets, ensembles).
- Inference per match (median 140 s match at 1080p60): the feature pass about 62 s wall, bound by the CPU decode (21 s of it on the GPU), plus about 15 s for the separate grid pass; then the 5-net head in about 3 s including loading. Weights `model/video_transcript_exp_ens5grid.pt` (191 MB, all members). Not tried on the CPU or on 720p30 copies.

Verification caveats (independent check, CPU only):
- Every headline number reproduces from `transcript/exp_tr_ens5grid_test_preds.parquet` with a separate matcher; optimal (Hungarian) matching gives the same F1.
- No train/val/test overlap; no undetected duplicate among same-character replay pairs (move-sequence similarity < 0.5).
- `video_players_test.pt` predates `duplicates.json`, so 5 old-test replays had a dropped copy in the player net's training. On `old_unleaked` (139) nothing moves: move F1@4 0.734, category 0.815.
- Label/video alignment: on 4 test replays (G0, G1, G2 old, G2 new) the motion onset around the mover peaks at offset 0 from the labelled start.
- The merge distance chose 8, the edge of its grid (4, 6, 8); larger was never tried.
- Early runs and the per-category variant also scored the test fold; selection used validation only.

Not solved yet:
- Moves without contact: whiffs 0.57 and interrupted moves 0.64 found; interrupted moves get the right move 0.60 of the time.
- Close starts (< 8 frames apart, e.g. 5P 5P, cancels): one start per 7.5 Hz tubelet. Next: 15 Hz tokens or two slots per tubelet.
- Throws (0.68 F1) and button/stance (0.71 / 0.69) lag the other levels; rare moves are scored as the `(rare)` catch-all only.
- Each 16-frame clip is encoded on its own (one-sided context at clip edges); the scene view is squashed, not letterboxed; during teleports, overlaps and cinematics both player squares can show the same player.
- Labels: G0/G1 object blocks and contact credit are noisier; HC Sniper shots and some Ramlethal openers are mislabelled in `openers.parquet` (build/moves bugs, not fixed).
- Not in `recognize.py`, the app or the CPU server; the feature pass needs a GPU (ROADMAP point 6).
