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

## Hitbox data

Research only and private: the logs and the database stay on the data drive and nothing here is published beyond this code.

- **Logging:** `GGST_LOG_BOXES=1` adds `p1_boxes`, `p2_boxes` and `obj_boxes` to the log; without it the log is unchanged. How boxes are read and transformed is ported from [AltimorTASDK/strivehitboxes](https://github.com/AltimorTASDK/strivehitboxes) (MIT, Copyright (c) 2021 Altimor), with offsets re-checked on 5.03. The format is described above `BoxRecord` in `Main.cs`. A game started through Steam does not inherit the caller's environment, so settings can also go in `%LOCALAPPDATA%\ggst_logger.env` (`NAME=value` lines); delete that file to go back to ordinary logging.
- **`drive/train_sweep.py`:** training mode (offline): every character tries every normal and every typable move of `dataset/moves.py`, whiffing from long range, then c.S and the throw next to the dummy. Keys go only to a recognised training HUD; the pause menu, the dialog and the character select are each checked, the cursor by its red frame. It stops on a locked or remote session or under 25 GB free, and waits while free commit is under 6 GB. Its reference screens are in `$GGST_INPUT_LOG_DIR/ref` and not in the repo.
- **`drive/box_batch.py`:** replays from Replay -> Search, no video, own manifest and state in the box log directory. Closes the game with WM_CLOSE only.
- **`dataset/hitbox_db.py`:** `build` makes the per-move database (`$GGST_DATA/ggst_hitboxes`), `verify` checks that logged hits happen where a hit box overlaps a hurt box, `export` writes `dustloop_hitboxes/gamedata/` for the local `compare.html`.

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
  - `hud_chars.py`: both characters from the game's HUD, per stretch of the video: the portrait in the corner, then the name on the bar, then `video_players`' visual vote as the last fallback. One art per character from the logged replays (`hud_chars_refs.npz`, next to the weights, not in the repository), tried as drawn and mirrored (recordings and streams draw P2's differently). Test fold, references built without it: portrait 440/440 sides at 1080p60 and on the 720p30 copies, the old visual vote 439/440. On the 14 tournament sets (alternate colours, never seen) the portrait and the name agree on all 44 sides of the 22 stretches and the visual vote is wrong on 8. `results_hud_chars.json`, `results_hud_chars_sets.json`.
  - `video_openers.continues_combo`: drops a detection when the attacker's combo counter was already showing and goes up through it (a new combo's first hit shows no number). `chain_eval.py --mid-combo` -> `results_mid_combo.json`. **Label note:** `dataset/build.py` does not count `CmnActZSpin` (spinning defender) as a damage state, so hits on a spinning defender are logged as openers although the game's counter goes on: 1.8 % of `openers.parquet`. Without them in the truth the rule takes the false starts kept from 152 to 142 for 0.0003 of the chain score; with them the score falls 0.0011 / 0.0019 (old / new), mostly those labels.
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
- Wired into `model/lab_analyze.py` (2026-11-04: `lab_result.json`, see `model/LAB_RESULT.md`): it runs on a single new video only with a GPU, so not on the CPU server; the feature pass needs a GPU (ROADMAP point 6). For a new video the feature pass reproduces the dataset features (cosine 1.000, checked on the first 2 s of a test replay); the head on the CPU reproduces the saved test predictions event for event.

### Inputs from video (2026-11)

ROADMAP point 3: each player's held direction and button presses per video frame (60 fps), from the video alone.

- **Labels** (`dataset/inputs_labels.py`, CPU, ~40 s): from the logs' `p*_rec` and `p*_face`, mapped to video frames with build.py's alignment like the transcript. Per frame and side: direction 1–9 relative to the facing, press edges and held bits of P/K/S/H/D, the log's state (move, idle, walk_f, walk_b, dash, guard, jump, hit) and a `live` flag (no round intro / win states). `ggst_dataset/inputs/labels/<stem>.npz` (26 MB for all 1,090), `inputs_motions.parquet` (the command read at every transcript event, like `verify_inputs.read_command`). On live frames: 5 is 42% of the frames, 4 19%, 1 17%, 6 12%; 467,762 presses.
- **Metric** (`model/inputs_eval.py score | selftest | baseline`), test fold old / new, live frames, same splits as the transcript:
  - direction accuracy per frame and per 7.5 Hz step, balanced, per state ("without a visible move" = every state but `move`);
  - presses as spotting per side and button at ±1 / 2 / 4 frames (local maxima over ±2 frames above a threshold tuned on val); recall split by whether a move of that side starts within 10 frames after the press;
  - commands (motion + button, read from the streams like `verify_inputs.read_command`: 236K, [4]6S, 2H) and the plain direction + button at the press, at the true move start (`oracle`) and at the start of the transcript prediction matched to the event at the move level within ±4 (`tr`, read at that start + 2 frames: 2 was best on val with the labels as predictions), against the move prior (the move's most common command in training). The reader on the labels agrees with the log-row reading on 98.7% of the events.
- **Model** (`model/video_inputs.py`): the transcript's inputs (V-JEPA 2 tokens with the grid, HUD, positions, characters) plus the transcript's posteriors per tubelet (category, offset, normal, outcome, top-8 moves through a move embedding; both sides); the transcript's trunk (local-window transformer), then each tubelet expanded to its 8 frames and two 60 fps convolutions; heads for direction (softmax) and presses (sigmoid, target spread ±1 frame). Posteriors: val / test from `ens5grid`; training replays out-of-fold from 4 single transcript nets with ens5grid's member setup, each trained on 3/4 of the train split (`video_inputs.py oof`, `video_transcript_exp_in_oof0-3.pt`), since ens5grid's own posteriors on its training replays are over-confident. 30 epochs, epoch and press threshold chosen on val.

Test fold (old 144 / new 76):

| | dir/frame | balanced | no visible move | press F1 ±1 | ±2 | ±4 | press recall ±2: move / no move | cmd given move | dir+button given move |
|---|---|---|---|---|---|---|---|---|---|
| direction 5 + one press per transcript event (`transcript_only`) | 0.443 / 0.403 | 0.111 | 0.369 / 0.326 | 0.188 / 0.186 | 0.252 / 0.250 | 0.319 / 0.321 | 0.40 / 0.02 | 0.305 / 0.277 | 0.341 / 0.311 |
| model without the posteriors (`nopost`) | 0.590 / 0.567 | 0.36 | 0.604 / 0.579 | 0.140 / 0.137 | 0.209 / 0.205 | 0.281 / 0.276 | 0.36 / 0.14 | 0.255 / 0.249 | 0.296 / 0.286 |
| **model (`base`)** | **0.597 / 0.572** | 0.37 | **0.606 / 0.578** | **0.191 / 0.190** | **0.276 / 0.276** | **0.357 / 0.361** | 0.44 / 0.09 | 0.424 / 0.405 | 0.493 / 0.477 |
| + event command head (`cmd`) | 0.594 / 0.564 | 0.35 | 0.601 / 0.569 | 0.162 / 0.169 | 0.237 / 0.246 | 0.312 / 0.323 | 0.44 / 0.11 | head 0.578 / 0.564 | head 0.641 / 0.633 |
| the move's usual command (move prior) | | | | | | | | **0.593 / 0.576** | **0.653 / 0.650** |

- Direction per state (base, old / new): idle 0.77 / 0.70, guard 0.75 / 0.73, walk back 0.74 / 0.72, dash 0.60 / 0.57, move 0.58 / 0.56, walk forward 0.54 / 0.57, hit 0.51 / 0.50, jump 0.44 / 0.44. Against each state's own most common direction (see the caveats below) the model adds little in idle and in moves and is far worse than the trivial rule while walking. Vertical part (up / neutral / down) 0.80, horizontal 0.70. Per 7.5 Hz step 0.600 / 0.575 (no better than per frame).
- Presses per button, F1 ±2: S 0.32, H 0.28, D 0.26, K 0.24, P 0.19. 60% of the logged presses start no visible move within 10 frames (mashed, buffered, kara, cancelled into nothing); the model finds 9% of them.
- Given a move the transcript spotted right (69% / 67% of the events with a command), its usual command beats everything read from the video: logged 236 and 214 motions 0.93 and 0.94 (old), but logged 41236 and 623 motions only 0.23 and 0.24 (players type 41236 for a 236 move, 6236 rolls into 623; the reading takes the longest motion). The per-move mode fitted on the test fold itself reaches only 0.60 (command) / 0.67 (direction + button): the remaining third is which variant the player typed, and nothing tried here sees it. Motion recognition for specials from the video equals move recognition.
- The posteriors are what gives the presses (F1 ±2 0.209 → 0.276; command reading 0.255 → 0.424); the direction gains little from them (0.590 → 0.597).
- Timing (`base`, transcript-matched moves, the press read at the matched start): within ±1 frame of the true press 0.48, ±2 0.69, ±4 0.89.

Stage 2, per-frame features (`model/inputs_frame_feats.py`, `model/inputs_refine.py`):
- 60 fps DINOv2-small CLS (384-d) of both player squares (cut and mirrored like the transcript's squares, positions interpolated from the 15 fps track), for val, test and 250 training replays (563): 2.4 h (4.9 M frames, ~570 frames/s wall: bound by the CPU decode and crop, 3 workers), 7.1 GB fp16 (`inputs/frame_feats/`). The features change sharply 1–2 frames before the labelled move start (the frame-to-frame change peaks at −2 and −1 for normals), so they carry frame-level timing.
- Dense model on the 250 replays, with vs without them (`ff` vs `ffctl`): F1 ±1 0.122 vs 0.125, ±2 0.181 vs 0.183, direction 0.554 vs 0.553: no gain. Warm-started from `cmd` (`ffinit`, 15 epochs): same as `cmd`.
- Window refiner around the transcript's matched starts (32 frames, both squares, conv; anchor jitter drawn from the transcript's start errors on val): start exactly right 0.336 → 0.376 / 0.346 → 0.378, within ±1 0.792 → 0.828 / 0.776 → 0.806, within ±2 0.933 → 0.925 / 0.919 → 0.912. The press is not improved (within ±1: 0.530 → 0.527): the press of a buffered move precedes its visible start by 0–8 frames. Trained with uniform ±6 jitter instead it is worse than the transcript (±1 0.740).

Costs (RTX 3090): 4 OOF transcript nets 30 min; posteriors 2 min; a dense model 7 min (`cmd` 9 min) on all features held on the GPU (~15 GB), test inference 15 s for 220 replays; frame features 2.4 h; refiner 1 min. Total about 3.7 h of GPU. Disk 8.1 GB under `ggst_dataset/inputs/` (7.1 GB of it the frame features).

Not solved:
- Presses without a visible move (60% of presses) and which motion variant was typed: not visible in these features, likely not at all.
- Directions inside moves, hitstun and jumps (0.44–0.58): the stick is not shown. Walks are not recognised either (0.54 / 0.74 where the state alone gives 0.99).
- The dense model overfits after ~20 epochs on 777 replays; no ensemble or seeds; train-time posteriors are weaker OOF single nets while test uses `ens5grid`; frame features for only 250 training replays; not tried on 720p30 copies; not in `recognize.py`, the app or the CPU server.

Verification caveats (independent check, CPU only, from the saved predictions):
- Reproduced: direction 0.597 / 0.572 (always 5: 0.443 / 0.403), press F1 ±2 0.276 / 0.276, `nopost` 0.590 / 0.567 and 0.209 / 0.205, move prior on matched events 0.593 / 0.576 and 0.653 / 0.650. Nothing from the log reaches the model at inference; OOF nets, move prior, thresholds and epochs come from train / val only. The `nopost` threshold (0.2) is at the edge of its grid, but on val it beats 0.15 and 0.1.
- Per state, model vs that state's most common direction (old / new): idle 0.77 / 0.70 vs 0.72 / 0.64 (always 5); move 0.58 / 0.56 vs 0.56 / 0.52; guard 0.75 / 0.73 vs 0.63 / 0.60 (always 1); dash 0.60 / 0.57 vs 0.37 / 0.40; jump 0.44 vs 0.25; hit 0.51 / 0.50 vs 0.36 / 0.32; walk forward 0.54 / 0.57 vs 0.99 / 1.00 (always 6); walk back 0.74 / 0.72 vs 0.99 (always 4). So "readable in neutral, not in the air or hitstun" is backwards: the gains are largest in jump, hit, dash and guard, and the model does not recognise walking.
- The command ceiling above (0.60 / 0.67) is over all test events, the prior only over matched ones. On the same matched events the ceiling is 0.633 / 0.637 (command) and 0.687 / 0.706 (direction + button), so the prior is 0.04–0.06 below it. Over all test events with a command the prior gets 0.558 / 0.540.
- Frame alignment: a 60 fps pixel check on 6 test replays puts the HUD HP change at label frame −1, 0 or +1 depending on the replay. The labels are right to about ±1 frame, which bounds the exact-frame numbers (refiner exact 0.34 → 0.38, press F1 ±1). The 1–2 frame lead of the DINOv2 features is within that, not a consistent offset.
- The refiner's test input uses the logged event's category instead of the predicted one; only one move of 1,325 (Leo bt.P) has two categories, so the effect is negligible.

### Search and state embeddings (2026-11)

ROADMAP point 4: search over the transcript's events, a state embedding of the video, and habit groups. All of it uses only the video at inference; the log makes the labels and the scores. Test fold old 144 / new 76.

**Event index** (`model/search_index.py build | query | eval | habits`, CPU, seconds):
- One row per transcript event: replay, side, frame / time, character and opponent, category, move, button/stance, outcome, score, opener, the previous and next event of each side (move, outcome, gap), each side's state at the start and `after_kd` (frames since the side woke up). Two copies are built with the same fields: `pred` (the `ens5grid` predictions; `opener` = a cross-fitted `detections.json` combo start of that side 4 frames before to 40 after the start; state from the state embedding below) and `log` (the logged events, moves outside the vocabulary folded to `(rare)`). Output: `search/search_index_{pred,log}_test[_<state>].parquet`.
- Queries: `query(ix, char="Sol", move="5H", outcome="whiff")`, `query(ix, char="Ky", opp_char="Sol", outcome="counter")` (Sol counter-hit by Ky), `query(ix, char="Sol", after=dict(who="own", outcome="block", within=40))`, `query(ix, side="P1", after_kd=True)`. The CLI prints the time in the recording and the neighbouring events.
- Scoring: query families are built from the logged test events (every key with ≥ 5 logged results). Each query runs on both indexes. A returned moment counts as right if it matches a logged one (same replay and side, ±8 frames, one-to-one). Micro P / R / F1 over all returned moments, k = 8 (k = 4 in brackets):

| family (queries old / new) | precision | recall | F1 (k=4) | mean per-query F1 | median count error |
|---|---|---|---|---|---|
| move + outcome, e.g. whiffed Sol 5H (1,352 / 753) | 0.822 / 0.831 | 0.756 / 0.750 | 0.787 / 0.788 (0.775 / 0.773) | 0.729 / 0.749 | 17% / 17% |
| – whiff only | 0.659 / 0.682 | 0.463 / 0.478 | 0.544 / 0.562 | | |
| – block / connected (hit or counter) / counter, old | 0.854 / 0.873 / 0.838 | 0.862 / 0.875 / 0.809 | 0.858 / 0.874 / 0.823 | | |
| move (871 / 633) | 0.793 / 0.788 | 0.724 / 0.707 | 0.757 / 0.745 (0.741 / 0.728) | 0.682 / 0.696 | 18% / 17% |
| X counter-hit by Y (184 / 103) | 0.854 / 0.872 | 0.818 / 0.829 | 0.836 / 0.850 (0.821 / 0.829) | 0.831 / 0.852 | 13% / 11% |
| combo opener with move M (355 / 202) | 0.874 / 0.869 | 0.711 / 0.725 | 0.784 / 0.791 (0.766 / 0.772) | 0.717 / 0.722 | 14% / 20% |
| next own move after a blocked one (34 / 30) | 0.822 / 0.843 | 0.859 / 0.842 | 0.840 / 0.842 (0.832 / 0.835) | 0.835 / 0.848 | 7% / 7% |
| first move after a knockdown (34 / 31; state from `clip_sup`) | 0.824 / 0.790 | 0.678 / 0.664 | 0.744 / 0.721 (0.697 / 0.680) | 0.742 / 0.729 | 17% / 18% |

- What works: hits, blocks, counter hits and openers come back at 0.8–0.87 F1. Counts are off by about 15% per query, which is good enough for "how often" stats. Whiff queries are the weak spot (recall 0.46–0.48), the transcript's own weak spot. Punish cannot be searched from the video: the transcript has no punish output (the chain's banner reader would add it).

**State labels** (`model/state_embed.py labels`, CPU, 30 s; `search/state/<stem>.npz`): per 7.5 Hz tubelet, the log row at its middle frame:
- per side: x, y, HP / 420, facing, and a class: knockdown (B/F/V Down loop/down/bound, QuickDown, the wake-up), hitstun (other damage states, throw tech), blockstun (guard states), attacking (any move), airborne, or neutral;
- the distance, and whether each side is cornered (within 0.3 of a wall and nearer to it than the opponent).
- Raw x drifts after wallbreaks: stage sections are centred on multiples of 1.6 (x 10^6 units), with walls at ±1.515. `stage_centre` picks the section by a Viterbi path over that grid. It agrees on 99.6% of the tubelets in segments where both walls are touched. Afterwards 0.1% of live tubelets fall outside the walls (players flying between sections in the wallbreak transition); those are dropped.
- The logs have no tension or burst values, so those are not in the state. The HUD's tension reading is an input.

**Model** (`model/state_embed.py train | eval | predict | similar`):
- Video tower: per tubelet, the transcript's frozen V-JEPA 2 tokens (scene and both player squares, plus the 2×2 grid) and the HUD/position aux. They go through a 2-layer transformer over 16 tubelets (2.1 s) centred on the moment, and the centre token becomes a 128-d embedding.
- State tower: an MLP over the state vector.
- Loss: symmetric InfoNCE over batches of 1,024 training tubelets (CLIP-style); `--sup 1` adds supervised state heads on the video embedding.
- Training: 40 × 300 steps, 4–6 min per run, with all training tokens held on the GPU (8.4 GB).
- Checks on the test fold, against frozen embeddings:
  - probes fitted on 150k tubelets of 200 training replays (linear, and a 2-layer MLP);
  - situation retrieval: 4,000 query moments, their top-10 nearest test moments from other replays, and how often those share the query's logged joint class (P1 class, P2 class; random 0.084), corner state and distance. Nearest neighbours in the logged state vector itself get a joint class of 1.00.

| embedding (old / new) | class per side (MLP probe) | balanced | who is hit | corner | distance MAE / R² (old) | retrieval@10: joint class | corner (old) | Δ distance (old) |
|---|---|---|---|---|---|---|---|---|
| raw tokens, PCA 256, no tower (`raw`) | 0.757 / 0.740 | 0.72 / 0.70 | 0.864 / 0.857 | 0.846 / 0.841 | 0.096 / 0.79 | 0.345 / 0.325 | 0.64 | 0.18 |
| CLIP only (`clip`) | 0.868 / 0.852 | 0.84 / 0.83 | 0.963 / 0.958 | 0.927 / 0.926 | 0.049 / 0.94 | 0.553 / 0.537 | 0.86 | 0.08 |
| supervised heads only (`sup`) | 0.899 / 0.885 | 0.88 / 0.87 | 0.971 / 0.967 | 0.935 / 0.933 | 0.055 / 0.93 | **0.752 / 0.721** | 0.91 | 0.16 |
| **CLIP + supervised (`clip_sup`)** | **0.899 / 0.885** | 0.88 / 0.87 | **0.972 / 0.970** | **0.936 / 0.936** | 0.051 / 0.94 | 0.662 / 0.643 | 0.90 | **0.09** |
| `clip_sup`, window 32 (batch 512) | 0.903 / 0.888 | 0.89 / 0.87 | 0.975 / 0.974 | 0.944 / 0.941 | 0.050 / 0.94 | 0.672 / 0.646 | 0.90 | 0.10 |
| `clip_sup`, batch 512 (its control) | 0.896 / 0.883 | 0.88 / 0.86 | 0.971 / 0.969 | 0.935 / 0.936 | 0.051 / 0.93 | 0.667 / 0.642 | 0.90 | 0.09 |
| `clip`, window 8 | 0.861 / 0.845 | 0.84 / 0.82 | 0.954 / 0.948 | 0.914 / 0.910 | 0.052 / 0.93 | 0.548 / 0.530 | 0.85 | 0.08 |

- Per class (`clip_sup`, MLP probe, old, P1/P2): neutral 0.86, attacking 0.92, blockstun 0.91, hitstun 0.93, knockdown 0.87, airborne 0.80. Stage x R² 0.90, HP MAE 0.024 of the bar.
- The tower is what does the work: a linear probe goes from 0.717 on the raw tokens to 0.899 with the tower.
- CLIP alone gives the best distance retrieval. The supervised heads give the best class retrieval, but they lose the distance (Δ 0.16). Together they get close to the best of both.
- `clip_sup` is the state stream used by the index and the habits. A 32-tubelet window adds about 0.005, within noise (validation tie).
- `similar --name clip_sup --stem S --time m:ss.s` lists moments like the given one in other replays (val + test embeddings from `predict`, 90 MB). Example: P1 blocking in the corner at distance 0.18 returns five corner-blockstun moments at distance 0.16–0.25 and one P2-corner hitstun.

**Habits** (`search_index.py habits --state clip_sup`, per character, test fold):
- Wake-ups (from the state stream: the end of a knockdown that is not followed by another hit). The option is the side's first event from 8 frames before to 45 after, or, without one, what the state shows: blocks, jumps, gets hit or waits.
  - The predicted wake-ups match the logged ones (±16 frames): precision 0.89, recall 0.85. On matched pairs the option is the same 0.67 of the time.
- Blocked strings (≥ 2 consecutive blocked moves of one side, each within 40 frames). The option is the attacker's next move within 60 frames, or "stops".
  - Found: precision 0.76, recall 0.83. Option the same on matched pairs 0.86.
- Per character, predicted vs logged option shares, as total variation distance weighted by situations: wake-up 0.092 by category, 0.215 by move; blocked string 0.081 / 0.169. For scale: two random halves of the same character's logged replays differ by 0.33 / 0.44 (move). So the model's error is smaller than the spread between samples of players, though the halves are smaller samples.
- Bias: the model reports "no move" after 0.63 of wake-ups where the log has 0.56 (old and new). It misses wake-up moves the way the transcript misses whiffs.
- Examples (log / pred top options):
  - Sol wake-up: blocks 0.34 / 0.30, waits 0.12 / 0.20, jumps 0.09 (log), 2P 0.07 (pred).
  - Chipp after a blocked string: stops 0.30 / 0.33, 236S 236K 0.17 / 0.17, 236S 236S 0.15 / 0.15.
  - A.B.A after a blocked string: 214K 0.26 / 0.24.
  - Timestamps per option: `model/results_exp_se_habits_clip_sup.json`.

Costs: about 1 GPU-hour for all of point 4 (9 trainings, 4–6 min each, plus evaluations of about 30 s); no new feature extraction. Disk: 104 MB under `ggst_dataset/search/`, 10 MB per tower checkpoint.

Not solved:
- Whiff search (recall 0.46–0.48) and wake-up moves are limited by the transcript, not by the index.
- Neutral, walking and dashing are one class here (the inputs model does not split them reliably: it reads walks worse than the walk state alone would).
- There is no tension or burst in the state: the log has neither.
- Habits are per character, not per player: the test fold has no player ids. Groups are option shares, not learned clusters.
- No punish key; the index is test-fold only (no index over unlabeled VODs yet); not in the app.
- The index covers only val / test. `lab_analyze.py` computes the state stream of a new video with the transcript (GPU only), and the UIs filter its events with the presets of `model/LAB_RESULT.md`. There is no index over many VODs yet; not tried on 720p30 copies; single seed. `state_meta.json` still holds an unused `wall` value from the first absolute-x estimate (corners come from `stage_centre` / `load_state`).

Verification caveats (independent check, CPU only):
- Reproduced: move + outcome F1 0.787 / 0.788 over 1,352 / 753 queries; `clip_sup` joint-class retrieval@10 0.658 / 0.647 with all test replays as the pool (reported 0.662 / 0.643). No leakage found (probes fitted on training replays, positions from the player net that never saw the test replays).
- The index F1 is optimistic: only keys with ≥ 5 logged results become queries, so wrong predictions under rare or log-absent keys are never counted. Over all (character, move, outcome) keys, `(rare)` excluded: P 0.788 / 0.789, R 0.726 / 0.711, **F1 0.756 / 0.748**.
- The random baseline (0.08) is weak. Random tubelets from other replays with the same joint class predicted by the `sup` heads already reach 0.597 / 0.604, so the embedding adds about 0.05 over its own class heads.
- The 0.899 / 0.885 class accuracy is an MLP probe. The class stream the index and habits actually use (the `sup` heads) is right 0.855 / 0.848 per side (P1 / P2) on old and 0.845 / 0.836 on new.
- Habits: the split-half TV (0.33 / 0.44) is not a noise floor for predicted-vs-logged TV (0.22 / 0.17): the halves have half the samples, and predicted and logged shares are measured on largely the same situations. It does not show that the habit shares are accurate; the matched-pair agreement (0.67 wake-up, 0.86 blocked string) is the better number.

### Language layer (2026-11)

ROADMAP point 5, local part: match facts, prompts in Italian and English, and a grounding checker for LLM answers. No network, no GPU; it reads predictions that already exist. Which LLM provider (if any) may receive the facts is not decided.

- **Scripts:** `model/lang_summary.py facts | run | check | verify` (pure Python, no torch); `model/lang_testdata.py` (CPU only, GPU hidden) rebuilds an `openers.json` in `recognize.py`'s schema for test-fold replays from the saved chain outputs (detector starts, v2 move-ensemble test probabilities, voted characters, banner reader), never the detections' log link, and copies that replay's transcript predictions and HUD parquet. Tests: `tests/test_lang_summary.py` (14, synthetic data, 0.2 s).
- **Facts** (`facts.json`, `facts_<lang>.md`): rounds and winners from the HUD, opener timeline with damage, KO and counter / punish per opener, each player's opener mix, top opening moves and biggest combos. With the transcript: attack outcome rates, most used attacks, what counter-hit each player and what they were doing, habits (next move after their own combo, after getting comboed, after blocking, blocked strings, each with n). Every fact has m:ss.s video timestamps; the facts record every timestamp, move, percentage and count they show (a test checks it equals the rendering).
- **Uncertain marks** (thresholds set by hand, not tuned): move p < 0.5, category p < 0.6 (alternatives listed), banner p < 0.6 ("unclear"), rare-class moves; rounds without a clean KO, with hearts disagreeing, a near double KO or an impossible first-to-2 sequence. Uncertain openers are left out of the top-move counts.
- **Prompts** (`prompt_<lang>.json / .txt`): system prompt + facts, asking for a 2–4 sentence summary and 3–5 numbered coaching notes, each citing [m:ss.s]; no frame data, no moves or characters outside the facts, no new numbers. 6.7k–8.5k characters, about 2,000–2,570 input tokens per match (chars / 3.3 estimate, no tokenizer offline, ±30%).
- **Providers:** `provider="none"` (the only one the CLI offers) writes the prompt. `ClaudeProvider` is a stub (model `claude-opus-5-5`, max_tokens, effort, server-side fallback configurable) that raises unless built with `enabled=True` **and** `GGST_LANG_ALLOW_NETWORK=1` is set; it has never been called, and the `anthropic` package is not installed. It sets `cache_control`, which is harmless but useless here: the ~400-token system prompt is below the minimum cacheable prefix.
- **Checker** (`check_answer`, CLI `check <answer.md> <facts.json>`). Errors: timestamps not in the facts (±1.5 s), characters not in the match, moves not in the facts (notation, Throw / Burst / RC, English names incl. short forms), frame-data claims, percentages not in the facts, coaching notes without a timestamp, and a move named next to a timestamp where the facts have another move (one move and one time in the sentence). Warnings: counts not in the facts, the move / time mismatch with several moves or times in a sentence, the note count.

Run on 6 test-fold replays (3 old, 3 new; `ggst_dataset/lang/<stem>/`, 1.6 MB), scored against the log (`verify`):
- Round winners right on all 6. Openers without an uncertain mark match the logged opener and move within ±6 frames 146 of 170 (86%); uncertain ones 10 of 15, so on these 6 the p < 0.5 mark does not separate good from bad readings well.
- Round winner rules on the whole HUD: train + val (870 replays, never looked at) 831 / 870 replays fully right; rounds without an uncertain mark 1,914 / 1,921 (99.6%), uncertain rounds 156 / 191. On the test fold 202 / 220 replays and 474 / 480 unmarked rounds, but the rules were changed while looking at test errors, so that is not a clean number.
- Checker on hand-written answers for one replay: good en and good it pass with 0 errors; a bad en answer fails with 10 errors (invented 2:30.0, Sol, 6H twice, Gun Flame, "-5 on block", 55%, Throw at 0:22.4 and 6H at 1:12.5 where the facts have other moves, a note without a timestamp) and 1 warning.

Verification caveats (independent check): no network call is possible and the tests pass. The checker catches invented m:ss timestamps (bracketed or not), invented notation (236236H), another character's move names (Volcanic Viper), a wrong move at a cited time and unknown characters. It misses:
- lowercase or case-varied names ("gun flame", "sol badguy"), slang ("DP", "super") and other notations ("5HS", "6HS", "cS");
- times written in seconds ("225 seconds in"), "N percent", numbers in words;
- frame-data claims without numbers ("very unsafe, punish it");
- a wrong match result or score in the summary ("P1 Asuka won 2-0" passes).

Not solved:
- The provider decision (what may leave the machine, which model); no real LLM answer has been generated or checked yet.
- The checker gaps above, which matter before any real answer is shown to a user.
- The uncertain threshold (p < 0.5) is not calibrated; recalibrate on val.
- One prompt for a whole set (citations like [M2 1:23.4]) is not built. The facts and prompts of one match are in `lab_analyze.py`'s `lab_result.json` (CPU server included); no LLM is called.
