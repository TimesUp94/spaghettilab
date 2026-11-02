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
| V-JEPA 2, last 8 blocks fine-tuned | running | | | | | |
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
| V-JEPA 2, last 8 blocks fine-tuned | running | | | | | |
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
- Decision: **stay on the ResNets** for the opener chain (`*_final.pt` unchanged). Pending the V-JEPA 2 fine-tune (`exp_bb_ft.py`, results in `model/results_exp_bb_vjepa2_large_ft.json`): adopt it, alone or in the ensemble, only if it beats v2's chain on both subsets by more than the seed noise. V-JEPA 2 ViT-L is the backbone for the whole-match embeddings (ROADMAP point 2).
