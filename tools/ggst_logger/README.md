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
    1. `video_openers.py`: combo starts and the attacker's side, from the CV pipeline's HUD readings plus the whole frame at 64�36 at every step (`video_tiny.py`; a small CNN for super flashes, cinematics with the HUD hidden and hitsparks), temporal conv + BiGRU. The image stream cut the false alarms by three quarters, so the chain detects at 0.1 instead of 0.3.
    2. `video_players.py`: both players' positions and both characters on 256×144 frames (ResNet18, CenterNet-style heatmap).
    3. `video_crops.py`: cuts the attacker crop strip around each detected start; `video_scene.py` adds the whole frame at 384×216 at the same moments (mirrored like the crop), for projectiles, summons and hits away from the attacker.
    4. `video_moves.py`: category, button/stance, exact move (restricted to the attacker's character), and "none" for a false start, from the crop and the whole frame (two ResNet18s, temporal fusion).
    5. `banner_reader.py`: counter / punish / neither from the game's own banner around each start: the yellow COUNTER / PUNISH word in the attacker's HUD box (the dataset build's templates), and a large counter hit's letterbox or big white COUNTER! (random forest on the scores of every 2nd frame from -60 to +60). It replaces `video_moves`' counter/punish head, which guessed from the attacker crop (0.65).
  - `recognize.py <video>`: runs the whole chain with the `*_final.pt` weights (trained on all 722 replays) and writes `openers.json`.
  - `degrade_eval.py`: the same chain on 720p30 3 Mbit/s copies of the test replays (a typical VOD).
  - `chain_eval.py`: the whole chain on the test fold with another detector (cuts crops only for new detections).
  - `error_report.py`: where the chain loses the test fold's openers (missed, dropped as "none", wrong move) by category, character and move. `video_moves.py` and `video_openers.py` take `--frac` for a learning curve.
  - Weights and results are in `$GGST_DATA/ggst_dataset/model/` and are not in the repo.
  - Test fold (144 held-out replays, 4,715 logged openers, bursts apart), from the 1080p60 recordings:
    - Starts found: 0.95 at 0.92 precision (HUD only: 0.94 at 0.83 at threshold 0.3; supers 0.84, HUD only 0.72; the app's damage events found 0.78 at 0.50).
    - Found and kept: 0.93, with 0.97 of the kept ones real.
    - Characters right on every replay.
    - Given found: category 0.91, button/stance 0.92, exact move 0.90 (0.73–1.0 by character), counter/punish/neither 0.96.
    - The whole chain gets 0.84 of all openers found with the right move (0.82 with the HUD-only detector, 0.79 with the attacker crop alone).
  - On the 720p30 copies of the same replays (`degrade_eval.py`), almost nothing is lost:
    - starts 0.95 found at 0.93 precision and 0.93 kept, with 0.97 of the kept ones real;
    - characters all right;
    - category 0.91, button/stance 0.92, exact move 0.89, counter/punish/neither 0.96;
    - whole chain 0.84.
