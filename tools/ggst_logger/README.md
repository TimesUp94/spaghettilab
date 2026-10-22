# Strive replay logger and recorder

Research tooling for a labelled combo-opener dataset. It plays Guilty Gear -Strive- replays, records clean video of them, and logs the game's own per-frame state from memory. The state covers inputs, each player's current move name, facing, position and HP. The visual models trained on this data must work from video alone; this is only how the training labels are made.

Windows only. **Never go online with the mod installed.** To play online, remove `dwmapi.dll` (the UE4SS loader) from the game folder.

## Parts

- **`Main.cs` / `GGSTInputLogger.csproj`:** a UE4SS .NET plugin.
  - It hooks the battle update and input functions, using the signatures from [WistfulHopes/GGSTReplayTakeover](https://github.com/WistfulHopes/GGSTReplayTakeover) (MIT).
  - In replay and training modes it writes one CSV row per game frame to `$GGST_INPUT_LOG_DIR` (default `G:\Projects\replanal-data\ggst_logs`).
  - It starts a new file whenever the game frame counter restarts.
  - F6 and F7 write memory snapshots (`dump_*.bin`) that are used to find new fields.
  - The offsets are for game version 5.03 and are listed at the top of `Main.cs`. A game patch can move them.
- **`drive/`:** drives the game.
  - `ggst.py` sends keyboard input and takes window screenshots. Screenshots use FFmpeg 9's `gfxcapture`, so they work while the window is covered.
  - `nav.py` is the menu navigator. It checks every screen against `ref/*.jpg` before it presses a key.
  - `batch.py` plays saved replays one by one, records each one, and appends to `ggst_rec/manifest.jsonl`.
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
- **`model/`:** first visual models trained on attacker crops around each opener: category, button and stance, and the exact move for the best-covered characters. Each script states how to run it.
