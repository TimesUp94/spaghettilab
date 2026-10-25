"""Export saved Guilty Gear -Strive- replays as video files.

Drives the running game from its title screen to Collection > Replay > Saved
replays, plays each replay and records the game window with FFmpeg. See
docs/strive-export.md and ``python -m replanal.strive_export --help``.

Modules: ``screens`` (screen recognition), ``exporter`` (the state machine),
``capture`` (FFmpeg discovery, screenshots, recording), ``rows`` (list OCR,
date parsing, file names), ``winio`` (Windows key input and window geometry).
"""
