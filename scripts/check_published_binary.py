#!/usr/bin/env python3
"""Check that a built app binary doesn't contain the local models path.

The builds we publish are compiled without the `local-models` cargo feature, so nothing in them can start
tools/ggst_logger/model/lab_analyze.py. These strings only exist in app/src-tauri/src/lab_local.rs, so finding
one means the feature got in. Runs after the builds in CI and in the release scripts. Standard library only.

    python scripts/check_published_binary.py <binary>            # exit 1 if a marker is found
    python scripts/check_published_binary.py <binary> --expect-local   # the other way round (our own build)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKERS = (b"lab_analyze.py", b"GGST_MODEL_DIR", b"[LAB] Spawning")


def found_markers(path: Path) -> list[str]:
    data = path.read_bytes()
    return [m.decode() for m in MARKERS if m in data]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary", type=Path)
    ap.add_argument("--expect-local", action="store_true", help="the binary must have the local path (our own build)")
    a = ap.parse_args(argv)
    if not a.binary.is_file():
        print(f"not found: {a.binary}", file=sys.stderr)
        return 2
    found = found_markers(a.binary)
    if a.expect_local:
        if len(found) != len(MARKERS):
            print(f"{a.binary}: expected the local models path, found only {found}", file=sys.stderr)
            return 1
        print(f"{a.binary}: has the local models path")
        return 0
    if found:
        print(f"{a.binary}: contains the local models path ({', '.join(found)}): built with --features local-models?", file=sys.stderr)
        return 1
    print(f"{a.binary}: no local models path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
