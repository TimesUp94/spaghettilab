"""Re-derive the combo list from an existing analysis run (no video pass).

Reads <out>/frames/<stem>_frames.parquet (needs combo_side/combo_count and
health columns), runs replanal.extractors.combo.derive_combos and writes
<out>/<stem>_combos.json plus the `combos` table in <out>/analysis.db.

Usage:
    .venv/Scripts/python.exe scripts/embed/combos_from_parquet.py output/exp17_combo 17_DIHLScarecrow_vs_QSalkazko
"""
from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))  # pyarrow without pandas
sys.modules["pandas"].__version__ = "0.0.0"  # type: ignore[attr-defined]

import pyarrow.parquet as pq  # noqa: E402

from replanal.extractors.combo import combos_to_records, derive_combos  # noqa: E402
from replanal.models import FrameData, HealthReading, Side  # noqa: E402
from replanal.storage import write_combos  # noqa: E402


def main():
    out, stem = Path(sys.argv[1]), sys.argv[2]
    t = pq.read_table(str(out / "frames" / f"{stem}_frames.parquet")).to_pydict()
    frames = []
    for i in range(len(t["frame_number"])):
        f = FrameData(frame_number=t["frame_number"][i], timestamp_ms=t["timestamp_ms"][i])
        if t["p1_health_pct"][i] is not None:
            f.p1_health = HealthReading(Side.P1, t["p1_health_pct"][i], 0, 0)
        if t["p2_health_pct"][i] is not None:
            f.p2_health = HealthReading(Side.P2, t["p2_health_pct"][i], 0, 0)
        if t["combo_side"][i] is not None:
            f.combo_side = Side(t["combo_side"][i])
            f.combo_count = t["combo_count"][i]
        frames.append(f)
    combos = derive_combos(frames)
    (out / f"{stem}_combos.json").write_text(json.dumps(combos_to_records(combos), indent=1))
    conn = sqlite3.connect(str(out / "analysis.db"))
    write_combos(conn, stem, combos)
    conn.close()
    print(f"{len(combos)} combos -> {out / f'{stem}_combos.json'}")


if __name__ == "__main__":
    main()
