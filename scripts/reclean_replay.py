"""Re-run health cleaning, damage events and combos from a frames parquet (no video decode).

Rebuilds the per-frame data from ``<dir>/frames/<replay_id>_frames.parquet`` (the
raw extractor readings are stored there), re-applies ``finalize_timer``,
``finalize_tension`` (from the raw ``timer_raw`` / ``p1/p2_tension_raw`` columns when
the parquet has them; older parquets only hold the cleaned values, which the
clean-ups leave as they are, apart from the tension median), ``HealthAggregator`` and
``derive_combos`` with the current code, and writes a fresh ``analysis.db``,
parquet and combos JSON into ``--output``.  Use after changing
``replanal/health_clean.py``, ``replanal/aggregator.py`` or the combo grouping.

Usage: python scripts/reclean_replay.py FRAMES.parquet --video-path PATH
           --duration-ms MS --frame-count N --output DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402

from replanal.aggregator import HealthAggregator  # noqa: E402
from replanal.config import PipelineConfig  # noqa: E402
from replanal.extractors.combo import combos_to_records, derive_combos  # noqa: E402
from replanal.extractors.tension import finalize_tension  # noqa: E402
from replanal.extractors.timer import finalize_timer  # noqa: E402
from replanal.models import FrameData, HealthReading, Side  # noqa: E402
from replanal.storage import (  # noqa: E402
    init_db, write_combos, write_damage_events, write_frame_data, write_frame_parquet, write_replay,
)


def frames_from_parquet(path: Path) -> list[FrameData]:
    rows = pq.read_table(path).to_pylist()
    frames = []
    for r in rows:
        f = FrameData(frame_number=r["frame_number"], timestamp_ms=r["timestamp_ms"])
        for side, key in ((Side.P1, "p1"), (Side.P2, "p2")):
            raw = r[f"{key}_health_raw"]
            if raw is None:
                continue
            reading = HealthReading(
                side=side, health_pct=raw, bar_pixels_filled=int(round(raw * 719)), bar_pixels_total=719,
                raw_pct=raw, pending_pct=r[f"{key}_pending_pct"] or 0.0, confidence=r[f"{key}_health_conf"] or 0.0,
            )
            setattr(f, f"{key}_health", reading)
        f.timer_value = r["timer_value"]
        f.p1_rounds_won = r["p1_rounds_won"]
        f.p2_rounds_won = r["p2_rounds_won"]
        f.p1_tension_pct = r["p1_tension_pct"]
        f.p2_tension_pct = r["p2_tension_pct"]
        f.combo_count = r["combo_count"]
        f.combo_side = Side(r["combo_side"]) if r["combo_side"] is not None else None
        # Readings before finalize_timer / finalize_tension (newer parquets).
        f.timer_raw = r.get("timer_raw")
        f.p1_tension_raw = r.get("p1_tension_raw")
        f.p2_tension_raw = r.get("p2_tension_raw")
        # The scene detector stops all later extractors on non-gameplay frames,
        # so those frames have no readings at all.
        f.is_gameplay = any(r[k] is not None for k in (
            "p1_health_raw", "p2_health_raw", "timer_value", "p1_tension_pct", "p2_tension_pct",
            "p1_rounds_won", "p2_rounds_won", "combo_count")) or any(
            r.get(k) is not None for k in ("timer_raw", "p1_tension_raw", "p2_tension_raw"))
        frames.append(f)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--video-path", required=True)
    ap.add_argument("--duration-ms", type=float, required=True)
    ap.add_argument("--frame-count", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = ap.parse_args()

    src = Path(args.parquet)
    replay_id = src.name.removesuffix("_frames.parquet")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig.from_yaml(Path(args.config))

    frames = frames_from_parquet(src)
    finalize_timer(frames)
    finalize_tension(frames)
    events = HealthAggregator(config.health_bar).aggregate(frames)
    combos = derive_combos(frames)

    db_path = out / "analysis.db"
    db_path.unlink(missing_ok=True)
    conn = init_db(db_path)
    write_replay(conn, replay_id, args.video_path, args.duration_ms, args.frame_count)
    write_frame_data(conn, replay_id, frames)
    write_damage_events(conn, replay_id, events)
    write_combos(conn, replay_id, combos)
    conn.close()
    (out / f"{replay_id}_combos.json").write_text(json.dumps(combos_to_records(combos), indent=1))
    write_frame_parquet(out / "frames", replay_id, frames)
    print(f"{replay_id}: {len(frames)} frames, {len(events)} damage events, {len(combos)} combos")


if __name__ == "__main__":
    main()
