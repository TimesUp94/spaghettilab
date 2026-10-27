"""Analyze a single GGS replay video.

Usage:
    python scripts/analyze_replay.py data/Replay_2025-12-16_21-13-59.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replanal.aggregator import HealthAggregator
from replanal.config import PipelineConfig
from replanal.extractors.health import HealthBarExtractor
from replanal.extractors.round_counter import RoundCounterExtractor
from replanal.extractors.scene import SceneDetector
from replanal.extractors.tension import TensionExtractor, finalize_tension
from replanal.extractors.timer import TimerExtractor, finalize_timer
from replanal.pipeline import ReplayPipeline
from replanal.extractors.combo import ComboCounterExtractor, combos_to_records, derive_combos
from replanal.storage import init_db, write_combos, write_damage_events, write_frame_data, write_frame_parquet, write_replay
from replanal.video import get_video_info
from replanal.viz import save_health_timeline


def main():
    parser = argparse.ArgumentParser(description="Analyze a GGS replay")
    parser.add_argument("video", help="Path to replay MP4")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--output", default=str(ROOT / "output"))
    parser.add_argument(
        "--sample-every", type=int, default=None,
        help="Process every Nth frame (default: config, 'auto' = round(video fps / 30))",
    )
    args = parser.parse_args()

    video_path = args.video
    config = PipelineConfig.from_yaml(Path(args.config))
    if args.sample_every is not None:
        config.sample_every_n_frames = max(1, args.sample_every)

    print(f"Analyzing: {video_path}", flush=True)
    info = get_video_info(video_path, config.fps, config.sample_every_n_frames)
    print(f"  Resolution: {info['width']}x{info['height']}, FPS: {info['effective_fps']:.2f}, Frames: {info['frame_count']}")
    print(f"  Sampling every {info['sample_every']} frame(s) = {info['effective_fps'] / info['sample_every']:.1f} samples/s")

    # Build pipeline with all extractors
    scene_det = SceneDetector()
    health_ext = HealthBarExtractor(config.health_bar)
    timer_ext = TimerExtractor()
    tension_ext = TensionExtractor()
    heart_ext = RoundCounterExtractor()
    combo_ext = ComboCounterExtractor()
    pipeline = ReplayPipeline(config, [scene_det, health_ext, timer_ext, tension_ext, heart_ext, combo_ext])

    # "Progress NN%" lines (whole percents) for the app's progress bar
    total = max(1, int(info["frame_count"]))
    last_pct = [-1]

    def progress(frame_number: int) -> None:
        pct = min(99, frame_number * 100 // total)
        if pct != last_pct[0]:
            last_pct[0] = pct
            print(f"  Progress {pct}%", flush=True)

    frames = pipeline.process_video(video_path, on_frame=progress)
    print(f"  Processed {len(frames)} frames")
    dropped = finalize_timer(frames)
    if dropped:
        print(f"  Dropped {dropped} single-digit timer readings that do not continue the countdown")
    ceilings = finalize_tension(frames)
    for attr, c in ceilings.items():
        if c is not None:
            print(f"  {attr[:2].upper()} tension bar covered beyond {c:.0%} (stream overlay): readings there dropped")

    # Aggregate damage events
    aggregator = HealthAggregator(config.health_bar)
    events = aggregator.aggregate(frames)
    print(f"  Detected {len(events)} damage events")

    for e in events:
        side_label = "P1" if e.target_side.value == 1 else "P2"
        print(f"    [{e.timestamp_ms/1000:.1f}s] {side_label} took {e.damage_pct:.1%} damage ({e.pre_health_pct:.1%} -> {e.post_health_pct:.1%})")

    # Store results
    output_dir = Path(args.output)
    replay_id = Path(video_path).stem
    db_path = output_dir / "analysis.db"

    conn = init_db(db_path)
    write_replay(conn, replay_id, video_path, info["duration_ms"], info["frame_count"])
    write_frame_data(conn, replay_id, frames)
    write_damage_events(conn, replay_id, events)

    # Combos from the on-screen hit counter (uses the cleaned health above)
    combos = derive_combos(frames)
    print(f"  Detected {len(combos)} combos (hit counter)")
    write_combos(conn, replay_id, combos)
    conn.close()
    import json

    combos_path = output_dir / f"{replay_id}_combos.json"
    combos_path.write_text(json.dumps(combos_to_records(combos), indent=1))
    print(f"  Combos -> {combos_path}")
    print(f"  Results -> {db_path}")

    # Write per-frame parquet
    parquet_dir = output_dir / "frames"
    parquet_path = write_frame_parquet(parquet_dir, replay_id, frames)
    print(f"  Parquet -> {parquet_path}")

    # Generate health timeline chart
    try:
        chart_path = save_health_timeline(frames, events, replay_id, output_dir)
        print(f"  Chart -> {chart_path}")
    except ImportError:
        print("  (matplotlib not installed, skipping chart)")


if __name__ == "__main__":
    main()
