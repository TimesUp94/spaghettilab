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
from replanal.extractors.timer import TimerExtractor
from replanal.pipeline import ReplayPipeline
from replanal.storage import init_db, write_damage_events, write_frame_data, write_frame_parquet, write_replay
from replanal.video import get_video_info
from replanal.viz import save_health_timeline


def main():
    parser = argparse.ArgumentParser(description="Analyze a GGS replay")
    parser.add_argument("video", help="Path to replay MP4")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--output", default=str(ROOT / "output"))
    args = parser.parse_args()

    video_path = args.video
    config = PipelineConfig.from_yaml(Path(args.config))

    print(f"Analyzing: {video_path}")
    info = get_video_info(video_path)
    print(f"  Resolution: {info['width']}x{info['height']}, FPS: {info['fps']:.0f}, Frames: {info['frame_count']}")

    # Build pipeline with all extractors
    scene_det = SceneDetector()
    health_ext = HealthBarExtractor(config.health_bar)
    timer_ext = TimerExtractor()
    heart_ext = RoundCounterExtractor()
    pipeline = ReplayPipeline(config, [scene_det, health_ext, timer_ext, heart_ext])

    frames = pipeline.process_video(video_path)
    print(f"  Processed {len(frames)} frames")

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
    conn.close()
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
