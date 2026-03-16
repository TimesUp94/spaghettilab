"""Batch-analyze all split tournament sets.

Usage:
    python scripts/analyze_sets.py data/sets/
    python scripts/analyze_sets.py data/sets/ --sample-every 3
"""
from __future__ import annotations

import argparse
import sys
import time
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


def analyze_one(video_path: str, config: PipelineConfig, output_dir: Path, db_conn) -> dict:
    """Analyze a single set. Returns summary stats."""
    replay_id = Path(video_path).stem
    info = get_video_info(video_path)
    duration_min = info["duration_ms"] / 60_000

    print(f"  {info['width']}x{info['height']}, {info['fps']:.0f}fps, {duration_min:.1f} min, {info['frame_count']} frames")

    # Scene detector MUST come first — it sets is_gameplay flag
    # Pipeline will skip remaining extractors on non-gameplay frames
    scene_det = SceneDetector()
    health_ext = HealthBarExtractor(config.health_bar)
    timer_ext = TimerExtractor()
    heart_ext = RoundCounterExtractor()
    pipeline = ReplayPipeline(config, [scene_det, health_ext, timer_ext, heart_ext])

    t0 = time.time()
    frames = pipeline.process_video(video_path)
    proc_time = time.time() - t0
    gameplay_frames = sum(1 for f in frames if f.is_gameplay)
    print(f"  Processed {len(frames)} frames ({gameplay_frames} gameplay) in {proc_time:.1f}s")

    aggregator = HealthAggregator(config.health_bar)
    events = aggregator.aggregate(frames)
    print(f"  {len(events)} damage events detected")

    # Store
    write_replay(db_conn, replay_id, video_path, info["duration_ms"], info["frame_count"])
    write_frame_data(db_conn, replay_id, frames)
    write_damage_events(db_conn, replay_id, events)

    parquet_dir = output_dir / "frames"
    write_frame_parquet(parquet_dir, replay_id, frames)

    try:
        save_health_timeline(frames, events, replay_id, output_dir)
    except ImportError:
        pass

    return {
        "replay_id": replay_id,
        "duration_min": duration_min,
        "frames": len(frames),
        "gameplay_frames": gameplay_frames,
        "events": len(events),
        "proc_time": proc_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch-analyze tournament sets")
    parser.add_argument("sets_dir", help="Directory containing set MP4 files")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--output", default=str(ROOT / "output"))
    parser.add_argument("--sample-every", type=int, default=2,
                        help="Process every Nth frame (default: 2 = 15fps)")
    args = parser.parse_args()

    sets_dir = Path(args.sets_dir)
    videos = sorted(sets_dir.glob("*.mp4"))
    if not videos:
        print(f"No MP4 files found in {sets_dir}")
        sys.exit(1)

    config = PipelineConfig.from_yaml(Path(args.config))
    config.sample_every_n_frames = args.sample_every

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "analysis.db"
    conn = init_db(db_path)

    print(f"Analyzing {len(videos)} sets (sample every {args.sample_every} frames)")
    print(f"Output: {output_dir}\n")

    summaries = []
    total_t0 = time.time()

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        try:
            summary = analyze_one(str(video), config, output_dir, conn)
            summaries.append(summary)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    conn.close()
    total_time = time.time() - total_t0

    # Summary table
    print("=" * 70)
    print(f"{'Set':<40} {'Dur':>5} {'Frames':>7} {'GP':>7} {'Events':>7} {'Time':>6}")
    print("-" * 75)
    for s in summaries:
        name = s["replay_id"][:37]
        print(f"{name:<40} {s['duration_min']:>4.1f}m {s['frames']:>7} {s['gameplay_frames']:>7} {s['events']:>7} {s['proc_time']:>5.0f}s")
    print("-" * 75)
    total_frames = sum(s["frames"] for s in summaries)
    total_gp = sum(s["gameplay_frames"] for s in summaries)
    total_events = sum(s["events"] for s in summaries)
    print(f"{'TOTAL':<40} {'':>5} {total_frames:>7} {total_gp:>7} {total_events:>7} {total_time:>5.0f}s")
    print(f"\nResults: {db_path}")


if __name__ == "__main__":
    main()
