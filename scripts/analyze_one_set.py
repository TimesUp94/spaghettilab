"""Analyze a single set file (wrapper around analyze_sets logic)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replanal.aggregator import HealthAggregator
from replanal.config import PipelineConfig
from replanal.extractors.health import HealthBarExtractor
from replanal.extractors.scene import SceneDetector
from replanal.pipeline import ReplayPipeline
from replanal.storage import init_db, write_damage_events, write_frame_data, write_replay
from replanal.video import get_video_info

VIDEO = str(ROOT / "data" / "sets" / "04_UF_GuessingGame_vs_Ricbionicle.mp4")
OUTPUT_DIR = ROOT / "output" / "sets"


def main():
    config = PipelineConfig.from_yaml(ROOT / "config" / "default.yaml")
    config.sample_every_n_frames = 1  # every frame

    info = get_video_info(VIDEO)
    replay_id = Path(VIDEO).stem
    print(f"Video: {info['width']}x{info['height']}, {info['fps']:.0f}fps, {info['frame_count']} frames")

    scene_det = SceneDetector()
    health_ext = HealthBarExtractor(config.health_bar)
    pipeline = ReplayPipeline(config, [scene_det, health_ext])

    t0 = time.time()
    frames = pipeline.process_video(VIDEO)
    proc_time = time.time() - t0
    gameplay_frames = sum(1 for f in frames if f.is_gameplay)
    print(f"Processed {len(frames)} frames ({gameplay_frames} gameplay) in {proc_time:.1f}s")

    aggregator = HealthAggregator(config.health_bar)
    events = aggregator.aggregate(frames)
    print(f"{len(events)} damage events detected")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUTPUT_DIR / "ricbionicle_test.db"
    conn = init_db(db_path)
    write_replay(conn, replay_id, VIDEO, info["duration_ms"], info["frame_count"])
    write_frame_data(conn, replay_id, frames)
    write_damage_events(conn, replay_id, events)
    conn.close()
    print(f"Results written to {db_path}")

    # Print HP at round-end candidates
    print("\nHP readings at key timestamps:")
    for f in frames:
        ts = f.timestamp_ms / 1000
        if f.p1_health and f.p2_health:
            p1 = f.p1_health.health_pct
            p2 = f.p2_health.health_pct
            # Print when either player is very low
            if p1 < 0.10 or p2 < 0.10:
                print(f"  t={ts:6.1f}s f={f.frame_number:4d}: P1={p1:.3f}  P2={p2:.3f}")


if __name__ == "__main__":
    main()
