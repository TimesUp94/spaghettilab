from __future__ import annotations

import sqlite3
from pathlib import Path

from replanal.models import DamageEvent, FrameData


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS replays (
            replay_id   TEXT PRIMARY KEY,
            video_path  TEXT,
            duration_ms REAL,
            frame_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS frame_data (
            replay_id     TEXT REFERENCES replays(replay_id),
            frame_number  INTEGER,
            timestamp_ms  REAL,
            p1_health_pct REAL,
            p2_health_pct REAL,
            timer_value   INTEGER,
            p1_rounds_won INTEGER,
            p2_rounds_won INTEGER,
            p1_tension_pct REAL,
            p2_tension_pct REAL,
            PRIMARY KEY (replay_id, frame_number)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS damage_events (
            event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id     TEXT REFERENCES replays(replay_id),
            timestamp_ms  REAL,
            frame_start   INTEGER,
            frame_end     INTEGER,
            target_side   INTEGER,
            damage_pct    REAL,
            pre_health_pct  REAL,
            post_health_pct REAL
        )
    """)
    # Migrations: add columns that may not exist in older DBs
    for col, coltype in [("p1_tension_pct", "REAL"), ("p2_tension_pct", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE frame_data ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


def write_replay(conn: sqlite3.Connection, replay_id: str, video_path: str, duration_ms: float, frame_count: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO replays (replay_id, video_path, duration_ms, frame_count) VALUES (?, ?, ?, ?)",
        (replay_id, video_path, duration_ms, frame_count),
    )
    conn.commit()


def write_frame_data(conn: sqlite3.Connection, replay_id: str, frames: list[FrameData]) -> None:
    """Write per-frame health data to SQLite."""
    conn.execute("DELETE FROM frame_data WHERE replay_id = ?", (replay_id,))
    conn.executemany(
        "INSERT INTO frame_data (replay_id, frame_number, timestamp_ms, p1_health_pct, p2_health_pct, timer_value, p1_rounds_won, p2_rounds_won, p1_tension_pct, p2_tension_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                replay_id,
                f.frame_number,
                f.timestamp_ms,
                f.p1_health.health_pct if f.p1_health else None,
                f.p2_health.health_pct if f.p2_health else None,
                f.timer_value,
                f.p1_rounds_won,
                f.p2_rounds_won,
                f.p1_tension_pct,
                f.p2_tension_pct,
            )
            for f in frames
        ],
    )
    conn.commit()


def write_damage_events(conn: sqlite3.Connection, replay_id: str, events: list[DamageEvent]) -> None:
    conn.execute("DELETE FROM damage_events WHERE replay_id = ?", (replay_id,))
    for e in events:
        conn.execute(
            "INSERT INTO damage_events (replay_id, timestamp_ms, frame_start, frame_end, target_side, damage_pct, pre_health_pct, post_health_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (replay_id, e.timestamp_ms, e.frame_start, e.frame_end, e.target_side.value, e.damage_pct, e.pre_health_pct, e.post_health_pct),
        )
    conn.commit()


def write_frame_parquet(output_dir: Path, replay_id: str, frames: list[FrameData]) -> Path:
    """Write per-frame data as a parquet file. Returns the output path."""
    import sys
    import types

    # Prevent pyarrow from importing a broken pandas installation
    _had_pandas = "pandas" in sys.modules
    _old_pandas = sys.modules.get("pandas")
    if not _had_pandas:
        _fake = types.ModuleType("pandas")
        _fake.__version__ = "0.0.0"  # type: ignore[attr-defined]
        sys.modules["pandas"] = _fake

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{replay_id}_frames.parquet"

        schema = pa.schema([
            ("frame_number", pa.int32()),
            ("timestamp_ms", pa.float64()),
            ("p1_health_pct", pa.float64()),
            ("p2_health_pct", pa.float64()),
            ("timer_value", pa.int32()),
            ("p1_rounds_won", pa.int32()),
            ("p2_rounds_won", pa.int32()),
            ("p1_tension_pct", pa.float64()),
            ("p2_tension_pct", pa.float64()),
        ])

        arrays = [
            pa.array([f.frame_number for f in frames], type=pa.int32()),
            pa.array([f.timestamp_ms for f in frames], type=pa.float64()),
            pa.array([f.p1_health.health_pct if f.p1_health else None for f in frames], type=pa.float64()),
            pa.array([f.p2_health.health_pct if f.p2_health else None for f in frames], type=pa.float64()),
            pa.array([f.timer_value for f in frames], type=pa.int32()),
            pa.array([f.p1_rounds_won for f in frames], type=pa.int32()),
            pa.array([f.p2_rounds_won for f in frames], type=pa.int32()),
            pa.array([f.p1_tension_pct for f in frames], type=pa.float64()),
            pa.array([f.p2_tension_pct for f in frames], type=pa.float64()),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, str(path))
        return path
    finally:
        if not _had_pandas:
            sys.modules.pop("pandas", None)
        elif _old_pandas is not None:
            sys.modules["pandas"] = _old_pandas
