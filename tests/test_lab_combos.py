"""lab_combos: an opener's combo_hits from the combos (no models, no video)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ggst_logger" / "model"))
import lab_combos as lc  # noqa: E402


def combo(attacker, start_s, hits):
    return dict(attacker=attacker, start_s=start_s, max_hits=hits)


def test_the_combo_after_the_opener_of_the_same_side():
    openers = [dict(time_s=40.0, attacker="P1"), dict(time_s=41.0, attacker="P2"), dict(time_s=60.0, attacker="P1")]
    n = lc.link_hits(openers, [combo("P1", 41.8, 9), combo("P2", 42.0, 3)])
    assert n == 2
    assert openers[0]["combo_hits"] == 9
    assert openers[1]["combo_hits"] == 3
    assert "combo_hits" not in openers[2]


def test_outside_the_window_or_past_the_next_opener():
    far = [dict(time_s=10.0, attacker="P1")]
    assert lc.link_hits(far, [combo("P1", 14.0, 5)]) == 0
    assert lc.link_hits(far, [combo("P1", 9.0, 5)]) == 0
    two = [dict(time_s=10.0, attacker="P1"), dict(time_s=11.5, attacker="P1")]
    assert lc.link_hits(two, [combo("P1", 12.0, 5)]) == 1
    assert "combo_hits" not in two[0]
    assert two[1]["combo_hits"] == 5


def test_a_parquet_without_counter_readings_gives_nothing(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = tmp_path / "x_frames.parquet"
    pq.write_table(pa.table({"frame_number": [1], "timestamp_ms": [33.3]}), p)
    openers = [dict(time_s=1.0, attacker="P1")]
    assert lc.add_combo_hits(openers, p) == 0
    assert "combo_hits" not in openers[0]


def test_a_broken_parquet_is_not_fatal(tmp_path):
    assert lc.add_combo_hits([dict(time_s=1.0, attacker="P1")], tmp_path / "missing.parquet") == 0
