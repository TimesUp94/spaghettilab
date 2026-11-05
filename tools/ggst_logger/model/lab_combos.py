"""The game's hit counter for lab_result.json: each opener's `combo_hits` (LAB_RESULT.md).

The counter's readings are in the HUD parquet of a current pipeline run; older parquets lack them and then
openers simply get no field. Combos come from replanal's derive_combos on the cleaned-up frames, exactly as
scripts/reclean_replay.py rebuilds them. Never fatal: an opener's hit count isn't worth losing a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The counter appears from the combo's second hit on, so its combo starts after the opener; the same numbers
# as the apps' fallback (app/src/lib/combos.ts, FGCScoreboard comboHits.js).
OPENER_BEFORE_S = 0.5
OPENER_AFTER_S = 3.0


def link_hits(openers: list[dict], combos: list[dict]) -> int:
    """Sets `combo_hits` on each opener with a combo: the first of the same attacker starting from just before
    the opener to OPENER_AFTER_S after it, and before the same side's next opener. combos: combos_to_records.
    Returns how many got one."""
    n = 0
    for side in ("P1", "P2"):
        own = sorted((o for o in openers if o.get("attacker") == side), key=lambda o: o["time_s"])
        starts = sorted((c for c in combos if c["attacker"] == side), key=lambda c: c["start_s"])
        for i, o in enumerate(own):
            nxt = own[i + 1]["time_s"] if i + 1 < len(own) else float("inf")
            until = min(o["time_s"] + OPENER_AFTER_S, nxt - OPENER_BEFORE_S)
            c = next((c for c in starts if o["time_s"] - OPENER_BEFORE_S <= c["start_s"] <= until), None)
            if c is not None and c.get("max_hits", 0) > 1:
                o["combo_hits"] = int(c["max_hits"])
                n += 1
    return n


def combos_from_parquet(parquet) -> list[dict]:
    """The combos of a frames parquet; [] when it has no counter readings (written before they existed)."""
    import pyarrow.parquet as pq
    if "combo_count" not in pq.read_schema(parquet).names:
        return []
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from replanal.aggregator import HealthAggregator
    from replanal.config import PipelineConfig
    from replanal.extractors.combo import combos_to_records, derive_combos
    from replanal.extractors.tension import finalize_tension
    from replanal.extractors.timer import finalize_timer
    from scripts.reclean_replay import frames_from_parquet
    frames = frames_from_parquet(Path(parquet))
    finalize_timer(frames)
    finalize_tension(frames)
    HealthAggregator(PipelineConfig.from_yaml(root / "config" / "default.yaml").health_bar).aggregate(frames)
    return combos_to_records(derive_combos(frames))


def add_combo_hits(openers: list[dict], parquet) -> int:
    """`combo_hits` on the openers from the parquet's counter readings; 0 on any problem."""
    try:
        return link_hits(openers, combos_from_parquet(parquet))
    except Exception as e:  # noqa: BLE001 - see the module's docstring
        print(f"combo hits skipped: {e}", flush=True)
        return 0
