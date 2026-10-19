"""Re-run the temporal cleaner on the raw health columns of a frames parquet (no video decode).
Usage: python health_reclean_parquet.py FRAMES.parquet OUT.npz"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from replanal.health_clean import clean_series  # noqa: E402

t = pq.read_table(sys.argv[1]).to_pydict()
a = {k: np.array([np.nan if v is None else v for v in t[k]], dtype=float)
     for k in ("p1_health_raw", "p2_health_raw", "p1_health_conf", "p2_health_conf", "p1_pending_pct", "p2_pending_pct")}
r = clean_series(a["p1_health_raw"], a["p2_health_raw"], np.nan_to_num(a["p1_health_conf"]), np.nan_to_num(a["p2_health_conf"]),
                 np.nan_to_num(a["p1_pending_pct"]), np.nan_to_num(a["p2_pending_pct"]))
print("resets at", [round(i / 30, 1) for i in r.resets])
for k in range(r.segment.max() + 1):
    idx = np.flatnonzero(r.segment == k)
    f1 = r.p1[idx]; f2 = r.p2[idx]
    print(f"seg {k} t {idx[0]/30:.1f}-{idx[-1]/30:.1f} final P1 {np.nanmin(f1) if (~np.isnan(f1)).any() else float('nan'):.3f} "
          f"P2 {np.nanmin(f2) if (~np.isnan(f2)).any() else float('nan'):.3f}")
np.savez_compressed(sys.argv[2], p1=r.p1, p2=r.p2, segment=r.segment)
