"""Apply replanal.health_clean to cached per-frame measurements (health_measure_cached.py),
mimicking HealthBarExtractor.extract()'s emission rules.  Writes CLEAN.npz with p1/p2/segment
and prints the resets.
Usage: python health_clean_cached.py MEAS.npz CLEAN.npz"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from replanal.extractors.health import HealthBarExtractor as E  # noqa: E402
from replanal.health_clean import clean_series  # noqa: E402
import time

d = np.load(sys.argv[1])
n = len(d["p1_fill"])
obs, conf = {}, {}
has = {s: (d[f"p{s}_fill"] >= E._EMPTY_PX) | (d[f"p{s}_pending"] >= E._EMPTY_PX) for s in (1, 2)}
for s, o in ((1, 2), (2, 1)):
    emit = (d[f"p{s}_cons"] >= E._MIN_CONSISTENCY) & (has[1] | has[2]) & (has[s] | has[o])
    obs[s] = np.where(emit, d[f"p{s}_fill"] / 719.0, np.nan)
    conf[s] = d[f"p{s}_cons"] * (1 - d[f"p{s}_occ"]) * (1 - d[f"p{s}_amb"])
t0 = time.time()
r = clean_series(obs[1], obs[2], conf[1], conf[2], d["p1_pending"] / 719.0, d["p2_pending"] / 719.0)
print(f"clean {time.time()-t0:.2f}s; resets at", [round(i / 30, 1) for i in r.resets])
np.savez_compressed(sys.argv[2], p1=r.p1, p2=r.p2, segment=r.segment, obs1=obs[1], obs2=obs[2], c1=conf[1], c2=conf[2])
