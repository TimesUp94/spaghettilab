"""Plot per-frame measurements (health_measure_cached.py) and optional cleaned series.
Usage: python health_plot_meas.py MEAS.npz OUT.png [t0 t1] [--clean CLEAN.npz]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUNDS_17 = [(12, 38), (42, 67), (72, 135), (148, 190), (194, 241), (248, 277), (289, 319), (324, 389)]
args = sys.argv[1:]
clean = None
if "--clean" in args:
    k = args.index("--clean"); clean = np.load(args[k + 1]); args = args[:k] + args[k + 2:]
d = np.load(args[0]); out = args[1]
t = np.arange(len(d["p1_fill"])) / 30.0
t0, t1 = (float(args[2]), float(args[3])) if len(args) > 3 else (0, t[-1])
sel = (t >= t0) & (t <= t1)
fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
for s, ax in zip((1, 2), axs):
    f = d[f"p{s}_fill"] / 719; p = (d[f"p{s}_fill"] + d[f"p{s}_pending"]) / 719
    conf = d[f"p{s}_cons"] * (1 - d[f"p{s}_occ"]) * (1 - d[f"p{s}_amb"])
    for a, b in ROUNDS_17:
        ax.axvspan(a, b, color="g", alpha=0.06)
    ax.scatter(t[sel], p[sel], s=1, c="salmon", label="fill+pending")
    sc = ax.scatter(t[sel], f[sel], s=2, c=conf[sel], cmap="viridis", vmin=0, vmax=1, label="fill (colour=conf)")
    if clean is not None:
        ax.plot(t[sel], clean[f"p{s}"][sel], "k-", lw=1, label="cleaned")
    ax.set_ylabel(f"P{s}"); ax.set_ylim(-0.05, 1.05); ax.legend(loc="lower left", fontsize=7)
fig.colorbar(sc, ax=axs)
plt.savefig(out, dpi=80)
