"""Validation accuracy against parameter count for every sweep run.

    python scripts/plot_efficiency.py --out paper/figures/efficiency.pdf
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

FAMILIES = [("gatv2", "GATv2", "#2a78d6", "o"), ("gat", "GAT", "#eb6834", "s"),
            ("sage", "GraphSAGE", "#1baf7a", "^"), ("expanded_gcn_k3", "Expanded GCN, K=3", "#eda100", "v"),
            ("expanded_gcn_k2", "Expanded GCN, K=2", "#e87ba4", "D"), ("gcn", "GCN", "#008300", "P"),
            ("gin", "GIN", "#6250d6", "X")]
YMIN = 83  # a collapsed GIN run (41.7%) is off the axis and reported in the text

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper/figures/efficiency.pdf")
args = ap.parse_args()

runs = {}
for path in glob.glob("runs/sweep_*/metrics.json"):
    run = os.path.dirname(path)
    m = json.load(open(path))
    cfg = yaml.safe_load(open(f"{run}/config.yaml"))["model"]
    family = cfg["conv"] + (f"_k{cfg['hops']}" if cfg["conv"] == "expanded_gcn" else "")
    runs.setdefault(family, []).append((m["params"] / 1e6, 100 * m["val"]["acc"]))

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.grid(color="#e5e4e0", lw=0.5)
ax.set_axisbelow(True)
for family, label, color, marker in FAMILIES:
    points = sorted(p for p in runs[family] if p[1] >= YMIN)
    ax.scatter(*zip(*points), c=color, marker=marker, s=26, lw=0.5, edgecolors="#fcfcfb", label=label, zorder=3)
ax.set_xscale("log")
ax.set_ylim(YMIN, 91.5)
ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel("Validation accuracy (%)")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
ax.minorticks_off()
ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0),
          handletextpad=0.1, columnspacing=0.8)
fig.tight_layout()
fig.savefig(args.out)
