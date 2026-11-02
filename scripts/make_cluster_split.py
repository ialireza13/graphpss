"""Homology-aware 80/10/10 split from RCSB sequence clusters.

Each graph is built from one chain (label chain 'A'), so each entry is assigned to the
RCSB cluster (clusters-by-entity-<k>.txt, MMseqs2) of that chain's entity
(data/clusters/chain_a_entity.json, from scripts/fetch_entities.py). Whole clusters go to
train / val / test, so no test chain shares >= k% sequence identity with a training chain.
Linking entries through *all* their entities instead merges complexes into one group
holding ~40% of the data, which makes a balanced split impossible.

    python scripts/make_cluster_split.py --identity 30
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from pssnet.data import GraphStore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="data/graphpss.npz")
ap.add_argument("--identity", type=int, default=30)
ap.add_argument("--clusters-dir", default="data/clusters")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-group-frac", type=float, default=0.05,
                help="fail if one group holds more than this fraction of the nodes")
args = ap.parse_args()

s = GraphStore.load(args.store)
ids = [str(x) for x in s.ids]
entity_of = json.load(open(f"{args.clusters_dir}/chain_a_entity.json"))
cluster_of_entity = {}
with open(f"{args.clusters_dir}/clusters-by-entity-{args.identity}.txt") as f:
    for c, line in enumerate(f):
        for m in line.split():
            cluster_of_entity[m] = c
# Entries without a known cluster become singleton groups.
roots = []
covered = set()
for i, pid in enumerate(ids):
    c = cluster_of_entity.get(entity_of.get(pid, ""))
    if c is None:
        roots.append(f"single:{pid}")
    else:
        roots.append(c)
        covered.add(i)
roots = np.array(roots, dtype=object)
sizes = np.diff(s.node_ptr)
group_nodes = Counter()
for r, n in zip(roots, sizes):
    group_nodes[r] += int(n)
groups = list(group_nodes)
largest = max(group_nodes.values()) / sizes.sum()
print(f"entries {len(ids)}, in cluster file {len(covered)}, groups {len(groups)}, "
      f"largest group {100 * largest:.2f}% of nodes")
if largest > args.max_group_frac:
    sys.exit("largest group too big for a balanced split; use a higher identity level")

# Assign whole groups in random order until each split reaches its share of nodes.
rng = np.random.default_rng(args.seed)
order = rng.permutation(len(groups))
targets = {"train": 0.8, "val": 0.1, "test": 0.1}
filled = {k: 0 for k in targets}
assign = {}
total = sizes.sum()
for gi in order:
    g = groups[gi]
    k = min(targets, key=lambda name: filled[name] / total / targets[name])
    assign[g] = k
    filled[k] += group_nodes[g]
split = {k: [] for k in targets}
for i, r in enumerate(roots):
    split[assign[r]].append(i)

out = f"data/splits/cluster{args.identity}_seed{args.seed}.json"
os.makedirs("data/splits", exist_ok=True)
with open(out, "w") as f:
    json.dump(split, f)
info = {
    "identity": args.identity, "entries": len(ids), "not_in_cluster_file": len(ids) - len(covered),
    "groups": len(groups), "largest_group_node_frac": float(largest),
    "graphs": {k: len(v) for k, v in split.items()},
    "node_frac": {k: float(filled[k] / total) for k in targets},
}
with open(out.replace(".json", "_info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
