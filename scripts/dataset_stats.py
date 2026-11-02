"""Dataset statistics for the paper: sizes, labels, degree, connectivity, contact separations."""
import json
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, "src")
from pssnet.data import CLASS_NAMES, GraphStore, load_or_make_split  # noqa: E402

s = GraphStore.load("data/graphpss.npz")
split = load_or_make_split("data/splits/seed0.json", len(s))
sizes = np.diff(s.node_ptr)
n_comp = np.empty(len(s), dtype=np.int32)
seps = np.zeros(6, dtype=np.int64)  # |i-j| = 1, 2, 3, 4, 5..10, >10
for g in range(len(s)):
    e = s.edges[s.edge_ptr[g]:s.edge_ptr[g + 1]].astype(np.int64)
    n = sizes[g]
    n_comp[g] = connected_components(csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)))[0]
    d = np.abs(e[:, 0] - e[:, 1])
    seps += np.array([(d == 1).sum(), (d == 2).sum(), (d == 3).sum(), (d == 4).sum(),
                      ((d >= 5) & (d <= 10)).sum(), (d > 10).sum()])
labels = np.bincount(s.y, minlength=3)
out = {
    "graphs": len(s), "nodes": int(sizes.sum()), "directed_edges": int(len(s.edges)),
    "size": {"mean": float(sizes.mean()), "median": float(np.median(sizes)),
             "min": int(sizes.min()), "max": int(sizes.max())},
    "mean_degree": float(len(s.edges) / sizes.sum()),
    "labels": {c: float(v / labels.sum()) for c, v in zip(CLASS_NAMES, labels)},
    "frac_connected": float((n_comp == 1).mean()),
    "mean_components": float(n_comp.mean()),
    "edge_separation_frac": dict(zip(["1", "2", "3", "4", "5-10", ">10"], (seps / seps.sum()).tolist())),
    "split_sizes": {k: len(v) for k, v in split.items()},
}
for k, ids in split.items():
    yy = np.concatenate([s.y[s.node_ptr[g]:s.node_ptr[g + 1]] for g in ids])
    out[f"majority_acc_{k}"] = float(np.bincount(yy, minlength=3).max() / len(yy))
json.dump(out, open("runs/dataset_stats.json", "w"), indent=2)
print(json.dumps(out, indent=2))
