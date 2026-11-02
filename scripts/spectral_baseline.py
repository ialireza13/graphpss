"""Spectral-clustering baseline on a sample of test graphs.

Each graph is clustered into 3 groups from its normalized-Laplacian eigenvectors; clusters
are then mapped to labels with the Hungarian algorithm using that graph's own labels. The
mapping uses test labels, so this is an optimistic upper bound for the method.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering

sys.path.insert(0, "src")
from pssnet.data import GraphStore, load_or_make_split  # noqa: E402


def one(args):
    y, e = args
    n = len(y)
    a = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    pred = SpectralClustering(3, affinity="precomputed", random_state=0, assign_labels="kmeans").fit_predict(a)
    cost = np.zeros((3, 3))
    for c in range(3):
        for t in range(3):
            cost[c, t] = -np.sum((pred == c) & (y == t))
    r, c = linear_sum_assignment(cost)
    return -cost[r, c].sum(), n


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="data/splits/cluster30_seed0.json")
args = ap.parse_args()
os.makedirs("runs", exist_ok=True)
s = GraphStore.load("data/graphpss.npz")
split = load_or_make_split(args.split, len(s))
test = np.random.default_rng(0).choice(split["test"], 2000, replace=False)
jobs = [(s.y[s.node_ptr[g]:s.node_ptr[g + 1]].astype(int),
         s.edges[s.edge_ptr[g]:s.edge_ptr[g + 1]].astype(int)) for g in test]
with ProcessPoolExecutor(12) as ex:
    res = list(ex.map(one, jobs, chunksize=8))
correct = sum(r[0] for r in res)
total = sum(r[1] for r in res)
out = {"split": args.split, "graphs": len(res), "nodes": int(total),
       "acc_oracle_matched": float(correct / total)}
tag = os.path.basename(args.split).replace(".json", "")
json.dump(out, open(f"runs/spectral_baseline_{tag}.json", "w"), indent=2)
print(out)
