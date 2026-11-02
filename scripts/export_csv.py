"""Export the store to the public CSV release (one .node_feature/.node_labels/.edges per protein)."""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "src")
from pssnet.data import GraphStore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="data/graphpss.npz")
ap.add_argument("--out", default="data/release/dataset")
ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
args = ap.parse_args()

s = GraphStore.load(args.store)
os.makedirs(args.out, exist_ok=True)
for i, pdb_id in enumerate(s.ids):
    y = s.y[s.node_ptr[i]:s.node_ptr[i + 1]]
    e = s.edges[s.edge_ptr[i]:s.edge_ptr[i + 1]]
    base = f"{args.out}/{pdb_id}"
    np.savetxt(f"{base}.node_feature", np.arange(1, len(y) + 1)[None], delimiter=",", fmt="%d")
    np.savetxt(f"{base}.node_labels", y, fmt="%d")
    np.savetxt(f"{base}.edges", e, delimiter=",", fmt="%d")
if args.zip:
    shutil.make_archive(args.out, "zip", os.path.dirname(args.out), os.path.basename(args.out))
