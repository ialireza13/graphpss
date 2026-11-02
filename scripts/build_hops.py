"""Precompute exact-k-hop pair lists for the expanded GCN (data/hops/)."""
import argparse
import json
import sys

sys.path.insert(0, "src")
from pssnet.data import GraphStore, build_hop_store  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="data/graphpss.npz")
ap.add_argument("--out", default="data/hops")
ap.add_argument("--k-max", type=int, default=3)
ap.add_argument("--workers", type=int, default=24)
args = ap.parse_args()
print(json.dumps(build_hop_store(GraphStore.load(args.store), args.out, args.k_max, args.workers), indent=2))
