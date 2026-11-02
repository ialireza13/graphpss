"""Build the compact GraphPSS store (data/graphpss.npz) from .pt files or the CSV release."""
import argparse
import json
import sys

sys.path.insert(0, "src")
from pssnet.data import build_store  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pt_cache", help="data/pt_cache (default) or a CSV release dir")
    ap.add_argument("--csv", action="store_true", help="--src is a CSV release directory")
    ap.add_argument("--out", default="data/graphpss.npz")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    print(json.dumps(build_store(args.src, args.out, args.workers, from_pt=not args.csv), indent=2))
