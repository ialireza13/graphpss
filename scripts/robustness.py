"""Evaluate trained checkpoints on the test split under contact-graph perturbations.

Tests the paper's claim that the model is robust to the contact threshold: removing or
adding long-range contacts mimics a smaller or larger threshold.
"""
import argparse
import json
import os
import sys

import torch
import torch.multiprocessing

# Perturbed batches recompute k-hop indices in the workers; the default fd-passing sharing
# strategy runs out of descriptors ("received 0 items of ancdata") on expanded GCN K=3.
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, "src")
from pssnet.data import GraphStore, load_or_make_split, make_loader  # noqa: E402
from pssnet.models import build_model  # noqa: E402
from pssnet.train import evaluate  # noqa: E402

SETTINGS = {
    "clean": None,
    "drop_long_10": {"drop_long": 0.1},
    "drop_long_25": {"drop_long": 0.25},
    "drop_long_50": {"drop_long": 0.5},
    "drop_long_100": {"drop_long": 1.0},
    "add_long_10": {"add_long": 0.1},
    "add_long_25": {"add_long": 0.25},
    "chain_only": {"chain_only": True},
}

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="runs/robustness.json")
args = ap.parse_args()

device = torch.device("cuda")
results = json.load(open(args.out)) if os.path.exists(args.out) else {}
store = None
for run in args.runs:
    ckpt = f"{run}/best.pt"
    if not os.path.exists(f"{run}/metrics.json") or run in results:
        continue
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state["config"]
    store = store or GraphStore.load(cfg["data"]["store"])
    split = load_or_make_split(cfg["data"]["split"], len(store))
    model = build_model(cfg).to(device)
    model.load_state_dict(state["state_dict"])
    hops = cfg["model"].get("hops", 1)
    results[run] = {}
    for name, pert in SETTINGS.items():
        loader = make_loader(store, split["test"], 64, False, hops, workers=8, perturb=pert, seed=123)
        m = evaluate(model, loader, device, amp=True)
        results[run][name] = {"acc": m["acc"], "macro_f1": m["macro_f1"]}
        print(run, name, f"{m['acc']:.4f}", flush=True)
    json.dump(results, open(args.out, "w"), indent=2)
