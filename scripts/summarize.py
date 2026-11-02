"""Collect runs/*/metrics.json into markdown tables (runs/results.md)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import yaml

groups = {"final": defaultdict(list), "sweep": defaultdict(list)}
for path in sorted(glob.glob("runs/*/metrics.json")):
    run = os.path.dirname(path)
    m = json.load(open(path))
    cfg = yaml.safe_load(open(f"{run}/config.yaml"))
    m["split"] = os.path.basename(cfg["data"]["split"]).replace(".json", "")
    m["model"] = cfg["model"]
    groups["sweep" if m["name"].startswith("sweep_") else "final"][m["name"]].append(m)


def fmt(vals):
    vals = [100 * v for v in vals]
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"


def table(by_name, key):
    rows = ["| Model | Split | Seeds | Params | Layers × width | Train acc | Val acc | Test acc | "
            "Test per-protein acc | Test macro-F1 | F1 coil / helix / sheet | Epochs (best) | Hours |",
            "|---" * 13 + "|"]
    for name, ms in sorted(by_name.items(), key=lambda kv: -np.mean([m[key]["acc"] for m in kv[1]])):
        md = ms[0]["model"]
        width = md["hidden"] * (md["heads"] if md["conv"] in ("gat", "gatv2") else 1)
        f1 = " / ".join(f"{100 * np.mean([m['test']['f1'][c] for m in ms]):.1f}" for c in ("coil", "helix", "sheet"))
        train = fmt([m["train"]["acc"] for m in ms]) if all("train" in m for m in ms) else "-"
        ppa = fmt([m["test"]["per_protein_acc"] for m in ms]) if all("per_protein_acc" in m["test"] for m in ms) else "-"
        rows.append(f"| {name} | {ms[0]['split']} | {len(ms)} | {ms[0]['params'] / 1e6:.2f}M | "
                    f"{md['layers']} × {width} | {train} | {fmt([m['val']['acc'] for m in ms])} | "
                    f"{fmt([m['test']['acc'] for m in ms])} | {ppa} | {fmt([m['test']['macro_f1'] for m in ms])} | "
                    f"{f1} | {ms[0]['epochs_run']} ({ms[0]['best_epoch']}) | "
                    f"{np.mean([m['train_hours'] for m in ms]):.1f} |")
    return rows


out = ["## Main runs (sorted by test accuracy)", ""] + table(groups["final"], "test")
if groups["sweep"]:
    out += ["", "## Sweep (25% of training data, 10 epochs; sorted by val accuracy)", ""] + table(groups["sweep"], "val")
for f in sorted(glob.glob("runs/contact_baselines_*.json")):
    r = json.load(open(f))
    out += ["", f"### Contact baselines ({os.path.basename(f)})", "",
            "| Method | Test acc | Per-protein acc | Macro-F1 |", "|---|---|---|---|"]
    out += [f"| {k} | {100 * v['acc']:.2f} | {100 * v['per_protein_acc']:.2f} | {100 * v['macro_f1']:.2f} |"
            for k, v in r.items() if k != "split"]
for f in sorted(glob.glob("runs/spectral_baseline*.json")) + ["runs/dataset_stats.json", "runs/label_audit/summary.json"]:
    if os.path.exists(f):
        out += ["", f"### {f}", "```json", open(f).read().strip(), "```"]
if os.path.exists("runs/robustness.json"):
    rob = json.load(open("runs/robustness.json"))
    settings = list(next(iter(rob.values())).keys())
    out += ["", "### Robustness (test acc under contact-graph perturbation)", "",
            "| Run | " + " | ".join(settings) + " |", "|---" * (len(settings) + 1) + "|"]
    out += [f"| {os.path.basename(run)} | " + " | ".join(f"{100 * r[s]['acc']:.1f}" for s in settings) + " |"
            for run, r in rob.items()]
open("runs/results.md", "w").write("\n".join(out) + "\n")
print("\n".join(out[:40]))
