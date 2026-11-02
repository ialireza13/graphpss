"""Pick the best sweep configuration per architecture and append the final runs to the queue.

Selection uses validation accuracy only. For each architecture family the best sweep run's
model settings are trained on the full training set with several seeds; ablations reuse the
best GATv2 / GCN / expanded-GCN settings.

    python scripts/select_configs.py configs/full.queue
"""
import glob
import json
import os
import sys

import yaml

FINAL = os.environ.get("FINAL_OVERRIDES", "train.max_epochs=20 train.patience=5")
SEEDS = (0, 1, 2)
MODEL_KEYS = ("conv", "pos_enc", "hidden", "layers", "heads", "fc_layers", "hops", "rel_pos")

queue_path = sys.argv[1]
best = {}
for path in glob.glob("runs/sweep_*/metrics.json"):
    run = os.path.dirname(path)
    m = json.load(open(path))
    cfg = yaml.safe_load(open(f"{run}/config.yaml"))["model"]
    family = cfg["conv"] + (f"_k{cfg['hops']}" if cfg["conv"] == "expanded_gcn" else "")
    if family not in best or m["val"]["acc"] > best[family][0]:
        best[family] = (m["val"]["acc"], cfg, os.path.basename(run))


def overrides(cfg, **changes):
    cfg = {**cfg, **changes}
    return " ".join(f"model.{k}={cfg[k]}" for k in MODEL_KEYS if k in cfg)


def extra(cfg):
    # Expanded GCN collates precomputed k-hop pairs, which needs more loader workers.
    return " train.workers=16" if cfg["conv"] == "expanded_gcn" else ""


names = {"gatv2": "pssnet_gatv2", "gat": "gat", "gcn": "gcn", "gin": "gin", "sage": "sage",
         "expanded_gcn_k2": "expanded_gcn_k2", "expanded_gcn_k3": "expanded_gcn_k3"}
lines = ["", "# ---- appended by scripts/select_configs.py ----", "# selected (val acc): " +
         ", ".join(f"{f}={b[2]} ({100 * b[0]:.2f})" for f, b in sorted(best.items()))]

ORDER = ("gatv2", "gcn", "expanded_gcn_k2", "expanded_gcn_k3", "gin", "sage", "gat")
SINGLE_SEED = {"gat"}  # side comparison, as expensive as GATv2


def final_line(family, seed, **changes):
    cfg = best[family][1]
    return f"{names[family]} {seed} {overrides(cfg, **changes)} {FINAL}{extra(cfg)}"


final_s0 = " ".join(f"runs/{names[f]}_s0" for f in ORDER if f in best)
lines.append("# A. Final comparison on the cluster split, seed 0, then evaluation")
lines += [final_line(f, 0) for f in ORDER if f in best]
MLP_POS = "model.conv=mlp model.fc_layers=3 model.hidden=2048 model.pos_enc=learned model.rel_pos=0"
MLP_CONTACTS = "model.conv=mlp model.fc_layers=3 model.hidden=2048 model.pos_enc=contacts model.rel_pos=16"
lines.append(f"mlp_pos 0 {MLP_POS} {FINAL}")
lines.append(f"mlp_contacts 0 {MLP_CONTACTS} {FINAL}")
lines += [f"! python scripts/robustness.py {final_s0} runs/mlp_pos_s0 runs/mlp_contacts_s0",
          "! python scripts/make_figures.py --run runs/pssnet_gatv2_s0 --out runs/figures",
          "! python scripts/summarize.py"]

lines.append("# B. Ablations (seed 0): the original absolute-index input instead of contact separations")
for family, name in (("gatv2", "gatv2_abspos"), ("gcn", "gcn_abspos"), ("expanded_gcn_k2", "egcn_k2_abspos")):
    if family in best:
        lines.append(final_line(family, 0, pos_enc="learned", rel_pos=0).replace(f"{names[family]} 0", f"{name} 0", 1))
lines.append("# Same models on the random (non-homology) split, to measure leakage")
for family in ("gatv2", "gcn", "expanded_gcn_k2"):
    if family in best:
        lines.append(final_line(family, 0).replace(f"{names[family]} 0", f"{names[family]}_randsplit 0", 1)
                     + " data.split=data/splits/seed0.json")
lines += ["! python scripts/contact_baselines.py --split data/splits/seed0.json",
          "! python scripts/summarize.py"]

lines.append("# C. Extra seeds for error bars")
for seed in SEEDS[1:]:
    lines += [final_line(f, seed) for f in ORDER if f in best and f not in SINGLE_SEED]
    lines.append(f"mlp_contacts {seed} {MLP_CONTACTS} {FINAL}")
lines += ["! python scripts/summarize.py", ""]
with open(queue_path, "a") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
