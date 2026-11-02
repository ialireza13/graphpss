"""Regenerate the paper figures from a trained checkpoint (proteins chosen by PDB id).

    python scripts/make_figures.py --run runs/pssnet_gatv2_s0 --out paper/figures
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, "src")
from pssnet.data import Collate, GraphStore, load_or_make_split  # noqa: E402
from pssnet.models import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/pssnet_gatv2_s0")
ap.add_argument("--out", default="runs/figures")
ap.add_argument("--samples", nargs="+", default=["6N5U", "2LKN", "1Q4S"])
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({"font.size": 14})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(f"{args.run}/best.pt", map_location=device, weights_only=False)
cfg = state["config"]
store = GraphStore.load(cfg["data"]["store"])
split = load_or_make_split(cfg["data"]["split"], len(store))
model = build_model(cfg).to(device).eval()
model.load_state_dict(state["state_dict"])
collate = Collate(store, cfg["model"].get("hops", 1))
test_set = set(split["test"])


def graph(pdb_id):
    i = store.index_of(pdb_id)
    assert store.ids[i] == pdb_id, f"{pdb_id} not in dataset"
    return i, collate([i]).to(device)


@torch.no_grad()
def forward(b, **kw):
    return model(b, **kw)


# Dataset distributions
sizes = np.diff(store.node_ptr)
plt.figure(figsize=(6, 3))
plt.hist(sizes, bins=40, color="darkslategray")
plt.xlabel("Graph size (residues)")
plt.ylabel("Proteins")
plt.tight_layout()
plt.savefig(f"{args.out}/dist_sizes.svg")
counts = np.bincount(store.y, minlength=3) / len(store.y)
plt.figure(figsize=(6, 3))
plt.bar([0, 1, 2], counts, color=["gray", "firebrick", "mediumblue"])
plt.xticks([0, 1, 2], ["Coil", r"$\alpha$-Helix", r"$\beta$-Sheet"])
plt.ylabel("Fraction of residues")
plt.tight_layout()
plt.savefig(f"{args.out}/dist_labels.svg")

# Sample predictions
cmap = ListedColormap(["white", "firebrick", "mediumblue"])
fig, axes = plt.subplots(len(args.samples), 1, figsize=(7, 1.7 * len(args.samples)))
info = {}
for ax, pid in zip(np.atleast_1d(axes), args.samples):
    i, b = graph(pid)
    pred = forward(b).argmax(1).cpu().numpy()
    y = b.y.cpu().numpy()
    info[pid] = {"in_test_split": i in test_set, "acc": float((pred == y).mean()), "n": int(len(y))}
    band = np.vstack([np.tile(y, (6, 1)), np.full((2, len(y)), np.nan), np.tile(pred, (6, 1))])
    ax.imshow(band, cmap=cmap, aspect="auto", vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks([2.5, 10.5], ["Real", "Pred"])
    ax.set_title(f"{pid} (acc {info[pid]['acc']:.2f})")
axes[-1].set_xlabel("Residue index")
fig.tight_layout()
fig.savefig(f"{args.out}/sample.svg")

# Attention map (GAT models only)
if cfg["model"]["conv"] in ("gat", "gatv2"):
    _, b = graph(args.samples[0])
    _, extra = forward(b, return_attention=True)
    n = b.num_nodes
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for layer, (ei, alpha) in enumerate(extra["attention"][-1:]):
        att = np.zeros((n, n))
        e = ei.cpu().numpy()
        att[e[1], e[0]] = alpha.float().mean(1).cpu().numpy()  # target attends to source
        ax[0].imshow(att, cmap="summer")
    adj = np.zeros((n, n))
    e = b.edge_index.cpu().numpy()
    adj[e[0], e[1]] = 1
    ax[0].set_title("Attention (last layer)")
    ax[1].imshow(adj, cmap="summer")
    ax[1].set_title("Adjacency")
    fig.tight_layout()
    fig.savefig(f"{args.out}/attention.svg")

    # Attention mass by sequence separation, over 500 test proteins
    sep_mass, sep_edges = np.zeros(2000), np.zeros(2000)
    for g in np.random.default_rng(0).choice(split["test"], 500, replace=False):
        bb = collate([int(g)]).to(device)
        _, ex = forward(bb, return_attention=True)
        ei, alpha = ex["attention"][-1]
        d = (ei[0] - ei[1]).abs().cpu().numpy()
        np.add.at(sep_mass, d, alpha.float().mean(1).cpu().numpy())
        np.add.at(sep_edges, d, 1)
    keep = slice(0, 30)
    plt.figure(figsize=(6, 3))
    plt.bar(np.arange(30), sep_mass[keep] / sep_mass.sum(), color="darkslategray", label="attention")
    plt.plot(np.arange(30), sep_edges[keep] / sep_edges.sum(), "o-", color="orange", label="edges")
    plt.xlabel("Sequence separation |i - j|")
    plt.ylabel("Fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out}/attention_by_separation.svg")
    info["attention_mass_sep_le_4"] = float(sep_mass[:5].sum() / sep_mass.sum())
    info["edge_frac_sep_le_4"] = float(sep_edges[:5].sum() / sep_edges.sum())

# t-SNE of latent representations
from sklearn.manifold import TSNE  # noqa: E402

for tag, ids in (("tsne", [store.index_of(args.samples[0])]),
                 ("tsne_10", np.random.default_rng(1).choice(split["test"], 10, replace=False).tolist())):
    b = collate([int(i) for i in ids]).to(device)
    _, ex = forward(b, return_embed=True)
    z = TSNE(2, random_state=0).fit_transform(ex["embed"]["gnn"].float().cpu().numpy())
    y = b.y.cpu().numpy()
    plt.figure(figsize=(6, 3.5))
    for c, m, col, lab in ((1, "^", "firebrick", r"$\alpha$-Helix"), (2, "s", "mediumblue", r"$\beta$-Sheet"),
                           (0, "o", "gray", "Coil")):
        plt.scatter(z[y == c, 0], z[y == c, 1], marker=m, c=col, s=8, label=lab)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(f"{args.out}/{tag}.svg")
    info[f"{tag}_proteins"] = [str(store.ids[i]) for i in ids]

json.dump(info, open(f"{args.out}/figures_info.json", "w"), indent=2)
print(json.dumps(info, indent=2))
