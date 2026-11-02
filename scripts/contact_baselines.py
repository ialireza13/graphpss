"""Non-GNN structural baselines on the contact graphs.

rule:  no training. With Cα contacts at 8 Å, helices close i->i+3 and i->i+4 contacts
       (≈5.0 and ≈6.2 Å), while extended strands do not (i->i+3 ≈ 10 Å). A residue is
       helix if it lies in a window j..j+4 with both (j, j+3) and (j, j+4) contacts; sheet
       if it is not helix, has no (i, i+3) contact on either side and has >= 2 long-range
       contacts (|i - j| > 4); coil otherwise.
gbt:   gradient-boosted trees on hand-crafted local contact features: contacts at sequence
       separations ±1..±max_sep (default 8; 16 matches the GNN input), number of long-range contacts, degree, relative position and
       chain length, for the residue and its neighbours at ±1, ±2.

    python scripts/contact_baselines.py --split data/splits/cluster30_seed0.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "src")
from pssnet.data import GraphStore, load_or_make_split  # noqa: E402
from pssnet.train import metrics_from_confusion  # noqa: E402

SEPS = [s for s in range(-8, 9) if s != 0]
STORE = None


def _init(path):
    global STORE
    STORE = GraphStore.load(path)


def graph_features(g):
    s = STORE
    n = s.num_nodes(g)
    e = s.edges[s.edge_ptr[g]:s.edge_ptr[g + 1]].astype(np.int64)
    keys = set((e[:, 0] * n + e[:, 1]).tolist())
    i = np.arange(n)
    contact = np.zeros((n, len(SEPS)), dtype=np.float32)
    for c, sep in enumerate(SEPS):
        j = i + sep
        ok = (j >= 0) & (j < n)
        contact[ok, c] = [(a * n + b) in keys for a, b in zip(i[ok], j[ok])]
    deg = np.bincount(e[:, 0], minlength=n).astype(np.float32)
    long = np.bincount(e[np.abs(e[:, 0] - e[:, 1]) > 4, 0], minlength=n).astype(np.float32)
    base = np.column_stack([contact, long, deg, i / max(n - 1, 1), np.full(n, n)])
    window = [np.roll(base, -w, axis=0) * ((i + w >= 0) & (i + w < n))[:, None] for w in (-2, -1, 1, 2)]
    feats = np.hstack([base] + window)

    # Rule baseline
    c3, c4 = contact[:, SEPS.index(3)], contact[:, SEPS.index(4)]
    turn = (c3 > 0) & (c4 > 0)
    helix = np.zeros(n, dtype=bool)
    for k in range(5):
        helix |= np.roll(turn, k) & (i >= k)
    no_turn = (contact[:, SEPS.index(3)] == 0) & (contact[:, SEPS.index(-3)] == 0)
    sheet = ~helix & no_turn & (long >= 2)
    rule = np.where(helix, 1, np.where(sheet, 2, 0))
    y = s.y[s.node_ptr[g]:s.node_ptr[g + 1]].astype(np.int64)
    return feats, y, rule


def collect(graph_ids, store_path, workers):
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(store_path,)) as ex:
        res = list(ex.map(graph_features, graph_ids, chunksize=64))
    return res


def confusion(y, p):
    return np.bincount(3 * y + p, minlength=9).reshape(3, 3)


def per_protein(res, preds):
    return float(np.mean([(p == r[1]).mean() for r, p in zip(res, preds)]))


ap = argparse.ArgumentParser()
ap.add_argument("--store", default="data/graphpss.npz")
ap.add_argument("--split", default="data/splits/cluster30_seed0.json")
ap.add_argument("--train-graphs", type=int, default=20000)
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--max-sep", type=int, default=8, help="contacts at separations +-1..max_sep")
ap.add_argument("--out", default=None)
args = ap.parse_args()
SEPS = [s for s in range(-args.max_sep, args.max_sep + 1) if s != 0]  # inherited by forked workers

os.makedirs("runs", exist_ok=True)
s = GraphStore.load(args.store)
split = load_or_make_split(args.split, len(s))
tag = os.path.basename(args.split).replace(".json", "")
suffix = "" if args.max_sep == 8 else f"_sep{args.max_sep}"
out_path = args.out or f"runs/contact_baselines_{tag}{suffix}.json"

train_ids = np.random.default_rng(0).choice(split["train"], args.train_graphs, replace=False)
train = collect(train_ids.tolist(), args.store, args.workers)
test = collect(split["test"], args.store, args.workers)

y_test = np.concatenate([r[1] for r in test])
rule_pred = [r[2] for r in test]
results = {"split": args.split}
results["rule"] = metrics_from_confusion(confusion(y_test, np.concatenate(rule_pred)),
                                         per_protein_acc=per_protein(test, rule_pred))

X = np.vstack([r[0] for r in train])
y = np.concatenate([r[1] for r in train])
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                     early_stopping=True, random_state=0)
clf.fit(X, y)
gbt_pred = [clf.predict(r[0]) for r in test]
results["gbt"] = metrics_from_confusion(confusion(y_test, np.concatenate(gbt_pred)),
                                        per_protein_acc=per_protein(test, gbt_pred))
results["gbt"]["train_residues"] = int(len(y))
json.dump(results, open(out_path, "w"), indent=2)
print(json.dumps({k: {m: v[m] for m in ("acc", "macro_f1", "per_protein_acc")}
                  for k, v in results.items() if k != "split"}, indent=2))
