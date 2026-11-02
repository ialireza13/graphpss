# GraphPSS and PSS-net

Protein secondary structure assignment from residue contact graphs. This repo contains
the code for **GraphPSS**, a node-classification benchmark of 172,891 protein contact
graphs with 54.2M nodes, and **PSS-net**, a graph attention model that reaches
**91.02 ± 0.03% Q3** on a homology-aware split using the contact graph alone.

The manuscript is in [paper/paper.pdf](paper/paper.pdf), the results in
[docs/results.md](docs/results.md), and the protocol in
[docs/experiments.md](docs/experiments.md).

## Dataset

- **Graph.** One node per residue of a protein's first chain. Two residues are connected
  when their Cα atoms are closer than 8 Å. No amino-acid identity or coordinates are used.
- **Node input.** Each residue starts from the sum of embeddings of its contacts' sequence
  separations, clipped at ±16. GAT and GATv2 edges also carry the separation.
- **Labels.** The PDB's own secondary-structure records: 0 = coil, 1 = helix, 2 = sheet.
  These are not DSSP output.
- **Splits.** `data/splits/cluster30_seed0.json` is the main 80/10/10 split, which keeps
  30%-identity sequence clusters apart. `data/splits/seed0.json` is a random split, used
  only to measure homology leakage.

The graphs are on Zenodo: [10.5281/zenodo.23829544](https://doi.org/10.5281/zenodo.23829544).
Put `graphpss.npz` in `data/` and skip the two optional build steps under Usage.
To regenerate the data, download mmCIF files from RCSB and run `scripts/build_graph.py`. It
reproduced the original graphs exactly on 299 random proteins, though later PDB remediation
can change individual entries, so comparisons should use the Zenodo release.

## Results

| Model | Test Q3 (%) |
|---|---|
| **PSS-net (GATv2)** | **91.02 ± 0.03** |
| GraphSAGE | 90.07 ± 0.05 |
| Expanded GCN, K = 3 | 89.70 ± 0.05 |
| Expanded GCN, K = 2 | 89.53 ± 0.08 |
| Gradient-boosted trees on contact features | 87.07 |
| GCN | 86.12 ± 0.11 |
| GIN | 85.87 ± 0.19 |
| MLP on own contacts, no message passing | 78.36 ± 0.01 |

## Layout

```
src/pssnet/   data.py (store, splits, batching, k-hop), models.py (all GNNs), train.py
configs/      base.yaml (PSS-net defaults), full.queue (every experiment in the paper)
scripts/      dataset, baseline, evaluation and figure scripts
data/splits/  the two fixed splits
docs/         protocol, results, per-run table
paper/        manuscript, figures
```

## Usage

```bash
pip install -e .                                    # torch and torch_geometric must match your CUDA
python scripts/build_graph.py data/cif/*.cif        # optional: mmCIF -> data/pt_cache
python scripts/build_store.py                       # optional: data/pt_cache -> data/graphpss.npz
python scripts/build_hops.py                        # k-hop pairs for the expanded GCN
PYTHONPATH=src python -m pssnet.train --config configs/base.yaml
PYTHONPATH=src python -m pssnet.train --config configs/base.yaml --set name=gcn model.conv=gcn
PYTHONPATH=src python -m pssnet.train --config configs/base.yaml \
    --set name=egcn_k2 model.conv=expanded_gcn model.hops=2
python scripts/run_queue.py configs/full.queue --lanes 1   # all experiments, one at a time
python scripts/summarize.py                                # runs/results.md
```

Early stopping and checkpoint selection use the validation set. The test set is evaluated
once, at the end of training.

## License

GPL-3.0. See [LICENSE](LICENSE).
