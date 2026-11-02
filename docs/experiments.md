# Experiment protocol

Everything runs in order from `configs/full.queue` through `scripts/run_all.sh`, one job
at a time on one A100. The queue can be restarted: finished runs and commands are skipped.

## Data and splits

- **Store:** `data/graphpss.npz` holds 172,891 graphs and 54.2M residues. The graphs use
  Cα contacts at 8 Å, the node input is the residue index, and the labels are the PDB's own helix and sheet records.
- **Main split:** `data/splits/cluster30_seed0.json`, 80/10/10 by residues.
  - Each entry is assigned to the RCSB 30%-identity cluster (MMseqs2) of its label chain
    A entity, the chain the graph was built from.
  - Whole clusters are assigned to train, val or test. That gives 21,700 groups, and the
    largest holds 1.6% of residues.
  - Test proteins whose label sequence is identical to a training protein: 0.9%, against
    16.9% under a random split.
- **Secondary split:** `data/splits/seed0.json`, a random 80/10/10 by entry. It is used
  only to measure how much homology leakage inflates accuracy.

## Training (all GNNs)

- **Optimisation:** Adam with lr 1e-3, cosine decay to 1e-4 over `max_epochs`, gradient
  clipping at 1.0, batches of 32 proteins, bf16 autocast.
- **Model selection:** early stopping on validation loss. The checkpoint with the best
  validation loss is evaluated once on test.
- **Metrics:**
  - residue accuracy (Q3)
  - mean per-protein accuracy
  - macro-F1 and per-class F1
  - confusion matrix
  - train accuracy of the selected checkpoint
- **Input (since the preliminary runs):** each residue starts from the sum over its contacts
  of a learned 32-d embedding of the sequence separation j − i, clipped at ±16. GAT/GATv2
  edges also carry a 16-d embedding of the separation. The original PSS-net input, a
  learned embedding of the absolute residue index, is kept as the `*_abspos` ablation.
- **Architecture:** input, then L × [conv → LayerNorm → ReLU], then an MLP head. Width is the per-node dimension after a conv; for GAT this is
  heads × per-head size.

## Stages

0. **CPU baselines on the cluster split:**
   - majority class;
   - a fixed contact rule;
   - gradient-boosted trees on local contact features (`scripts/contact_baselines.py`);
   - spectral clustering with per-graph Hungarian matching (an optimistic upper bound).
1. **Hyperparameter sweep** (38 runs): 25% of training proteins, 10 epochs, full val set.
   - Grid: layers ∈ {2, 3, 4} × width ∈ {512, 2048} for GATv2 (4 heads), GCN, GIN, SAGE
     and expanded GCN with K = 2 and K = 3.
   - Also GAT v1 with L ∈ {2, 3} at width 2048.
   - Selection uses validation accuracy only (`scripts/select_configs.py`).
2. **Final runs:** the selected configuration per architecture, full training set,
   20 epochs, patience 5.
   - Seeds 0, 1 and 2. GAT v1 gets seed 0 only.
   - Plus a graph-free position-only MLP.
3. **Ablations**, seed 0: the best GATv2, GCN and expanded GCN K = 2 with the original
   absolute-index input (`*_abspos`). `mlp_contacts`, the per-residue contact profile with
   no message passing, is part of the final runs.
4. **Leakage check:** best GATv2, GCN and expanded GCN K = 2 retrained on the random
   split, plus the contact baselines on that split.
5. **Evaluation:**
   - Robustness: test accuracy when 10–100% of long-range contacts (|i − j| ≥ 3) are
     dropped, when random long-range contacts are added, or when only chain edges are
     kept. This perturbs the edges; it is a proxy for changing the threshold, not the
     same thing.
   - Paper figures, with proteins selected by PDB id.
   - `runs/results.md`.

## Known limits to state in the paper

- **Labels are PDB records, not DSSP.** `scripts/build_graph.py` rebuilds them from
  mmCIF `_struct_conf` and `_struct_sheet_range`. It reproduces the stored labels on 299
  random proteins. They agree 93.0% with DSSP 4 on 294 proteins (H, G, I → helix, E →
  sheet); most differences are at helix ends. PDB records vary by depositor and software.
- **Node indices.** They count observed residues, not true sequence positions. 36.6% of
  chains have a backbone gap, but 99% of graphs are still connected.
- **Not a sequence-prediction result.** The labels are a function of the 3D structure,
  so accuracy should not be compared with sequence-only predictors such as JPred.
