# Final results (queue finished 2026-10-29 07:27 UTC)

The full per-run table is in `runs/results.md`, regenerated with `python scripts/summarize.py`.

## Setup

- **Split:** homology-aware split by 30%-identity sequence cluster unless marked "random".
- **Final runs:** 20 epochs with patience 5, using the configuration chosen on validation
  accuracy in the sweep.
- **Values:** test accuracy (%), mean ± std over seeds 0, 1 and 2 where 3 seeds ran.

## Main comparison

| Model | Config | Params | Test acc | Per-protein | Macro-F1 | F1 coil / helix / sheet |
|---|---|---|---|---|---|---|
| **PSS-net (GATv2)** | 4 × (4 heads × 128) | 1.72M | **91.02 ± 0.03** | 91.35 ± 0.04 | 90.82 ± 0.05 | 87.7 / 94.1 / 90.6 |
| GAT (1 seed) | 3 × (4 × 512) | 9.64M | 90.83 | 91.15 | 90.61 | 87.5 / 94.1 / 90.2 |
| GraphSAGE | 4 × 2048 | 29.5M | 90.07 ± 0.05 | 90.49 ± 0.06 | 89.83 ± 0.05 | 86.6 / 93.6 / 89.3 |
| Expanded GCN, K = 3 | 4 × 2048 | 16.9M | 89.70 ± 0.05 | 90.09 ± 0.04 | 89.41 ± 0.04 | 86.0 / 93.4 / 88.8 |
| Expanded GCN, K = 2 | 4 × 2048 | 16.9M | 89.53 ± 0.08 | 89.96 ± 0.07 | 89.24 ± 0.10 | 85.8 / 93.3 / 88.6 |
| GCN | 2 × 2048 | 8.5M | 86.12 ± 0.11 | 86.75 ± 0.11 | 85.68 ± 0.11 | 81.5 / 91.2 / 84.4 |
| GIN | 2 × 2048 | 37.9M | 85.87 ± 0.19 | 86.42 ± 0.20 | 85.43 ± 0.21 | 81.2 / 91.1 / 84.0 |
| Tree model on contact profile, ±16 | – | – | 87.07 | 87.25 | 86.36 | – |
| Tree model on contact profile, ±8 | – | – | 86.42 | 86.62 | 85.62 | 81.6 / 92.9 / 82.4 |
| MLP on own contact profile, no message passing | – | 8.5M | 78.36 ± 0.01 | 78.26 | 76.99 | 71.6 / 87.4 / 72.0 |
| Contact rule (helix / strand signatures) | – | – | 65.28 | 65.11 | 59.98 | 35.2 / 79.2 / 65.6 |
| Spectral clustering, oracle label matching | – | – | 44.5 | – | – | – |
| MLP on residue index only | – | 8.5M | 41.86 | 40.47 | 24.04 | – |
| Majority class (helix) | – | – | 41.1 | – | – | – |

## Input ablation (seed 0)

The original PSS-net input is a learned embedding of the absolute residue index. The new
input is the summed embedding of the node's contact separations, plus separation edge
features for GAT/GATv2.

| Model | Contact input | Absolute index | Δ |
|---|---|---|---|
| GATv2 | 91.05 | 84.63 | +6.4 |
| Expanded GCN, K = 2 | 89.47 | 82.24 | +7.2 |
| GCN | 86.04 | 80.28 | +5.8 |

Edge separation features alone, with identical node inputs, collapse to the majority
class: GATv2 uses edge features only in the attention logits, so every node stays
identical.

## Homology leakage (seed 0)

| Model | Cluster split | Random split | Δ |
|---|---|---|---|
| PSS-net (GATv2) | 91.05 | 94.36 | +3.3 |
| Expanded GCN, K = 2 | 89.47 | 94.14 | +4.7 |
| GCN | 86.04 | 92.54 | +6.5 |
| Tree model, ±8 | 86.42 | 87.13 | +0.7 |
| GATv2, absolute index (old PSS-net) | 84.05* | 89.70* | +5.7 |

\* 12-epoch preliminary runs.

## Robustness to edge perturbation (seed 0)

Each setting perturbs contacts with |i − j| ≥ 3 in the test graphs: dropping a fraction
of them, adding random ones, or keeping only chain edges.

| Model | Clean | Drop 10% | Drop 25% | Drop 50% | Drop all | Add 10% | Add 25% |
|---|---|---|---|---|---|---|---|
| GATv2 | 91.1 | 82.5 | 63.3 | 43.0 | 36.6 | 90.3 | 88.8 |
| GAT | 90.8 | 83.2 | 65.3 | 42.9 | 36.6 | 90.1 | 88.5 |
| SAGE | 90.1 | 82.5 | 65.0 | 43.3 | 36.6 | 89.1 | 86.7 |
| Expanded GCN, K = 3 | 89.7 | 82.2 | 67.4 | 46.0 | 36.6 | 88.7 | 86.9 |
| Expanded GCN, K = 2 | 89.5 | 82.1 | 66.0 | 44.2 | 36.5 | 88.5 | 86.7 |
| GCN | 86.0 | 79.5 | 66.4 | 47.5 | 36.6 | 83.9 | 80.1 |
| GIN | 86.0 | 80.5 | 68.9 | 50.6 | 36.6 | 83.9 | 80.2 |
| MLP on own contact profile | 78.4 | 72.9 | 63.6 | 49.2 | 36.6 | 77.4 | 76.0 |

## Points for the paper

1. **Headline.** PSS-net reaches 91.0% Q3 on a homology-aware split from the contact
   graph alone, with 1.7M parameters, against 87.1% for a tree model on the same contact
   profile.
2. **Message passing.** It adds about 13 points over the residue's own contact profile
   (91.0 vs 78.4).
3. **Expanded GCN.** It beats GCN by +3.4 (K = 2) and +3.6 (K = 3), at seed-level
   std ≈ 0.1. It is close to GraphSAGE and about 1.3 points below attention.
4. **Input.** Translation-invariant contact-separation input beats the absolute-index
   embedding by 6–7 points for every architecture.
5. **Leakage.** A random split inflates GNN accuracy by 3–6.5 points, but the tree model
   by only 0.7, so GNNs partly memorise homologous folds. The manuscript's 91.1% came
   from a random-style split.
6. **Robustness claim.** It needs rewording. The models tolerate spurious long-range
   contacts (−2 to −6 points at +25%) but not missing ones: removing the i→i+3 and i→i+4
   contacts that define a helix destroys the signal. This perturbs edges; it does not
   rebuild the graphs at another threshold.
7. **Scope.** DSSP labels are a function of the 3D structure. This is structure →
   secondary-structure annotation, not sequence-based prediction.

**Figures for the paper:** `runs/figures/`, regenerated from `runs/pssnet_gatv2_s0`.

## Error analysis (seed 0, all 17,870 test proteins)

From `scripts/analyze_errors.py`.

| Distance from a segment boundary | Share of residues | Accuracy | Share of errors |
|---|---|---|---|
| 0 | 27.4% | 81.5 | 56.7% |
| 1 | 21.2% | 92.4 | 18.0% |
| 2 | 14.8% | 94.7 | 8.7% |
| 3 or more | 36.6% | 96.0 | 16.5% |

Helices of 7+ residues are found 98–100% of the time, against 74% for 4–6 residues and
65% for 1–3. Strands of 4+ residues are found 99% of the time, against 85% for 1–3.
