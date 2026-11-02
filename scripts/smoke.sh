#!/usr/bin/env bash
# One short run per model type on a small subset, to check that everything runs and to time it.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for conv in gatv2 gat gcn gin sage mlp; do
  python -m pssnet.train --config configs/base.yaml --out runs/smoke/$conv \
    --set name=smoke_$conv model.conv=$conv train.max_epochs=1 train.subset=3000
done
for k in 2 3; do
  python -m pssnet.train --config configs/base.yaml --out runs/smoke/egcn$k \
    --set name=smoke_egcn$k model.conv=expanded_gcn model.hops=$k train.max_epochs=1 train.subset=3000
done
python -m pssnet.train --config configs/base.yaml --out runs/smoke/abspos \
  --set name=smoke_abspos model.pos_enc=learned model.rel_pos=0 train.max_epochs=1 train.subset=3000
