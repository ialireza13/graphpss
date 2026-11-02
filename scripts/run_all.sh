#!/usr/bin/env bash
# Full pipeline: every job runs in sequence from configs/full.queue (restartable).
cd "$(dirname "$0")/.."
export PYTHONPATH=src
mkdir -p logs runs
python scripts/run_queue.py configs/full.queue --lanes 1
echo "ALL DONE $(date)" >> logs/queue.log
