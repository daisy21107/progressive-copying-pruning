#!/bin/bash
set -euo pipefail

WORKSPACE="/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning"
PYTHON_BIN="$WORKSPACE/.venv/bin/python"
cd "$WORKSPACE"

echo "=========================================="
echo "RELAUNCHING MISSING CIFAR RUNS"
echo "Start time: $(date)"
echo "=========================================="

echo ""
echo "Stage 1/3: Running oneshot distillation (10 seeds)..."
"$PYTHON_BIN" scripts/run_seed_sweep.py \
  --config configs/oneshot_cifar100.yaml \
  --regime oneshot \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-std-override 0.2

echo ""
echo "Stage 2/3: Running progressive copying + pruning (10 seeds)..."
"$PYTHON_BIN" scripts/run_seed_sweep.py \
  --config configs/progressive_cifar100.yaml \
  --regime progressive \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-std-override 0.2

echo ""
echo "Stage 3/3: Running scratch training (seeds 5-9)..."
"$PYTHON_BIN" scripts/run_seed_sweep.py \
  --config configs/scratch_cifar100.yaml \
  --regime scratch \
  --seeds 5 6 7 8 9 \
  --noise-std-override 0.2

echo ""
echo "=========================================="
echo "ALL MISSING RUNS COMPLETE!"
echo "End time: $(date)"
echo "=========================================="
