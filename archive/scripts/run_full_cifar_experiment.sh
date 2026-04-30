#!/bin/bash
set -euo pipefail

WORKSPACE="/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning"
cd "$WORKSPACE"

echo "=========================================="
echo "FULL CIFAR-100 EXPERIMENT"
echo "Settings: noise_std=0.2, 31 runs total"
echo "Start time: $(date)"
echo "=========================================="

# Stage 1: Train teacher (seed 42)
echo ""
echo "[$(date '+%H:%M:%S')] Stage 1/4: Training teacher (seed=42)..."
.venv/bin/python -m src.main teacher \
  --config configs/teacher_cifar100.yaml \
  --seed 42 \
  --out-dir runs/cifar_teacher_final

echo "[$(date '+%H:%M:%S')] Teacher training complete!"

# Stage 2-4: Run 10-seed sweeps for each regime
echo ""
echo "[$(date '+%H:%M:%S')] Stage 2/4: Running one-shot distillation + post-hoc pruning (10 seeds)..."
.venv/bin/python scripts/run_seed_sweep.py \
  --config configs/oneshot_cifar100.yaml \
  --regime oneshot \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-std-override 0.2

echo "[$(date '+%H:%M:%S')] One-shot distillation + post-hoc pruning complete!"

echo ""
echo "[$(date '+%H:%M:%S')] Stage 3/4: Running progressive copying + pruning (10 seeds)..."
.venv/bin/python scripts/run_seed_sweep.py \
  --config configs/progressive_cifar100.yaml \
  --regime progressive \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-std-override 0.2

echo "[$(date '+%H:%M:%S')] Progressive copying complete!"

echo ""
echo "[$(date '+%H:%M:%S')] Stage 4/4: Running from-scratch training + post-hoc pruning (10 seeds)..."
.venv/bin/python scripts/run_seed_sweep.py \
  --config configs/scratch_cifar100.yaml \
  --regime scratch \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-std-override 0.2

echo "[$(date '+%H:%M:%S')] From-scratch training + post-hoc pruning complete!"

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run aggregation: python scripts/aggregate_cifar_results.py"
echo "2. Check results in results/"
