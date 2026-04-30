#!/bin/bash
# Full experiment launch: teacher (seed 42) + 3 regimes (10 seeds each)

cd "$(dirname "$0")"
VENV=/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning/.venv/bin/python

echo "================================"
echo "STEP 1: Train teacher (seed=42)"
echo "================================"
$VENV scripts/run_seed_sweep.py \
  --config configs/teacher_noise0.40.yaml \
  --regime teacher \
  --seeds 42

echo ""
echo "================================"
echo "STEP 2: One-shot distillation (10 seeds)"
echo "================================"
$VENV scripts/run_seed_sweep.py \
  --config configs/oneshot_noise0.40.yaml \
  --regime oneshot \
  --seeds 0 1 2 3 4 5 6 7 8 9

echo ""
echo "================================"
echo "STEP 3: Progressive copy+prune (10 seeds)"
echo "================================"
$VENV scripts/run_seed_sweep.py \
  --config configs/progressive_noise0.40.yaml \
  --regime progressive \
  --seeds 0 1 2 3 4 5 6 7 8 9

echo ""
echo "================================"
echo "STEP 4: From-scratch baseline (10 seeds)"
echo "================================"
$VENV scripts/run_seed_sweep.py \
  --config configs/scratch_noise0.40.yaml \
  --regime scratch \
  --seeds 0 1 2 3 4 5 6 7 8 9

echo ""
echo "================================"
echo "STEP 5: Aggregate all results"
echo "================================"
$VENV -m src.utils.summarize_seeds --pattern "runs/teacher_noise0.40_seed42/metrics.csv"
$VENV -m src.utils.summarize_seeds --pattern "runs/oneshot_noise0.40_seed*/metrics.csv"
$VENV -m src.utils.summarize_seeds --pattern "runs/progressive_noise0.40_seed*/metrics.csv"
$VENV -m src.utils.summarize_seeds --pattern "runs/scratch_noise0.40_seed*/metrics.csv"

echo ""
echo "✅ FULL EXPERIMENT COMPLETE"
