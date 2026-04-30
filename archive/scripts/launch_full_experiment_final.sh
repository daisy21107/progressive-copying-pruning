#!/bin/bash
# Full experiment: Teacher + 3 regimes (oneshot, progressive, scratch) × 10 seeds each
# Total: 1 + 3×10 = 31 runs
# Expected runtime: ~3-4 hours

set -e  # Exit on error

REPO_DIR="/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning"
PYTHON="$REPO_DIR/.venv/bin/python"
CONFIG_DIR="$REPO_DIR/configs"

echo "================================================================================"
echo "FULL EXPERIMENT RUN: Progressive Copying with Iterative Pruning"
echo "================================================================================"
echo "Settings: canvas=34, noise=0.15"
echo "Teacher seed: 42"
echo "Regime seeds: 0-9 (10 seeds each)"
echo "Total runs: 1 (teacher) + 10 (oneshot) + 10 (progressive) + 10 (scratch) = 31"
echo "================================================================================"
echo ""

# ============================================================================
# Stage 1: Train Teacher (deterministic, seed=42)
# ============================================================================
echo "[Stage 1/4] Training Teacher..."
echo "============================================================================"

$PYTHON scripts/run_seed_sweep.py \
  --config "$CONFIG_DIR/teacher_noise0.15_canvas34.yaml" \
  --regime teacher \
  --seeds 42

TEACHER_CKPT="$REPO_DIR/runs/teacher_noise0.15_canvas34_seed42/teacher.pt"
if [ ! -f "$TEACHER_CKPT" ]; then
  echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
  exit 1
fi

echo "✅ Teacher trained successfully: $TEACHER_CKPT"
echo ""

# ============================================================================
# Stage 2: One-Shot Distillation (10 seeds)
# ============================================================================
echo "[Stage 2/4] Running One-Shot Distillation (10 seeds)..."
echo "============================================================================"

$PYTHON scripts/run_seed_sweep.py \
  --config "$CONFIG_DIR/oneshot_noise0.15_canvas34.yaml" \
  --regime oneshot \
  --seeds 0 1 2 3 4 5 6 7 8 9

echo "✅ One-shot distillation completed (10 seeds)"
echo ""

# ============================================================================
# Stage 3: Progressive Copying with Pruning (10 seeds)
# ============================================================================
echo "[Stage 3/4] Running Progressive Copying with Pruning (10 seeds)..."
echo "============================================================================"

$PYTHON scripts/run_seed_sweep.py \
  --config "$CONFIG_DIR/progressive_noise0.15_canvas34.yaml" \
  --regime progressive \
  --seeds 0 1 2 3 4 5 6 7 8 9

echo "✅ Progressive copying completed (10 seeds)"
echo ""

# ============================================================================
# Stage 4: From-Scratch Training (10 seeds)  
# ============================================================================
echo "[Stage 4/4] Running From-Scratch Training (10 seeds)..."
echo "============================================================================"

$PYTHON scripts/run_seed_sweep.py \
  --config "$CONFIG_DIR/scratch_noise0.15_canvas34.yaml" \
  --regime scratch \
  --seeds 0 1 2 3 4 5 6 7 8 9

echo "✅ From-scratch training completed (10 seeds)"
echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo "================================================================================"
echo "✅ ALL EXPERIMENTS COMPLETED!"
echo "================================================================================"
echo ""
echo "Results Summary:"
echo "  Teacher:     runs/teacher_noise0.15_canvas34_seed42/"
echo "  One-Shot:    runs/oneshot_noise0.15_canvas34_seed{0..9}/"
echo "  Progressive: runs/progressive_noise0.15_canvas34_seed{0..9}/"
echo "  Scratch:     runs/scratch_noise0.15_canvas34_seed{0..9}/"
echo ""
echo "Next steps: Run aggregation and analysis script"
echo "================================================================================"
