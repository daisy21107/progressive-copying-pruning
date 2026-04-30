#!/bin/bash
# Continue full experiment from stage 2 (teacher already complete)
# Remaining: 3 regimes × 10 seeds = 30 runs

set -e  # Exit on error

REPO_DIR="/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning"
PYTHON="$REPO_DIR/.venv/bin/python"
CONFIG_DIR="$REPO_DIR/configs"

echo "================================================================================"
echo "CONTINUING FULL EXPERIMENT: Stages 2-4"
echo "================================================================================"
echo "Settings: canvas=34, noise=0.15"
echo "Teacher: ✅ Already complete (runs/teacher_seed42/)"
echo "Remaining: 10 (oneshot) + 10 (progressive) + 10 (scratch) = 30 runs"
echo "================================================================================"
echo ""

# Verify teacher exists
TEACHER_CKPT="$REPO_DIR/runs/teacher_seed42/teacher.pt"
if [ ! -f "$TEACHER_CKPT" ]; then
  echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
  exit 1
fi
echo "✅ Teacher checkpoint verified: $TEACHER_CKPT"
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
echo "  Teacher:     runs/teacher_seed42/"
echo "  One-Shot:    runs/oneshot_seed{0..9}/"
echo "  Progressive: runs/progressive_seed{0..9}/"
echo "  Scratch:     runs/scratch_seed{0..9}/"
echo ""
echo "Next step: python scripts/aggregate_final_results.py"
echo "================================================================================"
