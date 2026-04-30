#!/bin/bash
# Quick calibration sweep for random positioning canvas size
# Tests 32, 36, 40, 44 to find 70-85% accuracy range

cd "$(dirname "$0")"
VENV=/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning/.venv/bin/python

echo "================================"
echo "Random Position Canvas Calibration"
echo "================================"

for canvas in 32 36 40 44; do
    echo ""
    echo "Canvas size: $canvas (5 seeds)"
    $VENV scripts/run_seed_sweep.py \
      --config configs/scratch_random_pos_canvas${canvas}.yaml \
      --regime scratch \
      --seeds 0 1 2 3 4
    
    # Aggregate
    echo "  Aggregating results..."
    $VENV -m src.utils.summarize_seeds --pattern "runs/scratch_random_pos_canvas${canvas}_seed*/metrics.csv"
done

echo ""
echo "✅ Calibration complete. Review results above to select optimal canvas_size."
