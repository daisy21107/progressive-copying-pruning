#!/bin/bash
# Smart calibration loop - tests progressively until finding stable zone

set -e

CONFIG="configs/scratch_noise0.40.yaml"
REGIME="scratch"
SEEDS="0 1 2 3 4"

# Test candidates in smart order (avoiding cliff)
CANDIDATES=(0.51 0.55 0.50 0.56)

for noise in "${CANDIDATES[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing noise_std=$noise"
    echo "========================================"
    
    rm -rf runs/scratch_seed[0-4]
    
    /Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning/.venv/bin/python \
        scripts/run_seed_sweep.py \
        --config "$CONFIG" \
        --regime "$REGIME" \
        --seeds $SEEDS \
        --noise-std-override "$noise"
    
    echo ""
    echo "Results for noise_std=$noise:"
    /Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning/.venv/bin/python \
        -m src.utils.summarize_seeds --pattern "runs/scratch_seed[0-4]/metrics.csv"
    
    # Extract mean accuracy
    RESULTS=$(/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning/.venv/bin/python \
        -m src.utils.summarize_seeds --pattern "runs/scratch_seed[0-4]/metrics.csv" 2>&1)
    
    MEAN_ACC=$(echo "$RESULTS" | grep "val_acc:" | grep -oE "0\.[0-9]+" | head -1)
    
    echo "Mean accuracy: $MEAN_ACC"
    
    # Decision logic
    if (( $(echo "$MEAN_ACC > 0.70 && $MEAN_ACC < 0.85" | bc -l) )); then
        echo "✅ FOUND STABLE ZONE at noise_std=$noise"
        echo "Using this noise level for main experiment!"
        exit 0
    else
        echo "❌ Not in target range [0.70, 0.85]"
        echo "Continuing to next candidate..."
    fi
done

echo ""
echo "⚠️  No ideal candidate found. Use manual selection or try different range."
