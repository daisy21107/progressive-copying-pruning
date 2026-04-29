#!/bin/bash
# Quick script to check calibration progress

echo "========================================="
echo "CALIBRATION PROGRESS CHECK"
echo "========================================="
echo ""

for noise in 0.52 0.53 0.54 0.55 0.58; do
    count=$(ls -d runs/scratch_seed* 2>/dev/null | grep -c "seed[0-4]$" || echo 0)
    echo "noise_std=$noise: $count/5 runs completed"
    
    # Show last metrics if available
    if [ -f "runs/scratch_seed4/metrics.csv" ]; then
        last_acc=$(tail -1 runs/scratch_seed4/metrics.csv | cut -d',' -f3)
        echo "  Latest val_acc: $last_acc"
    fi
done

echo ""
echo "Total directories: $(ls -d runs/scratch_seed* 2>/dev/null | wc -l)"
echo ""
echo "To see full results when done:"
echo "  for noise in 0.52 0.53 0.54 0.55 0.58; do"
echo "    python -m src.utils.summarize_seeds --pattern \"runs/scratch_seed*/metrics.csv\""
echo "  done"
