#!/usr/bin/env python3
"""Monitor CIFAR experiment progress."""

import os
from pathlib import Path
import time

def count_completed_runs():
    """Count completed training runs."""
    regimes = ['oneshot', 'progressive', 'scratch']
    completed = 0
    total = 0
    
    for regime in regimes:
        for seed in range(10):
            total += 1
            run_dir = Path(f"runs/{regime}_noise0.200_seed{seed}")
            metrics_file = run_dir / "metrics.csv"
            if metrics_file.exists():
                # Check if has 5 epochs (final run)
                with open(metrics_file) as f:
                    lines = len(f.readlines()) - 1  # Minus header
                if lines >= 5:
                    completed += 1
    
    return completed, total

def check_teacher():
    """Check if teacher completed."""
    teacher_dir = Path("runs/cifar_teacher_final")
    metrics_file = teacher_dir / "metrics.csv"
    if metrics_file.exists() and metrics_file.stat().st_size > 100:
        return True
    return False

def main():
    print("=" * 60)
    print("CIFAR-100 EXPERIMENT PROGRESS")
    print("=" * 60)
    print()
    
    teacher_done = check_teacher()
    completed, total = count_completed_runs()
    
    print(f"Teacher:      {'✓ DONE' if teacher_done else '⏳ Training...'}")
    print(f"Student runs: {completed}/{total} completed")
    print()
    
    if teacher_done and completed < total:
        eta_mins = (total - completed) * 5
        print(f"Estimated remaining time: ~{eta_mins} minutes")
    elif completed == total:
        print("✓ ALL EXPERIMENTS COMPLETED!")
        print()
        print("Next step: python scripts/aggregate_cifar_results.py")
    elif not teacher_done:
        print("Teacher still training... (~5 minutes)")
    
    print()
    print("Monitor live:")
    print("  tail -f runs/cifar_full_experiment.log")
    
    print()
    print("Last check: $(date '+%H:%M:%S')")

if __name__ == '__main__':
    main()
