#!/usr/bin/env python3
"""
Aggregate results from full FYP experiment and compute statistics.
Usage: python scripts/aggregate_final_results.py
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def read_metrics(metrics_file: str) -> float:
    """Read final epoch validation accuracy from metrics.csv."""
    if not os.path.exists(metrics_file):
        return None
    try:
        with open(metrics_file, newline="") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                return None
            last_row = rows[-1]

            if "val_acc" in last_row and last_row["val_acc"] != "":
                return float(last_row["val_acc"])

            for fallback_key in ("accuracy", "acc", "val_accuracy"):
                if fallback_key in last_row and last_row[fallback_key] != "":
                    return float(last_row[fallback_key])

            return None
    except Exception as e:
        print(f"  ERROR reading {metrics_file}: {e}")
        return None

def aggregate_regime(regime: str, seeds: List[int], setting: str) -> Dict:
    """Aggregate results for a single regime."""
    
    # Use simple naming pattern: {regime}_seed{N}
    if regime == "teacher":
        pattern = f"runs/teacher_seed{seeds[0]}"
    else:
        pattern = f"runs/{regime}_seed"
    
    results = []
    for seed in seeds:
        if regime == "teacher":
            metrics_file = f"{pattern}/metrics.csv"
        else:
            metrics_file = f"{pattern}{seed}/metrics.csv"
        
        acc = read_metrics(metrics_file)
        if acc is not None:
            results.append((seed, acc))
            print(f"  Seed {seed:2d}: {acc:.4f}")
    
    if not results:
        print(f"  ❌ No completed runs for {regime}")
        return None
    
    accs = [a for _, a in results]
    mean = np.mean(accs)
    std = np.std(accs)
    min_acc = np.min(accs)
    max_acc = np.max(accs)
    
    return {
        "regime": regime,
        "completed": len(results),
        "total": len(seeds),
        "mean": float(mean),
        "std": float(std),
        "min": float(min_acc),
        "max": float(max_acc),
        "seeds": results,
    }

def main():
    os.chdir("/Users/jungwonbae/Desktop/fyp/fyp-progressive-copying-pruning")
    
    setting = "noise0.15_canvas34"
    
    print("\n" + "="*70)
    print("AGGREGATING FULL EXPERIMENT RESULTS")
    print("="*70)
    print(f"Setting: {setting}")
    print()
    
    results = {}
    
    # Teacher [1 seed]
    print("[Teacher] seed=42")
    results["teacher"] = aggregate_regime("teacher", [42], setting)
    print()
    
    # One-Shot [10 seeds]
    print("[One-Shot Distillation] seeds=0-9")
    results["oneshot"] = aggregate_regime("oneshot", list(range(10)), setting)
    print()
    
    # Progressive [10 seeds]
    print("[Progressive Copying + Pruning] seeds=0-9")
    results["progressive"] = aggregate_regime("progressive", list(range(10)), setting)
    print()
    
    # Scratch [10 seeds]
    print("[From-Scratch Training] seeds=0-9")
    results["scratch"] = aggregate_regime("scratch", list(range(10)), setting)
    print()
    
    # Summary table
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"{'Regime':<20} {'Completed':<12} {'Mean':<10} {'Std':<10} {'Range':<20}")
    print("-"*70)
    
    for regime in ["teacher", "oneshot", "progressive", "scratch"]:
        stats = results[regime]
        if stats:
            completed = stats["completed"]
            total = stats["total"]
            mean = stats["mean"]
            std = stats["std"]
            min_acc = stats["min"]
            max_acc = stats["max"]
            
            status = f"{completed}/{total}"
            range_str = f"[{min_acc:.4f}, {max_acc:.4f}]"
            print(f"{regime:<20} {status:<12} {mean:.4f}     {std:.4f}     {range_str:<20}")
    
    print("="*70)
    print()
    
    # Save detailed results
    results_file = f"results_{setting}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Detailed results saved to: {results_file}")
    
    # Key findings
    print()
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    if results["teacher"] and results["oneshot"]:
        teacher_acc = results["teacher"]["mean"]
        oneshot_acc = results["oneshot"]["mean"]
        gap = teacher_acc - oneshot_acc
        print(f"Teacher accuracy: {teacher_acc:.4f}")
        print(f"One-shot student accuracy: {oneshot_acc:.4f}")
        print(f"Student-Teacher gap: {gap:.4f} ({abs(gap)*100:.2f}%)")
        print()
    
    if results["progressive"] and results["scratch"]:
        progressive_acc = results["progressive"]["mean"]
        scratch_acc = results["scratch"]["mean"]
        improvement = progressive_acc - scratch_acc
        print(f"Progressive accuracy: {progressive_acc:.4f}")
        print(f"Scratch accuracy: {scratch_acc:.4f}")
        if improvement > 0:
            print(f"✅ Progressive IMPROVES over scratch: +{improvement:.4f} ({improvement*100:.2f}%)")
        else:
            print(f"❌ Progressive does NOT improve over scratch: {improvement:.4f}")
    
    print("="*70)

if __name__ == "__main__":
    main()
