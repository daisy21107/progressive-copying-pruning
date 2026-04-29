#!/usr/bin/env python3
"""Aggregate CIFAR-100 experimental results across seeds."""

import csv
import json
import os
from pathlib import Path
import numpy as np
try:
    from paths import get_cifar_runs_dir, get_cifar_results_dir
except ImportError:
    from scripts.paths import get_cifar_runs_dir, get_cifar_results_dir

RUNS_DIR = get_cifar_runs_dir()
RESULTS_DIR = get_cifar_results_dir(create=True)

def load_final_metrics(run_dir):
    """Load final epoch metrics from a run."""
    metrics_file = run_dir / "metrics.csv"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return None
    
    # Get last epoch
    final = rows[-1]
    return {
        'val_acc': float(final['val_acc']),
        'val_kl': float(final.get('val_kl', 0)) if final.get('val_kl') else 0,
        'val_agree': float(final.get('val_agree', 0)) if final.get('val_agree') else 0,
    }

def find_teacher_result():
    """Find teacher result from cifar_teacher_final or other locations."""
    possible_paths = [
        RUNS_DIR / "cifar_teacher_final" / "metrics.csv",
        RUNS_DIR / "teacher_cifar100_seed42" / "metrics.csv",
        Path("runs/cifar_teacher_final/metrics.csv"),
        Path("runs/teacher_cifar100_seed42/metrics.csv"),
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                final = rows[-1]
                return float(final['val_acc'])
    
    return None

def aggregate_regime(regime, seeds):
    """Aggregate results across seeds for one regime."""
    results = []
    
    for seed in seeds:
        run_dir = RUNS_DIR / f"{regime}_noise0.200_seed{seed}"
        metrics = load_final_metrics(run_dir)
        if metrics:
            results.append(metrics)
    
    if not results:
        return None
    
    val_accs = [r['val_acc'] for r in results]
    
    return {
        'n_seeds': len(results),
        'val_acc': {
            'mean': float(np.mean(val_accs)),
            'std': float(np.std(val_accs)),
            'min': float(np.min(val_accs)),
            'max': float(np.max(val_accs)),
            'values': [float(v) for v in val_accs],
        }
    }

def main():
    # Aggregate each regime
    regimes = {
        'teacher': [42],
        'oneshot': list(range(10)),
        'progressive': list(range(10)),
        'scratch': list(range(10)),
    }
    
    results = {}
    
    print("=" * 60)
    print("CIFAR-100 EXPERIMENTAL RESULTS")
    print("=" * 60)
    print()
    
    # Teacher
    print("TEACHER (seed 42)")
    print("-" * 40)
    teacher_acc = find_teacher_result()
    if teacher_acc:
        results['teacher'] = {'val_acc': teacher_acc}
        print(f"  Val Accuracy: {teacher_acc:.2%}")
    else:
        print(f"  Waiting for teacher result...")
    print()
    
    # Student regimes
    for regime in ['oneshot', 'progressive', 'scratch']:
        print(f"{regime.upper()}")
        print("-" * 40)
        
        agg = aggregate_regime(regime, list(range(10)))
        
        if agg:
            results[regime] = agg
            acc = agg['val_acc']
            print(f"  Seeds: {agg['n_seeds']}/10")
            print(f"  Val Accuracy: {acc['mean']:.2%} ± {acc['std']:.2%}")
            print(f"  Range: [{acc['min']:.2%}, {acc['max']:.2%}]")
        else:
            print(f"  No results found yet")
        
        print()
    
    # Compare distillation methods
    print("=" * 60)
    print("DISTILLATION COMPARISON")
    print("=" * 60)
    print()
    
    if 'oneshot' in results and 'progressive' in results:
        oneshot_mean = results['oneshot']['val_acc']['mean']
        prog_mean = results['progressive']['val_acc']['mean']
        diff = prog_mean - oneshot_mean
        
        print(f"Progressive: {prog_mean:.2%}")
        print(f"One-shot:    {oneshot_mean:.2%}")
        print(f"Difference:  {diff:+.2%}")
        print()
        
        if abs(diff) > 0.01:  # >1% difference
            winner = "Progressive" if diff > 0 else "One-shot"
            print(f"✓ {winner} outperforms (>{abs(diff):.1%})")
        else:
            print(f"≈ Similar performance (within 1%)")
    
    print()
    
    # Compare vs scratch
    if 'oneshot' in results and 'scratch' in results:
        oneshot_mean = results['oneshot']['val_acc']['mean']
        scratch_mean = results['scratch']['val_acc']['mean']
        diff = oneshot_mean - scratch_mean
        
        print(f"One-shot vs Scratch: {diff:+.2%}")
        print(f"  Interpretation: Distillation {'helps' if diff > 0 else 'hurts'}")
    
    if 'progressive' in results and 'scratch' in results:
        prog_mean = results['progressive']['val_acc']['mean']
        scratch_mean = results['scratch']['val_acc']['mean']
        diff = prog_mean - scratch_mean
        
        print(f"Progressive vs Scratch: {diff:+.2%}")
        print(f"  Interpretation: Progressive {'helps' if diff > 0 else 'hurts'}")
    
    print()
    
    # Save results to JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / 'cifar_final_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {out_path}")

if __name__ == '__main__':
    main()
