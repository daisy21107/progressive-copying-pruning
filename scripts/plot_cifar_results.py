#!/usr/bin/env python3
"""Plot CIFAR-100 experimental results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
try:
    from paths import get_cifar_results_dir
except ImportError:
    from scripts.paths import get_cifar_results_dir

RESULTS_DIR = get_cifar_results_dir(create=True)

def main():
    results_file = RESULTS_DIR / 'cifar_final_results.json'
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Wait for experiment to complete first!")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Create comparison plot
    regimes = ['oneshot', 'progressive', 'scratch']
    
    # Check if we have results for all regimes
    available = [r for r in regimes if r in results and 'val_acc' in results[r]]
    
    if not available:
        print("No results found yet")
        return
    
    means = [results[r]['val_acc']['mean'] * 100 for r in available]
    stds = [results[r]['val_acc']['std'] * 100 for r in available]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(available))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', ' ').title() for r in available])
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-100 Distillation Comparison (noise_std=0.2, 10 seeds each)', fontsize=14, fontweight='bold')
    ax.axhline(y=74, color='red', linestyle='--', alpha=0.5, label='Target (74%)')
    ax.set_ylim([70, 76])
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.1f}%\n±{std:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = RESULTS_DIR / 'cifar_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {comparison_path}")
    
    # Create detailed comparison if we have progressive and oneshot
    if 'progressive' in results and 'oneshot' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Distribution of accuracies
        oneshot_vals = results['oneshot']['val_acc']['values']
        prog_vals = results['progressive']['val_acc']['values']
        
        ax1.boxplot([oneshot_vals, prog_vals], tick_labels=['One-shot', 'Progressive'])
        ax1.set_ylabel('Validation Accuracy', fontsize=12)
        ax1.set_title('Distribution Across Seeds', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Violin plot comparison
        positions = [1, 2]
        parts = ax2.violinplot([oneshot_vals, prog_vals], positions=positions, showmeans=True)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(['One-shot', 'Progressive'])
        ax2.set_ylabel('Validation Accuracy', fontsize=12)
        ax2.set_title('Accuracy Distribution (Violin Plot)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        detailed_path = RESULTS_DIR / 'cifar_comparison_detailed.png'
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {detailed_path}")
    
    print("\nPlots created successfully!")

if __name__ == '__main__':
    main()
