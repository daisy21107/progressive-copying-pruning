#!/usr/bin/env python3
"""
Analyze RQ1 Results: Progressive vs One-Shot vs Dense

Generates:
- Summary statistics (mean ± std)
- Comparison plots
- Statistical tests
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def load_run_metrics(run_dir: Path) -> Dict:
    """Load metrics from a single run directory."""
    metrics_file = run_dir / "metrics.csv"
    if not metrics_file.exists():
        return None
    
    df = pd.read_csv(metrics_file)
    
    # Extract final metrics
    final_row = df.iloc[-1]
    
    return {
        "seed": int(run_dir.name.split("_seed")[-1]),
        "val_acc": float(final_row.get("val_acc", np.nan)),
        "val_kl": float(final_row.get("val_kl", np.nan)),
        "val_agree": float(final_row.get("val_agree", np.nan)),
        "epochs": len(df),
        "metrics_df": df,  # Keep full dataframe for plotting
    }


def aggregate_results(runs_dir: Path, methods: List[str]) -> Dict[str, List[Dict]]:
    """Aggregate metrics across all runs."""
    results = {method: [] for method in methods}
    
    for method in methods:
        for seed in range(10):
            run_dir = runs_dir / f"{method}_seed{seed}"
            if run_dir.exists():
                metrics = load_run_metrics(run_dir)
                if metrics is not None:
                    results[method].append(metrics)
    
    return results


def compute_statistics(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Compute summary statistics."""
    stats = []
    
    for method in sorted(results.keys()):
        data = results[method]
        
        if not data:
            continue
        
        accs = [d["val_acc"] for d in data]
        kls = [d["val_kl"] for d in data if not np.isnan(d["val_kl"])]
        agrees = [d["val_agree"] for d in data if not np.isnan(d["val_agree"])]
        
        stats.append({
            "method": method,
            "n_runs": len(data),
            "acc_mean": np.mean(accs) if accs else np.nan,
            "acc_std": np.std(accs) if accs else np.nan,
            "acc_min": np.min(accs) if accs else np.nan,
            "acc_max": np.max(accs) if accs else np.nan,
            "kl_mean": np.mean(kls) if kls else np.nan,
            "kl_std": np.std(kls) if kls else np.nan,
            "agree_mean": np.mean(agrees) if agrees else np.nan,
            "agree_std": np.std(agrees) if agrees else np.nan,
        })
    
    return pd.DataFrame(stats)


def print_summary_table(stats_df: pd.DataFrame):
    """Print nicely formatted summary table."""
    print("\n" + "=" * 90)
    print("RQ1 RESULTS SUMMARY")
    print("=" * 90)
    print()
    print(f"{'Method':<15} | {'Accuracy (%)':<20} | {'KL Div.':<15} | {'Agreement':<12}")
    print("-" * 90)
    
    for _, row in stats_df.iterrows():
        acc_str = f"{row['acc_mean']:.2f} ± {row['acc_std']:.2f}" if not np.isnan(row['acc_mean']) else "N/A"
        kl_str = f"{row['kl_mean']:.4f} ± {row['kl_std']:.4f}" if not np.isnan(row['kl_mean']) else "N/A"
        agree_str = f"{row['agree_mean']:.4f} ± {row['agree_std']:.4f}" if not np.isnan(row['agree_mean']) else "N/A"
        
        print(f"{row['method']:<15} | {acc_str:<20} | {kl_str:<15} | {agree_str:<12}")
    
    print("=" * 90)
    print()


def compare_methods(results: Dict[str, List[Dict]]):
    """Perform pairwise comparisons."""
    print("PAIRWISE COMPARISONS (paired t-test)")
    print("-" * 60)
    
    methods = sorted(results.keys())
    
    try:
        from scipy import stats as sp_stats
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                accs1 = [d["val_acc"] for d in results[method1]]
                accs2 = [d["val_acc"] for d in results[method2]]
                
                if len(accs1) == len(accs2):
                    t_stat, p_val = sp_stats.ttest_rel(accs1, accs2)
                    diff = np.mean(accs1) - np.mean(accs2)
                    significance = "* p<0.05" if p_val < 0.05 else ""
                    
                    print(f"{method1:12s} vs {method2:12s}: "
                          f"Δ={diff:+.2f}% (t={t_stat:6.3f}, p={p_val:.4f}) {significance}")
    
    except ImportError:
        print("(scipy not available for statistical tests)")
    
    print()


def save_results(results: Dict[str, List[Dict]], output_dir: Path):
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed = {}
    for method, data in results.items():
        detailed[method] = [{k: v for k, v in d.items() if k != "metrics_df"} 
                           for d in data]
    
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(detailed, f, indent=2)
    
    # Save summary CSV
    all_rows = []
    for method, data in results.items():
        for d in data:
            all_rows.append({
                "method": method,
                "seed": d["seed"],
                "val_acc": d["val_acc"],
                "val_kl": d["val_kl"],
                "val_agree": d["val_agree"],
            })
    
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(output_dir / "all_results.csv", index=False)
    
    print(f"✓ Saved detailed results to {output_dir}/detailed_results.json")
    print(f"✓ Saved summary CSV to {output_dir}/all_results.csv")


def plot_results(results: Dict[str, List[Dict]], output_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping plots)")
        return
    
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Final accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = sorted(results.keys())
    accs_by_method = {m: [d["val_acc"] for d in results[m]] for m in methods}
    
    positions = range(len(methods))
    boxes = [accs_by_method[m] for m in methods]
    bp = ax.boxplot(boxes, labels=methods, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel("Final Accuracy (%)", fontsize=12)
    ax.set_title("RQ1: Final Accuracy Comparison", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy_comparison.png", dpi=150)
    print(f"✓ Saved plot: {plot_dir}/accuracy_comparison.png")
    plt.close()
    
    # Plot 2: KL Divergence dynamics (if available)
    if any(d.get("metrics_df") is not None for d in sum(results.values(), [])):
        fig, axes = plt.subplots(1, len(methods), figsize=(14, 4))
        
        for ax, method in zip(axes if len(methods) > 1 else [axes], methods):
            for data in results[method]:
                df = data.get("metrics_df")
                if df is not None and "val_kl" in df.columns:
                    ax.plot(df["epoch"], df["val_kl"], alpha=0.5, label=f"seed {data['seed']}")
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel("KL Divergence")
            ax.set_title(f"{method.capitalize()}")
            ax.grid(alpha=0.3)
        
        plt.suptitle("KL Divergence Dynamics", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "kl_dynamics.png", dpi=150)
        print(f"✓ Saved plot: {plot_dir}/kl_dynamics.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RQ1 results"
    )
    parser.add_argument(
        "--runs-dir", 
        type=Path, 
        default="runs",
        help="Directory containing experiment runs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results/rq1_summary",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dense", "progressive", "oneshot"],
        help="Methods to analyze"
    )
    
    args = parser.parse_args()
    
    # Load and aggregate results
    print(f"Loading results from {args.runs_dir}...")
    results = aggregate_results(args.runs_dir, args.methods)
    
    for method, data in results.items():
        print(f"  {method}: {len(data)} runs")
    
    # Compute statistics
    stats_df = compute_statistics(results)
    
    # Print summary
    print_summary_table(stats_df)
    
    # Comparisons
    compare_methods(results)
    
    # Save results
    save_results(results, args.output_dir)
    
    # Generate plots
    plot_results(results, args.output_dir)
    
    print(f"\n✓ Analysis complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
