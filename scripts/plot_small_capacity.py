#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    in_path = Path("results/small_capacity_sweep.json")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run aggregate_small_capacity.py first.")

    with in_path.open("r") as handle:
        results = json.load(handle)

    capacities = [3, 5, 10, 15, 25]

    def series(regime):
        means = []
        stds = []
        for cap in capacities:
            entry = results.get(str(cap), {}).get(regime)
            if entry is None:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(entry["mean"] * 100.0)
                stds.append(entry["std"] * 100.0)
        return np.array(means), np.array(stds)

    prog_means, prog_stds = series("progressive")
    one_means, one_stds = series("oneshot")
    scr_means, scr_stds = series("scratch")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.errorbar(capacities, prog_means, yerr=prog_stds, marker="o", linewidth=2.2, capsize=4, label="Progressive")
    ax1.errorbar(capacities, one_means, yerr=one_stds, marker="s", linewidth=2.2, capsize=4, label="One-shot")
    ax1.errorbar(capacities, scr_means, yerr=scr_stds, marker="^", linewidth=2.2, capsize=4, label="Scratch")
    ax1.set_xlabel("Model Capacity (%)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Small-Capacity Accuracy")
    ax1.set_xticks(capacities)
    ax1.grid(alpha=0.3)
    ax1.legend()

    gaps = prog_means - one_means
    colors = ["#2ca02c" if g > 0 else "#d62728" for g in gaps]
    bars = ax2.bar([str(c) for c in capacities], gaps, color=colors, alpha=0.85)
    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.set_xlabel("Model Capacity (%)")
    ax2.set_ylabel("Progressive - One-shot (points)")
    ax2.set_title("Progressive Advantage")
    ax2.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, gaps):
        if np.isnan(value):
            continue
        y = value + 0.2 if value >= 0 else value - 0.2
        va = "bottom" if value >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2.0, y, f"{value:+.1f}", ha="center", va=va, fontsize=9)

    out_path = Path("docs/meeting_prep/small_capacity_results.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()