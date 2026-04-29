"""Analyze results and create figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_bars(df, ax, colors):
    grouped = df.groupby("method_label")
    methods = list(colors.keys())
    means = [grouped.get_group(m)["final_accuracy"].mean() for m in methods]
    stds = [grouped.get_group(m)["final_accuracy"].std() for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("(A) Final Performance", fontsize=12, fontweight="bold")
    ax.set_ylim([70, 80])
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_cka_bars(df, ax, colors):
    grouped = df.groupby("method_label")
    methods = list(colors.keys())
    means = [grouped.get_group(m)["avg_cka"].mean() for m in methods]
    stds = [grouped.get_group(m)["avg_cka"].std() for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=10)
    ax.set_ylabel("CKA Similarity", fontsize=12, fontweight="bold")
    ax.set_title("(B) Representation Preservation", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_accuracy_vs_cka(df, ax, colors):
    for method in colors.keys():
        data = df[df["method_label"] == method]
        ax.scatter(
            data["avg_cka"],
            data["final_accuracy"],
            s=150,
            alpha=0.6,
            color=colors[method],
            label=method,
            edgecolors="black",
            linewidth=1.5,
        )

        mean_cka = data["avg_cka"].mean()
        mean_acc = data["final_accuracy"].mean()
        ax.scatter(
            mean_cka,
            mean_acc,
            s=300,
            marker="*",
            color=colors[method],
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )

    ax.set_xlabel("CKA Similarity to Teacher", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("(C) Performance vs Preservation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)


def create_publication_figure(df):
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    colors = {
        "Dense (0% sparse)": "#2E86AB",
        "Progressive (90% sparse)": "#A23B72",
        "One-shot (90% sparse)": "#F18F01",
    }

    ax1 = fig.add_subplot(gs[0, 0])
    plot_accuracy_bars(df, ax1, colors)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_cka_bars(df, ax2, colors)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_accuracy_vs_cka(df, ax3, colors)

    plt.suptitle(
        "Training Trajectory Affects Teacher Replication Quality",
        fontsize=16,
        fontweight="bold",
    )

    output_dir = Path("docs/rq1_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "publication_figure.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    results_dir = Path("results/rq1")

    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        print("Download results from Colab/Drive into results/rq1 first")
        return

    results = []
    methods = {
        "dense": "Dense (0% sparse)",
        "progressive": "Progressive (90% sparse)",
        "oneshot_posthoc": "One-shot (90% sparse)",
    }

    for method_key, method_label in methods.items():
        for seed in [0, 1, 2]:
            run_dir = results_dir / f"runs/rq1_{method_key}_seed{seed}"
            if not run_dir.exists():
                print(f"Warning: {run_dir} not found, skipping")
                continue

            metrics = pd.read_csv(run_dir / "metrics.csv")
            cka_file = run_dir / "cka_scores.json"

            avg_cka = None
            if cka_file.exists():
                with open(cka_file, "r") as f:
                    cka = json.load(f)
                avg_cka = cka.get("average", None)

            results.append(
                {
                    "method": method_key,
                    "method_label": method_label,
                    "seed": seed,
                    "final_accuracy": float(metrics.iloc[-1]["val_acc"]) * 100.0,
                    "avg_cka": avg_cka,
                }
            )

    df = pd.DataFrame(results)
    if df.empty:
        print("No RQ1 results found.")
        return

    print("\n" + "=" * 70)
    print("RQ1 RESULTS SUMMARY")
    print("=" * 70)

    summary = df.groupby("method_label").agg(
        {
            "final_accuracy": ["mean", "std"],
            "avg_cka": ["mean", "std"],
        }
    )
    print(summary)

    create_publication_figure(df)

    print("\nAnalysis complete")
    print("Figure saved to docs/rq1_results/publication_figure.png")


if __name__ == "__main__":
    main()
