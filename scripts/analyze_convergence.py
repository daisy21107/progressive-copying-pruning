#!/usr/bin/env python3
"""Analyze convergence study results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    conv_dir = Path("runs/convergence")

    if not conv_dir.exists():
        print(f"Error: {conv_dir} not found!")
        return

    results = {}
    for path in sorted(conv_dir.glob("progressive_e*")):
        try:
            epochs = int(path.name.split("_e")[1])
        except (IndexError, ValueError):
            continue

        metrics_file = path / "metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            results[epochs] = df

    if not results:
        print("No results found!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for epochs in sorted(results.keys()):
        df = results[epochs]
        ax1.plot(df["epoch"], df["val_acc"] * 100, marker="o", label=f"{epochs}e", linewidth=2)

    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax1.set_title("Convergence Study", fontweight="bold", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    epochs_list = sorted(results.keys())
    final_accs = [results[e].iloc[-1]["val_acc"] * 100 for e in epochs_list]

    ax2.plot(epochs_list, final_accs, "o-", linewidth=3, markersize=10)
    ax2.set_xlabel("Training Epochs", fontweight="bold")
    ax2.set_ylabel("Final Accuracy (%)", fontweight="bold")
    ax2.set_title("Final Performance vs Training Length", fontweight="bold", fontsize=14)
    ax2.grid(True, alpha=0.3)

    for x, y in zip(epochs_list, final_accs):
        ax2.text(x, y + 0.1, f"{y:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("convergence_results.png", dpi=200, bbox_inches="tight")
    print("✓ Saved: convergence_results.png")
    plt.show()

    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)

    for epochs in sorted(results.keys()):
        df = results[epochs]
        final = df.iloc[-1]["val_acc"] * 100

        if len(df) >= 5:
            last5 = df.iloc[-5:]["val_acc"].values
            improvement = (last5[-1] - last5[0]) * 100

            status = "✓ Plateaued" if improvement < 0.1 else "⚡ Slowing" if improvement < 0.5 else "⚠️ Still climbing"

            print(f"{epochs:2d} epochs: {final:.2f}% ({improvement:+.2f}% last 5) {status}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    for epochs in sorted(results.keys()):
        df = results[epochs]
        if len(df) >= 5:
            last5 = df.iloc[-5:]["val_acc"].values
            improvement = (last5[-1] - last5[0]) * 100
            if improvement < 0.5:
                print(f"✓ Use {epochs} epochs (performance plateaued)")
                break
    else:
        print("⚠️ Consider testing longer (still improving at max epochs)")
    print("=" * 70)


if __name__ == "__main__":
    main()
