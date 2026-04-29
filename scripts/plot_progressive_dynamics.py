#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
try:
    from paths import get_cifar_runs_dir, get_cifar_results_dir
except ImportError:
    from scripts.paths import get_cifar_runs_dir, get_cifar_results_dir

RUNS = get_cifar_runs_dir()
OUT = get_cifar_results_dir(create=True)
OUT.mkdir(exist_ok=True)


def load_seed(seed: int):
    p = RUNS / f"progressive_noise0.200_seed{seed}" / "metrics.csv"
    if not p.exists():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=int)
    val_acc = np.array([float(r["val_acc"]) for r in rows], dtype=float)
    val_kl = np.array([float(r["val_kl"]) for r in rows], dtype=float) if "val_kl" in rows[0] else None
    sparsity = np.array([float(r["actual_sparsity"]) for r in rows], dtype=float) if "actual_sparsity" in rows[0] else None
    return {"epochs": epochs, "val_acc": val_acc, "val_kl": val_kl, "sparsity": sparsity}


def main():
    all_data = []
    for seed in range(10):
        d = load_seed(seed)
        if d is not None:
            all_data.append((seed, d))

    if not all_data:
        raise RuntimeError("No progressive run data found.")

    # Seed-0 figure for quick inspection
    seed0 = dict(all_data).get(0, all_data[0][1])
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(seed0["epochs"], seed0["val_acc"], marker="o", linewidth=2, label="Val Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.grid(alpha=0.3)
    ax1.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Prune step")
    ax1.axvline(x=6, color="red", linestyle="--", alpha=0.7)

    if seed0["sparsity"] is not None:
        ax2 = ax1.twinx()
        ax2.plot(seed0["epochs"], seed0["sparsity"], color="purple", marker="x", linewidth=1.8, label="Actual Sparsity")
        ax2.set_ylabel("Sparsity")

    ax1.set_title("Progressive Dynamics (seed 0)")
    fig.tight_layout()
    fig.savefig(OUT / "progressive_dynamics_seed0.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Mean ± std across seeds
    epochs = all_data[0][1]["epochs"]
    val_acc_mat = np.stack([d["val_acc"] for _, d in all_data], axis=0)
    mean_acc = val_acc_mat.mean(axis=0)
    std_acc = val_acc_mat.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, mean_acc, linewidth=2, label="Mean Val Acc")
    ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, label="±1 std")
    ax.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Prune step")
    ax.axvline(x=6, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Progressive Dynamics (10 seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "progressive_dynamics_mean_std.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Fidelity dynamics if available
    if all_data[0][1]["val_kl"] is not None:
        kl_mat = np.stack([d["val_kl"] for _, d in all_data], axis=0)
        mean_kl = kl_mat.mean(axis=0)
        std_kl = kl_mat.std(axis=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, mean_kl, linewidth=2, label="Mean Val KL")
        ax.fill_between(epochs, mean_kl - std_kl, mean_kl + std_kl, alpha=0.2, label="±1 std")
        ax.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Prune step")
        ax.axvline(x=6, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("KL(teacher || student)")
        ax.set_title("Progressive Fidelity Dynamics (10 seeds)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "progressive_fidelity_dynamics_kl.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("Saved:")
    print(OUT / "progressive_dynamics_seed0.png")
    print(OUT / "progressive_dynamics_mean_std.png")
    if all_data[0][1]["val_kl"] is not None:
        print(OUT / "progressive_fidelity_dynamics_kl.png")


if __name__ == "__main__":
    main()
