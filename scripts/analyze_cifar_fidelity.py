#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import numpy as np
try:
    from paths import get_cifar_runs_dir, get_cifar_results_dir
except ImportError:
    from scripts.paths import get_cifar_runs_dir, get_cifar_results_dir

RUNS = get_cifar_runs_dir()
RESULTS = get_cifar_results_dir(create=True)
RESULTS.mkdir(exist_ok=True)


def load_final_row(regime: str, seed: int):
    metrics_path = RUNS / f"{regime}_noise0.200_seed{seed}" / "metrics.csv"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def summarize_regime(regime: str, metric_names):
    rows = [load_final_row(regime, seed) for seed in range(10)]
    rows = [r for r in rows if r is not None]
    summary = {"n": len(rows), "metrics": {}}

    for metric in metric_names:
        vals = []
        for r in rows:
            if metric in r and r[metric] not in (None, ""):
                try:
                    vals.append(float(r[metric]))
                except ValueError:
                    pass
        if vals:
            summary["metrics"][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary


def main():
    metrics = ["val_acc", "val_kl", "val_logit_mse", "val_prob_mse", "val_agree"]

    oneshot = summarize_regime("oneshot", metrics)
    progressive = summarize_regime("progressive", metrics)
    scratch = summarize_regime("scratch", ["val_acc"])

    out = {
        "oneshot": oneshot,
        "progressive": progressive,
        "scratch": scratch,
    }

    (RESULTS / "cifar_fidelity_summary.json").write_text(json.dumps(out, indent=2))

    print("=== CIFAR Fidelity Summary (final epoch per run) ===")
    for regime, data in out.items():
        print(f"\n[{regime}] seeds={data['n']}")
        for metric, stats in data["metrics"].items():
            print(
                f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f} "
                f"(range {stats['min']:.6f} .. {stats['max']:.6f})"
            )

    if oneshot["metrics"] and progressive["metrics"]:
        print("\n=== Progressive - Oneshot (mean deltas) ===")
        shared = sorted(set(oneshot["metrics"]).intersection(progressive["metrics"]))
        for metric in shared:
            delta = progressive["metrics"][metric]["mean"] - oneshot["metrics"][metric]["mean"]
            print(f"  {metric}: {delta:+.6f}")


if __name__ == "__main__":
    main()
