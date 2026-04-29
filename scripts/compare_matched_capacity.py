#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_progressive_metrics(run_dir: Path) -> Dict[str, float]:
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {metrics_path}")
    last = rows[-1]
    return {
        "val_acc": float(last["val_acc"]),
        "val_kl": float(last["val_kl"]),
        "val_logit_mse": float(last["val_logit_mse"]),
        "val_prob_mse": float(last["val_prob_mse"]),
        "val_agree": float(last["val_agree"]),
        "actual_sparsity": float(last["actual_sparsity"]),
        "target_sparsity": float(last["target_sparsity"]),
    }


def load_pruned_metrics(run_dir: Path) -> Dict[str, float]:
    metrics_path = run_dir / "pruned_metrics.json"
    with metrics_path.open("r") as handle:
        payload = json.load(handle)
    post = payload["post_prune"]
    return {
        "val_acc": float(post["val_acc"]),
        "val_kl": float(post.get("val_kl", np.nan)),
        "val_logit_mse": float(post.get("val_logit_mse", np.nan)),
        "val_prob_mse": float(post.get("val_prob_mse", np.nan)),
        "val_agree": float(post.get("val_agree", np.nan)),
        "actual_sparsity": float(post["actual_sparsity"]),
        "target_sparsity": float(payload["target_sparsity"]),
    }


def summarize_group(values: List[Dict[str, float]]) -> Dict[str, Dict[str, float] | List[Dict[str, float]]]:
    metric_names = [
        "val_acc",
        "val_kl",
        "val_logit_mse",
        "val_prob_mse",
        "val_agree",
        "actual_sparsity",
        "target_sparsity",
    ]
    summary: Dict[str, Dict[str, float] | List[Dict[str, float]]] = {"per_seed": values}
    for name in metric_names:
        data = np.array([row[name] for row in values], dtype=float)
        summary[name] = {
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
        }
    return summary


def collect_progressive(base_dir: Path, prefix: str, seeds: List[int]) -> List[Dict[str, float]]:
    rows = []
    for seed in seeds:
        run_dir = base_dir / f"{prefix}{seed}"
        row = load_progressive_metrics(run_dir)
        row["seed"] = seed
        rows.append(row)
    return rows


def collect_pruned(base_dir: Path, prefix: str, seeds: List[int]) -> List[Dict[str, float]]:
    rows = []
    for seed in seeds:
        run_dir = base_dir / f"{prefix}{seed}"
        row = load_pruned_metrics(run_dir)
        row["seed"] = seed
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched-capacity results across progressive, one-shot, and scratch runs.")
    parser.add_argument("--progressive-base", required=True, help="Directory containing progressive seed runs.")
    parser.add_argument("--progressive-prefix", default="progressive_noise0.200_seed", help="Seeded progressive run prefix.")
    parser.add_argument("--oneshot-base", required=True, help="Directory containing pruned one-shot seed runs.")
    parser.add_argument("--oneshot-prefix", default="oneshot_pruned_seed", help="Seeded pruned one-shot run prefix.")
    parser.add_argument("--scratch-base", required=True, help="Directory containing pruned scratch seed runs.")
    parser.add_argument("--scratch-prefix", default="scratch_pruned_seed", help="Seeded pruned scratch run prefix.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)), help="Seeds to aggregate.")
    parser.add_argument("--out", required=True, help="Path to output JSON summary.")
    args = parser.parse_args()

    progressive_rows = collect_progressive(Path(args.progressive_base), args.progressive_prefix, args.seeds)
    oneshot_rows = collect_pruned(Path(args.oneshot_base), args.oneshot_prefix, args.seeds)
    scratch_rows = collect_pruned(Path(args.scratch_base), args.scratch_prefix, args.seeds)

    results = {
        "progressive": summarize_group(progressive_rows),
        "oneshot_pruned": summarize_group(oneshot_rows),
        "scratch_pruned": summarize_group(scratch_rows),
        "comparisons": {
            "progressive_minus_oneshot_acc": float(
                np.nanmean([row["val_acc"] for row in progressive_rows]) - np.nanmean([row["val_acc"] for row in oneshot_rows])
            ),
            "progressive_minus_scratch_acc": float(
                np.nanmean([row["val_acc"] for row in progressive_rows]) - np.nanmean([row["val_acc"] for row in scratch_rows])
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()