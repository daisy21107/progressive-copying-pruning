#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np


CAPACITIES = [3, 5, 10, 15, 25]
SEEDS = [0, 1, 2, 3, 4]


def read_last_val_acc(path: Path):
    if not path.exists():
        return None
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return float(rows[-1]["val_acc"])


def summarize(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    results = {}

    for cap in CAPACITIES:
        cap_key = str(cap)
        results[cap_key] = {}
        for regime in ["progressive", "oneshot", "scratch"]:
            vals = []
            for seed in SEEDS:
                path = Path(f"runs/{regime}_{cap}pct_seed{seed}/metrics.csv")
                value = read_last_val_acc(path)
                if value is not None:
                    vals.append(value)
            if vals:
                results[cap_key][regime] = summarize(vals)

    print("\n" + "=" * 84)
    print(f"{'Capacity':<10} {'Progressive':<24} {'One-shot':<24} {'Scratch':<24}")
    print("=" * 84)

    for cap in CAPACITIES:
        key = str(cap)
        row = results.get(key, {})

        def fmt(regime):
            s = row.get(regime)
            if s is None:
                return "N/A"
            return f"{s['mean']*100:.2f}% +- {s['std']*100:.2f}%"

        print(f"{cap:>3}%       {fmt('progressive'):<24} {fmt('oneshot'):<24} {fmt('scratch'):<24}")

    print("=" * 84)
    print("\nProgressive advantage vs one-shot:")
    for cap in CAPACITIES:
        key = str(cap)
        row = results.get(key, {})
        if "progressive" in row and "oneshot" in row:
            gap = (row["progressive"]["mean"] - row["oneshot"]["mean"]) * 100.0
            print(f"{cap:>3}% capacity: {gap:+.2f} points")

    out_path = Path("results/small_capacity_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()