#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
BASE = Path("runs")


def read_final_val_acc(csv_path: Path):
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["val_acc"])


def main():
    results = []
    print("\n=== CIFAR-100 Noise Calibration (scratch, seed=0) ===")
    for noise in NOISE_LEVELS:
        tag = f"noise{noise:.3f}"
        run_dir = BASE / f"scratch_{tag}_seed0"
        val_acc = read_final_val_acc(run_dir / "metrics.csv")
        status = "done" if val_acc is not None else "missing"
        results.append({
            "noise_std": noise,
            "run_dir": str(run_dir),
            "status": status,
            "val_acc": val_acc,
        })
        if val_acc is None:
            print(f"noise={noise:.1f} -> MISSING")
        else:
            print(f"noise={noise:.1f} -> val_acc={val_acc:.4f}")

    done = [r for r in results if r["val_acc"] is not None]
    print("\n=== Recommendation ===")
    target = [r for r in done if 0.70 <= r["val_acc"] <= 0.90]
    if target:
        best = sorted(target, key=lambda r: abs(r["val_acc"] - 0.80))[0]
        print(
            f"Found target-range setting: noise_std={best['noise_std']:.1f} "
            f"(val_acc={best['val_acc']:.4f})"
        )
    elif done:
        closest = sorted(done, key=lambda r: abs(r["val_acc"] - 0.80))[0]
        print(
            f"No exact 70-90% hit yet; closest is noise_std={closest['noise_std']:.1f} "
            f"(val_acc={closest['val_acc']:.4f})"
        )
    else:
        print("No completed runs yet.")

    out_json = Path("results_cifar_noise_sweep_seed0.json")
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to: {out_json}")


if __name__ == "__main__":
    main()
