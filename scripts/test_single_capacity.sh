#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

CAPACITY="${1:-10}"
SEED="${2:-0}"

echo "=========================================="
echo "TEST: Single Capacity Point (${CAPACITY}%)"
echo "=========================================="

cd "$ROOT_DIR"

"$PYTHON" -m src.main progressive \
  --config "configs/small_capacity_sweep/progressive_${CAPACITY}pct.yaml" \
  --seed "$SEED" \
  --out-dir "runs/test_progressive_${CAPACITY}pct_seed${SEED}"

"$PYTHON" -m src.main oneshot \
  --config "configs/small_capacity_sweep/oneshot_${CAPACITY}pct.yaml" \
  --seed "$SEED" \
  --out-dir "runs/test_oneshot_${CAPACITY}pct_seed${SEED}"

"$PYTHON" -m src.main scratch \
  --config "configs/small_capacity_sweep/scratch_${CAPACITY}pct.yaml" \
  --seed "$SEED" \
  --out-dir "runs/test_scratch_${CAPACITY}pct_seed${SEED}"

export CAPACITY_VALUE="$CAPACITY"
export SEED_VALUE="$SEED"

"$PYTHON" - <<'PYTHON'
import csv
import os
from pathlib import Path


def last_row(path: Path):
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1]


cap = os.environ["CAPACITY_VALUE"]
seed = os.environ["SEED_VALUE"]

prog = last_row(Path(f"runs/test_progressive_{cap}pct_seed{seed}/metrics.csv"))
one = last_row(Path(f"runs/test_oneshot_{cap}pct_seed{seed}/metrics.csv"))
scr = last_row(Path(f"runs/test_scratch_{cap}pct_seed{seed}/metrics.csv"))

print("\nResults:")
print(f"Progressive: {float(prog['val_acc']):.2%} (sparsity {float(prog['actual_sparsity']):.2%})")
print(f"One-shot:    {float(one['val_acc']):.2%}")
print(f"Scratch:     {float(scr['val_acc']):.2%}")
PYTHON

echo "\nDone."