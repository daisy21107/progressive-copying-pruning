#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

CAPACITIES=(3 5 10 15 25)
SEEDS=(0 1 2 3 4)

echo "=========================================="
echo "SMALL CAPACITY SWEEP"
echo "Capacities: ${CAPACITIES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="

cd "$ROOT_DIR"
mkdir -p logs

for cap in "${CAPACITIES[@]}"; do
  echo "\n=== ${cap}% capacity ==="

  echo "[1/3] progressive"
  for seed in "${SEEDS[@]}"; do
    "$PYTHON" -m src.main progressive \
      --config "configs/small_capacity_sweep/progressive_${cap}pct.yaml" \
      --seed "$seed" \
      --out-dir "runs/progressive_${cap}pct_seed${seed}"
  done

  echo "[2/3] oneshot"
  for seed in "${SEEDS[@]}"; do
    "$PYTHON" -m src.main oneshot \
      --config "configs/small_capacity_sweep/oneshot_${cap}pct.yaml" \
      --seed "$seed" \
      --out-dir "runs/oneshot_${cap}pct_seed${seed}"
  done

  echo "[3/3] scratch"
  for seed in "${SEEDS[@]}"; do
    "$PYTHON" -m src.main scratch \
      --config "configs/small_capacity_sweep/scratch_${cap}pct.yaml" \
      --seed "$seed" \
      --out-dir "runs/scratch_${cap}pct_seed${seed}"
  done

  echo "Completed ${cap}% capacity"
done

echo "\nAll runs complete."
echo "Next: $PYTHON scripts/aggregate_small_capacity.py"