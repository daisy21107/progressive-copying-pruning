#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "CONVERGENCE STUDY"
echo "Testing epochs: 20, 30, 40, 50, 60"
echo ""

for epochs in 20 30 40 50 60; do
    echo "=== Testing ${epochs} epochs ==="

    .venv/bin/python -m src.main progressive \
        --config configs/rq1/progressive_90pct.yaml \
        --epochs "${epochs}" \
        --seed 0 \
        --out-dir "runs/convergence/progressive_e${epochs}"

    echo "✓ ${epochs} epochs done"
    echo ""
done

echo "ALL CONVERGENCE TESTS COMPLETE"
echo "Run: .venv/bin/python scripts/analyze_convergence.py"
