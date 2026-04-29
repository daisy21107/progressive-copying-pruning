#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Testing RQ1 implementation locally (2 epochs per run)..."

mk_cfg() {
  local src="$1"
  local dst="$2"
  sed -E 's/^epochs:[[:space:]]*[0-9]+/epochs: 2/' "$src" > "$dst"
}

TMP_DENSE="$(mktemp /tmp/rq1_dense_XXXX.yaml)"
TMP_ONE="$(mktemp /tmp/rq1_oneshot_XXXX.yaml)"
TMP_PROG="$(mktemp /tmp/rq1_progressive_XXXX.yaml)"

mk_cfg "configs/rq1/dense.yaml" "$TMP_DENSE"
mk_cfg "configs/rq1/oneshot_90pct.yaml" "$TMP_ONE"
mk_cfg "configs/rq1/progressive_90pct.yaml" "$TMP_PROG"

echo "=== Testing Dense ==="
.venv/bin/python -m src.main dense \
  --config "$TMP_DENSE" \
  --seed 0 \
  --out-dir runs/test_dense_local

echo "=== Testing One-shot Post-hoc ==="
.venv/bin/python -m src.main oneshot_posthoc \
  --config "$TMP_ONE" \
  --seed 0 \
  --out-dir runs/test_oneshot_local

echo "=== Testing Progressive ==="
.venv/bin/python -m src.main progressive \
  --config "$TMP_PROG" \
  --seed 0 \
  --out-dir runs/test_progressive_local

rm -f "$TMP_DENSE" "$TMP_ONE" "$TMP_PROG"

echo "All methods tested. Check runs/test_* for outputs."
