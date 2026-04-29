#!/bin/bash

# run_rq1_full.sh: Execute complete RQ1 experiment (all methods, 10 seeds)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
METHODS=("dense" "progressive" "oneshot")
SEEDS=(0 1 2 3 4 5 6 7 8 9)
TEACHER_CONFIG="configs/teacher/teacher.yaml"
TEACHER_CKPT="runs/teacher/teacher.pt"
RQ1_CONFIG_DIR="configs/rq1"

# Map method names to config files
declare -A CONFIG_MAP
CONFIG_MAP["dense"]="$RQ1_CONFIG_DIR/dense.yaml"
CONFIG_MAP["progressive"]="$RQ1_CONFIG_DIR/progressive_90pct.yaml"
CONFIG_MAP["oneshot"]="$RQ1_CONFIG_DIR/oneshot_90pct.yaml"

# Helper functions
log_section() {
    echo -e "${BLUE}===================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================================${NC}"
}

log_step() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Create output directory
mkdir -p "$PROJECT_ROOT/runs"

##############################################################################
# STEP 1: Train Teacher (if not exists)
##############################################################################

log_section "STEP 1: Teacher Training"

if [ -f "$TEACHER_CKPT" ]; then
    log_warn "Teacher checkpoint already exists: $TEACHER_CKPT"
    read -p "Re-train teacher? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -m src.main teacher \
            --config "$TEACHER_CONFIG" \
            --out-dir runs/teacher \
            --seed 0
        log_step "Teacher training complete"
    fi
else
    log_step "Teacher checkpoint not found. Training..."
    python -m src.main teacher \
        --config "$TEACHER_CONFIG" \
        --out-dir runs/teacher \
        --seed 0
    log_step "Teacher training complete"
fi

# Verify teacher checkpoint exists
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
    exit 1
fi

##############################################################################
# STEP 2: Run All Methods × Seeds
##############################################################################

log_section "STEP 2: RQ1 Experiments (All Methods × Seeds)"

TOTAL_RUNS=$((${#METHODS[@]} * ${#SEEDS[@]}))
CURRENT_RUN=0

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        OUT_DIR="runs/${method}_seed${seed}"
        CONFIG="${CONFIG_MAP[$method]}"
        
        echo ""
        log_step "[$CURRENT_RUN/$TOTAL_RUNS] Running $method (seed=$seed)"
        
        python -m src.main "$method" \
            --config "$CONFIG" \
            --out-dir "$OUT_DIR" \
            --seed "$seed"
        
        # Verify run completed
        if [ ! -f "$OUT_DIR/metrics.csv" ]; then
            log_warn "Run may have failed: $OUT_DIR/metrics.csv not found"
        fi
    done
done

log_step "All experiments complete!"

##############################################################################
# STEP 3: Aggregate Results
##############################################################################

log_section "STEP 3: Aggregating Results"

mkdir -p results/rq1_summary

# Simple aggregation: extract final accuracies
python3 << 'EOF'
import os
import pandas as pd
import json
from pathlib import Path

runs_dir = Path("runs")
results = {method: [] for method in ["dense", "progressive", "oneshot"]}

for method in ["dense", "progressive", "oneshot"]:
    for seed in range(10):
        run_dir = runs_dir / f"{method}_seed{seed}"
        metrics_file = run_dir / "metrics.csv"
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            final_acc = df["val_acc"].iloc[-1] if len(df) > 0 else None
            final_kl = df["val_kl"].iloc[-1] if "val_kl" in df.columns and len(df) > 0 else None
            
            results[method].append({
                "seed": seed,
                "val_acc": final_acc,
                "val_kl": final_kl
            })

# Write summary
for method, data in results.items():
    accs = [d["val_acc"] for d in data if d["val_acc"] is not None]
    kls = [d["val_kl"] for d in data if d["val_kl"] is not None]
    
    if accs:
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((x - mean_acc)**2 for x in accs) / len(accs))**0.5
        print(f"{method:12s}: {mean_acc:6.3f} ± {std_acc:.3f} % (n={len(accs)})")
    
    if kls:
        mean_kl = sum(kls) / len(kls)
        std_kl = (sum((x - mean_kl)**2 for x in kls) / len(kls))**0.5
        print(f"              KL: {mean_kl:6.4f} ± {std_kl:.4f}")

# Save detailed results
with open("results/rq1_summary/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Aggregation complete. See results/rq1_summary/results.json")
EOF

log_step "Results aggregated"

##############################################################################
# STEP 4: Summary Table
##############################################################################

log_section "FINAL RESULTS"

echo ""
echo "Summary (run 'python scripts/analyze_rq1.py' for detailed analysis):"
echo ""

python3 << 'EOF'
import json
from pathlib import Path

results_file = Path("results/rq1_summary/results.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    print("Method        | Accuracy (%)       | KL Divergence")
    print("-" * 60)
    
    for method in ["dense", "progressive", "oneshot"]:
        if method in results and results[method]:
            accs = [d["val_acc"] for d in results[method] if d["val_acc"] is not None]
            kls = [d["val_kl"] for d in results[method] if d["val_kl"] is not None]
            
            acc_str = f"{sum(accs)/len(accs):.2f} ± {(sum((x-sum(accs)/len(accs))**2 for x in accs)/len(accs))**0.5:.2f}" if accs else "N/A"
            kl_str = f"{sum(kls)/len(kls):.4f} ± {(sum((x-sum(kls)/len(kls))**2 for x in kls)/len(kls))**0.5:.4f}" if kls else "N/A"
            
            print(f"{method:13s} | {acc_str:18s} | {kl_str}")
else:
    print("Results file not found. Run aggregation first.")
EOF

echo ""
log_step "RQ1 experiment suite complete!"

echo ""
echo "Next steps:"
echo "  1. Check results: cat results/rq1_summary/results.json"
echo "  2. Generate plots: python scripts/analyze_rq1.py"
echo "  3. View summary: head -5 results/rq1_summary/summary.csv"
