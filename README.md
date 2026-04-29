# Progressive Copying with Iterative Pruning

Empirical investigation of progressive capacity reduction during knowledge distillation.

**Main Question:** Can a student model maintain teacher performance while being gradually pruned during training, rather than all at once after training?

**Methods Compared:**
- **Progressive:** Student gradually pruned during distillation (proposed)
- **One-Shot Post-Hoc:** Student pruned after distillation completes (baseline)
- **Dense:** No pruning, standard distillation (oracle)

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download/Prepare Data

```bash
# CIFAR-100 will be auto-downloaded on first run
# No manual setup needed
```

### 3. Train Teacher (Reference Model)

```bash
python -m src.main teacher --config configs/teacher/teacher.yaml
```

This trains a teacher on ground-truth labels and outputs:
- `runs/teacher/student.pt` — trained teacher checkpoint
- `runs/teacher/metrics.csv` — training curves

### 4. Run RQ1: Method Comparison

```bash
# Run all three methods with 10 random seeds
bash scripts/run_rq1_full.sh

# Or run single method:
python -m src.main dense --config configs/rq1/dense.yaml
python -m src.main progressive --config configs/rq1/progressive_90pct.yaml
python -m src.main oneshot --config configs/rq1/oneshot_90pct.yaml
```

### 5. Analyze Results

```bash
python scripts/analyze_rq1.py
```

Outputs:
- Summary statistics (mean ± std by method)
- Plots: accuracy, KL divergence, CKA similarity
- Final results table

---

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── src/
│   ├── main.py                       # CLI entry point
│   ├── methods/                      # Training methods
│   │   ├── train_teacher.py          # Teacher training
│   │   ├── distill.py                # Dense distillation baseline
│   │   ├── progressive_copy_prune.py # Main: progressive pruning
│   │   ├── oneshot_posthoc_prune.py  # One-shot post-hoc baseline
│   │   └── pruning.py                # Magnitude-based pruning utilities
│   │
│   ├── models/
│   │   └── cnn_pair.py              # PairClassifier (2-input CNN)
│   │
│   ├── tasks/
│   │   ├── cifar100_pairs.py        # CIFAR-100 dual-input task
│   │   └── dual_mnist.py            # MNIST dual-input task (legacy)
│   │
│   ├── metrics/
│   │   ├── fidelity.py              # KL, logit MSE, agreement
│   │   ├── repsim.py                # CKA similarity
│   │   └── modularity.py            # (stub) specialization metrics
│   │
│   └── utils/
│       └── train_utils.py           # Logging, device setup
│
├── configs/
│   ├── teacher/                     # Teacher training configs
│   │   ├── teacher.yaml
│   │   └── teacher_cifar100.yaml
│   │
│   └── rq1/                         # RQ1 experiment configs
│       ├── dense.yaml
│       ├── progressive_90pct.yaml
│       └── oneshot_90pct.yaml
│
├── scripts/
│   ├── run_rq1_full.sh             # Run complete RQ1 sweep
│   ├── analyze_rq1.py              # Analysis script
│   └── ... (other analysis/utility scripts)
│
├── results/
│   ├── rq1_summary/
│   │   └── results.csv             # Final results (after analysis)
│   └── (run artifacts)
│
├── runs/                           # Experiment outputs (auto-created)
│   ├── teacher/
│   ├── oneshot_seed0/ ... seed9/
│   ├── progressive_seed0/ ... seed9/
│   └── dense_seed0/ ... seed9/
│
├── notebooks/                      # Jupyter analysis notebooks
│
├── docs/                           # Extended documentation
│   ├── METHODS.md                 # Method descriptions
│   └── REPRODUCTION.md            # Full reproduction guide
│
└── archive/                       # Old experiments (for reference)
    ├── configs/
    ├── scripts/
    └── results/
```

---

## Method Descriptions

### Dense Distillation (Baseline)

**File:** `src/methods/distill.py`

Standard knowledge distillation: student trained to match teacher logits over all epochs. No pruning.

- **Input:** Pre-trained teacher checkpoint
- **Cost:** Full capacity student throughout
- **Result:** Establishes upper bound on performance

### Progressive Copying + Pruning (Main)

**File:** `src/methods/progressive_copy_prune.py`

Student starts at full capacity and is gradually pruned at specified epochs while maintaining distillation loss.

**Key Features:**
- Gradual magnitude-based pruning at each epoch (or at milestones)
- Final target: 90% sparsity
- Student learns to maintain teacher fidelity as capacity decreases
- Outputs: KL(teacher || student), CKA similarity over time

**Configuration:**

```yaml
# Example: prune 1% per epoch toward 90% target
progressive:
  target_sparsity: 0.90           # 90% of weights zero
  prune_per_epoch: 0.01           # 1% → 90% over ~90 epochs
  fixed_target_sparsity: true     # Re-compute sparsity each epoch
  rerandomize: false              # Don't re-init pruned weights
```

### One-Shot Post-Hoc (Baseline)

**File:** `src/methods/oneshot_posthoc_prune.py`

Standard distillation followed by magnitude-based pruning.

1. Train dense student for N epochs
2. Prune to 90% at end of training
3. Fine-tune for M epochs (optional)

**Result:** Moderate performance drop vs. progressive (78.7 vs 79.5%).

---

## Configuration

All experiments controlled via YAML configs in `configs/`.

### Teacher Config Example

```yaml
# configs/teacher/teacher.yaml
seed: 0
epochs: 50
lr: 1e-3

# Dataset
task: cifar100_pairs        # or dual_mnist
noise_std: 0.0              # Gaussian noise (optional)

# Architecture
model:
  in_channels: 3            # 3 for CIFAR, 1 for MNIST
  width: 32                # CNN channels
  hidden: 128               # Dense layer dims
  shared_encoder: true      # Share weights across 2 inputs

# Checkpointing
checkpoint_name: teacher.pt
```

### RQ1 Config Example

```yaml
# configs/rq1/progressive_90pct.yaml
seed: 0
epochs: 40
lr: 1e-3
temperature: 1.0

task: cifar100_pairs

model:
  width: 32
  hidden: 128

# For distill, oneshot, progressive methods:
teacher_ckpt: "runs/teacher/teacher.pt"  # Path to trained teacher

# Progressive-specific:
progressive:
  target_sparsity: 0.90
  prune_per_epoch: 0.01
  fixed_target_sparsity: true

# Compute CKA (similarity of hidden representations)
compute_cka: true
```

---

## Running Experiments

### Single Experiment

```bash
# Train a teacher
python -m src.main teacher --config configs/teacher/teacher.yaml

# Run progressive method (single seed)
python -m src.main progressive --config configs/rq1/progressive_90pct.yaml --out-dir runs/progressive_seed0
```

**Output Directory Structure:**

```
runs/progressive_seed0/
├── config.yaml              # Snapshot of config used
├── metrics.csv              # Per-epoch metrics
├── student.pt               # Final student checkpoint
├── cka_scores.json         # CKA by layer (if compute_cka=true)
└── log.txt                 # Stdout log
```

### Batch Experiment (All Seeds)

**Provided script** automates 10 seed runs for each method:

```bash
bash scripts/run_rq1_full.sh
```

This:
1. Trains teacher (if not exists)
2. Runs all 3 methods × 10 seeds
3. Aggregates results → `results/rq1_summary/results.csv`

### Custom Sweep

Modify `scripts/run_rq1_full.sh` or create your own script:

```bash
for seed in {0..9}; do
  python -m src.main progressive \
    --config configs/rq1/progressive_90pct.yaml \
    --out-dir runs/progressive_seed$seed \
    --seed $seed
done
```

---

## Analyzing Results

### Generate Summary

```bash
python scripts/analyze_rq1.py --runs-dir runs --output results/rq1_summary
```

**Outputs:**
- `summary.csv` — mean ± std accuracy, KL, CKA by method
- `plots/` — accuracy curves, KL dynamics, CKA similarity
- `final_results.json` — detailed per-seed metrics

### Manual Analysis

```python
import pandas as pd
import json

# Load all seed results
for seed in range(10):
    metrics = pd.read_csv(f"runs/progressive_seed{seed}/metrics.csv")
    with open(f"runs/progressive_seed{seed}/cka_scores.json") as f:
        cka = json.load(f)
    print(f"Seed {seed}: final acc={metrics['val_acc'].iloc[-1]:.3f}, "
          f"final KL={metrics['val_kl'].iloc[-1]:.4f}")
```

---

## Metrics

**Fidelity (KL Divergence):**
$$KL(p_{\text{teacher}} || p_{\text{student}}) = \sum_i p_i \log(p_i / q_i)$$

Lower is better. Measures how well student matches teacher output distribution.

**Agreement Rate:**
$$\text{Agree} = \frac{1}{N} \sum_i \mathbb{1}[\arg\max p_i = \arg\max q_i]$$

Fraction of test samples where teacher and student have same predicted class.

**CKA (Canonical Correlation Analysis):**
Measures representational similarity between teacher and student hidden activations. 1.0 = identical, 0.0 = independent.

**Sparsity:**
Fraction of weights that are exactly zero after pruning.

---

### Dependencies

See `requirements.txt`. Key packages:
- `torch`, `torchvision` — deep learning
- `pandas` — metrics logging
- `pyyaml` — config parsing
- `tqdm` — progress bars

