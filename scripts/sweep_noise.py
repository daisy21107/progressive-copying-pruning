#!/usr/bin/env python3
"""
Sweep script to test different noise levels and find a suitable difficulty regime.

Usage:
  python scripts/sweep_noise.py --base-config configs/scratch.yaml [--noise-levels 0.0 0.1 0.2 ...]
  python scripts/sweep_noise.py --base-config configs/teacher.yaml --regimes teacher scratch

Outputs a table of final validation accuracy/agreement across noise levels.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def run_experiment(regime: str, config_path: str, out_dir: str) -> Tuple[Optional[float], Dict[str, str]]:
    """Run a single experiment and extract final metrics.
    
    Returns:
        (final_val_acc or final_agreement, metrics_dict)
    """
    cmd = [sys.executable, "-m", "src.main", regime, "--config", config_path]
    result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ERROR running {regime}: {result.stderr[:200]}")
        return None, {}
    
    # Try to read metrics.csv from output
    metrics_file = Path(out_dir) / "metrics.csv"
    if not metrics_file.exists():
        print(f"    WARNING: no metrics.csv found at {metrics_file}")
        return None, {}
    
    try:
        with open(metrics_file, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, {}

        metrics = rows[-1]

        if "val_agree" in metrics and metrics.get("val_agree") not in (None, ""):
            val_metric = float(metrics["val_agree"])
        elif "val_acc" in metrics and metrics.get("val_acc") not in (None, ""):
            val_metric = float(metrics["val_acc"])
        else:
            return None, metrics

        return val_metric, metrics
    except Exception as e:
        print(f"    ERROR parsing metrics: {e}")
        return None, {}


def sweep_noise(base_config_path: str, noise_levels: List[float], regimes: List[str], epochs_override: int = None):
    """Sweep over noise levels and collect results.
    
    Args:
        base_config_path: Path to base YAML config
        noise_levels: List of noise_std values to test
        regimes: List of regimes to run (e.g., ["teacher", "scratch"])
        epochs_override: Optional epochs override for quick sweep (e.g., 3)
    """
    base_cfg = load_config(base_config_path)
    
    print(f"\n{'='*80}")
    print(f"Noise Sweep: base_config={base_config_path}, noise_levels={noise_levels}")
    print(f"Regimes: {regimes}")
    if epochs_override:
        print(f"Epochs override: {epochs_override}")
    print(f"{'='*80}\n")
    
    results = {regime: {} for regime in regimes}
    
    for noise_std in noise_levels:
        print(f"\n>>> noise_std={noise_std}")
        
        for regime in regimes:
            cfg = dict(base_cfg)
            
            # Ensure data section exists
            if "data" not in cfg:
                cfg["data"] = {}
            
            cfg["data"]["noise_std"] = noise_std
            cfg["data"]["noise_mode"] = "fixed"
            cfg["data"]["noise_seed"] = 2026
            
            # Override epochs if specified
            if epochs_override:
                cfg["epochs"] = epochs_override
            
            # Use timestamped out_dir to avoid collisions
            ts = int(time.time() * 1000) % 1000000
            out_dir = f"runs/sweep_noise_std{noise_std:.2f}_{regime}_{ts}"
            cfg["out_dir"] = out_dir
            
            # Write temp config safely under system temp dir
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f"_{regime}_sweep.yaml",
                delete=False,
            ) as temp_file:
                save_config(cfg, temp_file.name)
                temp_config = temp_file.name
            
            try:
                print(f"  [{regime}] running... ", end="", flush=True)
                val_metric, _ = run_experiment(regime, temp_config, out_dir)
                
                if val_metric is not None:
                    results[regime][noise_std] = val_metric
                    print(f"val_metric={val_metric:.4f}")
                else:
                    print("FAILED")
                    results[regime][noise_std] = None
            finally:
                # Clean up temp config
                if os.path.exists(temp_config):
                    os.remove(temp_config)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE:")
    print(f"{'='*80}")
    
    # Header
    header = "noise_std".ljust(12)
    for regime in regimes:
        header += f" | {regime:>12}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for noise_std in noise_levels:
        row = f"{noise_std:<12.2f}"
        for regime in regimes:
            val = results[regime].get(noise_std)
            if val is not None:
                row += f" | {val:>12.4f}"
            else:
                row += f" | {'FAILED':>12}"
        print(row)
    
    print(f"{'='*80}")
    print("\nRecommendation:")
    print("  - Pick a noise_std where student scratch val_acc is ~0.70–0.85 (challenging but convergent)")
    print("  - Example: if noise_std=0.25 gives ~0.75 val_acc, use that for future runs")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Sweep noise levels to find suitable difficulty")
    parser.add_argument("--base-config", required=True, help="Base YAML config path (e.g., configs/scratch.yaml)")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4],
                        help="Noise std values to sweep")
    parser.add_argument("--regimes", nargs="+", default=["scratch"], 
                        help="Regimes to run (e.g., teacher scratch oneshot progressive)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for quick sweep (e.g., 3 for fast testing)")
    
    args = parser.parse_args()
    
    sweep_noise(args.base_config, args.noise_levels, args.regimes, args.epochs)


if __name__ == "__main__":
    main()
