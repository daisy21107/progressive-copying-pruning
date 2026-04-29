#!/usr/bin/env python3
"""
Run seed sweep for seed variance experiments.

Usage:
  # Run one regime with 10 seeds
  python scripts/run_seed_sweep.py --config configs/scratch_noise0.40.yaml --regime scratch --seeds 0 1 2 3 4 5 6 7 8 9
  
  # Run all regimes (oneshot, progressive, scratch) with 10 seeds
  python scripts/run_seed_sweep.py --config-base configs/ --regimes oneshot progressive scratch --seeds 0 1 2 3 4 5 6 7 8 9
  
  # Quick test with 2 seeds
  python scripts/run_seed_sweep.py --config configs/scratch_noise0.40.yaml --regime scratch --seeds 0 1
"""

import argparse
import subprocess
import sys
import os
import tempfile
from pathlib import Path
import yaml


def run_single(
    regime: str,
    config_path: str,
    seed: int,
    base_out_dir: str = "runs",
    epochs_override: int = None,
    noise_std_override: float = None,
    canvas_size_override: int = None,
    bg_noise_std_override: float = None,
    label_rule_override: str = None,
) -> bool:
    """Run a single experiment with specified seed.
    
    Returns:
        True if successful, False otherwise
    """
    tags = []
    if epochs_override is not None:
        tags.append(f"epochs{int(epochs_override)}")
    if noise_std_override is not None:
        tags.append(f"noise{noise_std_override:.3f}")
    if canvas_size_override is not None:
        tags.append(f"canvas{canvas_size_override}")
    if bg_noise_std_override is not None:
        tags.append(f"bg{bg_noise_std_override:.3f}")
    if label_rule_override is not None:
        tags.append(f"label{label_rule_override}")
    if tags:
        out_dir = os.path.join(base_out_dir, f"{regime}_{'_'.join(tags)}_seed{seed}")
    else:
        out_dir = os.path.join(base_out_dir, f"{regime}_seed{seed}")
    
    print(f"\n{'='*80}")
    print(f"Running: {regime} with seed={seed}")
    print(f"Config: {config_path}")
    print(f"Output: {out_dir}")
    print(f"{'='*80}\n")
    
    cfg_path = config_path
    temp_file = None
    if epochs_override is not None or noise_std_override is not None or canvas_size_override is not None or bg_noise_std_override is not None or label_rule_override is not None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("data", {})
        if epochs_override is not None:
            cfg["epochs"] = int(epochs_override)
        if noise_std_override is not None:
            cfg["data"]["noise_std"] = float(noise_std_override)
        if canvas_size_override is not None:
            cfg["data"]["random_position"] = True
            cfg["data"]["canvas_size"] = int(canvas_size_override)
        if bg_noise_std_override is not None:
            cfg["data"]["bg_noise_std"] = float(bg_noise_std_override)
        if label_rule_override is not None:
            cfg["data"]["label_rule"] = str(label_rule_override)
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(cfg, temp_file, default_flow_style=False)
        temp_file.close()
        cfg_path = temp_file.name

    cmd = [
        sys.executable, "-m", "src.main", regime,
        "--config", cfg_path,
        "--seed", str(seed),
        "--out-dir", out_dir
    ]
    if epochs_override is not None:
        cmd.extend(["--epochs", str(int(epochs_override))])
    
    result = subprocess.run(cmd, cwd=".")
    if temp_file is not None:
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {regime} seed={seed}")
        return False
    else:
        print(f"\n✅ SUCCESS: {regime} seed={seed} → {out_dir}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Seed sweep runner")
    parser.add_argument("--config", type=str, help="Single config file to use")
    parser.add_argument("--config-base", type=str, help="Base directory with noise configs (e.g., configs/)")
    parser.add_argument("--regime", type=str, help="Single regime to run")
    parser.add_argument("--regimes", nargs="+", help="Multiple regimes to run")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="List of seeds to sweep")
    parser.add_argument("--base-out-dir", type=str, default="runs", help="Base output directory")
    parser.add_argument("--epochs-override", type=int, default=None,
                        help="Override epochs in config for all runs")
    parser.add_argument("--noise-std-override", type=float, default=None,
                        help="Override noise_std in config for all runs")
    parser.add_argument("--canvas-size-override", type=int, default=None,
                        help="Override canvas_size and enable random_position")
    parser.add_argument("--bg-noise-std-override", type=float, default=None,
                        help="Override bg_noise_std for all runs")
    parser.add_argument("--label-rule-override", type=str, default=None,
                        help="Override label_rule in config for all runs")
    args = parser.parse_args()
    
    # Determine regimes to run
    if args.regimes:
        regimes = args.regimes
    elif args.regime:
        regimes = [args.regime]
    else:
        print("ERROR: Must specify --regime or --regimes")
        sys.exit(1)
    
    # Determine config(s) to use
    if args.config:
        # Single config for all regimes (must be appropriate)
        configs = {regime: args.config for regime in regimes}
    elif args.config_base:
        # Auto-detect configs from base directory
        configs = {}
        for regime in regimes:
            config_path = os.path.join(args.config_base, f"{regime}_noise0.40.yaml")
            if not os.path.exists(config_path):
                print(f"ERROR: Config not found: {config_path}")
                sys.exit(1)
            configs[regime] = config_path
    else:
        print("ERROR: Must specify --config or --config-base")
        sys.exit(1)
    
    # Run all combinations
    total = len(regimes) * len(args.seeds)
    success = 0
    failures = []
    
    print(f"\n{'='*80}")
    print(f"SEED SWEEP PLAN")
    print(f"{'='*80}")
    print(f"Regimes: {regimes}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs override: {args.epochs_override}")
    print(f"Total runs: {total}")
    print(f"Configs:")
    for regime, config in configs.items():
        print(f"  {regime}: {config}")
    print(f"{'='*80}\n")
    
    for regime in regimes:
        for seed in args.seeds:
            ok = run_single(
                regime,
                configs[regime],
                seed,
                args.base_out_dir,
                args.epochs_override,
                args.noise_std_override,
                args.canvas_size_override,
                args.bg_noise_std_override,
                args.label_rule_override,
            )
            if ok:
                success += 1
            else:
                failures.append((regime, seed))
    
    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {success}/{total}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for regime, seed in failures:
            print(f"  - {regime} seed={seed}")
    else:
        print("🎉 All runs completed successfully!")
    print(f"{'='*80}\n")
    
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
