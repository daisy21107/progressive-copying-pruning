import argparse
import os
import yaml
from datetime import datetime

from src.methods.train_teacher import run as run_teacher
from src.methods.distill import run as run_oneshot, run_dense
from src.methods.oneshot_posthoc_prune import run as run_oneshot_posthoc
from src.methods.progressive_copy_prune import run as run_progressive
from src.utils.train_utils import set_seed, ensure_outdir, save_config_snapshot


def load_config(path: str) -> dict:
	with open(path, "r") as f:
		return yaml.safe_load(f)


def main():
	parser = argparse.ArgumentParser(description="FYP Progressive Copying with Pruning")
	parser.add_argument(
		"regime",
		choices=["teacher", "oneshot", "oneshot_posthoc", "dense", "progressive", "scratch"],
		help="Run regime",
	)
	parser.add_argument("--config", required=True, help="Path to YAML config")
	parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
	parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
	parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
	args = parser.parse_args()

	cfg = load_config(args.config)
	if args.epochs is not None:
		cfg["epochs"] = int(args.epochs)

	# Enforce trajectory-based comparison defaults:
	# one-shot and scratch train at full capacity, then prune post-hoc.
	if args.regime in ("oneshot", "scratch"):
		posthoc = cfg.get("posthoc_prune", {})
		if not isinstance(posthoc, dict):
			posthoc = {}
		target_default = float(cfg.get("progressive", {}).get("target_sparsity", 0.90))
		posthoc.setdefault("enabled", True)
		posthoc.setdefault("target_sparsity", target_default)
		posthoc.setdefault("fixed_target_sparsity", True)
		posthoc.setdefault("rerandomize", False)
		cfg["posthoc_prune"] = posthoc

	if args.regime == "scratch":
		cfg.setdefault("checkpoint_name", "student.pt")

	seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
	set_seed(seed)
	
	# Update config with overridden seed for snapshot
	cfg["seed"] = seed

	time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
	if args.out_dir is not None:
		out_dir = args.out_dir
	else:
		out_dir = cfg.get("out_dir", os.path.join("runs", f"{args.regime}-{time_tag}"))
	ensure_outdir(out_dir)
	save_config_snapshot(cfg, out_dir)

	if args.regime == "teacher":
		run_teacher(cfg, out_dir)
	elif args.regime == "oneshot":
		run_oneshot(cfg, out_dir)
	elif args.regime == "oneshot_posthoc":
		run_oneshot_posthoc(cfg, out_dir)
	elif args.regime == "dense":
		run_dense(cfg, out_dir)
	elif args.regime == "progressive":
		run_progressive(cfg, out_dir)
	elif args.regime == "scratch":
		# Reuse supervised training runner for scratch baseline.
		run_teacher(cfg, out_dir)
	else:
		raise ValueError(f"Unknown regime: {args.regime}")


if __name__ == "__main__":
	main()
