#!/usr/bin/env python3

import copy
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload):
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def scaled_dim(base_dim: int, target_capacity: float, min_dim: int) -> int:
    ratio = target_capacity ** 0.5
    return max(min_dim, int(round(base_dim * ratio)))


def main():
    capacities = [3, 5, 10, 15, 25]
    out_dir = Path("configs/small_capacity_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    progressive_base = load_yaml(Path("configs/progressive_cifar100.yaml"))
    oneshot_base = load_yaml(Path("configs/oneshot_cifar100.yaml"))
    scratch_base = load_yaml(Path("configs/scratch_cifar100.yaml"))

    base_width = int(oneshot_base["student_model"]["width"])
    base_hidden = int(oneshot_base["student_model"]["hidden"])

    for cap in capacities:
        target_capacity = cap / 100.0
        final_sparsity = 1.0 - target_capacity

        prog_cfg = copy.deepcopy(progressive_base)
        prog_cfg["epochs"] = 20
        prog_cfg.setdefault("progressive", {})
        prog_cfg["progressive"].update({
            "fixed_target_sparsity": True,
            "target_sparsity": final_sparsity,
            "prune_milestones": [
                {"epoch": 5, "target_sparsity": round(final_sparsity * 0.33, 6)},
                {"epoch": 10, "target_sparsity": round(final_sparsity * 0.66, 6)},
                {"epoch": 15, "target_sparsity": round(final_sparsity, 6)},
            ],
        })
        dump_yaml(out_dir / f"progressive_{cap}pct.yaml", prog_cfg)

        width = scaled_dim(base_width, target_capacity, min_dim=3)
        hidden = scaled_dim(base_hidden, target_capacity, min_dim=12)

        one_cfg = copy.deepcopy(oneshot_base)
        one_cfg["epochs"] = 20
        one_cfg["student_model"]["width"] = width
        one_cfg["student_model"]["hidden"] = hidden
        dump_yaml(out_dir / f"oneshot_{cap}pct.yaml", one_cfg)

        scr_cfg = copy.deepcopy(scratch_base)
        scr_cfg["epochs"] = 20
        scr_cfg["model"]["width"] = width
        scr_cfg["model"]["hidden"] = hidden
        dump_yaml(out_dir / f"scratch_{cap}pct.yaml", scr_cfg)

        print(
            f"{cap:>2}% | progressive sparsity@15={final_sparsity:.1%} "
            f"| one-shot/scratch width={width} hidden={hidden}"
        )

    print(f"\nWrote small-capacity configs to {out_dir}")


if __name__ == "__main__":
    main()