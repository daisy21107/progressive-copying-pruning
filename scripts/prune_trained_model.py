#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.pruning import current_global_sparsity, remove_pruning_reparam, set_global_sparsity
from src.metrics.fidelity import fidelity_metrics, probability_mse
from src.models.cnn_pair import PairClassifier
from src.tasks import get_dataloaders
from src.utils.train_utils import get_device


def build_model(model_cfg: Dict[str, Any], num_classes: int, device: torch.device) -> PairClassifier:
    return PairClassifier(
        num_classes=num_classes,
        width=int(model_cfg.get("width", 32)),
        hidden=int(model_cfg.get("hidden", 128)),
        shared_encoder=bool(model_cfg.get("shared_encoder", True)),
        in_channels=int(model_cfg.get("in_channels", 1)),
    ).to(device)


def resolve_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("student_model", cfg.get("model", {}))


def load_checkpoint_model(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> PairClassifier:
    model = build_model(resolve_model_cfg(cfg), num_classes, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def load_teacher(
    cfg: Dict[str, Any],
    teacher_ckpt: Optional[Path],
    num_classes: int,
    device: torch.device,
) -> Optional[PairClassifier]:
    teacher_path = teacher_ckpt
    if teacher_path is None:
        raw_path = cfg.get("teacher_ckpt")
        if raw_path:
            teacher_path = Path(raw_path)

    if teacher_path is None or not teacher_path.exists():
        return None

    teacher_cfg = cfg.get("teacher_model", cfg.get("model", {}))
    teacher = build_model(teacher_cfg, num_classes, device)
    state = torch.load(teacher_path, map_location=device)
    teacher.load_state_dict(state["model_state"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def evaluate_model(
    model: PairClassifier,
    data_loader,
    device: torch.device,
    teacher: Optional[PairClassifier],
    temperature: float,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    n_batches = 0
    fid_kl = 0.0
    fid_mse = 0.0
    fid_prob_mse = 0.0
    fid_agree = 0.0

    with torch.no_grad():
        for x1, x2, y in data_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logits = model(x1, x2)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
            n_batches += 1

            if teacher is not None:
                t_logits = teacher(x1, x2)
                metrics = fidelity_metrics(t_logits, logits, temperature)
                fid_kl += metrics["kl"]
                fid_mse += metrics["logit_mse"]
                fid_agree += metrics["agree"]
                fid_prob_mse += float(probability_mse(t_logits, logits, temperature).item())

    result = {
        "val_acc": correct / max(1, total),
        "actual_sparsity": current_global_sparsity(model),
    }
    if teacher is not None:
        result.update({
            "val_kl": fid_kl / max(1, n_batches),
            "val_logit_mse": fid_mse / max(1, n_batches),
            "val_prob_mse": fid_prob_mse / max(1, n_batches),
            "val_agree": fid_agree / max(1, n_batches),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune a trained checkpoint to a target sparsity and re-evaluate it.")
    parser.add_argument("--checkpoint", required=True, help="Path to a dense checkpoint (teacher.pt or student.pt).")
    parser.add_argument("--config", required=True, help="Path to the config used for the checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Directory to write the pruned checkpoint and metrics.")
    parser.add_argument("--target-sparsity", type=float, required=True, help="Global sparsity target in [0, 1].")
    parser.add_argument("--teacher-ckpt", default=None, help="Optional teacher checkpoint for fidelity evaluation.")
    parser.add_argument("--temperature", type=float, default=None, help="Override fidelity temperature.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)

    device = get_device(cfg)
    _train_loader, test_loader, num_classes = get_dataloaders(cfg)
    temperature = float(args.temperature if args.temperature is not None else cfg.get("temperature", 1.0))

    model = load_checkpoint_model(cfg, checkpoint_path, num_classes, device)
    teacher = load_teacher(cfg, Path(args.teacher_ckpt) if args.teacher_ckpt else None, num_classes, device)

    pre_metrics = evaluate_model(model, test_loader, device, teacher, temperature)
    set_global_sparsity(model, amount=float(args.target_sparsity))
    post_metrics = evaluate_model(model, test_loader, device, teacher, temperature)
    remove_pruning_reparam(model)
    final_sparsity = current_global_sparsity(model)

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "teacher_ckpt": args.teacher_ckpt or cfg.get("teacher_ckpt"),
        "target_sparsity": float(args.target_sparsity),
        "pre_prune": pre_metrics,
        "post_prune": {
            **post_metrics,
            "actual_sparsity": final_sparsity,
        },
    }

    torch.save(
        {
            "model_state": model.state_dict(),
            "summary": summary,
        },
        out_dir / "pruned_model.pt",
    )
    with (out_dir / "pruned_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()