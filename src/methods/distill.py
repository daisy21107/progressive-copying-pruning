"""
Dense Knowledge Distillation (One-Shot Baseline)

Standard knowledge distillation without any pruning.
Trains a student to match teacher outputs over all epochs.

This is the baseline that establishes the upper bound on performance
(full capacity student matching teacher logits).
"""

from typing import Dict, Any
from copy import deepcopy
import json
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.tasks import get_dataloaders
from src.utils.train_utils import get_device, CSVLogger
from src.utils.batch_utils import unpack_batch, forward_logits
from src.utils.model_factory import build_classifier
from src.metrics.fidelity import kl_teacher_student, fidelity_metrics, probability_mse
from src.methods.pruning import (
    set_global_sparsity,
    set_global_sparsity_absolute,
    rerandomize_pruned_weights,
)
from src.metrics.repsim import compute_cka_similarity


def load_teacher(cfg: Dict[str, Any], device: torch.device, num_classes: int) -> nn.Module:
    """Load and freeze a pretrained teacher model.
    
    Args:
        cfg: Configuration dict with 'teacher_ckpt' path.
        device: Device to load model onto.
        num_classes: Number of output classes.
    
    Returns:
        Frozen teacher model in eval mode.
    """
    teacher = build_classifier(cfg, num_classes, role="teacher").to(device)
    ckpt_path = cfg.get("teacher_ckpt")
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError("teacher_ckpt not found; train a teacher first.")
    state = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(state["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def run(cfg: Dict[str, Any], out_dir: str):
    """Train student via knowledge distillation (dense baseline).
    
    Loads pretrained teacher, trains student to match teacher logits.
    Optionally applies post-hoc pruning after training.
    
    Args:
        cfg: Config dict with keys:
            - teacher_ckpt: path to teacher checkpoint
            - lr: learning rate (default 1e-3)
            - epochs: number of distillation epochs
            - temperature: KD temperature (default 1.0)
            - posthoc_prune: dict with enable/target_sparsity (optional)
            - compute_cka: whether to compute CKA (default False)
        out_dir: Output directory for logs and checkpoints.
    """
    device = get_device(cfg)
    train_loader, test_loader, num_classes = get_dataloaders(cfg)

    teacher = load_teacher(cfg, device, num_classes)

    # Student can be smaller; configurable via cfg["student_model"] else cfg["model"]
    student = build_classifier(cfg, num_classes, role="student").to(device)

    optim = Adam(student.parameters(), lr=float(cfg.get("lr", 1e-3)))
    epochs = int(cfg.get("epochs", 10))
    T = float(cfg.get("temperature", 1.0))

    logger = CSVLogger(out_dir)

    def eval_student(model: nn.Module):
        model.eval()
        fid_kl = 0.0
        fid_mse = 0.0
        fid_prob_mse = 0.0
        fid_ag = 0.0
        correct = 0
        total = 0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Fidelity Eval"):
                inputs, y = unpack_batch(batch)
                inputs = tuple(t.to(device) for t in inputs)
                y = y.to(device)
                t_logits = forward_logits(teacher, inputs)
                s_logits = forward_logits(model, inputs)
                m = fidelity_metrics(t_logits, s_logits, T)
                fid_kl += m["kl"]
                fid_mse += m["logit_mse"]
                fid_ag += m["agree"]
                fid_prob_mse += float(probability_mse(t_logits, s_logits, T).item())
                preds = s_logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
                n_val += 1

        return {
            "val_kl": fid_kl / max(1, n_val),
            "val_logit_mse": fid_mse / max(1, n_val),
            "val_prob_mse": fid_prob_mse / max(1, n_val),
            "val_agree": fid_ag / max(1, n_val),
            "val_acc": correct / max(1, total),
        }

    last_train_kl = 0.0

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Distill E{epoch}"):
            inputs, _y = unpack_batch(batch)
            inputs = tuple(t.to(device) for t in inputs)
            optim.zero_grad()
            with torch.no_grad():
                t_logits = forward_logits(teacher, inputs)
            s_logits = forward_logits(student, inputs)
            loss = kl_teacher_student(t_logits, s_logits, T)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        last_train_kl = train_loss
        metrics = eval_student(student)

        logger.log({
            "epoch": epoch,
            "train_kl": train_loss,
            "val_kl": metrics["val_kl"],
            "val_logit_mse": metrics["val_logit_mse"],
            "val_prob_mse": metrics["val_prob_mse"],
            "val_agree": metrics["val_agree"],
            "val_acc": metrics["val_acc"],
        })

    posthoc = cfg.get("posthoc_prune", {})
    if bool(posthoc.get("enabled", False)):
        target_sparsity = float(posthoc.get("target_sparsity", 0.90))
        fixed_target_sparsity = bool(posthoc.get("fixed_target_sparsity", True))
        rerand = bool(posthoc.get("rerandomize", False))

        if fixed_target_sparsity:
            set_global_sparsity_absolute(student, amount=target_sparsity)
        else:
            set_global_sparsity(student, amount=target_sparsity)
        if rerand:
            rerandomize_pruned_weights(student)

        metrics = eval_student(student)
        logger.log({
            "epoch": epochs + 1,
            "train_kl": last_train_kl,
            "val_kl": metrics["val_kl"],
            "val_logit_mse": metrics["val_logit_mse"],
            "val_prob_mse": metrics["val_prob_mse"],
            "val_agree": metrics["val_agree"],
            "val_acc": metrics["val_acc"],
        })

    if bool(cfg.get("compute_cka", False)):
        cka_scores = compute_cka_similarity(student, teacher, test_loader, device)
        with open(os.path.join(out_dir, "cka_scores.json"), "w") as f:
            json.dump(cka_scores, f, indent=2)

    # Save student
    torch.save({"model_state": student.state_dict()}, os.path.join(out_dir, "student.pt"))
    logger.close()


def train_dense_baseline(cfg: Dict[str, Any], out_dir: str):
    """Dense baseline: distillation with no pruning and optional CKA export."""
    dense_cfg = deepcopy(cfg)
    posthoc = dense_cfg.get("posthoc_prune", {})
    if not isinstance(posthoc, dict):
        posthoc = {}
    posthoc["enabled"] = False
    dense_cfg["posthoc_prune"] = posthoc
    dense_cfg.setdefault("compute_cka", True)
    run(dense_cfg, out_dir)


def run_dense(cfg: Dict[str, Any], out_dir: str):
    """Entry point: run dense distillation baseline."""
    train_dense_baseline(cfg, out_dir)
