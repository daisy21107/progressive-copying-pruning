"""
Progressive Copying with Gradual Pruning

Implements the main method: progressive capacity reduction during distillation.

Student starts at full capacity and is gradually pruned with magnitude-based
unstructured pruning at scheduled epochs (e.g., epochs 5, 10, 15) while
maintaining knowledge distillation loss.

Key characteristics:
- Student initialized to full capacity (dense)
- Gradual pruning at configured epochs
- Final target: 90% sparsity
- Uses KL divergence loss (no ground-truth labels needed)
- Maintains representational similarity with teacher via CKA metric
"""

from typing import Dict, Any
import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.tasks import get_dataloaders
from src.utils.train_utils import get_device, CSVLogger
from src.utils.batch_utils import unpack_batch, forward_logits
from src.utils.model_factory import build_classifier
from src.metrics.fidelity import kl_teacher_student, fidelity_metrics, probability_mse
from src.metrics.repsim import compute_cka_similarity
from src.methods.pruning import (
    set_global_sparsity,
    set_global_sparsity_absolute,
    current_global_sparsity,
    rerandomize_pruned_weights,
)


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
    """Progressive copying + pruning: student gradually pruned during distillation.
    
    Student starts at full capacity. At each epoch (or milestone), magnitude-based
    pruning is applied to increase sparsity toward target. Student is trained
    to match teacher logits throughout via KL divergence loss.
    
    Args:
        cfg: Config dict with keys:
            - teacher_ckpt: path to teacher checkpoint
            - lr: learning rate (default 1e-3)
            - epochs: total distillation epochs
            - temperature: KD temperature (default 1.0)
            - progressive: dict with:
                - target_sparsity: final sparsity target (e.g., 0.90)
                - prune_per_epoch: sparsity increment per epoch (override if milestones used)
                - prune_milestones: list of epochs to prune (optional)
                - rerandomize: whether to rerandomize pruned weights (default False)
                - fixed_target_sparsity: re-compute sparsity each epoch (default True)
            - compute_cka: whether to compute CKA (default False)
        out_dir: Output directory for logs and checkpoints.
    """
    device = get_device(cfg)
    train_loader, test_loader, num_classes = get_dataloaders(cfg)

    teacher = load_teacher(cfg, device, num_classes)

    student = build_classifier(cfg, num_classes, role="student").to(device)

    optim = Adam(student.parameters(), lr=float(cfg.get("lr", 1e-3)))
    epochs = int(cfg.get("epochs", 20))
    T = float(cfg.get("temperature", 1.0))

    prog = cfg.get("progressive", {})
    prune_per_epoch = float(prog.get("prune_per_epoch", 0.01))
    target_sparsity = float(prog.get("target_sparsity", 0.90))
    rerand = bool(prog.get("rerandomize", False))
    fixed_target_sparsity = bool(prog.get("fixed_target_sparsity", True))
    prune_interval = int(prog.get("prune_interval", 1))
    prune_start_epoch = int(prog.get("prune_start_epoch", 1))
    prune_per_event = float(prog.get("prune_per_event", prune_per_epoch))
    milestones = prog.get("prune_milestones", [])

    milestone_targets = {}
    for event in milestones:
        if not isinstance(event, dict):
            continue
        if "epoch" not in event:
            continue
        if "target_sparsity" in event:
            sp = float(event["target_sparsity"])
        elif "sparsity" in event:
            sp = float(event["sparsity"])
        else:
            continue
        milestone_targets[int(event["epoch"])] = max(0.0, min(1.0, sp))

    logger = CSVLogger(out_dir)
    current_target = 0.0

    for epoch in range(1, epochs + 1):
        should_prune = False
        if milestone_targets:
            if epoch in milestone_targets:
                current_target = milestone_targets[epoch]
                should_prune = True
        else:
            if epoch >= prune_start_epoch and ((epoch - prune_start_epoch) % max(1, prune_interval) == 0):
                current_target = min(target_sparsity, current_target + prune_per_event)
                should_prune = True

        if should_prune:
            if fixed_target_sparsity:
                set_global_sparsity_absolute(student, amount=current_target)
            else:
                set_global_sparsity(student, amount=current_target)
            if rerand:
                rerandomize_pruned_weights(student)

        student.train()
        total_kl = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Prog E{epoch}"):
            inputs, _y = unpack_batch(batch)
            inputs = tuple(t.to(device) for t in inputs)
            optim.zero_grad()
            with torch.no_grad():
                t_logits = forward_logits(teacher, inputs)
            s_logits = forward_logits(student, inputs)
            loss = kl_teacher_student(t_logits, s_logits, T)
            loss.backward()
            optim.step()
            total_kl += loss.item()
            n_batches += 1

        train_kl = total_kl / max(1, n_batches)

        # Eval fidelity and task accuracy
        student.eval()
        fid_kl = 0.0
        fid_mse = 0.0
        fid_prob_mse = 0.0
        fid_ag = 0.0
        correct = 0
        total = 0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fidelity E{epoch}"):
                inputs, y = unpack_batch(batch)
                inputs = tuple(t.to(device) for t in inputs)
                y = y.to(device)
                t_logits = forward_logits(teacher, inputs)
                s_logits = forward_logits(student, inputs)
                m = fidelity_metrics(t_logits, s_logits, T)
                fid_kl += m["kl"]
                fid_mse += m["logit_mse"]
                fid_ag += m["agree"]
                fid_prob_mse += float(probability_mse(t_logits, s_logits, T).item())
                preds = s_logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
                n_val += 1

        val_acc = correct / max(1, total)
        sparsity = current_global_sparsity(student)
        logger.log({
            "epoch": epoch,
            "train_kl": train_kl,
            "val_kl": fid_kl / max(1, n_val),
            "val_logit_mse": fid_mse / max(1, n_val),
            "val_prob_mse": fid_prob_mse / max(1, n_val),
            "val_agree": fid_ag / max(1, n_val),
            "val_acc": val_acc,
            "target_sparsity": current_target,
            "actual_sparsity": sparsity,
        })

    # Save pruned student
    if bool(cfg.get("compute_cka", False)):
        cka_scores = compute_cka_similarity(student, teacher, test_loader, device)
        with open(os.path.join(out_dir, "cka_scores.json"), "w") as f:
            json.dump(cka_scores, f, indent=2)

    torch.save({"model_state": student.state_dict()}, os.path.join(out_dir, "student.pt"))
    logger.close()
