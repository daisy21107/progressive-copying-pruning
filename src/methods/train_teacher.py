from typing import Dict, Any
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.tasks import get_dataloaders
from src.models.cnn_pair import PairClassifier
from src.utils.train_utils import get_device, CSVLogger
from src.methods.pruning import (
    set_global_sparsity,
    set_global_sparsity_absolute,
    rerandomize_pruned_weights,
)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy.
    
    Args:
        logits: Model outputs of shape (batch_size, num_classes)
        labels: Ground-truth labels of shape (batch_size,)
    
    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def run(cfg: Dict[str, Any], out_dir: str):
    """Train a teacher model on ground-truth labels.
    
    Args:
        cfg: Config dict with keys:
            - model: dict with architecture (width, hidden, etc.)
            - lr: learning rate (default 1e-3)
            - epochs: number of training epochs (default 10)
            - checkpoint_name: output file name (default 'teacher.pt')
        out_dir: Directory to save checkpoint and metrics log.
    """
    device = get_device(cfg)
    train_loader, test_loader, num_classes = get_dataloaders(cfg)

    model_cfg = cfg.get("model", {})
    width = int(model_cfg.get("width", 32))
    hidden = int(model_cfg.get("hidden", 128))
    shared = bool(model_cfg.get("shared_encoder", True))
    in_channels = int(model_cfg.get("in_channels", 1))

    model = PairClassifier(
        num_classes=num_classes,
        width=width,
        hidden=hidden,
        shared_encoder=shared,
        in_channels=in_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    epochs = int(cfg.get("epochs", 10))

    logger = CSVLogger(out_dir)
    best_val = 0.0
    checkpoint_name = str(cfg.get("checkpoint_name", "teacher.pt"))
    checkpoint_path = os.path.join(out_dir, checkpoint_name)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        for x1, x2, y in tqdm(train_loader, desc=f"Train E{epoch}"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_acc += accuracy(logits.detach(), y)
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        train_acc = total_acc / max(1, n_batches)

        # Eval
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for x1, x2, y in tqdm(test_loader, desc=f"Val E{epoch}"):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits, y)
                n_val += 1

        val_loss /= max(1, n_val)
        val_acc /= max(1, n_val)

        logger.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict()}, checkpoint_path)

    posthoc = cfg.get("posthoc_prune", {})
    if bool(posthoc.get("enabled", False)):
        target_sparsity = float(posthoc.get("target_sparsity", 0.90))
        fixed_target_sparsity = bool(posthoc.get("fixed_target_sparsity", True))
        rerand = bool(posthoc.get("rerandomize", False))

        if fixed_target_sparsity:
            set_global_sparsity_absolute(model, amount=target_sparsity)
        else:
            set_global_sparsity(model, amount=target_sparsity)
        if rerand:
            rerandomize_pruned_weights(model)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for x1, x2, y in tqdm(test_loader, desc="Val PostPrune"):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits, y)
                n_val += 1

        val_loss /= max(1, n_val)
        val_acc /= max(1, n_val)

        logger.log({
            "epoch": epochs + 1,
            "train_loss": 0.0,
            "train_acc": 0.0,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        torch.save({"model_state": model.state_dict()}, checkpoint_path)

    logger.close()
