"""
True One-Shot Pruning: Prune at initialization, then train from scratch.
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
from src.methods.pruning import set_global_sparsity_absolute, current_global_sparsity


def load_teacher(cfg: Dict[str, Any], device: torch.device, num_classes: int):
    """Load pretrained teacher model."""
    teacher = build_classifier(cfg, num_classes, role="teacher")
    ckpt = torch.load(cfg['teacher_ckpt'], map_location=device)
    teacher.load_state_dict(ckpt['model_state'])
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def run(cfg: Dict[str, Any], out_dir: str) -> None:
    """
    TRUE One-Shot Pruning: Prune at initialization, train sparse network from scratch.
    
    Key difference from progressive:
    - One-shot: Prune once at epoch 0, then train with FIXED mask
    - Progressive: Gradually increase sparsity during training
    """
    
    device = get_device(cfg)
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Data
    train_loader, test_loader, num_classes = get_dataloaders(cfg)
    
    # Models
    teacher = load_teacher(cfg, device, num_classes)
    student = build_classifier(cfg, num_classes, role="student")
    student.to(device)
    
    # Training params
    lr = cfg['lr']
    temperature = cfg.get('temperature', 2.0)
    epochs = cfg['epochs']
    target_sparsity = cfg.get('target_sparsity', 0.9)
    
    # ========== CRITICAL: PRUNE AT INITIALIZATION ==========
    print("="*70)
    print("ONE-SHOT PRUNING AT INITIALIZATION")
    print("="*70)
    print(f"Target sparsity: {target_sparsity*100:.0f}%")
    
    # Prune BEFORE training
    set_global_sparsity_absolute(student, amount=target_sparsity)
    actual_sparsity = current_global_sparsity(student)
    print(f"Actual sparsity: {actual_sparsity*100:.2f}%")
    print("="*70)
    
    # Optimizer
    optimizer = Adam(student.parameters(), lr=lr)
    logger = CSVLogger(out_dir)
    
    # ========== TRAIN SPARSE NETWORK FROM SCRATCH ==========
    for epoch in range(1, epochs + 1):
        student.train()
        total_kl = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"OneShot E{epoch}")
        for batch in pbar:
            inputs, _y = unpack_batch(batch)
            inputs = tuple(t.to(device) for t in inputs)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                t_logits = forward_logits(teacher, inputs)
            
            s_logits = forward_logits(student, inputs)
            loss = kl_teacher_student(t_logits, s_logits, temperature)
            
            loss.backward()
            optimizer.step()
            
            total_kl += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_kl = total_kl / max(1, n_batches)
        
        # Validation
        student.eval()
        fid_kl = 0.0
        fid_mse = 0.0
        fid_prob_mse = 0.0
        fid_ag = 0.0
        correct = 0
        total = 0
        n_val = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Eval E{epoch}", leave=False):
                inputs, y = unpack_batch(batch)
                inputs = tuple(t.to(device) for t in inputs)
                y = y.to(device)
                
                t_logits = forward_logits(teacher, inputs)
                s_logits = forward_logits(student, inputs)
                
                m = fidelity_metrics(t_logits, s_logits, temperature)
                fid_kl += m["kl"]
                fid_mse += m["logit_mse"]
                fid_ag += m["agree"]
                fid_prob_mse += float(probability_mse(t_logits, s_logits, temperature).item())
                
                preds = s_logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
                n_val += 1
        
        val_acc = correct / max(1, total)
        
        logger.log({
            "epoch": epoch,
            "train_kl": train_kl,
            "val_kl": fid_kl / max(1, n_val),
            "val_logit_mse": fid_mse / max(1, n_val),
            "val_prob_mse": fid_prob_mse / max(1, n_val),
            "val_agree": fid_ag / max(1, n_val),
            "val_acc": val_acc,
            "sparsity": actual_sparsity,
        })
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}: Val Acc={val_acc*100:.2f}%, KL={fid_kl/max(1,n_val):.4f}")
    
    # CKA
    if cfg.get('compute_cka', False):
        print("\nComputing CKA...")
        cka_scores = compute_cka_similarity(student, teacher, test_loader, device)
        with open(os.path.join(out_dir, 'cka_scores.json'), 'w') as f:
            json.dump(cka_scores, f, indent=2)
    
    # Save
    torch.save({
        'model_state': student.state_dict(),
        'final_acc': val_acc,
        'sparsity': actual_sparsity
    }, os.path.join(out_dir, 'student.pt'))
    
    logger.close()
    
    print(f"\n{'='*70}")
    print("ONE-SHOT TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final accuracy: {val_acc*100:.2f}%")
    print(f"Sparsity: {actual_sparsity*100:.2f}%")
    print(f"Saved to: {out_dir}")