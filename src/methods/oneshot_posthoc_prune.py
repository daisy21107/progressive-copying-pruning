"""
One-shot distillation with post-hoc pruning + fine-tuning.
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
from src.metrics.fidelity import kl_teacher_student, fidelity_metrics
from src.metrics.repsim import compute_cka_similarity
from src.methods.pruning import set_global_sparsity, current_global_sparsity

def load_teacher(cfg: Dict[str, Any], device: torch.device, num_classes: int):
    """Load pretrained teacher model."""
    teacher = build_classifier(cfg, num_classes, role="teacher")
    ckpt = torch.load(cfg['teacher_ckpt'], map_location=device)
    teacher.load_state_dict(ckpt['model_state'])
    teacher.to(device)
    teacher.eval()
    return teacher

def validate(student, teacher, test_loader, device, temperature):
    """Run validation and compute metrics."""
    student.eval()
    teacher.eval()
    
    total_kl = 0.0
    total_mse = 0.0
    total_agree = 0.0
    correct = 0
    total = 0
    batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = unpack_batch(batch)
            inputs = tuple(t.to(device) for t in inputs)
            labels = labels.to(device)

            student_logits = forward_logits(student, inputs)
            teacher_logits = forward_logits(teacher, inputs)
            
            # Fidelity metrics
            metrics = fidelity_metrics(teacher_logits, student_logits, temperature)
            total_kl += metrics['kl']
            total_mse += metrics['logit_mse']
            total_agree += metrics['agree']
            
            # Accuracy
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            batches += 1
    
    return {
        'kl_divergence': total_kl / batches,
        'logit_mse': total_mse / batches,
        'agreement': total_agree / batches,
        'accuracy': correct / total
    }

def run(cfg: Dict[str, Any], out_dir: str) -> None:
    """
    One-shot distillation with post-hoc pruning and fine-tuning.
    
    Strategy:
    1. Train at full capacity (35 epochs)
    2. Apply pruning to target sparsity
    3. Fine-tune sparse network (5 epochs)
    Total: 40 epochs
    """
    
    device = get_device(cfg)
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Data - MATCH PROGRESSIVE EXACTLY
    train_loader, test_loader, num_classes = get_dataloaders(cfg)
    
    # Models
    teacher = load_teacher(cfg, device, num_classes)
    student = build_classifier(cfg, num_classes, role="student")
    student.to(device)
    
    # Training params
    lr = cfg['lr']
    temperature = cfg.get('temperature', 2.0)
    total_epochs = cfg['epochs']
    target_sparsity = cfg.get('target_sparsity', 0.9)
    
    # Split: 35 dense + 5 finetune
    dense_epochs = total_epochs - 5
    
    optimizer = Adam(student.parameters(), lr=lr)
    logger = CSVLogger(out_dir)
    
    print("="*70)
    print("ONE-SHOT WITH FINE-TUNING")
    print("="*70)
    print(f"Dense: {dense_epochs} epochs")
    print(f"Prune to: {target_sparsity*100:.0f}%")
    print(f"Finetune: {total_epochs - dense_epochs} epochs")
    print("="*70)
    
    # ========== PHASE 1: DENSE TRAINING ==========
    for epoch in range(1, dense_epochs + 1):
        student.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Dense E{epoch}/{dense_epochs}")
        for batch in pbar:
            inputs, labels = unpack_batch(batch)
            inputs = tuple(t.to(device) for t in inputs)
            labels = labels.to(device)

            student_logits = forward_logits(student, inputs)

            with torch.no_grad():
                teacher_logits = forward_logits(teacher, inputs)
            
            loss = kl_teacher_student(teacher_logits, student_logits, temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        val_metrics = validate(student, teacher, test_loader, device, temperature)
        
        avg_train_loss = train_loss_sum / train_batches
        
        logger.log({
            'epoch': epoch,
            'phase': 'dense',
            'train_loss': avg_train_loss,
            'val_kl': val_metrics['kl_divergence'],
            'val_logit_mse': val_metrics['logit_mse'],
            'val_acc': val_metrics['accuracy'],
            'val_agree': val_metrics['agreement'],
            'sparsity': 0.0
        })
        
        print(f"Dense E{epoch}: Loss={avg_train_loss:.4f}, Acc={val_metrics['accuracy']*100:.2f}%")
    
    pre_prune_acc = val_metrics['accuracy'] * 100
    
    # ========== PHASE 2: PRUNING ==========
    print(f"\n{'='*70}")
    print(f"PRUNING TO {target_sparsity*100:.0f}%")
    print('='*70)
    
    set_global_sparsity(student, target_sparsity)
    actual_sparsity = current_global_sparsity(student)
    print(f"Actual sparsity: {actual_sparsity*100:.2f}%")
    
    # Eval post-pruning
    post_prune_metrics = validate(student, teacher, test_loader, device, temperature)
    post_prune_acc = post_prune_metrics['accuracy'] * 100
    
    print(f"Pre-pruning:  {pre_prune_acc:.2f}%")
    print(f"Post-pruning: {post_prune_acc:.2f}% ({post_prune_acc - pre_prune_acc:+.2f}%)")
    
    # ========== PHASE 3: FINE-TUNING ==========
    print(f"\n{'='*70}")
    print("FINE-TUNING SPARSE NETWORK")
    print('='*70)
    
    optimizer = Adam(student.parameters(), lr=lr * 0.1)
    
    for epoch in range(dense_epochs + 1, total_epochs + 1):
        student.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Finetune E{epoch}/{total_epochs}")
        for batch in pbar:
            inputs, labels = unpack_batch(batch)
            inputs = tuple(t.to(device) for t in inputs)
            labels = labels.to(device)

            student_logits = forward_logits(student, inputs)

            with torch.no_grad():
                teacher_logits = forward_logits(teacher, inputs)
            
            loss = kl_teacher_student(teacher_logits, student_logits, temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        val_metrics = validate(student, teacher, test_loader, device, temperature)
        
        avg_train_loss = train_loss_sum / train_batches
        
        logger.log({
            'epoch': epoch,
            'phase': 'finetune',
            'train_loss': avg_train_loss,
            'val_kl': val_metrics['kl_divergence'],
            'val_logit_mse': val_metrics['logit_mse'],
            'val_acc': val_metrics['accuracy'],
            'val_agree': val_metrics['agreement'],
            'sparsity': actual_sparsity
        })
        
        print(f"Finetune E{epoch}: Loss={avg_train_loss:.4f}, Acc={val_metrics['accuracy']*100:.2f}%")
    
    final_acc = val_metrics['accuracy'] * 100
    recovery = final_acc - post_prune_acc
    
    # ========== CKA ==========
    print(f"\nComputing CKA...")
    cka_scores = compute_cka_similarity(student, teacher, test_loader, device)
    
    with open(os.path.join(out_dir, 'cka_scores.json'), 'w') as f:
        json.dump(cka_scores, f, indent=2)
    
    # ========== SUMMARY ==========
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    print(f"Pre-pruning:       {pre_prune_acc:.2f}%")
    print(f"Post-pruning:      {post_prune_acc:.2f}% ({post_prune_acc - pre_prune_acc:+.2f}%)")
    print(f"After fine-tuning: {final_acc:.2f}% ({recovery:+.2f}% recovery)")
    print(f"CKA similarity:    {cka_scores.get('average', 'N/A'):.4f}")
    print('='*70)
    
    # Save
    torch.save({
        'model_state': student.state_dict(),
        'final_metrics': val_metrics,
        'cka_scores': cka_scores,
        'config': cfg
    }, os.path.join(out_dir, 'student.pt'))
    
    with open(os.path.join(out_dir, 'config_snapshot.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    
    logger.close()
    print(f"\n✓ Saved to {out_dir}")