from typing import Dict
import torch
import torch.nn.functional as F


def softmax_with_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.softmax(logits / T, dim=-1)


def kl_teacher_student(teacher_logits: torch.Tensor, student_logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    pt = softmax_with_temperature(teacher_logits.detach(), T)
    log_ps = F.log_softmax(student_logits / T, dim=-1)
    kl = F.kl_div(log_ps, pt, reduction="batchmean") * (T * T)
    return kl


def logit_mse(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_logits, teacher_logits.detach())


def agreement_rate(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    t_pred = teacher_logits.argmax(dim=-1)
    s_pred = student_logits.argmax(dim=-1)
    return (t_pred == s_pred).float().mean()


def fidelity_metrics(teacher_logits: torch.Tensor, student_logits: torch.Tensor, T: float = 1.0) -> Dict[str, float]:
    with torch.no_grad():
        return {
            "kl": float(kl_teacher_student(teacher_logits, student_logits, T).item()),
            "logit_mse": float(logit_mse(teacher_logits, student_logits).item()),
            "agree": float(agreement_rate(teacher_logits, student_logits).item()),
        }


def probability_mse(teacher_logits: torch.Tensor, student_logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """MSE between teacher and student probabilities (post-softmax)."""
    pt = softmax_with_temperature(teacher_logits.detach(), T)
    ps = softmax_with_temperature(student_logits, T)
    return F.mse_loss(ps, pt)
