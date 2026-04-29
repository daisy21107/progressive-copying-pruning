"""Representation similarity metrics (CKA, SVCCA placeholder)."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K_X = X @ X.T
    K_Y = Y @ Y.T

    hsic_xy = torch.trace(K_X @ K_Y)
    hsic_xx = torch.trace(K_X @ K_X)
    hsic_yy = torch.trace(K_Y @ K_Y)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if float(denom.item()) == 0.0:
        return 0.0
    return float((hsic_xy / denom).item())


def get_layer_activations(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    layer_name: str,
    max_samples: int = 1000,
) -> torch.Tensor:
    """Extract flattened activations from a layer over a dataloader."""
    activations: List[torch.Tensor] = []
    samples_collected = 0

    def hook_fn(_module, _inputs, output):
        act = output.detach().cpu()
        if len(act.shape) == 4:
            act = act.mean(dim=[2, 3])
        elif len(act.shape) == 3:
            act = act.mean(dim=1)
        activations.append(act)

    layers = dict(model.named_modules())
    if layer_name not in layers:
        raise KeyError(f"Layer not found: {layer_name}")
    handle = layers[layer_name].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for x1, x2, _ in dataloader:
            if samples_collected >= max_samples:
                break
            x1, x2 = x1.to(device), x2.to(device)
            _ = model(x1, x2)
            samples_collected += x1.shape[0]

    handle.remove()

    if not activations:
        raise RuntimeError(f"No activations captured for layer: {layer_name}")

    return torch.cat(activations, dim=0)[:max_samples]


def compute_cka_similarity(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader,
    device: torch.device,
    layer_pairs: List[Tuple[str, str]] | None = None,
    max_samples: int = 1000,
) -> Dict[str, float | None]:
    """Compute layer-wise and average linear CKA between student and teacher."""
    if layer_pairs is None:
        layer_pairs = [
            ("enc1.conv1", "enc1.conv1"),
            ("enc1.conv2", "enc1.conv2"),
            ("enc1.conv3", "enc1.conv3"),
            ("enc1.conv4", "enc1.conv4"),
            ("fc1", "fc1"),
        ]

    similarities: Dict[str, float | None] = {}

    for student_layer, teacher_layer in tqdm(layer_pairs, desc="CKA"):
        key = f"{student_layer}__{teacher_layer}"
        try:
            student_acts = get_layer_activations(student, dataloader, device, student_layer, max_samples)
            teacher_acts = get_layer_activations(teacher, dataloader, device, teacher_layer, max_samples)
            cka_score = linear_CKA(student_acts, teacher_acts)
            similarities[key] = cka_score
        except Exception:
            similarities[key] = None

    valid = [v for v in similarities.values() if v is not None]
    similarities["average"] = float(np.mean(valid)) if valid else None
    return similarities


def cka_stub():
    return None


def svcca_stub():
    return None
