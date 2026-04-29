from typing import Iterable, List
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def parameters_to_prune(model: nn.Module, param_types: Iterable[str] = ("weight",)) -> List:
    modules = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            for ptype in param_types:
                if hasattr(module, ptype):
                    modules.append((module, ptype))
    return modules


def set_global_sparsity(model: nn.Module, amount: float, param_types: Iterable[str] = ("weight",)):
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0:
        return
    params = parameters_to_prune(model, param_types)
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def set_global_sparsity_absolute(model: nn.Module, amount: float, param_types: Iterable[str] = ("weight",)):
    """Set an absolute global sparsity target by resetting pruning reparameterization first.

    This avoids compounding sparsity when pruning is called repeatedly across epochs.
    """
    remove_pruning_reparam(model, param_types)
    set_global_sparsity(model, amount=amount, param_types=param_types)


def remove_pruning_reparam(model: nn.Module, param_types: Iterable[str] = ("weight",)):
    for module, ptype in parameters_to_prune(model, param_types):
        if hasattr(module, f"{ptype}_orig") and hasattr(module, f"{ptype}_mask"):
            prune.remove(module, ptype)


def current_global_sparsity(model: nn.Module, param_types: Iterable[str] = ("weight",)) -> float:
    total = 0
    zeros = 0
    for module, ptype in parameters_to_prune(model, param_types):
        tensor = getattr(module, ptype)
        total += tensor.numel()
        zeros += (tensor == 0).sum().item()
    return zeros / max(1, total)


def rerandomize_pruned_weights(model: nn.Module, param_types: Iterable[str] = ("weight",)):
    with torch.no_grad():
        for module, ptype in parameters_to_prune(model, param_types):
            # If pruning reparam is active, module has <ptype> = <ptype>_orig * <ptype>_mask
            w_attr = f"{ptype}_orig" if hasattr(module, f"{ptype}_orig") else ptype
            m_attr = f"{ptype}_mask" if hasattr(module, f"{ptype}_mask") else None
            w = getattr(module, w_attr)
            if m_attr is not None:
                mask = getattr(module, m_attr)
                # Rerandomize where mask == 0
                std = w.std().item() if w.numel() > 1 else 0.02
                noise = torch.randn_like(w) * std
                w[mask == 0] = noise[mask == 0]
            else:
                # No mask reparam; treat zeros as pruned
                std = w.std().item() if w.numel() > 1 else 0.02
                noise = torch.randn_like(w) * std
                w[w == 0] = noise[w == 0]
