from typing import Tuple

import torch


def unpack_batch(batch) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    if len(batch) == 2:
        x, y = batch
        return (x,), y
    if len(batch) == 3:
        x1, x2, y = batch
        return (x1, x2), y
    raise ValueError(f"Unsupported batch structure with {len(batch)} items")


def forward_logits(model, inputs: Tuple[torch.Tensor, ...]):
    if len(inputs) == 1:
        return model(inputs[0])
    return model(*inputs)