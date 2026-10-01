from __future__ import annotations

import torch
from torch.nn import functional as F


def ordinal_cdf_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean normalized ranked-probability loss for ordered classes 0..100."""
    if logits.ndim != 2 or logits.shape[1] != 101:
        raise ValueError(f"expected logits shaped [batch, 101], got {tuple(logits.shape)}")
    if target.ndim != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError("target must be a length-batch vector")
    probabilities = logits.softmax(dim=1)
    predicted_cdf = probabilities.cumsum(dim=1)[:, :100]
    thresholds = torch.arange(100, device=target.device).unsqueeze(0)
    target_cdf = (target.unsqueeze(1) <= thresholds).to(logits.dtype)
    return (predicted_cdf - target_cdf).square().mean()


def arithmetic_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ordinal_lambda: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if ordinal_lambda < 0:
        raise ValueError("ordinal_lambda must be non-negative")
    cross_entropy = F.cross_entropy(logits, target)
    ordinal = ordinal_cdf_loss(logits, target)
    total = cross_entropy + ordinal_lambda * ordinal
    parts = {
        "loss": total.detach(),
        "cross_entropy": cross_entropy.detach(),
        "ordinal": ordinal.detach(),
    }
    return total, parts

