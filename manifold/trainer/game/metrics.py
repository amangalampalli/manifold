"""Training and evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def perturbation_auc(
    pred: torch.Tensor, target: torch.Tensor, times: torch.Tensor
) -> torch.Tensor:
    """Area under the mean perturbation magnitude curve."""
    perturbation = torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=-1)
    if perturbation.numel() <= 1:
        return perturbation.sum() * 0.0
    return torch.trapz(perturbation, times.to(pred.device))


def post_control_perturbation_auc(
    pred: torch.Tensor, target: torch.Tensor, times: torch.Tensor
) -> torch.Tensor:
    """Perturbation AUC after the initial corrupted state."""
    if pred.size(0) <= 2:
        return perturbation_auc(pred, target, times)
    return perturbation_auc(pred[1:], target[1:], times[1:])


def trajectory_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def belief_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)
