"""Data types for synthetic graph-manifold experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphSignalSample:
    h0: torch.Tensor
    target_trajectory: torch.Tensor
    labels: torch.Tensor
    times: torch.Tensor
