"""Policy modules for manifold training."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

BaselinePolicy = Literal["random", "greedy", "neural"]


class SourcePolicy(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, *, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(h))


class ControlPolicy(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        control_dim: int,
        *,
        residual_scale: float,
        analytic_gain: float,
        community_gain: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.residual_scale = residual_scale
        self.analytic_gain = analytic_gain
        self.community_gain = community_gain
        feature_dim = 4 * latent_dim + 2
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gate_head = nn.Linear(hidden_dim, latent_dim)
        self.residual_head = nn.Linear(hidden_dim, control_dim)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -8.0)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        h: torch.Tensor,
        analytic_control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trunk = self.trunk(features)
        gate = torch.sigmoid(self.gate_head(trunk))
        residual = self.residual_scale * torch.tanh(self.residual_head(trunk))
        lap = features[:, self.latent_dim : 2 * self.latent_dim]
        damping = torch.zeros(
            h.size(0), self.control_dim, dtype=h.dtype, device=h.device
        )
        shared_dim = min(self.latent_dim, self.control_dim)
        damping[:, :shared_dim] = -gate[:, :shared_dim] * lap[:, :shared_dim]
        if analytic_control is None:
            analytic_control = torch.zeros_like(damping)
        return analytic_control + damping + residual
