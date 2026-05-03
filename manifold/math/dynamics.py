"""Continuous-time neural sheaf dynamics."""

from __future__ import annotations

import torch
from torch import nn
from torchdiffeq import odeint

from manifold.math.sheaf import SparseSheafOperator


class GNNMessagePassingBlock(nn.Module):
    """Mean-aggregating message passing block for latent node dynamics."""

    def __init__(self, latent_dim: int, edge_attr_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim + edge_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index.to(h.device)
        edge_attr = edge_attr.to(h.device)
        messages = self.message_mlp(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        degree = torch.zeros(h.size(0), dtype=h.dtype, device=h.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        agg = agg / degree.clamp_min(1.0).unsqueeze(-1)
        return self.update_mlp(torch.cat([h, agg], dim=-1))


class NeuralSheafODE(nn.Module):
    """Neural ODE with GNN drift, sheaf consistency, and control input."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        edge_attr_dim: int = 4,
        hidden_dim: int = 64,
        control_dim: int = 16,
        sheaf_lambda: float = 0.15,
        restriction_scale: float = 0.08,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.sheaf_lambda = sheaf_lambda
        self.gnn = GNNMessagePassingBlock(latent_dim, edge_attr_dim, hidden_dim)
        self.sheaf = SparseSheafOperator(
            latent_dim,
            edge_attr_dim,
            hidden_dim=hidden_dim,
            restriction_scale=restriction_scale,
        )
        self.control_projection = nn.Linear(control_dim, latent_dim)
        self._init_control_projection()
        self._edge_index: torch.Tensor | None = None
        self._edge_attr: torch.Tensor | None = None
        self._control: torch.Tensor | None = None

    def _init_control_projection(self) -> None:
        nn.init.zeros_(self.control_projection.weight)
        nn.init.zeros_(self.control_projection.bias)
        shared_dim = min(self.control_dim, self.latent_dim)
        with torch.no_grad():
            self.control_projection.weight[:shared_dim, :shared_dim] = torch.eye(
                shared_dim
            )

    def set_context(
        self,
        *,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        control: torch.Tensor | None = None,
    ) -> None:
        self._edge_index = edge_index
        self._edge_attr = edge_attr
        self._control = control

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        del t
        if self._edge_index is None or self._edge_attr is None:
            raise RuntimeError(
                "NeuralSheafODE context is missing edge_index or edge_attr"
            )
        edge_index = self._edge_index.to(h.device)
        edge_attr = self._edge_attr.to(h.device)
        drift = self.gnn(h, edge_index, edge_attr)
        sheaf_term, _ = self.sheaf.apply_laplacian(h, edge_index, edge_attr)
        control = self._control
        if control is None:
            control_effect = torch.zeros_like(h)
        else:
            control_effect = self.control_projection(control.to(h.device))
        return drift - self.sheaf_lambda * sheaf_term + control_effect

    def rollout(
        self,
        h0: torch.Tensor,
        times: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        control: torch.Tensor | None = None,
        step_size: float | None = None,
    ) -> torch.Tensor:
        """Integrate with fixed-step RK4."""
        self.set_context(edge_index=edge_index, edge_attr=edge_attr, control=control)
        options = {"step_size": step_size} if step_size is not None else None
        return odeint(self, h0, times.to(h0.device), method="rk4", options=options)
