"""Sparse neural sheaf operators."""

from __future__ import annotations

import torch
from torch import nn

from manifold.utils.devices import sparse_mm


class RestrictionMapLayer(nn.Module):
    """Edge-conditioned linear restriction maps for sheaf coboundaries."""

    def __init__(
        self,
        latent_dim: int,
        edge_attr_dim: int,
        *,
        hidden_dim: int = 64,
        restriction_scale: float = 0.08,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.restriction_scale = restriction_scale
        self.net = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * latent_dim * latent_dim),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_count = edge_attr.size(0)
        raw = self.net(edge_attr).view(edge_count, 2, self.latent_dim, self.latent_dim)
        eye = torch.eye(self.latent_dim, dtype=edge_attr.dtype, device=edge_attr.device)
        maps = eye.view(
            1, 1, self.latent_dim, self.latent_dim
        ) + self.restriction_scale * torch.tanh(raw)
        return maps[:, 0], maps[:, 1]


class SparseSheafOperator(nn.Module):
    """Build and apply a sparse sheaf Laplacian as delta^T delta."""

    def __init__(
        self,
        latent_dim: int,
        edge_attr_dim: int,
        *,
        hidden_dim: int = 64,
        restriction_scale: float = 0.08,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.restriction_map_layer = RestrictionMapLayer(
            latent_dim,
            edge_attr_dim,
            hidden_dim=hidden_dim,
            restriction_scale=restriction_scale,
        )

    def build_coboundary(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        """Construct sparse coboundary delta with shape [E*d, N*d]."""
        device = edge_attr.device
        dtype = edge_attr.dtype
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_count = edge_index.size(1)
        d = self.latent_dim
        if edge_count == 0:
            empty = torch.empty(2, 0, dtype=torch.long, device=device)
            values = torch.empty(0, dtype=dtype, device=device)
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    empty, values, (0, num_nodes * d), device=device
                ).coalesce()

        left_maps, right_maps = self.restriction_map_layer(edge_attr)
        rows, cols_left, cols_right = _block_indices(edge_index, num_nodes, d, device)
        indices = torch.cat(
            [
                torch.stack([rows, cols_left], dim=0),
                torch.stack([rows, cols_right], dim=0),
            ],
            dim=1,
        )
        values = torch.cat([left_maps.reshape(-1), -right_maps.reshape(-1)], dim=0)
        with torch.sparse.check_sparse_tensor_invariants(False):
            return torch.sparse_coo_tensor(
                indices,
                values,
                (edge_count * d, num_nodes * d),
                device=device,
                dtype=dtype,
            ).coalesce()

    def apply_laplacian(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply L_s h as delta^T delta h without materializing dense L_s."""
        num_nodes = h.size(0)
        delta = self.build_coboundary(edge_index, edge_attr, num_nodes=num_nodes)
        flat_h = h.reshape(num_nodes * self.latent_dim, 1)
        residual = sparse_mm(delta, flat_h)
        lap = sparse_mm(delta.transpose(0, 1), residual)
        return lap.reshape_as(h), residual.reshape(edge_index.size(1), self.latent_dim)


def _block_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del num_nodes
    edge_count = edge_index.size(1)
    matrix_entries = latent_dim * latent_dim
    row_offsets = (
        torch.arange(edge_count, device=device) * latent_dim
    ).repeat_interleave(matrix_entries)
    row_inside = (
        torch.arange(latent_dim, device=device)
        .repeat_interleave(latent_dim)
        .repeat(edge_count)
    )
    col_inside = (
        torch.arange(latent_dim, device=device).repeat(latent_dim).repeat(edge_count)
    )
    rows = row_offsets + row_inside
    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    cols_left = src.repeat_interleave(matrix_entries) * latent_dim + col_inside
    cols_right = dst.repeat_interleave(matrix_entries) * latent_dim + col_inside
    return rows, cols_left, cols_right
