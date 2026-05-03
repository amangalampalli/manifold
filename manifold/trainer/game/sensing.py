"""Active sparse sensing and belief correction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BeliefUpdate:
    h_corr: torch.Tensor
    probe_nodes: torch.Tensor
    observed_mask: torch.Tensor


class ActiveSensingBeliefUpdater(nn.Module):
    """Policy-directed k-hop sensing with a learned gain correction."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int = 64,
        k_hop: int = 2,
        budget: int = 8,
        noise_std: float = 0.03,
    ) -> None:
        super().__init__()
        self.k_hop = k_hop
        self.budget = budget
        self.noise_std = noise_std
        self.probe_policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gain = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.gain.weight)
        self.gain.weight.data.mul_(0.2)

    def select_probe_nodes(self, h_pred: torch.Tensor) -> torch.Tensor:
        count = min(self.budget, h_pred.size(0))
        scores = self.probe_policy(h_pred).squeeze(-1)
        return torch.topk(scores, k=count).indices

    def observation_mask(
        self,
        edge_index: torch.Tensor,
        *,
        num_nodes: int,
        probe_nodes: torch.Tensor,
    ) -> torch.Tensor:
        return k_hop_observation_mask(
            edge_index,
            num_nodes=num_nodes,
            probe_nodes=probe_nodes,
            k_hop=self.k_hop,
            device=probe_nodes.device,
        )

    def forward(
        self,
        h_pred: torch.Tensor,
        y_true: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> BeliefUpdate:
        probe_nodes = self.select_probe_nodes(h_pred)
        observed_mask = self.observation_mask(
            edge_index.to(h_pred.device),
            num_nodes=h_pred.size(0),
            probe_nodes=probe_nodes,
        )
        noise = (
            self.noise_std * torch.randn_like(y_true)
            if self.noise_std > 0
            else torch.zeros_like(y_true)
        )
        y = y_true + noise
        innovation = torch.zeros_like(h_pred)
        innovation[observed_mask] = y[observed_mask] - h_pred[observed_mask]
        h_corr = h_pred + self.gain(innovation)
        return BeliefUpdate(
            h_corr=h_corr, probe_nodes=probe_nodes, observed_mask=observed_mask
        )


def k_hop_observation_mask(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    probe_nodes: torch.Tensor,
    k_hop: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a node visibility mask for probe-centered k-hop neighborhoods."""
    edge_cpu = edge_index.detach().cpu()
    adjacency = [set() for _ in range(num_nodes)]
    for src, dst in edge_cpu.t().tolist():
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    visible = set(int(node) for node in probe_nodes.detach().cpu().tolist())
    frontier = set(visible)
    for _ in range(k_hop):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(adjacency[node])
        next_frontier -= visible
        visible.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    if visible:
        mask[torch.tensor(sorted(visible), dtype=torch.long)] = True
    return mask.to(device)
