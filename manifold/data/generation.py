"""Synthetic graph-manifold signal generation."""

from __future__ import annotations

import networkx as nx
import torch

from manifold.utils.config import GraphConfig, ModelConfig
from manifold.data.types import GraphSignalSample


class SyntheticGraphDataset:
    """Reproducible stochastic-block graph with smooth latent trajectories."""

    def __init__(
        self,
        graph_config: GraphConfig,
        model_config: ModelConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.graph_config = graph_config
        self.model_config = model_config
        self.device = torch.device(device)
        self._generator = torch.Generator(device="cpu").manual_seed(graph_config.seed)
        self.edge_index, self.edge_attr, self.labels = self._build_graph()
        self.community_centers = torch.randn(
            graph_config.num_communities,
            model_config.latent_dim,
            generator=self._generator,
            dtype=torch.float32,
        )

    @property
    def num_nodes(self) -> int:
        return self.graph_config.num_nodes

    def to(self, device: torch.device | str) -> "SyntheticGraphDataset":
        self.device = torch.device(device)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.labels = self.labels.to(self.device)
        self.community_centers = self.community_centers.to(self.device)
        return self

    def sample(self, *, steps: int, dt: float) -> GraphSignalSample:
        """Sample a clean target trajectory for recovery."""
        device = self.device
        labels_cpu = self.labels.detach().cpu()
        centers = self.community_centers.to(device)[labels_cpu].to(device)
        h0 = centers + 0.25 * torch.randn(
            self.num_nodes,
            self.model_config.latent_dim,
            generator=self._generator,
            dtype=torch.float32,
        ).to(device)
        times = torch.linspace(
            0.0, steps * dt, steps + 1, dtype=torch.float32, device=device
        )
        trajectory = []
        for time_value in times:
            decay = torch.exp(-1.3 * time_value)
            wave = 0.04 * torch.sin(
                time_value * (labels_cpu.to(device).float().unsqueeze(-1) + 1.0)
            )
            trajectory.append(centers + decay * (h0 - centers) + wave)
        return GraphSignalSample(
            h0=h0,
            target_trajectory=torch.stack(trajectory, dim=0),
            labels=(self.labels % self.model_config.num_classes).to(device),
            times=times,
        )

    def _build_graph(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.graph_config
        sizes = _balanced_sizes(cfg.num_nodes, cfg.num_communities)
        probs = [
            [
                cfg.p_in if row == col else cfg.p_out
                for col in range(cfg.num_communities)
            ]
            for row in range(cfg.num_communities)
        ]
        graph = nx.stochastic_block_model(sizes, probs, seed=cfg.seed)
        graph.add_nodes_from(range(cfg.num_nodes))
        for node in range(cfg.num_nodes):
            graph.add_edge(node, (node + 1) % cfg.num_nodes)

        labels = []
        for community, size in enumerate(sizes):
            labels.extend([community] * size)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        directed_edges: list[tuple[int, int]] = []
        attrs: list[list[float]] = []
        for src, dst in sorted(graph.edges()):
            for left, right in ((src, dst), (dst, src)):
                directed_edges.append((left, right))
                same = float(label_tensor[left] == label_tensor[right])
                friction = 0.04 if same else 0.18
                distortion = torch.rand((), generator=self._generator).item() * (
                    0.05 if same else 0.16
                )
                community_gap = abs(
                    int(label_tensor[left]) - int(label_tensor[right])
                ) / max(1, cfg.num_communities - 1)
                attrs.append([same, friction, distortion, community_gap])

        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
        if cfg.edge_attr_dim != 4:
            if cfg.edge_attr_dim < 4:
                edge_attr = edge_attr[:, : cfg.edge_attr_dim]
            else:
                pad = torch.zeros(
                    edge_attr.size(0), cfg.edge_attr_dim - 4, dtype=edge_attr.dtype
                )
                edge_attr = torch.cat([edge_attr, pad], dim=-1)
        return edge_index, edge_attr, label_tensor


def _balanced_sizes(total: int, groups: int) -> list[int]:
    base = total // groups
    remainder = total % groups
    return [base + (1 if idx < remainder else 0) for idx in range(groups)]
