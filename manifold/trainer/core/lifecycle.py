"""Trainer lifecycle and epoch loop."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from manifold.utils.config import ExperimentConfig
from manifold.data.generation import SyntheticGraphDataset
from manifold.utils.devices import resolve_device, sparse_backend_status
from manifold.math.dynamics import NeuralSheafODE
from manifold.trainer.game.sensing import ActiveSensingBeliefUpdater
from manifold.trainer.support.policies import ControlPolicy, SourcePolicy
from manifold.trainer.support.utils import TrainingHistory


class TrainerLifecycleMixin:
    def __init__(
        self, config: ExperimentConfig, *, device: torch.device | None = None
    ) -> None:
        self.config = config
        self.device = device or resolve_device(config.device)
        torch.manual_seed(config.seed)

        self.dataset = SyntheticGraphDataset(
            config.graph, config.model, device=self.device
        ).to(self.device)
        self.model = NeuralSheafODE(
            latent_dim=config.model.latent_dim,
            edge_attr_dim=config.graph.edge_attr_dim,
            hidden_dim=config.model.hidden_dim,
            control_dim=config.model.control_dim,
            sheaf_lambda=config.model.sheaf_lambda,
            restriction_scale=config.model.restriction_scale,
        ).to(self.device)
        self.sensing = ActiveSensingBeliefUpdater(
            latent_dim=config.model.latent_dim,
            hidden_dim=config.model.hidden_dim,
            k_hop=config.sensing.k_hop,
            budget=config.sensing.budget,
            noise_std=config.sensing.noise_std,
        ).to(self.device)
        self.source_policy = SourcePolicy(
            config.model.latent_dim,
            config.model.hidden_dim,
            scale=config.training.source_scale,
        ).to(self.device)
        self.control_policy = ControlPolicy(
            config.model.latent_dim,
            config.model.hidden_dim,
            config.model.control_dim,
            residual_scale=config.training.controller_residual_scale,
            analytic_gain=config.training.analytic_sheaf_gain,
            community_gain=config.training.analytic_community_gain,
        ).to(self.device)
        self.classifier = torch.nn.Linear(
            config.model.latent_dim, config.model.num_classes
        ).to(self.device)

        controller_parameters = list(self.model.parameters())
        controller_parameters += list(self.sensing.parameters())
        controller_parameters += list(self.control_policy.parameters())
        controller_parameters += list(self.classifier.parameters())
        self.controller_optimizer = torch.optim.Adam(
            controller_parameters, lr=config.training.controller_lr
        )
        self.source_optimizer = torch.optim.Adam(
            self.source_policy.parameters(), lr=config.training.source_lr
        )
        self.history = TrainingHistory()
        self._cached_boundary_nodes: torch.Tensor | None = None

    def startup_report(self) -> dict[str, object]:
        sparse_status = sparse_backend_status(self.device)
        return {
            "device": str(self.device),
            "mps_available": torch.backends.mps.is_available(),
            "sparse_mm_supported": sparse_status.sparse_mm_supported,
            "sparse_backend_message": sparse_status.message,
            "seed": self.config.seed,
            "num_nodes": self.config.graph.num_nodes,
            "num_edges": int(self.dataset.edge_index.size(1)),
            "rk4_steps": self.config.training.trajectory_steps,
        }

    def train(self, *, progress: bool = False) -> TrainingHistory:
        epochs = range(1, self.config.training.epochs + 1)
        iterator = tqdm(epochs, desc="training", unit="epoch", disable=not progress)
        for epoch in iterator:
            metrics = self.train_epoch(epoch)
            self.history.metrics.append(metrics)
            if progress:
                iterator.set_postfix(
                    auc=f"{metrics['perturbation_auc']:.4f}",
                    source_auc=f"{metrics['source_auc']:.4f}",
                    mse=f"{metrics['trajectory_mse']:.4f}",
                )
        return self.history

    def train_epoch(self, epoch: int) -> dict[str, float]:
        source_loss = torch.tensor(0.0, device=self.device)
        source_auc = torch.tensor(0.0, device=self.device)
        boundary_nodes_used = torch.tensor(0.0, device=self.device)
        perturbation_norm = torch.tensor(0.0, device=self.device)
        source_frozen = self._source_frozen(epoch)
        if source_frozen:
            source_auc, boundary_nodes_used, perturbation_norm = self._source_eval()
        else:
            for _ in range(self.config.training.source_steps):
                source_loss, source_auc, boundary_nodes_used, perturbation_norm = (
                    self._source_step()
                )

        controller_loss = torch.tensor(0.0, device=self.device)
        auc = torch.tensor(0.0, device=self.device)
        post_control_auc = torch.tensor(0.0, device=self.device)
        mse = torch.tensor(0.0, device=self.device)
        final_belief_mse = torch.tensor(0.0, device=self.device)
        ce = torch.tensor(0.0, device=self.device)
        control_energy = torch.tensor(0.0, device=self.device)
        control_target_mse = torch.tensor(0.0, device=self.device)
        observed_fraction = torch.tensor(0.0, device=self.device)
        for _ in range(self.config.training.controller_steps):
            (
                controller_loss,
                auc,
                post_control_auc,
                mse,
                final_belief_mse,
                ce,
                control_energy,
                control_target_mse,
                observed_fraction,
            ) = self._controller_step()

        return {
            "epoch": float(epoch),
            "source_loss": float(source_loss.detach().cpu()),
            "source_auc": float(source_auc.detach().cpu()),
            "controller_loss": float(controller_loss.detach().cpu()),
            "perturbation_auc": float(auc.detach().cpu()),
            "post_control_perturbation_auc": float(post_control_auc.detach().cpu()),
            "trajectory_mse": float(mse.detach().cpu()),
            "final_belief_mse": float(final_belief_mse.detach().cpu()),
            "belief_ce": float(ce.detach().cpu()),
            "control_energy": float(control_energy.detach().cpu()),
            "control_target_mse": float(control_target_mse.detach().cpu()),
            "boundary_nodes_used": float(boundary_nodes_used.detach().cpu()),
            "perturbation_norm": float(perturbation_norm.detach().cpu()),
            "observed_fraction": float(observed_fraction.detach().cpu()),
            "source_frozen": float(source_frozen),
        }

    def iter_metrics(self):
        yield from self.history.metrics
