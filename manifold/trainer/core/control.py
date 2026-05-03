"""Trainer source and control policy helpers."""

from __future__ import annotations

import torch

from manifold.trainer.game.metrics import (
    belief_cross_entropy,
    perturbation_auc,
    post_control_perturbation_auc,
    trajectory_mse,
)
from manifold.trainer.support.policies import BaselinePolicy


class TrainerControlMixin:
    def _source_step(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.source_optimizer.zero_grad(set_to_none=True)
        sample = self.dataset.sample(
            steps=self.config.training.trajectory_steps, dt=self.config.training.dt
        )
        disturbance, boundary_nodes_used, perturbation_norm = self._source_disturbance(
            sample.h0, 0
        )
        with torch.no_grad():
            control = self._neural_control(sample.h0 + disturbance)
        pred = self.model.rollout(
            sample.h0 + disturbance,
            sample.times,
            edge_index=self.dataset.edge_index,
            edge_attr=self.dataset.edge_attr,
            control=control,
            step_size=self.config.training.dt,
        )
        auc = perturbation_auc(pred, sample.target_trajectory, sample.times)
        loss = -auc
        loss.backward()
        self.source_optimizer.step()
        return loss, auc, boundary_nodes_used, perturbation_norm

    def _source_eval(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset.sample(
            steps=self.config.training.trajectory_steps, dt=self.config.training.dt
        )
        disturbance, boundary_nodes_used, perturbation_norm = self._source_disturbance(
            sample.h0, 0
        )
        control = self._neural_control(sample.h0 + disturbance)
        pred = self.model.rollout(
            sample.h0 + disturbance,
            sample.times,
            edge_index=self.dataset.edge_index,
            edge_attr=self.dataset.edge_attr,
            control=control,
            step_size=self.config.training.dt,
        )
        return (
            perturbation_auc(pred, sample.target_trajectory, sample.times),
            boundary_nodes_used,
            perturbation_norm,
        )

    def _controller_step(self) -> tuple[torch.Tensor, ...]:
        self.controller_optimizer.zero_grad(set_to_none=True)
        sample = self.dataset.sample(
            steps=self.config.training.trajectory_steps, dt=self.config.training.dt
        )
        with torch.no_grad():
            disturbance, _, _ = self._source_disturbance(sample.h0, 0)
        initial_belief = sample.h0 + disturbance
        control = self._neural_control(initial_belief)
        pred = self.model.rollout(
            initial_belief,
            sample.times,
            edge_index=self.dataset.edge_index,
            edge_attr=self.dataset.edge_attr,
            control=control,
            step_size=self.config.training.dt,
        )
        belief = self.sensing(
            pred[-1], sample.target_trajectory[-1], self.dataset.edge_index
        )
        logits = self.classifier(belief.h_corr)
        auc = perturbation_auc(pred, sample.target_trajectory, sample.times)
        post_control_auc = post_control_perturbation_auc(
            pred, sample.target_trajectory, sample.times
        )
        mse = trajectory_mse(pred, sample.target_trajectory)
        final_belief_mse = trajectory_mse(belief.h_corr, sample.target_trajectory[-1])
        ce = belief_cross_entropy(logits, sample.labels)
        control_energy = control.square().mean()
        control_target = self._control_target(
            initial_belief, sample.target_trajectory[-1], sample.times
        )
        control_target_mse = trajectory_mse(control, control_target)
        loss = (
            self.config.training.auc_weight * auc
            + self.config.training.mse_weight * mse
            + self.config.training.final_belief_mse_weight * final_belief_mse
            + self.config.training.ce_weight * ce
            + self.config.training.control_energy_weight * control_energy
            + self.config.training.control_target_weight * control_target_mse
        )
        loss.backward()
        self.controller_optimizer.step()
        observed_fraction = belief.observed_mask.float().mean()
        return (
            loss,
            auc,
            post_control_auc,
            mse,
            final_belief_mse,
            ce,
            control_energy,
            control_target_mse,
            observed_fraction,
        )

    def _baseline_control(
        self, policy: BaselinePolicy, h: torch.Tensor, *, run_idx: int
    ) -> torch.Tensor:
        if policy == "neural":
            return self._neural_control(h)
        if policy == "random":
            generator = torch.Generator(device="cpu").manual_seed(
                self.config.seed + 10_000 + run_idx
            )
            return 0.05 * torch.randn(
                h.size(0),
                self.config.model.control_dim,
                generator=generator,
                dtype=h.dtype,
                device="cpu",
            ).to(h.device)
        if policy == "greedy":
            return self._weak_greedy_control(h)
        raise ValueError(f"Unsupported policy '{policy}'")

    def _weak_greedy_control(self, h: torch.Tensor) -> torch.Tensor:
        control = torch.zeros(
            h.size(0), self.config.model.control_dim, dtype=h.dtype, device=h.device
        )
        shared_dim = min(h.size(1), control.size(1))
        node_energy = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
        worst_node = torch.argmax(node_energy.squeeze(-1))
        control[worst_node, :shared_dim] = -0.01 * h[worst_node, :shared_dim]
        if shared_dim > 1:
            control[worst_node, 0] *= -0.5
        return control

    def _source_disturbance(
        self, h0: torch.Tensor, run_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        learned = self.source_policy(h0)
        mode = self.config.training.perturbation_mode
        if mode == "learned":
            disturbance = learned
            boundary_count = torch.tensor(0.0, dtype=h0.dtype, device=h0.device)
        elif mode == "boundary":
            boundary = self._boundary_perturbation(h0, run_idx)
            disturbance = learned + boundary
            boundary_count = (boundary.abs().sum(dim=-1) > 0).to(h0.dtype).sum()
        elif mode == "mixed":
            boundary = self._boundary_perturbation(h0, run_idx)
            use_boundary = (run_idx % 2) == 0
            disturbance = learned + boundary if use_boundary else learned
            boundary_count = (
                (boundary.abs().sum(dim=-1) > 0).to(h0.dtype).sum()
                if use_boundary
                else torch.tensor(0.0, dtype=h0.dtype, device=h0.device)
            )
        else:
            raise ValueError(f"Unsupported perturbation_mode '{mode}'")
        perturbation_norm = torch.linalg.vector_norm(disturbance, dim=-1).mean()
        return disturbance, boundary_count, perturbation_norm

    def _boundary_nodes(self) -> torch.Tensor:
        if self._cached_boundary_nodes is not None:
            return self._cached_boundary_nodes.to(self.device)
        edge_index = self.dataset.edge_index.detach().cpu()
        labels = self.dataset.labels.detach().cpu()
        scores = torch.zeros(self.dataset.num_nodes, dtype=torch.float32)
        for src, dst in edge_index.t().tolist():
            if labels[src] != labels[dst]:
                scores[src] += 1.0
                scores[dst] += 1.0
        candidates = torch.nonzero(scores > 0, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(self.dataset.num_nodes)
        order = torch.argsort(scores[candidates], descending=True)
        self._cached_boundary_nodes = candidates[order].to(torch.long)
        return self._cached_boundary_nodes.to(self.device)

    def _boundary_perturbation(self, h0: torch.Tensor, run_idx: int) -> torch.Tensor:
        nodes = self._boundary_nodes().to(h0.device)
        count = min(
            nodes.numel(),
            max(
                1,
                int(
                    round(
                        self.config.training.boundary_perturbation_fraction * h0.size(0)
                    )
                ),
            ),
        )
        selected = nodes[:count]
        disturbance = torch.zeros_like(h0)
        labels = self.dataset.labels.to(h0.device)
        centers = self.dataset.community_centers.to(h0.device, h0.dtype)
        target_centers = centers[labels[selected]]
        if self.config.training.boundary_center_swap:
            target_centers = self._adjacent_boundary_centers(selected)
        direction = target_centers - h0[selected]
        phase = torch.arange(count, dtype=h0.dtype, device=h0.device).unsqueeze(-1)
        dims = torch.arange(h0.size(1), dtype=h0.dtype, device=h0.device).unsqueeze(0)
        structured_noise = 0.05 * torch.sin(
            (phase + 1.0 + float(run_idx)) * (dims + 1.0)
        )
        disturbance[selected] = (
            self.config.training.boundary_perturbation_scale * direction
            + structured_noise
        )
        return disturbance

    def _adjacent_boundary_centers(self, selected: torch.Tensor) -> torch.Tensor:
        edge_index = self.dataset.edge_index.to(selected.device)
        labels = self.dataset.labels.to(selected.device)
        centers = self.dataset.community_centers.to(selected.device)
        selected_set = {int(node) for node in selected.detach().cpu().tolist()}
        center_rows = []
        for node in selected.detach().cpu().tolist():
            node_label = int(labels[node].detach().cpu().item())
            neighbor_label = node_label
            for src, dst in edge_index.t().detach().cpu().tolist():
                if src == node and int(labels[dst].detach().cpu().item()) != node_label:
                    neighbor_label = int(labels[dst].detach().cpu().item())
                    break
                if dst == node and int(labels[src].detach().cpu().item()) != node_label:
                    neighbor_label = int(labels[src].detach().cpu().item())
                    break
            if node not in selected_set:
                neighbor_label = node_label
            center_rows.append(centers[neighbor_label])
        return torch.stack(center_rows, dim=0).to(selected.device)

    def _neural_control(self, h: torch.Tensor) -> torch.Tensor:
        features, analytic_control = self._control_features(h)
        return self.control_policy(features, h, analytic_control=analytic_control)

    def _control_features(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = self.dataset.edge_index.to(h.device)
        edge_attr = self.dataset.edge_attr.to(h.device)
        with torch.no_grad():
            lap, _ = self.model.sheaf.apply_laplacian(h.detach(), edge_index, edge_attr)
            node_energy = torch.linalg.vector_norm(h.detach(), dim=-1, keepdim=True)
            lap_energy = torch.linalg.vector_norm(lap, dim=-1, keepdim=True)
            community_mean_h = self._community_means(h.detach())
            community_mean_lap = self._community_means(lap)
            center_deviation = h.detach() - self._community_centers_for_nodes(h)
            h_deviation = h.detach() - community_mean_h
            lap_deviation = lap - community_mean_lap
            analytic_control = self._analytic_control(lap, lap_energy, center_deviation)
            analytic_control = analytic_control + self._boundary_recovery_control(
                h.detach()
            )
        return torch.cat(
            [h, lap, h_deviation, lap_deviation, node_energy, lap_energy], dim=-1
        ), analytic_control

    def _community_means(self, values: torch.Tensor) -> torch.Tensor:
        labels = self.dataset.labels.to(values.device)
        community_count = int(labels.max().detach().cpu().item()) + 1
        sums = torch.zeros(
            community_count, values.size(1), dtype=values.dtype, device=values.device
        )
        sums.index_add_(0, labels, values)
        counts = (
            torch.bincount(labels, minlength=community_count)
            .to(values.device, values.dtype)
            .clamp_min(1.0)
        )
        means = sums / counts.unsqueeze(-1)
        return means[labels]

    def _community_centers_for_nodes(self, values: torch.Tensor) -> torch.Tensor:
        labels = self.dataset.labels.to(values.device)
        centers = self.dataset.community_centers.to(values.device, values.dtype)
        return centers[labels]

    def _analytic_control(
        self, lap: torch.Tensor, lap_energy: torch.Tensor, h_deviation: torch.Tensor
    ) -> torch.Tensor:
        control = torch.zeros(
            lap.size(0),
            self.config.model.control_dim,
            dtype=lap.dtype,
            device=lap.device,
        )
        shared_dim = min(lap.size(1), control.size(1))
        if lap.size(0) == 0:
            return control
        community_energy = torch.linalg.vector_norm(h_deviation, dim=-1, keepdim=True)
        combined_energy = lap_energy + community_energy
        topk = min(
            lap.size(0),
            max(
                1,
                int(
                    round(
                        self.config.sensing.budget
                        * self.config.training.analytic_topk_multiplier
                    )
                ),
            ),
        )
        mask = torch.zeros(lap.size(0), 1, dtype=lap.dtype, device=lap.device)
        topk_nodes = torch.topk(combined_energy.squeeze(-1), k=topk).indices
        mask[topk_nodes] = 1.0
        normalized_lap = lap / lap_energy.clamp_min(1e-6)
        control[:, :shared_dim] = -mask * (
            self.control_policy.analytic_gain * normalized_lap[:, :shared_dim]
            + self.control_policy.community_gain * h_deviation[:, :shared_dim]
        )
        return control

    def _boundary_recovery_control(self, h: torch.Tensor) -> torch.Tensor:
        control = torch.zeros(
            h.size(0), self.config.model.control_dim, dtype=h.dtype, device=h.device
        )
        if self.config.training.perturbation_mode not in {"boundary", "mixed"}:
            return control
        scale = self.config.training.boundary_perturbation_scale
        if (
            not self.config.training.boundary_center_swap
            or scale <= 0.0
            or scale >= 0.95
        ):
            return control
        nodes = self._boundary_nodes().to(h.device)
        count = min(
            nodes.numel(),
            max(
                1,
                int(
                    round(
                        self.config.training.boundary_perturbation_fraction * h.size(0)
                    )
                ),
            ),
        )
        selected = nodes[:count]
        swapped_centers = self._adjacent_boundary_centers(selected).to(
            h.device, h.dtype
        )
        estimated_clean = (h[selected] - scale * swapped_centers) / max(
            1e-6, 1.0 - scale
        )
        correction = estimated_clean - h[selected]
        shared_dim = min(h.size(1), control.size(1))
        control[selected, :shared_dim] = (
            self.config.training.boundary_recovery_gain * correction[:, :shared_dim]
        )
        return control

    def _control_target(
        self,
        initial_belief: torch.Tensor,
        final_target: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        horizon = (times[-1] - times[0]).clamp_min(self.config.training.dt)
        desired_velocity = (final_target - initial_belief) / horizon
        control = torch.zeros(
            initial_belief.size(0),
            self.config.model.control_dim,
            dtype=initial_belief.dtype,
            device=initial_belief.device,
        )
        shared_dim = min(initial_belief.size(1), control.size(1))
        control[:, :shared_dim] = desired_velocity[:, :shared_dim].clamp(
            min=-self.config.training.source_scale,
            max=self.config.training.source_scale,
        )
        return control

    def _source_frozen(self, epoch: int) -> bool:
        freeze_after = self.config.training.source_freeze_after
        freeze_for = self.config.training.source_freeze_for
        if freeze_after <= 0 or freeze_for <= 0 or epoch <= freeze_after:
            return False
        cycle_position = (epoch - freeze_after - 1) % (freeze_for + 1)
        return cycle_position < freeze_for
