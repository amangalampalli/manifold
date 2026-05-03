"""Trainer evaluation, export, and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from manifold.data.types import GraphSignalSample
from manifold.trainer.game.metrics import (
    belief_cross_entropy,
    perturbation_auc,
    post_control_perturbation_auc,
    trajectory_mse,
)
from manifold.trainer.support.policies import BaselinePolicy
from manifold.trainer.support.utils import (
    EvalCase,
    comparison_deltas,
    community_layout,
    eval_cases_from_payload,
    eval_cases_to_payload,
    write_json_file,
)


class TrainerEvaluationMixin:
    @torch.no_grad()
    def evaluate_policy(
        self, policy: BaselinePolicy, *, runs: int = 3, progress: bool = False
    ) -> dict[str, float]:
        if runs <= 0:
            raise ValueError("runs must be positive")

        auc_values: list[torch.Tensor] = []
        post_control_auc_values: list[torch.Tensor] = []
        mse_values: list[torch.Tensor] = []
        ce_values: list[torch.Tensor] = []
        observed_values: list[torch.Tensor] = []
        iterator = tqdm(
            range(runs), desc=f"eval:{policy}", unit="run", disable=not progress
        )
        for run_idx in iterator:
            sample = self.dataset.sample(
                steps=self.config.training.trajectory_steps, dt=self.config.training.dt
            )
            disturbance, _, _ = self._source_disturbance(sample.h0, run_idx)
            initial_belief = sample.h0 + disturbance
            control = self._baseline_control(policy, initial_belief, run_idx=run_idx)
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
            auc_values.append(
                perturbation_auc(pred, sample.target_trajectory, sample.times)
            )
            post_control_auc_values.append(
                post_control_perturbation_auc(
                    pred, sample.target_trajectory, sample.times
                )
            )
            mse_values.append(trajectory_mse(pred, sample.target_trajectory))
            ce_values.append(belief_cross_entropy(logits, sample.labels))
            observed_values.append(belief.observed_mask.float().mean())
            if progress:
                iterator.set_postfix(
                    auc=f"{auc_values[-1].detach().cpu().item():.4f}",
                    mse=f"{mse_values[-1].detach().cpu().item():.4f}",
                )

        return {
            "policy": policy,
            "runs": float(runs),
            "perturbation_auc": float(torch.stack(auc_values).mean().cpu()),
            "post_control_perturbation_auc": float(
                torch.stack(post_control_auc_values).mean().cpu()
            ),
            "trajectory_mse": float(torch.stack(mse_values).mean().cpu()),
            "belief_ce": float(torch.stack(ce_values).mean().cpu()),
            "observed_fraction": float(torch.stack(observed_values).mean().cpu()),
        }

    @torch.no_grad()
    def compare_policies(
        self,
        policies: tuple[BaselinePolicy, ...] = ("random", "greedy", "neural"),
        *,
        runs: int = 3,
        eval_cases: list[EvalCase] | None = None,
        progress: bool = False,
    ) -> list[dict[str, float]]:
        if runs <= 0:
            raise ValueError("runs must be positive")
        if eval_cases is not None:
            runs = len(eval_cases)
            if runs <= 0:
                raise ValueError("eval_cases must not be empty")

        aggregates = {
            policy: {
                "auc": [],
                "post_control_auc": [],
                "mse": [],
                "ce": [],
                "observed": [],
                "boundary": [],
                "perturbation_norm": [],
            }
            for policy in policies
        }
        cases = eval_cases if eval_cases is not None else self.make_eval_cases(runs)
        run_iter = tqdm(
            enumerate(cases),
            total=runs,
            desc="paired-eval",
            unit="run",
            disable=not progress,
        )
        for run_idx, case in run_iter:
            sample = case.sample
            disturbance = case.disturbance
            boundary_nodes_used = case.boundary_nodes_used
            perturbation_norm = case.perturbation_norm
            initial_belief = sample.h0 + disturbance
            for policy in policies:
                control = self._baseline_control(
                    policy, initial_belief, run_idx=run_idx
                )
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
                aggregates[policy]["auc"].append(
                    perturbation_auc(pred, sample.target_trajectory, sample.times)
                )
                aggregates[policy]["post_control_auc"].append(
                    post_control_perturbation_auc(
                        pred, sample.target_trajectory, sample.times
                    )
                )
                aggregates[policy]["mse"].append(
                    trajectory_mse(pred, sample.target_trajectory)
                )
                aggregates[policy]["ce"].append(
                    belief_cross_entropy(logits, sample.labels)
                )
                aggregates[policy]["observed"].append(
                    belief.observed_mask.float().mean()
                )
                aggregates[policy]["boundary"].append(boundary_nodes_used)
                aggregates[policy]["perturbation_norm"].append(perturbation_norm)

        results: list[dict[str, float]] = []
        for policy in policies:
            policy_values = aggregates[policy]
            results.append(
                {
                    "policy": policy,
                    "runs": float(runs),
                    "perturbation_auc": float(
                        torch.stack(policy_values["auc"]).mean().cpu()
                    ),
                    "post_control_perturbation_auc": float(
                        torch.stack(policy_values["post_control_auc"]).mean().cpu()
                    ),
                    "trajectory_mse": float(
                        torch.stack(policy_values["mse"]).mean().cpu()
                    ),
                    "belief_ce": float(torch.stack(policy_values["ce"]).mean().cpu()),
                    "observed_fraction": float(
                        torch.stack(policy_values["observed"]).mean().cpu()
                    ),
                    "boundary_nodes_used": float(
                        torch.stack(policy_values["boundary"]).mean().cpu()
                    ),
                    "perturbation_norm": float(
                        torch.stack(policy_values["perturbation_norm"]).mean().cpu()
                    ),
                }
            )
        results.extend(comparison_deltas(results))
        return results

    @torch.no_grad()
    def make_eval_cases(self, runs: int) -> list[EvalCase]:
        if runs <= 0:
            raise ValueError("runs must be positive")
        cases = []
        for run_idx in range(runs):
            sample = self.dataset.sample(
                steps=self.config.training.trajectory_steps, dt=self.config.training.dt
            )
            disturbance, boundary_nodes_used, perturbation_norm = (
                self._source_disturbance(sample.h0, run_idx)
            )
            cases.append(
                EvalCase(
                    sample=sample,
                    disturbance=disturbance.detach().clone(),
                    boundary_nodes_used=boundary_nodes_used.detach().clone(),
                    perturbation_norm=perturbation_norm.detach().clone(),
                )
            )
        return cases

    def save_checkpoint(
        self, path: str | Path, *, epoch: int, metrics: dict[str, float]
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "metrics": metrics,
                "config": self.config,
                "model": self.model.state_dict(),
                "sensing": self.sensing.state_dict(),
                "source_policy": self.source_policy.state_dict(),
                "control_policy": self.control_policy.state_dict(),
                "classifier": self.classifier.state_dict(),
                "controller_optimizer": self.controller_optimizer.state_dict(),
                "source_optimizer": self.source_optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(
        self, path: str | Path, *, load_optimizers: bool = False
    ) -> dict[str, object]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model"])
        self.sensing.load_state_dict(checkpoint["sensing"])
        self.source_policy.load_state_dict(checkpoint["source_policy"])
        self.control_policy.load_state_dict(checkpoint["control_policy"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        if load_optimizers:
            self.controller_optimizer.load_state_dict(
                checkpoint["controller_optimizer"]
            )
            self.source_optimizer.load_state_dict(checkpoint["source_optimizer"])
        return checkpoint

    def save_eval_cases(self, path: str | Path, eval_cases: list[EvalCase]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(eval_cases_to_payload(eval_cases), path)

    def load_eval_cases(self, path: str | Path) -> list[EvalCase]:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        return eval_cases_from_payload(payload, self.device)

    def save_graph_artifacts(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        torch.save(
            {
                "edge_index": self.dataset.edge_index.detach().cpu(),
                "edge_attr": self.dataset.edge_attr.detach().cpu(),
                "labels": self.dataset.labels.detach().cpu(),
                "community_centers": self.dataset.community_centers.detach().cpu(),
                "num_nodes": self.dataset.num_nodes,
                "num_communities": self.config.graph.num_communities,
            },
            run_dir / "graph.pt",
        )
        write_json_file(
            run_dir / "graph_layout.json",
            community_layout(self.dataset.labels.detach().cpu()),
        )

    @torch.no_grad()
    def export_rollouts(
        self,
        run_dir: str | Path,
        *,
        policies: tuple[BaselinePolicy, ...],
        eval_cases: list[EvalCase],
        progress: bool = False,
    ) -> list[dict[str, object]]:
        run_dir = Path(run_dir)
        rollout_dir = run_dir / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, object]] = []
        iterator = tqdm(
            enumerate(eval_cases),
            total=len(eval_cases),
            desc="export-rollouts",
            unit="run",
            disable=not progress,
        )
        for run_idx, case in iterator:
            sample = case.sample
            initial_belief = sample.h0 + case.disturbance
            for policy in policies:
                control = self._baseline_control(
                    policy, initial_belief, run_idx=run_idx
                )
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
                node_error = torch.linalg.vector_norm(
                    pred - sample.target_trajectory, dim=-1
                )
                mean_error = node_error.mean(dim=-1)
                filename = f"{policy}_run_{run_idx:03d}.pt"
                torch.save(
                    {
                        "policy": policy,
                        "run_idx": run_idx,
                        "pred_trajectory": pred.detach().cpu(),
                        "target_trajectory": sample.target_trajectory.detach().cpu(),
                        "initial_clean_state": sample.h0.detach().cpu(),
                        "initial_belief": initial_belief.detach().cpu(),
                        "disturbance": case.disturbance.detach().cpu(),
                        "control": control.detach().cpu(),
                        "probe_nodes": belief.probe_nodes.detach().cpu(),
                        "observed_mask": belief.observed_mask.detach().cpu(),
                        "labels": sample.labels.detach().cpu(),
                        "times": sample.times.detach().cpu(),
                        "node_error": node_error.detach().cpu(),
                        "mean_error": mean_error.detach().cpu(),
                        "boundary_nodes_used": case.boundary_nodes_used.detach().cpu(),
                        "perturbation_norm": case.perturbation_norm.detach().cpu(),
                    },
                    rollout_dir / filename,
                )
                manifest.append(
                    {
                        "policy": policy,
                        "run_idx": run_idx,
                        "path": str(Path("rollouts") / filename),
                        "final_mean_error": float(mean_error[-1].detach().cpu()),
                        "mean_error_auc": float(
                            torch.trapz(mean_error, sample.times).detach().cpu()
                        ),
                    }
                )
        write_json_file(run_dir / "rollouts_manifest.json", manifest)
        return manifest

    @torch.no_grad()
    def export_closed_loop_rollouts(
        self,
        run_dir: str | Path,
        *,
        policies: tuple[BaselinePolicy, ...],
        eval_cases: list[EvalCase],
        progress: bool = False,
    ) -> list[dict[str, object]]:
        run_dir = Path(run_dir)
        rollout_dir = run_dir / "closed_loop_rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, object]] = []
        iterator = tqdm(
            enumerate(eval_cases),
            total=len(eval_cases),
            desc="closed-loop-export",
            unit="run",
            disable=not progress,
        )
        for run_idx, case in iterator:
            sample = case.sample
            initial_belief = sample.h0 + case.disturbance
            for policy in policies:
                payload = self._closed_loop_payload(
                    policy, sample, initial_belief, run_idx
                )
                filename = f"{policy}_closed_loop_run_{run_idx:03d}.pt"
                torch.save(payload, rollout_dir / filename)
                mean_error = payload["mean_error"]
                manifest.append(
                    {
                        "policy": policy,
                        "run_idx": run_idx,
                        "path": str(Path("closed_loop_rollouts") / filename),
                        "final_mean_error": float(mean_error[-1]),
                        "mean_error_auc": float(
                            torch.trapz(mean_error, payload["times"])
                        ),
                    }
                )
        write_json_file(run_dir / "closed_loop_rollouts_manifest.json", manifest)
        write_json_file(
            run_dir / "closed_loop_run_rankings.json",
            _closed_loop_run_rankings(manifest),
        )
        return manifest

    def _closed_loop_payload(
        self,
        policy: BaselinePolicy,
        sample: GraphSignalSample,
        initial_belief: torch.Tensor,
        run_idx: int,
    ) -> dict[str, object]:
        states = [initial_belief]
        controls = []
        corrected_states = []
        probe_nodes = []
        observed_masks = []
        current = initial_belief
        for step_idx in range(sample.times.numel() - 1):
            control = self._baseline_control(
                policy, current, run_idx=run_idx * 10_000 + step_idx
            )
            local_times = sample.times[step_idx : step_idx + 2] - sample.times[step_idx]
            pred_window = self.model.rollout(
                current,
                local_times,
                edge_index=self.dataset.edge_index,
                edge_attr=self.dataset.edge_attr,
                control=control,
                step_size=self.config.training.dt,
            )
            predicted_next = pred_window[-1]
            belief = self.sensing(
                predicted_next,
                sample.target_trajectory[step_idx + 1],
                self.dataset.edge_index,
            )
            current = belief.h_corr
            states.append(current)
            controls.append(control)
            corrected_states.append(belief.h_corr)
            probe_nodes.append(belief.probe_nodes)
            observed_masks.append(belief.observed_mask)
        pred = torch.stack(states, dim=0)
        node_error = torch.linalg.vector_norm(pred - sample.target_trajectory, dim=-1)
        mean_error = node_error.mean(dim=-1)
        if controls:
            control_tensor = torch.stack(controls, dim=0)
            corrected_tensor = torch.stack(corrected_states, dim=0)
            probe_tensor = torch.stack(probe_nodes, dim=0)
            observed_tensor = torch.stack(observed_masks, dim=0)
        else:
            control_tensor = torch.empty(
                0,
                initial_belief.size(0),
                self.config.model.control_dim,
                device=self.device,
            )
            corrected_tensor = torch.empty_like(
                control_tensor[..., : initial_belief.size(1)]
            )
            probe_tensor = torch.empty(
                0, self.config.sensing.budget, dtype=torch.long, device=self.device
            )
            observed_tensor = torch.empty(
                0, initial_belief.size(0), dtype=torch.bool, device=self.device
            )
        return {
            "policy": policy,
            "mode": "closed_loop",
            "run_idx": run_idx,
            "pred_trajectory": pred.detach().cpu(),
            "target_trajectory": sample.target_trajectory.detach().cpu(),
            "initial_clean_state": sample.h0.detach().cpu(),
            "initial_belief": initial_belief.detach().cpu(),
            "controls": control_tensor.detach().cpu(),
            "corrected_states": corrected_tensor.detach().cpu(),
            "probe_nodes": probe_tensor.detach().cpu(),
            "observed_masks": observed_tensor.detach().cpu(),
            "labels": sample.labels.detach().cpu(),
            "times": sample.times.detach().cpu(),
            "node_error": node_error.detach().cpu(),
            "mean_error": mean_error.detach().cpu(),
        }


def _closed_loop_run_rankings(
    manifest: list[dict[str, object]],
) -> list[dict[str, float | int]]:
    by_run: dict[int, dict[str, dict[str, object]]] = {}
    for item in manifest:
        run_idx = int(item["run_idx"])
        policy = str(item["policy"])
        by_run.setdefault(run_idx, {})[policy] = item

    rankings = []
    for run_idx, policies in sorted(by_run.items()):
        if "neural" not in policies or "greedy" not in policies:
            continue
        neural = policies["neural"]
        greedy = policies["greedy"]
        random = policies.get("random")
        greedy_auc = float(greedy["mean_error_auc"])
        neural_auc = float(neural["mean_error_auc"])
        greedy_final = float(greedy["final_mean_error"])
        neural_final = float(neural["final_mean_error"])
        row: dict[str, float | int] = {
            "run_idx": run_idx,
            "greedy_mean_error_auc": greedy_auc,
            "neural_mean_error_auc": neural_auc,
            "neural_minus_greedy_mean_error_auc": neural_auc - greedy_auc,
            "neural_vs_greedy_mean_error_auc_improvement_pct": 100.0
            * (greedy_auc - neural_auc)
            / max(abs(greedy_auc), 1e-12),
            "greedy_final_mean_error": greedy_final,
            "neural_final_mean_error": neural_final,
            "neural_minus_greedy_final_mean_error": neural_final - greedy_final,
            "neural_vs_greedy_final_mean_error_improvement_pct": 100.0
            * (greedy_final - neural_final)
            / max(abs(greedy_final), 1e-12),
        }
        if random is not None:
            row["random_mean_error_auc"] = float(random["mean_error_auc"])
            row["random_final_mean_error"] = float(random["final_mean_error"])
        rankings.append(row)

    return sorted(
        rankings,
        key=lambda item: float(item["neural_vs_greedy_mean_error_auc_improvement_pct"]),
        reverse=True,
    )
