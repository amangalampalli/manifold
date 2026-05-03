"""Utility records and serialization helpers for manifold training."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import torch

from manifold.data.types import GraphSignalSample


@dataclass
class TrainingHistory:
    metrics: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class EvalCase:
    sample: GraphSignalSample
    disturbance: torch.Tensor
    boundary_nodes_used: torch.Tensor
    perturbation_norm: torch.Tensor


def comparison_deltas(results: list[dict[str, float]]) -> list[dict[str, float]]:
    by_policy = {item["policy"]: item for item in results}
    if "neural" not in by_policy or "greedy" not in by_policy:
        return []
    neural = by_policy["neural"]
    greedy = by_policy["greedy"]
    return [
        {
            "policy": "neural_minus_greedy",
            "runs": neural["runs"],
            "neural_minus_greedy_auc": neural["perturbation_auc"]
            - greedy["perturbation_auc"],
            "neural_minus_greedy_post_control_auc": neural[
                "post_control_perturbation_auc"
            ]
            - greedy["post_control_perturbation_auc"],
            "neural_minus_greedy_mse": neural["trajectory_mse"]
            - greedy["trajectory_mse"],
            "neural_vs_greedy_auc_improvement_pct": 100.0
            * (greedy["perturbation_auc"] - neural["perturbation_auc"])
            / max(abs(greedy["perturbation_auc"]), 1e-12),
            "neural_vs_greedy_mse_improvement_pct": 100.0
            * (greedy["trajectory_mse"] - neural["trajectory_mse"])
            / max(abs(greedy["trajectory_mse"]), 1e-12),
            "neural_vs_greedy_post_control_auc_improvement_pct": 100.0
            * (
                greedy["post_control_perturbation_auc"]
                - neural["post_control_perturbation_auc"]
            )
            / max(abs(greedy["post_control_perturbation_auc"]), 1e-12),
            "perturbation_auc": neural["perturbation_auc"] - greedy["perturbation_auc"],
            "post_control_perturbation_auc": neural["post_control_perturbation_auc"]
            - greedy["post_control_perturbation_auc"],
            "trajectory_mse": neural["trajectory_mse"] - greedy["trajectory_mse"],
            "belief_ce": neural["belief_ce"] - greedy["belief_ce"],
            "observed_fraction": neural["observed_fraction"]
            - greedy["observed_fraction"],
            "boundary_nodes_used": neural.get("boundary_nodes_used", 0.0),
            "perturbation_norm": neural.get("perturbation_norm", 0.0),
        }
    ]


def eval_cases_to_payload(eval_cases: list[EvalCase]) -> list[dict[str, torch.Tensor]]:
    payload = []
    for case in eval_cases:
        payload.append(
            {
                "h0": case.sample.h0.detach().cpu(),
                "target_trajectory": case.sample.target_trajectory.detach().cpu(),
                "labels": case.sample.labels.detach().cpu(),
                "times": case.sample.times.detach().cpu(),
                "disturbance": case.disturbance.detach().cpu(),
                "boundary_nodes_used": case.boundary_nodes_used.detach().cpu(),
                "perturbation_norm": case.perturbation_norm.detach().cpu(),
            }
        )
    return payload


def eval_cases_from_payload(
    payload: list[dict[str, torch.Tensor]], device: torch.device
) -> list[EvalCase]:
    cases = []
    for item in payload:
        sample = GraphSignalSample(
            h0=item["h0"].to(device),
            target_trajectory=item["target_trajectory"].to(device),
            labels=item["labels"].to(device),
            times=item["times"].to(device),
        )
        cases.append(
            EvalCase(
                sample=sample,
                disturbance=item["disturbance"].to(device),
                boundary_nodes_used=item["boundary_nodes_used"].to(device),
                perturbation_norm=item["perturbation_norm"].to(device),
            )
        )
    return cases


def community_layout(labels: torch.Tensor) -> dict[str, object]:
    labels = labels.to(torch.long)
    community_count = int(labels.max().item()) + 1 if labels.numel() else 0
    positions: dict[str, list[float]] = {}
    for community in range(community_count):
        nodes = torch.nonzero(labels == community, as_tuple=False).flatten().tolist()
        if not nodes:
            continue
        community_angle = 2.0 * math.pi * community / max(1, community_count)
        center = (
            torch.tensor(
                [math.cos(community_angle), math.sin(community_angle)],
                dtype=torch.float32,
            )
            * 5.0
        )
        radius = 0.45 + 0.035 * len(nodes) ** 0.5
        for offset, node in enumerate(nodes):
            node_angle = 2.0 * math.pi * offset / max(1, len(nodes))
            local = (
                torch.tensor(
                    [math.cos(node_angle), math.sin(node_angle)], dtype=torch.float32
                )
                * radius
            )
            point = center + local
            positions[str(node)] = [float(point[0]), float(point[1])]
    return {
        "type": "community_radial",
        "positions": positions,
        "num_nodes": int(labels.numel()),
        "num_communities": community_count,
    }


def write_json_file(path: Path, payload: object) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
