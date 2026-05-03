from __future__ import annotations

import json
from pathlib import Path

import torch

from manifold.viz import plot_all


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _rollout(policy: str, run_idx: int, scale: float) -> dict:
    steps = 4
    nodes = 6
    dim = 3
    pred = torch.full((steps, nodes, dim), scale)
    target = torch.zeros_like(pred)
    node_error = torch.linalg.vector_norm(pred - target, dim=-1)
    return {
        "policy": policy,
        "mode": "closed_loop",
        "run_idx": run_idx,
        "pred_trajectory": pred,
        "target_trajectory": target,
        "initial_clean_state": target[0],
        "initial_belief": pred[0],
        "controls": torch.full((steps - 1, nodes, dim), scale * 0.1),
        "corrected_states": pred[:-1],
        "probe_nodes": torch.zeros((steps - 1, 2), dtype=torch.long),
        "observed_masks": torch.ones((steps - 1, nodes), dtype=torch.bool),
        "labels": torch.tensor([0, 0, 0, 1, 1, 1]),
        "times": torch.arange(steps, dtype=torch.float32),
        "node_error": node_error,
        "mean_error": node_error.mean(dim=1),
    }


def test_plot_all_generates_manifest_and_figures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    fig_dir = tmp_path / "figs"

    _write_jsonl(
        run_dir / "logs" / "metrics.jsonl",
        [
            {"epoch": 1, "controller_loss": 1.0, "trajectory_mse": 0.4, "control_energy": 0.2},
            {"epoch": 2, "controller_loss": 0.8, "trajectory_mse": 0.3, "control_energy": 0.25},
        ],
    )
    _write_jsonl(
        run_dir / "logs" / "eval.jsonl",
        [
            {
                "epoch": 2,
                "comparison": [
                    {"policy": "greedy", "post_control_perturbation_auc": 1.0, "trajectory_mse": 0.5},
                    {"policy": "neural", "post_control_perturbation_auc": 0.5, "trajectory_mse": 0.25},
                    {
                        "policy": "neural_minus_greedy",
                        "neural_vs_greedy_auc_improvement_pct": 50.0,
                        "neural_vs_greedy_mse_improvement_pct": 50.0,
                    },
                ],
            }
        ],
    )

    closed = run_dir / "closed-loop-test"
    rollouts = closed / "closed_loop_rollouts"
    rollouts.mkdir(parents=True)
    for policy, scale in [("random", 1.5), ("greedy", 1.0), ("neural", 0.5)]:
        torch.save(_rollout(policy, 3, scale), rollouts / f"{policy}_closed_loop_run_003.pt")
    torch.save(
        {
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            "edge_attr": torch.zeros(4, 4),
            "labels": torch.tensor([0, 0, 0, 1, 1, 1]),
            "community_centers": torch.zeros(2, 3),
            "num_nodes": 6,
            "num_communities": 2,
        },
        closed / "graph.pt",
    )
    (closed / "graph_layout.json").write_text(
        json.dumps({"positions": [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]})
    )
    (closed / "closed_loop_run_rankings.json").write_text(
        json.dumps([{"run_idx": 3, "auc_improvement_pct": 50.0, "mse_improvement_pct": 75.0}])
    )

    result = plot_all(
        run_dir=run_dir,
        fig_dir=fig_dir,
        closed_loop_dirs=["closed-loop-test"],
        selected_runs={"closed-loop-test": 3},
    )

    assert (fig_dir / "plot_manifest.json").exists()
    assert "training_losses.png" in result.generated
    assert "selected_runs_neural_vs_greedy.png" in result.generated
    assert any(path.suffix == ".png" for path in fig_dir.iterdir())
