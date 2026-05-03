"""Argument parsing and CLI config overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

from manifold.utils.config import (
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    SensingConfig,
    TrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train neural sheaf ODE graph signal stabilization."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "compare", "export-closed-loop", "export-chatgpt-baseline"],
        default="train",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default=None
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs for smoke runs.",
    )
    parser.add_argument(
        "--num-nodes", type=int, default=None, help="Override graph.num_nodes."
    )
    parser.add_argument(
        "--num-communities",
        type=int,
        default=None,
        help="Override graph.num_communities.",
    )
    parser.add_argument("--p-in", type=float, default=None, help="Override graph.p_in.")
    parser.add_argument(
        "--p-out", type=float, default=None, help="Override graph.p_out."
    )
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=None,
        help="Override training.trajectory_steps.",
    )
    parser.add_argument(
        "--sensing-budget", type=int, default=None, help="Override sensing.budget."
    )
    parser.add_argument(
        "--source-steps", type=int, default=None, help="Override training.source_steps."
    )
    parser.add_argument(
        "--controller-steps",
        type=int,
        default=None,
        help="Override training.controller_steps.",
    )
    parser.add_argument("--final-belief-mse-weight", type=float, default=None)
    parser.add_argument("--control-energy-weight", type=float, default=None)
    parser.add_argument("--control-target-weight", type=float, default=None)
    parser.add_argument(
        "--perturbation-mode", choices=["learned", "boundary", "mixed"], default=None
    )
    parser.add_argument("--boundary-perturbation-scale", type=float, default=None)
    parser.add_argument("--boundary-perturbation-fraction", type=float, default=None)
    parser.add_argument("--boundary-recovery-gain", type=float, default=None)
    parser.add_argument("--source-freeze-after", type=int, default=None)
    parser.add_argument("--source-freeze-for", type=int, default=None)
    parser.add_argument("--analytic-sheaf-gain", type=float, default=None)
    parser.add_argument("--analytic-community-gain", type=float, default=None)
    parser.add_argument("--analytic-topk-multiplier", type=float, default=None)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["random", "greedy", "neural"],
        default=["random", "greedy", "neural"],
        help="Policies to score in compare mode.",
    )
    parser.add_argument(
        "--eval-runs", type=int, default=3, help="Evaluation trajectories per policy."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Compare policies every N epochs in compare mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for run artifacts.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional run folder name."
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Existing run directory for export-closed-loop.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint to load for export-closed-loop.",
    )
    parser.add_argument(
        "--closed-loop-output-name", type=str, default="closed-loop-eval"
    )
    parser.add_argument(
        "--rollout-spec",
        action="append",
        default=[],
        help="ChatGPT baseline spec as label:closed-loop-folder:run_idx, e.g. 48:closed-loop-eval-48-t8:14.",
    )
    parser.add_argument("--chatgpt-model", type=str, default="gpt-5.5")
    parser.add_argument("--chatgpt-reasoning", type=str, default="high")
    parser.add_argument("--chatgpt-output-name", type=str, default="chatgpt55-high-comparison")
    parser.add_argument("--chatgpt-fake-runner", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--chatgpt-replay-raw-responses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute ChatGPT controls from saved raw_responses instead of calling codex exec.",
    )
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--save-run", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--eval-set-path",
        type=Path,
        default=None,
        help="Load a saved paired eval set instead of sampling one.",
    )
    parser.add_argument(
        "--fresh-eval-set",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample a fresh eval set for export-closed-loop instead of reusing the saved run eval set.",
    )
    parser.add_argument(
        "--export-rollouts", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--early-stopping", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-metric",
        choices=[
            "neural_vs_greedy_auc_improvement_pct",
            "neural_vs_greedy_post_control_auc_improvement_pct",
            "neural_vs_greedy_mse_improvement_pct",
        ],
        default="neural_vs_greedy_post_control_auc_improvement_pct",
    )
    return parser


def with_cli_overrides(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    graph = GraphConfig(
        num_nodes=args.num_nodes
        if args.num_nodes is not None
        else config.graph.num_nodes,
        num_communities=args.num_communities
        if args.num_communities is not None
        else config.graph.num_communities,
        p_in=args.p_in if args.p_in is not None else config.graph.p_in,
        p_out=args.p_out if args.p_out is not None else config.graph.p_out,
        edge_attr_dim=config.graph.edge_attr_dim,
        seed=config.graph.seed,
    )
    model = ModelConfig(
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        control_dim=config.model.control_dim,
        sheaf_lambda=config.model.sheaf_lambda,
        restriction_scale=config.model.restriction_scale,
        num_classes=config.model.num_classes,
    )
    sensing = SensingConfig(
        k_hop=config.sensing.k_hop,
        budget=args.sensing_budget
        if args.sensing_budget is not None
        else config.sensing.budget,
        noise_std=config.sensing.noise_std,
    )
    training = TrainingConfig(
        epochs=args.epochs if args.epochs is not None else config.training.epochs,
        trajectory_steps=args.trajectory_steps
        if args.trajectory_steps is not None
        else config.training.trajectory_steps,
        dt=config.training.dt,
        controller_lr=config.training.controller_lr,
        source_lr=config.training.source_lr,
        source_steps=args.source_steps
        if args.source_steps is not None
        else config.training.source_steps,
        controller_steps=args.controller_steps
        if args.controller_steps is not None
        else config.training.controller_steps,
        auc_weight=config.training.auc_weight,
        mse_weight=config.training.mse_weight,
        final_belief_mse_weight=args.final_belief_mse_weight
        if args.final_belief_mse_weight is not None
        else config.training.final_belief_mse_weight,
        ce_weight=config.training.ce_weight,
        control_energy_weight=args.control_energy_weight
        if args.control_energy_weight is not None
        else config.training.control_energy_weight,
        control_target_weight=args.control_target_weight
        if args.control_target_weight is not None
        else config.training.control_target_weight,
        source_scale=config.training.source_scale,
        perturbation_mode=args.perturbation_mode
        if args.perturbation_mode is not None
        else config.training.perturbation_mode,
        boundary_perturbation_scale=args.boundary_perturbation_scale
        if args.boundary_perturbation_scale is not None
        else config.training.boundary_perturbation_scale,
        boundary_perturbation_fraction=args.boundary_perturbation_fraction
        if args.boundary_perturbation_fraction is not None
        else config.training.boundary_perturbation_fraction,
        boundary_center_swap=config.training.boundary_center_swap,
        boundary_recovery_gain=args.boundary_recovery_gain
        if args.boundary_recovery_gain is not None
        else config.training.boundary_recovery_gain,
        source_freeze_after=args.source_freeze_after
        if args.source_freeze_after is not None
        else config.training.source_freeze_after,
        source_freeze_for=args.source_freeze_for
        if args.source_freeze_for is not None
        else config.training.source_freeze_for,
        controller_residual_scale=config.training.controller_residual_scale,
        analytic_sheaf_gain=args.analytic_sheaf_gain
        if args.analytic_sheaf_gain is not None
        else config.training.analytic_sheaf_gain,
        analytic_community_gain=args.analytic_community_gain
        if args.analytic_community_gain is not None
        else config.training.analytic_community_gain,
        analytic_topk_multiplier=args.analytic_topk_multiplier
        if args.analytic_topk_multiplier is not None
        else config.training.analytic_topk_multiplier,
        log_every=config.training.log_every,
    )
    return ExperimentConfig(
        device=config.device,
        seed=config.seed,
        graph=graph,
        model=model,
        sensing=sensing,
        training=training,
    )
