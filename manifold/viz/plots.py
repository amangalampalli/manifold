"""Generate publication/demo plots from saved manifold run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


POLICY_COLORS = {
    "random": "#777777",
    "greedy": "#d95f02",
    "neural": "#1b9e77",
    "chatgpt": "#7570b3",
    "ChatGPT 5.5 High": "#7570b3",
}


def _case_label(folder: str, run_idx: int | None = None) -> str:
    if "48" in folder:
        base = "48-node graph"
    elif "20" in folder or "512" in folder:
        base = "512-node graph"
    else:
        base = folder.replace("-", " ")
    return f"{base}, rollout {run_idx}" if run_idx is not None else base


def _case_slug(folder: str, run_idx: int | None = None) -> str:
    if "48" in folder:
        base = "48_node_graph"
    elif "20" in folder or "512" in folder:
        base = "512_node_graph"
    else:
        base = folder.replace("-", "_")
    return f"{base}_rollout_{run_idx:03d}" if run_idx is not None else base


@dataclass
class PlotResult:
    fig_dir: Path
    generated: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def add_generated(self, path: Path) -> None:
        self.generated.append(path.name)

    def add_skipped(self, name: str, reason: str) -> None:
        self.skipped[name] = reason

    def write_manifest(self) -> Path:
        path = self.fig_dir / "plot_manifest.json"
        payload = {
            "fig_dir": str(self.fig_dir),
            "generated": sorted(self.generated),
            "skipped": self.skipped,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


def plot_all(
    run_dir: str | Path,
    fig_dir: str | Path = "manifold/docs/figs",
    closed_loop_dirs: Iterable[str] | None = None,
    selected_runs: dict[str, int] | None = None,
    include_fake_chatgpt: bool = False,
) -> PlotResult:
    """Generate every plot supported by the artifacts present in ``run_dir``."""

    run_dir = Path(run_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    result = PlotResult(fig_dir=fig_dir)

    selected_runs = selected_runs or {
        "closed-loop-eval-48-t8": 14,
        "closed-loop-eval-20-t8": 15,
    }
    closed_loop_names = list(closed_loop_dirs or _discover_closed_loop_dirs(run_dir))

    _plot_training_metrics(run_dir, fig_dir, result)
    _plot_eval_metrics(run_dir, fig_dir, result)
    _plot_closed_loop_rankings(run_dir, fig_dir, result, closed_loop_names)
    _plot_selected_closed_loop_runs(run_dir, fig_dir, result, selected_runs)
    _plot_token_control_efficiency(run_dir, fig_dir, result, selected_runs)
    _plot_graph_diagnostics(run_dir, fig_dir, result, closed_loop_names, selected_runs)
    _plot_chatgpt_timing(run_dir, fig_dir, result, include_fake=include_fake_chatgpt)
    _plot_artifact_sizes(run_dir, fig_dir, result)

    result.write_manifest()
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_pt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _save(fig: plt.Figure, fig_dir: Path, filename: str, result: PlotResult) -> None:
    path = fig_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    result.add_generated(path)


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            xs.append(float(row.get("epoch", len(xs) + 1)))
            ys.append(float(value))
    return xs, ys


def _plot_training_metrics(run_dir: Path, fig_dir: Path, result: PlotResult) -> None:
    rows = _read_jsonl(run_dir / "logs" / "metrics.jsonl")
    if not rows:
        result.add_skipped("training metrics", "missing logs/metrics.jsonl")
        return

    groups = {
        "training_losses.png": [
            ("controller_loss", "Controller loss"),
            ("source_loss", "Source loss"),
        ],
        "training_recovery_metrics.png": [
            ("post_control_perturbation_auc", "Post-control AUC"),
            ("trajectory_mse", "Trajectory MSE"),
            ("final_belief_mse", "Final belief MSE"),
        ],
        "training_control_metrics.png": [
            ("control_energy", "Control energy"),
            ("control_target_mse", "Control-target MSE"),
            ("observed_fraction", "Observed fraction"),
        ],
        "training_source_metrics.png": [
            ("source_auc", "Source AUC"),
            ("perturbation_norm", "Perturbation norm"),
            ("boundary_nodes_used", "Boundary nodes"),
            ("source_frozen", "Source frozen"),
        ],
    }

    for filename, keys in groups.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for key, label in keys:
            xs, ys = _series(rows, key)
            if ys:
                ax.plot(xs, ys, label=label, linewidth=2)
                plotted = True
        if not plotted:
            plt.close(fig)
            result.add_skipped(filename, "no matching training fields")
            continue
        ax.set_title(filename.replace(".png", "").replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()
        _save(fig, fig_dir, filename, result)


def _plot_eval_metrics(run_dir: Path, fig_dir: Path, result: PlotResult) -> None:
    rows = _read_jsonl(run_dir / "logs" / "eval.jsonl")
    if not rows:
        result.add_skipped("eval metrics", "missing logs/eval.jsonl")
        return

    by_policy: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        epoch = float(row.get("epoch", len(by_policy) + 1))
        for item in row.get("comparison", []):
            policy = item.get("policy")
            if not isinstance(policy, str):
                continue
            by_policy.setdefault(policy, []).append({"epoch": epoch, **item})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    deltas = by_policy.get("neural_minus_greedy", [])
    for key, label in [
        ("neural_vs_greedy_auc_improvement_pct", "AUC improvement %"),
        ("neural_vs_greedy_mse_improvement_pct", "MSE improvement %"),
        ("neural_vs_greedy_post_control_auc_improvement_pct", "Post-control AUC improvement %"),
    ]:
        xs = [row["epoch"] for row in deltas if isinstance(row.get(key), (int, float))]
        ys = [float(row[key]) for row in deltas if isinstance(row.get(key), (int, float))]
        if ys:
            ax.plot(xs, ys, label=label, linewidth=2)
    if ax.lines:
        ax.axhline(0.0, color="#333333", linewidth=1, alpha=0.5)
        ax.set_title("Neural Improvement Over Greedy During Eval")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Improvement (%)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        _save(fig, fig_dir, "eval_neural_vs_greedy_improvement.png", result)
    else:
        plt.close(fig)
        result.add_skipped("eval_neural_vs_greedy_improvement.png", "missing neural-minus-greedy eval rows")

    for metric, filename, ylabel in [
        ("post_control_perturbation_auc", "eval_policy_post_control_auc.png", "Post-control AUC"),
        ("trajectory_mse", "eval_policy_trajectory_mse.png", "Trajectory MSE"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for policy in ["random", "greedy", "neural"]:
            rows_for_policy = by_policy.get(policy, [])
            xs = [row["epoch"] for row in rows_for_policy if isinstance(row.get(metric), (int, float))]
            ys = [float(row[metric]) for row in rows_for_policy if isinstance(row.get(metric), (int, float))]
            if ys:
                ax.plot(xs, ys, label=policy, color=POLICY_COLORS.get(policy), linewidth=2)
        if ax.lines:
            ax.set_title(ylabel + " By Policy During Eval")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
            _save(fig, fig_dir, filename, result)
        else:
            plt.close(fig)
            result.add_skipped(filename, f"missing {metric} eval rows")


def _discover_closed_loop_dirs(run_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "closed_loop_rollouts").exists()
    )


def _policy_path(closed_loop_dir: Path, policy: str, run_idx: int) -> Path:
    return closed_loop_dir / "closed_loop_rollouts" / f"{policy}_closed_loop_run_{run_idx:03d}.pt"


def _plot_closed_loop_rankings(
    run_dir: Path,
    fig_dir: Path,
    result: PlotResult,
    closed_loop_names: list[str],
) -> None:
    summaries: list[dict[str, float | str]] = []
    for name in closed_loop_names:
        rankings = _load_json(run_dir / name / "closed_loop_run_rankings.json")
        if not rankings:
            continue
        top = rankings[: min(10, len(rankings))]
        labels = [str(item.get("run_idx", idx)) for idx, item in enumerate(top)]
        auc = [_pick_float(item, ["auc_improvement_pct", "neural_vs_greedy_auc_improvement_pct"]) for item in top]
        mse = [_pick_float(item, ["mse_improvement_pct", "neural_vs_greedy_mse_improvement_pct"]) for item in top]
        final = [_pick_float(item, ["final_error_improvement_pct", "final_improvement_pct"]) for item in top]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        width = 0.26
        ax.bar(x - width, auc, width, label="AUC improvement")
        ax.bar(x, mse, width, label="MSE improvement")
        if any(not math.isnan(v) for v in final):
            ax.bar(x + width, final, width, label="Final-error improvement")
        ax.axhline(0.0, color="#333333", linewidth=1, alpha=0.6)
        ax.set_title(f"Top Closed-loop Runs: {_case_label(name)}")
        ax.set_xlabel("Run")
        ax.set_ylabel("Neural over greedy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        _save(fig, fig_dir, f"{name}_top_run_improvements.png", result)

        for item in rankings:
            summaries.append(
                {
                    "folder": name,
                    "run_idx": float(item.get("run_idx", -1)),
                    "auc": _pick_float(item, ["auc_improvement_pct", "neural_vs_greedy_auc_improvement_pct"]),
                    "mse": _pick_float(item, ["mse_improvement_pct", "neural_vs_greedy_mse_improvement_pct"]),
                    "final": _pick_float(item, ["final_error_improvement_pct", "final_improvement_pct"]),
                }
            )

    if not summaries:
        result.add_skipped("closed-loop rankings", "no closed_loop_run_rankings.json files found")
        return

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for name in sorted({str(row["folder"]) for row in summaries}):
        rows = [row for row in summaries if row["folder"] == name]
        ax.scatter([row["auc"] for row in rows], [row["mse"] for row in rows], label=_case_label(name), alpha=0.75, s=36)
    ax.axhline(0.0, color="#333333", linewidth=1, alpha=0.4)
    ax.axvline(0.0, color="#333333", linewidth=1, alpha=0.4)
    ax.set_title("Closed-loop Run Tradeoff")
    ax.set_xlabel("AUC improvement over greedy (%)")
    ax.set_ylabel("Trajectory MSE improvement over greedy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, fig_dir, "closed_loop_auc_mse_tradeoff.png", result)


def _pick_float(row: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def _rollout_metrics(payload: dict[str, Any]) -> dict[str, float]:
    mean_error = _to_numpy(payload.get("mean_error"))
    node_error = _to_numpy(payload.get("node_error"))
    pred = payload.get("pred_trajectory")
    target = payload.get("target_trajectory")
    controls = _to_numpy(payload.get("controls"))
    masks = _to_numpy(payload.get("observed_masks"))

    trajectory_mse = float("nan")
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        trajectory_mse = float(torch.mean((pred.cpu() - target.cpu()) ** 2).item())
    return {
        "auc": _auc(mean_error),
        "trajectory_mse": trajectory_mse,
        "final_error": float(mean_error[-1]) if mean_error.size else float("nan"),
        "mean_node_error": float(np.nanmean(node_error)) if node_error.size else float("nan"),
        "control_norm": float(np.nanmean(np.linalg.norm(controls, axis=-1))) if controls.size else float("nan"),
        "observed_fraction": float(np.nanmean(masks)) if masks.size else float("nan"),
    }


def _cumulative_auc(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.zeros_like(values, dtype=float)
    if values.size > 1:
        increments = 0.5 * (values[1:] + values[:-1])
        out[1:] = np.cumsum(increments)
    return out


def _ratio_text(baseline: float, candidate: float) -> str:
    if not math.isfinite(baseline) or not math.isfinite(candidate) or candidate <= 0:
        return "n/a"
    return f"{baseline / candidate:.2f}x better"


def _label_bars(ax: plt.Axes, *, fmt: str = "{:.3g}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not math.isfinite(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if value is None:
        return np.asarray([])
    return np.asarray(value)


def _auc(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return float(values[0])
    return float(np.trapezoid(values, dx=1.0))


def _plot_selected_closed_loop_runs(
    run_dir: Path,
    fig_dir: Path,
    result: PlotResult,
    selected_runs: dict[str, int],
) -> None:
    selected_metrics: list[dict[str, Any]] = []
    for name, run_idx in selected_runs.items():
        case_label = _case_label(name, run_idx)
        case_slug = _case_slug(name, run_idx)
        closed_loop_dir = run_dir / name
        if not closed_loop_dir.exists():
            result.add_skipped(f"{name} selected run", "closed-loop folder missing")
            continue
        payloads: dict[str, dict[str, Any]] = {}
        for policy in ["random", "greedy", "neural"]:
            payload = _load_pt(_policy_path(closed_loop_dir, policy, run_idx))
            if payload is not None:
                payloads[policy] = payload
        payloads.update(_load_chatgpt_rollouts_for(run_dir, name, run_idx))
        if not payloads:
            result.add_skipped(f"{name} run {run_idx}", "no rollout payloads found")
            continue

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for policy, payload in payloads.items():
            mean_error = _to_numpy(payload.get("mean_error"))
            if mean_error.size:
                ax.plot(
                    np.arange(mean_error.size),
                    mean_error,
                    label=_policy_label(policy),
                    color=POLICY_COLORS.get(policy, POLICY_COLORS.get(_policy_label(policy))),
                    linewidth=2,
                )
        if ax.lines:
            ax.set_title(f"Closed-loop Mean Error: {case_label}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Mean node error")
            ax.grid(True, alpha=0.25)
            ax.legend()
            _save(fig, fig_dir, f"{case_slug}_mean_error.png", result)
        else:
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        policies = list(payloads)
        metrics = {policy: _rollout_metrics(payload) for policy, payload in payloads.items()}
        for metric, ax, title in [
            ("auc", axes[0], "Mean-error AUC"),
            ("trajectory_mse", axes[1], "Trajectory MSE"),
        ]:
            values = [metrics[policy][metric] for policy in policies]
            ax.bar(
                [_policy_label(policy) for policy in policies],
                values,
                color=[POLICY_COLORS.get(policy, POLICY_COLORS.get(_policy_label(policy), "#4c78a8")) for policy in policies],
            )
            ax.set_title(title)
            _label_bars(ax, fmt="{:.3f}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=20)
        fig.suptitle(f"Policy Score Comparison: {case_label}")
        _save(fig, fig_dir, f"{case_slug}_policy_scores.png", result)

        for policy, metric_values in metrics.items():
            selected_metrics.append({"folder": name, "run_idx": run_idx, "policy": policy, **metric_values})

        _plot_control_and_sensing(fig_dir, result, name, run_idx, payloads)
        _plot_auc_curves(fig_dir, result, name, run_idx, payloads)
        _plot_error_histograms(fig_dir, result, name, run_idx, payloads)
        _plot_node_level_advantage_histogram(fig_dir, result, name, run_idx, payloads)

    if selected_metrics:
        _plot_selected_summary(fig_dir, result, selected_metrics)


def _policy_label(policy: str) -> str:
    return "ChatGPT 5.5 High" if policy == "chatgpt" else policy


def _load_chatgpt_rollouts_for(run_dir: Path, closed_loop_name: str, run_idx: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = run_dir / "chatgpt55-high-comparison"
    rollouts = base / "chatgpt_rollouts"
    if not rollouts.exists():
        return out
    candidates = sorted(rollouts.glob(f"*{closed_loop_name}*run_{run_idx:03d}*.pt"))
    if not candidates:
        candidates = sorted(rollouts.glob(f"*run_{run_idx:03d}*.pt"))
    if candidates:
        payload = _load_pt(candidates[0])
        if payload is not None:
            out["chatgpt"] = payload
    return out


def _plot_control_and_sensing(
    fig_dir: Path,
    result: PlotResult,
    name: str,
    run_idx: int,
    payloads: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plotted = False
    for policy, payload in payloads.items():
        controls = _to_numpy(payload.get("controls"))
        masks = _to_numpy(payload.get("observed_masks"))
        color = POLICY_COLORS.get(policy, POLICY_COLORS.get(_policy_label(policy)))
        if controls.size:
            control_norm = np.linalg.norm(controls, axis=-1).mean(axis=1)
            axes[0].plot(control_norm, label=_policy_label(policy), color=color, linewidth=2)
            plotted = True
        if masks.size:
            axes[1].plot(masks.mean(axis=1), label=_policy_label(policy), color=color, linewidth=2)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_title("Control Norm")
    axes[0].set_xlabel("Control step")
    axes[0].set_ylabel("Mean L2 norm")
    axes[1].set_title("Observed Fraction")
    axes[1].set_xlabel("Control step")
    axes[1].set_ylabel("Visible node fraction")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"Closed-loop Efficiency Signals: {_case_label(name, run_idx)}")
    _save(fig, fig_dir, f"{_case_slug(name, run_idx)}_control_sensing.png", result)


def _plot_auc_curves(
    fig_dir: Path,
    result: PlotResult,
    name: str,
    run_idx: int,
    payloads: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    metrics = {policy: _rollout_metrics(payload) for policy, payload in payloads.items()}
    plotted = False
    for policy, payload in payloads.items():
        mean_error = _to_numpy(payload.get("mean_error"))
        if not mean_error.size:
            continue
        color = POLICY_COLORS.get(policy, POLICY_COLORS.get(_policy_label(policy)))
        steps = np.arange(mean_error.size)
        axes[0].plot(steps, mean_error, label=_policy_label(policy), color=color, linewidth=2)
        axes[1].plot(steps, _cumulative_auc(mean_error), label=_policy_label(policy), color=color, linewidth=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_title("Instantaneous Error")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean node error")
    axes[1].set_title("Cumulative AUC")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Area under error curve")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"AUC Curves: {_case_label(name, run_idx)}")
    _save(fig, fig_dir, f"{_case_slug(name, run_idx)}_auc_curves.png", result)


def _plot_error_histograms(
    fig_dir: Path,
    result: PlotResult,
    name: str,
    run_idx: int,
    payloads: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plotted = False
    metrics = {policy: _rollout_metrics(payload) for policy, payload in payloads.items()}
    for policy, payload in payloads.items():
        node_error = _to_numpy(payload.get("node_error"))
        if not node_error.size:
            continue
        color = POLICY_COLORS.get(policy, POLICY_COLORS.get(_policy_label(policy)))
        axes[0].hist(node_error[-1].reshape(-1), bins=28, alpha=0.45, label=_policy_label(policy), color=color)
        axes[1].hist(node_error.reshape(-1), bins=34, alpha=0.42, label=_policy_label(policy), color=color)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_title("Final Node Error Distribution")
    axes[0].set_xlabel("Node error")
    axes[0].set_ylabel("Node count")
    axes[1].set_title("All-step Node Error Distribution")
    axes[1].set_xlabel("Node error")
    axes[1].set_ylabel("Node-step count")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle(f"Error Histograms: {_case_label(name, run_idx)}")
    _save(fig, fig_dir, f"{_case_slug(name, run_idx)}_error_histograms.png", result)


def _plot_node_level_advantage_histogram(
    fig_dir: Path,
    result: PlotResult,
    name: str,
    run_idx: int,
    payloads: dict[str, dict[str, Any]],
) -> None:
    if "neural" not in payloads:
        return
    neural = _to_numpy(payloads["neural"].get("node_error"))
    if not neural.size:
        return
    comparisons = [policy for policy in ["greedy", "chatgpt", "random"] if policy in payloads]
    if not comparisons:
        return
    fig, axes = plt.subplots(1, len(comparisons), figsize=(5.3 * len(comparisons), 4.8), squeeze=False)
    for ax, policy in zip(axes[0], comparisons):
        baseline = _to_numpy(payloads[policy].get("node_error"))
        if not baseline.size:
            continue
        improvement = baseline.reshape(-1) - neural.reshape(-1)
        ax.hist(improvement, bins=34, color=POLICY_COLORS.get(policy), alpha=0.72)
        ax.axvline(0.0, color="#111111", linewidth=1)
        pct_positive = 100.0 * float((improvement > 0).mean())
        baseline_mse = _rollout_metrics(payloads[policy])["trajectory_mse"]
        neural_mse = _rollout_metrics(payloads["neural"])["trajectory_mse"]
        ax.set_title(f"Neural Error Reduction vs {_policy_label(policy)}")
        ax.set_xlabel("Baseline node-error minus neural")
        ax.set_ylabel("Node-step count")
        ax.text(
            0.03,
            0.94,
            f"baseline MSE {baseline_mse:.3f}\nneural MSE {neural_mse:.3f}\n{pct_positive:.0f}% node-steps lower",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
        )
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Node-level Advantage Histograms: {_case_label(name, run_idx)}")
    _save(fig, fig_dir, f"{_case_slug(name, run_idx)}_node_advantage_histograms.png", result)


def _plot_selected_summary(fig_dir: Path, result: PlotResult, rows: list[dict[str, Any]]) -> None:
    neural_rows = [row for row in rows if row["policy"] == "neural"]
    bars: list[tuple[str, float, float]] = []
    for neural in neural_rows:
        greedy = next(
            (
                row
                for row in rows
                if row["folder"] == neural["folder"] and row["run_idx"] == neural["run_idx"] and row["policy"] == "greedy"
            ),
            None,
        )
        if greedy is None:
            continue
        auc_gain = _improvement_pct(greedy["auc"], neural["auc"])
        mse_gain = _improvement_pct(greedy["trajectory_mse"], neural["trajectory_mse"])
        bars.append((_case_label(str(neural["folder"]), int(neural["run_idx"])), auc_gain, mse_gain))
    if not bars:
        return
    x = np.arange(len(bars))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.35
    ax.bar(x - width / 2, [row[1] for row in bars], width, label="AUC improvement")
    ax.bar(x + width / 2, [row[2] for row in bars], width, label="MSE improvement")
    ax.axhline(0.0, color="#333333", linewidth=1, alpha=0.6)
    ax.set_title("Selected Demo Runs: Neural Improvement Over Greedy")
    ax.set_ylabel("Improvement (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in bars])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    _save(fig, fig_dir, "selected_runs_neural_vs_greedy.png", result)
    _plot_readable_error_values(fig_dir, result, rows)
    return

    value_rows: list[tuple[str, str, float, float]] = []
    for neural in neural_rows:
        for policy_name in ["neural", "greedy", "chatgpt", "random"]:
            policy_row = next(
                (
                    row
                    for row in rows
                    if row["folder"] == neural["folder"]
                    and row["run_idx"] == neural["run_idx"]
                    and row["policy"] == policy_name
                ),
                None,
            )
            if policy_row is None:
                continue
            value_rows.append(
                (
                    _case_label(str(neural["folder"]), int(neural["run_idx"])),
                    _policy_label(policy_name),
                    policy_row["auc"],
                    policy_row["trajectory_mse"],
                )
            )
    if not value_rows:
        return
    labels = [f"{row[0]}\n{row[1]}" for row in value_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(labels)), 5.2))
    width = 0.35
    ax.bar(x - width / 2, [row[2] for row in value_rows], width, label="AUC")
    ax.bar(x + width / 2, [row[3] for row in value_rows], width, label="Trajectory MSE")
    _label_bars(ax, fmt="{:.3f}")
    ax.set_title("Selected Demo Runs: Real Error Values")
    ax.set_ylabel("Lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    _save(fig, fig_dir, "selected_runs_real_error_values.png", result)
    _plot_readable_error_values(fig_dir, result, rows)


def _plot_readable_error_values(fig_dir: Path, result: PlotResult, rows: list[dict[str, Any]]) -> None:
    policies = ["neural", "greedy", "chatgpt", "random"]
    policy_labels = [_policy_label(policy) for policy in policies]
    cases = []
    for row in rows:
        case = (str(row["folder"]), int(row["run_idx"]))
        if case not in cases:
            cases.append(case)

    for folder, run_idx in cases:
        case_rows = {
            str(row["policy"]): row
            for row in rows
            if str(row["folder"]) == folder and int(row["run_idx"]) == run_idx
        }
        available = [policy for policy in policies if policy in case_rows]
        if not available:
            continue
        labels = [_policy_label(policy) for policy in available]
        auc_values = [case_rows[policy]["auc"] for policy in available]
        mse_values = [case_rows[policy]["trajectory_mse"] for policy in available]
        colors = [POLICY_COLORS.get(policy, "#4c78a8") for policy in available]

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        axes[0].bar(labels, auc_values, color=colors)
        axes[0].set_title("AUC")
        axes[0].set_ylabel("Lower is better")
        _label_bars(axes[0], fmt="{:.3f}")

        axes[1].bar(labels, mse_values, color=colors)
        axes[1].set_title("Trajectory MSE")
        axes[1].set_ylabel("Lower is better")
        _label_bars(axes[1], fmt="{:.3f}")

        for ax in axes:
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=18)
        fig.suptitle(f"Real Error Values: {_case_label(folder, run_idx)}")
        _save(fig, fig_dir, f"{_case_slug(folder, run_idx)}_real_error_values.png", result)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    case_labels = [_case_label(folder, run_idx).replace(", ", "\n") for folder, run_idx in cases]
    neural_auc = []
    greedy_auc = []
    neural_mse = []
    greedy_mse = []
    for folder, run_idx in cases:
        case_rows = {
            str(row["policy"]): row
            for row in rows
            if str(row["folder"]) == folder and int(row["run_idx"]) == run_idx
        }
        if "neural" in case_rows and "greedy" in case_rows:
            neural_auc.append(case_rows["neural"]["auc"])
            greedy_auc.append(case_rows["greedy"]["auc"])
            neural_mse.append(case_rows["neural"]["trajectory_mse"])
            greedy_mse.append(case_rows["greedy"]["trajectory_mse"])
        else:
            neural_auc.append(float("nan"))
            greedy_auc.append(float("nan"))
            neural_mse.append(float("nan"))
            greedy_mse.append(float("nan"))
    x = np.arange(len(cases))
    width = 0.35
    axes[0].bar(x - width / 2, neural_auc, width, label="Neural", color=POLICY_COLORS["neural"])
    axes[0].bar(x + width / 2, greedy_auc, width, label="Greedy", color=POLICY_COLORS["greedy"])
    axes[0].set_title("AUC")
    axes[0].set_ylabel("Lower is better")
    _label_bars(axes[0], fmt="{:.3f}")
    axes[1].bar(x - width / 2, neural_mse, width, label="Neural", color=POLICY_COLORS["neural"])
    axes[1].bar(x + width / 2, greedy_mse, width, label="Greedy", color=POLICY_COLORS["greedy"])
    axes[1].set_title("Trajectory MSE")
    axes[1].set_ylabel("Lower is better")
    _label_bars(axes[1], fmt="{:.3f}")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Neural vs Greedy: Real Error Values")
    _save(fig, fig_dir, "neural_vs_greedy_real_error_values.png", result)


def _improvement_pct(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or baseline == 0 or not math.isfinite(candidate):
        return float("nan")
    return 100.0 * (baseline - candidate) / abs(baseline)


def _plot_token_control_efficiency(
    run_dir: Path,
    fig_dir: Path,
    result: PlotResult,
    selected_runs: dict[str, int],
) -> None:
    rows = _efficiency_rows(run_dir, selected_runs)
    if not rows:
        result.add_skipped("token_control_efficiency", "missing ChatGPT raw responses or selected rollouts")
        return

    for row in rows:
        labels = ["Neural", "ChatGPT 5.5 High", "Greedy"]
        colors = [POLICY_COLORS["neural"], POLICY_COLORS["chatgpt"], POLICY_COLORS["greedy"]]
        active_values = [row["neural_active_nodes"], row["chatgpt_active_nodes"], row["greedy_active_nodes"]]
        norm_values = [row["neural_mean_norm"], row["chatgpt_mean_norm"], row["greedy_mean_norm"]]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        axes[0].bar(labels, active_values, color=colors)
        axes[0].set_title("Active Node-steps")
        axes[0].set_ylabel("Count")
        axes[0].text(
            0.03,
            0.94,
            f"neural {int(row['neural_active_nodes']):,}\nChatGPT {int(row['chatgpt_active_nodes']):,}\ngreedy {int(row['greedy_active_nodes']):,}",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
        )

        axes[1].bar(labels, norm_values, color=colors)
        axes[1].set_title("Mean Control Density")
        axes[1].set_ylabel("Mean L2 norm per node-step")
        axes[1].set_yscale("log")
        axes[1].text(
            0.03,
            0.94,
            f"neural {row['neural_mean_norm']:.3f}\nChatGPT {row['chatgpt_mean_norm']:.3f}\ngreedy {row['greedy_mean_norm']:.5f}",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
        )

        for ax in axes:
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=15)
        fig.suptitle(f"Control Efficiency: {_case_label(str(row['folder']), int(row['run_idx']))}")
        _save(fig, fig_dir, f"{_case_slug(str(row['folder']), int(row['run_idx']))}_token_control_efficiency.png", result)

    if len(rows) > 1:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        labels = [row["label"] for row in rows]
        x = np.arange(len(rows))
        width = 0.35
        ax.bar(
            x - width / 2,
            [row["neural_active_nodes"] for row in rows],
            width,
            label="Neural active node-steps",
        )
        ax.bar(
            x + width / 2,
            [row["chatgpt_active_nodes"] for row in rows],
            width,
            label="ChatGPT active node-steps",
        )
        _label_bars(ax, fmt="{:.0f}")
        ax.set_title("Active Control Coverage Across Demo Graphs")
        ax.set_ylabel("Active node-steps")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        _save(fig, fig_dir, "selected_runs_control_coverage_values.png", result)


def _efficiency_rows(run_dir: Path, selected_runs: dict[str, int]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    raw_dir = run_dir / "chatgpt55-high-comparison" / "raw_responses"
    if not raw_dir.exists():
        return rows
    for folder, run_idx in selected_runs.items():
        label = _chatgpt_label_for(folder, run_idx)
        if label is None:
            continue
        neural = _load_pt(_policy_path(run_dir / folder, "neural", run_idx))
        greedy = _load_pt(_policy_path(run_dir / folder, "greedy", run_idx))
        chatgpt = _load_chatgpt_rollouts_for(run_dir, folder, run_idx).get("chatgpt")
        if neural is None or greedy is None or chatgpt is None:
            continue
        chatgpt_tokens = _chatgpt_token_total(raw_dir, label)
        if chatgpt_tokens <= 0:
            continue
        neural_stats = _control_stats(neural)
        greedy_stats = _control_stats(greedy)
        chatgpt_stats = _control_stats(chatgpt)
        rows.append(
            {
                "label": _case_label(folder, run_idx),
                "folder": folder,
                "run_idx": run_idx,
                "chatgpt_tokens": chatgpt_tokens,
                "tokens_per_chatgpt_action": chatgpt_tokens / max(1.0, chatgpt_stats["active_nodes"]),
                "neural_active_nodes": neural_stats["active_nodes"],
                "chatgpt_active_nodes": chatgpt_stats["active_nodes"],
                "greedy_active_nodes": greedy_stats["active_nodes"],
                "neural_mean_norm": neural_stats["mean_norm"],
                "chatgpt_mean_norm": chatgpt_stats["mean_norm"],
                "greedy_mean_norm": greedy_stats["mean_norm"],
                "neural_vs_chatgpt_active_x": neural_stats["active_nodes"] / max(1.0, chatgpt_stats["active_nodes"]),
                "neural_vs_greedy_active_x": neural_stats["active_nodes"] / max(1.0, greedy_stats["active_nodes"]),
                "neural_vs_chatgpt_norm_x": neural_stats["mean_norm"] / max(1e-12, chatgpt_stats["mean_norm"]),
                "neural_vs_greedy_norm_x": neural_stats["mean_norm"] / max(1e-12, greedy_stats["mean_norm"]),
            }
        )
    return rows


def _chatgpt_label_for(folder: str, run_idx: int) -> str | None:
    if "48" in folder:
        return f"48_run_{run_idx:03d}"
    if "20" in folder or "512" in folder:
        return f"512_run_{run_idx:03d}"
    return None


def _chatgpt_token_total(raw_dir: Path, label: str) -> float:
    import re

    total = 0
    for path in sorted(raw_dir.glob(f"{label}_step_*.txt")):
        match = re.search(r"tokens used\s*\n([0-9,]+)", path.read_text(encoding="utf-8"))
        if match:
            total += int(match.group(1).replace(",", ""))
    return float(total)


def _control_stats(payload: dict[str, Any]) -> dict[str, float]:
    controls = _to_numpy(payload.get("controls"))
    if controls.size == 0:
        return {"active_nodes": 0.0, "mean_norm": 0.0}
    node_norm = np.linalg.norm(controls, axis=-1)
    return {
        "active_nodes": float((node_norm > 1e-8).sum()),
        "mean_norm": float(node_norm.mean()),
    }


def _plot_graph_diagnostics(
    run_dir: Path,
    fig_dir: Path,
    result: PlotResult,
    closed_loop_names: list[str],
    selected_runs: dict[str, int],
) -> None:
    plotted_any = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for idx, name in enumerate(closed_loop_names[:2]):
        graph = _load_pt(run_dir / name / "graph.pt")
        if not graph:
            continue
        edge_index = graph.get("edge_index")
        labels = graph.get("labels")
        if not isinstance(edge_index, torch.Tensor) or not isinstance(labels, torch.Tensor):
            continue
        degrees = torch.bincount(edge_index[0].long(), minlength=int(labels.numel())).cpu().numpy()
        axes[0].hist(degrees, bins=min(30, max(5, int(degrees.max()) + 1)), alpha=0.55, label=_case_label(name))
        counts = torch.bincount(labels.long()).cpu().numpy()
        axes[1].plot(np.arange(len(counts)), counts, marker="o", linewidth=1.5, label=_case_label(name))
        plotted_any = True
    if plotted_any:
        axes[0].set_title("Graph Degree Distribution")
        axes[0].set_xlabel("Degree")
        axes[0].set_ylabel("Nodes")
        axes[1].set_title("Community Sizes")
        axes[1].set_xlabel("Community")
        axes[1].set_ylabel("Nodes")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend()
        _save(fig, fig_dir, "graph_degree_and_communities.png", result)
    else:
        plt.close(fig)
        result.add_skipped("graph_degree_and_communities.png", "missing graph.pt files")

    for name, run_idx in selected_runs.items():
        _plot_graph_layout_error(run_dir, fig_dir, result, name, run_idx)


def _plot_graph_layout_error(
    run_dir: Path,
    fig_dir: Path,
    result: PlotResult,
    name: str,
    run_idx: int,
) -> None:
    layout = _load_json(run_dir / name / "graph_layout.json")
    payload = _load_pt(_policy_path(run_dir / name, "neural", run_idx))
    if not layout or payload is None:
        result.add_skipped(f"{_case_slug(name, run_idx)}_layout_error.png", "missing layout or neural rollout")
        return

    positions = _layout_positions(layout)
    errors = _to_numpy(payload.get("node_error"))
    labels = _to_numpy(payload.get("labels"))
    if positions.size == 0 or errors.size == 0:
        result.add_skipped(f"{_case_slug(name, run_idx)}_layout_error.png", "layout/error arrays empty")
        return

    final_error = errors[-1]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=final_error,
        s=18 if len(final_error) > 128 else 48,
        cmap="magma",
        alpha=0.86,
        edgecolors="none",
    )
    if labels.size:
        centers = []
        for label in sorted(set(labels.astype(int).tolist())):
            pts = positions[labels.astype(int) == label]
            if pts.size:
                centers.append((label, pts[:, 0].mean(), pts[:, 1].mean()))
        for label, x, y in centers:
            ax.text(x, y, str(label), ha="center", va="center", fontsize=7, color="#111111")
    ax.set_title(f"Neural Final Node Error: {_case_label(name, run_idx)}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, label="Final node error")
    _save(fig, fig_dir, f"{_case_slug(name, run_idx)}_layout_error.png", result)


def _layout_positions(layout: dict[str, Any]) -> np.ndarray:
    for key in ["positions", "pos", "layout"]:
        value = layout.get(key)
        if isinstance(value, dict):
            ordered = sorted(value.items(), key=lambda item: int(item[0]))
            return np.asarray([coords for _, coords in ordered], dtype=float)
        if isinstance(value, list):
            return np.asarray(value, dtype=float)
    if "x" in layout and "y" in layout:
        return np.column_stack([np.asarray(layout["x"], dtype=float), np.asarray(layout["y"], dtype=float)])
    return np.asarray([])


def _plot_chatgpt_timing(run_dir: Path, fig_dir: Path, result: PlotResult, include_fake: bool = False) -> None:
    candidates = [run_dir / "chatgpt55-high-comparison"]
    if include_fake:
        candidates.extend(sorted(run_dir.glob("chatgpt55-high-comparison*fake*")))
    timing_path = next((path / "timing.json" for path in candidates if (path / "timing.json").exists()), None)
    if timing_path is None:
        result.add_skipped("chatgpt_wallclock.png", "missing real chatgpt55-high-comparison/timing.json")
        return
    timing = _load_json(timing_path) or {}
    rows = _flatten_timing(timing)
    if not rows:
        result.add_skipped("chatgpt_wallclock.png", "timing.json has no per-step timing rows")
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [row["label"] for row in rows]
    values = [row["seconds"] for row in rows]
    ax.bar(labels, values, color="#7570b3")
    ax.set_title("ChatGPT 5.5 High Wall-clock Time")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, fig_dir, "chatgpt_wallclock.png", result)


def _flatten_timing(timing: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rollouts = timing.get("rollouts")
    if isinstance(rollouts, list):
        for rollout in rollouts:
            if not isinstance(rollout, dict):
                continue
            label = str(rollout.get("run_label", f"run_{len(rows)}"))
            total = rollout.get("chatgpt_rollout_wall_time_sec")
            if isinstance(total, (int, float)):
                rows.append({"label": f"{label} total", "seconds": float(total)})
            step_times = rollout.get("chatgpt_step_wall_time_sec")
            if isinstance(step_times, list) and step_times:
                rows.append({"label": f"{label} avg step", "seconds": float(np.mean(step_times))})
        total = timing.get("total_wall_time_sec")
        if isinstance(total, (int, float)):
            rows.append({"label": "parallel total", "seconds": float(total)})
        return rows
    for key, value in timing.items():
        if isinstance(value, (int, float)):
            rows.append({"label": key, "seconds": float(value)})
        elif isinstance(value, dict):
            total = value.get("total_wall_time_sec") or value.get("wall_time_sec") or value.get("total_seconds")
            if isinstance(total, (int, float)):
                rows.append({"label": key, "seconds": float(total)})
    return rows[:24]


def _plot_artifact_sizes(run_dir: Path, fig_dir: Path, result: PlotResult) -> None:
    folders = [path for path in run_dir.iterdir() if path.is_dir()]
    folders.append(run_dir)
    rows = []
    for folder in folders:
        size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
        rows.append((folder.name, size / (1024 * 1024)))
    rows = sorted(rows, key=lambda item: item[1], reverse=True)[:12]
    if not rows:
        result.add_skipped("artifact_storage_sizes.png", "no artifact folders found")
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([row[0] for row in rows], [row[1] for row in rows], color="#4c78a8")
    ax.set_title("Run Artifact Storage Footprint")
    ax.set_ylabel("MiB")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, fig_dir, "artifact_storage_sizes.png", result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate plots from manifold run artifacts.")
    parser.add_argument("--run-dir", default="outputs/demo-512-c12-final")
    parser.add_argument("--fig-dir", default="manifold/docs/figs")
    parser.add_argument(
        "--closed-loop-dir",
        action="append",
        dest="closed_loop_dirs",
        help="Closed-loop artifact folder to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--selected-run",
        action="append",
        default=[],
        metavar="FOLDER:RUN",
        help="Selected demo rollout, for example closed-loop-eval-48-t8:14.",
    )
    parser.add_argument("--include-fake-chatgpt", action="store_true")
    return parser


def _parse_selected_runs(items: list[str]) -> dict[str, int] | None:
    if not items:
        return None
    selected: dict[str, int] = {}
    for item in items:
        folder, sep, run = item.partition(":")
        if not sep:
            raise ValueError(f"Expected FOLDER:RUN for --selected-run, got {item!r}")
        selected[folder] = int(run)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = plot_all(
        run_dir=args.run_dir,
        fig_dir=args.fig_dir,
        closed_loop_dirs=args.closed_loop_dirs,
        selected_runs=_parse_selected_runs(args.selected_run),
        include_fake_chatgpt=args.include_fake_chatgpt,
    )
    print(json.dumps({"fig_dir": str(result.fig_dir), "generated": len(result.generated), "skipped": result.skipped}, indent=2))
    return 0
