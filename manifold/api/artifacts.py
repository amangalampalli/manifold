"""Closed-loop artifact loading and browser-safe serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import torch


POLICIES = ("random", "greedy", "neural", "chatgpt")
DEFAULT_ARTIFACT_ROOT = Path("outputs/demo-512-c12-final")


@dataclass(frozen=True)
class ClosedLoopRun:
    id: str
    path: Path
    label: str
    default_run_idx: int = 0


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a requested closed-loop artifact cannot be found."""


class ClosedLoopArtifactStore:
    """Loads closed-loop export artifacts and converts them for the frontend."""

    def __init__(self, root: str | Path | None = None) -> None:
        env_root = os.getenv("MANIFOLD_ARTIFACT_ROOT")
        self.root = Path(root or env_root or DEFAULT_ARTIFACT_ROOT)

    def list_runs(self) -> list[dict[str, Any]]:
        return [self.summary(run.id) for run in self._runs()]

    def summary(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        manifest = self._manifest(run.path)
        by_policy = self._policy_averages(manifest)
        chatgpt_summary = self._chatgpt_policy_summary(run)
        if chatgpt_summary:
            by_policy["chatgpt"] = chatgpt_summary
        greedy = by_policy.get("greedy")
        neural = by_policy.get("neural")
        chatgpt = by_policy.get("chatgpt")

        deltas: dict[str, float] = {}
        if greedy and neural:
            deltas["mean_error_auc_reduction_pct"] = _improvement_pct(
                greedy["mean_error_auc"], neural["mean_error_auc"]
            )
            deltas["final_error_reduction_pct"] = _improvement_pct(
                greedy["final_mean_error"], neural["final_mean_error"]
            )
            deltas["mean_error_mse_reduction_pct"] = self._mse_reduction_pct(run.path)
        if chatgpt and neural:
            deltas["chatgpt_auc_reduction_pct"] = _improvement_pct(
                chatgpt["mean_error_auc"], neural["mean_error_auc"]
            )

        headline_metrics = _headline_metrics(deltas)
        config = _read_json_if_present(run.path / "config.json")
        startup = _read_json_if_present(run.path / "startup.json")
        source = _read_json_if_present(run.path / "source.json")
        if not config and source.get("source_run_dir"):
            config = _read_json_if_present(Path(source["source_run_dir"]) / "config.json")

        return {
            "id": run.id,
            "label": run.label,
            "defaultRunIdx": run.default_run_idx,
            "artifactPath": str(run.path),
            "policies": by_policy,
            "deltas": deltas,
            "headlineMetrics": headline_metrics,
            "sensing": {
                "budget": _nested_get(config, ("sensing", "budget")),
                "kHop": _nested_get(config, ("sensing", "k_hop")),
                "noiseStd": _nested_get(config, ("sensing", "noise_std")),
            },
            "model": {
                "sheafLambda": _nested_get(config, ("model", "sheaf_lambda")),
                "latentDim": _nested_get(config, ("model", "latent_dim")),
            },
            "graph": {
                "numNodes": startup.get("num_nodes") or _nested_get(config, ("graph", "num_nodes")),
                "numCommunities": startup.get("num_communities")
                or _nested_get(config, ("graph", "num_communities")),
                "pIn": _nested_get(config, ("graph", "p_in")),
                "pOut": _nested_get(config, ("graph", "p_out")),
            },
            "source": source,
        }

    def graph(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        graph_path = run.path / "graph.pt"
        if not graph_path.exists():
            raise ArtifactNotFoundError(f"missing graph artifact: {graph_path}")
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        labels = graph["labels"].to(torch.long).tolist()
        layout = _read_json_if_present(run.path / "graph_layout.json")
        positions = layout.get("positions", {})
        edge_index = graph["edge_index"].to(torch.long)
        edge_attr = graph["edge_attr"].to(torch.float32)

        nodes = []
        for idx, label in enumerate(labels):
            xy = positions.get(str(idx), [0.0, 0.0])
            nodes.append(
                {
                    "id": idx,
                    "community": int(label),
                    "x": _round(float(xy[0])),
                    "y": _round(float(xy[1])),
                }
            )

        seen: set[tuple[int, int]] = set()
        edges = []
        for edge_idx in range(edge_index.shape[1]):
            source = int(edge_index[0, edge_idx])
            target = int(edge_index[1, edge_idx])
            left, right = sorted((source, target))
            if left == right or (left, right) in seen:
                continue
            seen.add((left, right))
            attr = edge_attr[edge_idx].tolist()
            edges.append(
                {
                    "source": left,
                    "target": right,
                    "sameCommunity": bool(round(float(attr[0]))) if attr else False,
                    "friction": _round(float(attr[1])) if len(attr) > 1 else 0.0,
                    "distortion": _round(float(attr[2])) if len(attr) > 2 else 0.0,
                    "communityGap": _round(float(attr[3])) if len(attr) > 3 else 0.0,
                }
            )

        return {
            "runId": run.id,
            "numNodes": int(graph["num_nodes"]),
            "numCommunities": int(graph["num_communities"]),
            "nodes": nodes,
            "edges": edges,
        }

    def rollout(self, run_id: str, policy: str, run_idx: int = 0) -> dict[str, Any]:
        if policy not in POLICIES:
            raise ArtifactNotFoundError(f"unknown policy: {policy}")
        if policy == "chatgpt":
            return self._chatgpt_rollout(run_id, run_idx)
        run = self._run(run_id)
        manifest = self._manifest(run.path)
        entry = next(
            (
                item
                for item in manifest
                if item.get("policy") == policy and int(item.get("run_idx", -1)) == run_idx
            ),
            None,
        )
        if entry is None:
            raise ArtifactNotFoundError(f"missing rollout for {policy} run {run_idx}")

        payload_path = run.path / str(entry["path"])
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        node_error = payload["node_error"].to(torch.float32)
        pred = payload["pred_trajectory"].to(torch.float32)
        controls = payload.get("controls", torch.empty(0)).to(torch.float32)
        corrected = payload.get("corrected_states", torch.empty(0)).to(torch.float32)
        initial_perturbation = torch.linalg.vector_norm(
            payload["initial_belief"].to(torch.float32)
            - payload["initial_clean_state"].to(torch.float32),
            dim=-1,
        )

        return {
            "runId": run.id,
            "policy": policy,
            "runIdx": run_idx,
            "times": _tensor_list(payload["times"]),
            "meanError": _tensor_list(payload["mean_error"]),
            "nodeError": _tensor_matrix(node_error),
            "initialPerturbation": _tensor_list(initial_perturbation),
            "stateProjection": _projection(pred),
            "controlEnergy": _per_step_norm(controls),
            "controlMagnitude": _tensor_matrix(torch.linalg.vector_norm(controls, dim=-1))
            if controls.numel()
            else [],
            "correctionEnergy": _per_step_delta(corrected, pred[1:]),
            "residualEnergy": _per_step_delta(
                corrected, payload["target_trajectory"].to(torch.float32)[1:]
            ),
            "probeNodes": _int_rows(payload.get("probe_nodes", torch.empty(0))),
            "observedNodes": _mask_rows(payload.get("observed_masks", torch.empty(0))),
            "finalMeanError": _round(float(payload["mean_error"][-1])),
            "meanErrorAuc": _round(float(torch.trapz(payload["mean_error"], payload["times"]))),
        }

    def _chatgpt_rollout(self, run_id: str, run_idx: int) -> dict[str, Any]:
        run = self._run(run_id)
        entry = self._chatgpt_entry(run, run_idx)
        comparison_root = self.root / "chatgpt55-high-comparison"
        payload = torch.load(
            comparison_root / str(entry["path"]), map_location="cpu", weights_only=False
        )
        serialized = self._serialize_payload(run.id, "chatgpt", run_idx, payload)
        serialized["chatgptDecisions"] = payload.get("chatgpt_decisions", [])
        serialized["modelLabel"] = payload.get("policy", "ChatGPT")
        serialized["validDecisions"] = int(entry.get("valid_decisions", 0))
        return serialized

    def telemetry(self, run_id: str, run_idx: int = 0) -> dict[str, Any]:
        neural = self.rollout(run_id, "neural", run_idx)
        lines = []
        for step_idx, time_value in enumerate(neural["times"]):
            if step_idx == 0:
                lines.append(
                    {
                        "step": step_idx,
                        "message": "belief initialized from disturbed manifold state",
                    }
                )
                continue
            probes = neural["probeNodes"][step_idx - 1]
            observed = neural["observedNodes"][step_idx - 1]
            lines.append(
                {
                    "step": step_idx,
                    "message": (
                        f"t+{time_value:.2f} probe={probes[:5]} "
                        f"observed={len(observed)} control={neural['controlEnergy'][step_idx - 1]:.4f}"
                    ),
                }
            )
        return {"runId": run_id, "lines": lines}

    def _serialize_payload(
        self, run_id: str, policy: str, run_idx: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        node_error = payload["node_error"].to(torch.float32)
        pred = payload["pred_trajectory"].to(torch.float32)
        controls = payload.get("controls", torch.empty(0)).to(torch.float32)
        corrected = payload.get("corrected_states", torch.empty(0)).to(torch.float32)
        initial_perturbation = torch.linalg.vector_norm(
            payload["initial_belief"].to(torch.float32)
            - payload["initial_clean_state"].to(torch.float32),
            dim=-1,
        )
        return {
            "runId": run_id,
            "policy": policy,
            "runIdx": run_idx,
            "times": _tensor_list(payload["times"]),
            "meanError": _tensor_list(payload["mean_error"]),
            "nodeError": _tensor_matrix(node_error),
            "initialPerturbation": _tensor_list(initial_perturbation),
            "stateProjection": _projection(pred),
            "controlEnergy": _per_step_norm(controls),
            "controlMagnitude": _tensor_matrix(torch.linalg.vector_norm(controls, dim=-1))
            if controls.numel()
            else [],
            "correctionEnergy": _per_step_delta(corrected, pred[1:]),
            "residualEnergy": _per_step_delta(
                corrected, payload["target_trajectory"].to(torch.float32)[1:]
            ),
            "probeNodes": _int_rows(payload.get("probe_nodes", torch.empty(0))),
            "observedNodes": _mask_rows(payload.get("observed_masks", torch.empty(0))),
            "finalMeanError": _round(float(payload["mean_error"][-1])),
            "meanErrorAuc": _round(float(torch.trapz(payload["mean_error"], payload["times"]))),
        }

    def _runs(self) -> list[ClosedLoopRun]:
        root = self.root
        manifest_name = "closed_loop_rollouts_manifest.json"
        if (root / manifest_name).exists():
            paths = [root]
        elif root.exists():
            paths = sorted(path.parent for path in root.glob(f"**/{manifest_name}"))
        else:
            paths = []
        preferred = [
            path
            for path in paths
            if path.name in {"closed-loop-eval-48-t8", "closed-loop-eval-20-t8"}
        ]
        if preferred:
            paths = preferred
        runs = [self._run_from_path(path) for path in paths]
        return _dedupe_run_ids(sorted(runs, key=_run_sort_key))

    def _run(self, run_id: str) -> ClosedLoopRun:
        for run in self._runs():
            if run.id == run_id:
                return run
        raise ArtifactNotFoundError(f"unknown run: {run_id}")

    def _run_from_path(self, path: Path) -> ClosedLoopRun:
        if path.name == "closed-loop-eval-48-t8":
            label = "48 node"
            run_id = "graph-48"
            default_run_idx = 14
        elif path.name == "closed-loop-eval-20-t8":
            label = "512 node"
            run_id = "graph-512"
            default_run_idx = 15
        elif path.name == "closed-loop-eval" and path.parent.name == "demo-512-c12-final":
            label = "512 node"
            run_id = "graph-512-legacy"
            default_run_idx = 0
        elif path.name == "closed-loop-eval" and path.parent:
            label = path.parent.name
            run_id = _slug(label)
            default_run_idx = 0
        elif path.name in {"closed-loop", "closed_loop_eval"} and path.parent:
            label = path.parent.name
            run_id = _slug(label)
            default_run_idx = 0
        else:
            label = path.name
            run_id = _slug(label)
            default_run_idx = 0
        return ClosedLoopRun(
            id=run_id, path=path, label=label, default_run_idx=default_run_idx
        )

    def _manifest(self, run_path: Path) -> list[dict[str, Any]]:
        manifest_path = run_path / "closed_loop_rollouts_manifest.json"
        if not manifest_path.exists():
            raise ArtifactNotFoundError(f"missing closed-loop manifest: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _policy_averages(self, manifest: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
        for item in manifest:
            policy = item.get("policy")
            if policy in grouped:
                grouped[policy].append(item)

        averages = {}
        for policy, items in grouped.items():
            if not items:
                continue
            averages[policy] = {
                "runs": float(len(items)),
                "final_mean_error": _round(
                    sum(float(item["final_mean_error"]) for item in items) / len(items)
                ),
                "mean_error_auc": _round(
                    sum(float(item["mean_error_auc"]) for item in items) / len(items)
                ),
            }
        return averages

    def _mse_reduction_pct(self, run_path: Path) -> float:
        try:
            greedy = self._load_policy_payload(run_path, "greedy")
            neural = self._load_policy_payload(run_path, "neural")
        except ArtifactNotFoundError:
            return 0.0
        greedy_mse = float((greedy["node_error"].to(torch.float32) ** 2).mean())
        neural_mse = float((neural["node_error"].to(torch.float32) ** 2).mean())
        return _improvement_pct(greedy_mse, neural_mse)

    def _chatgpt_policy_summary(self, run: ClosedLoopRun) -> dict[str, float] | None:
        try:
            entry = self._chatgpt_entry(run, run.default_run_idx)
        except ArtifactNotFoundError:
            return None
        return {
            "runs": 1.0,
            "final_mean_error": _round(float(entry["final_mean_error"])),
            "mean_error_auc": _round(float(entry["mean_error_auc"])),
        }

    def _chatgpt_entry(self, run: ClosedLoopRun, run_idx: int) -> dict[str, Any]:
        manifest_path = (
            self.root / "chatgpt55-high-comparison" / "chatgpt_rollouts_manifest.json"
        )
        if not manifest_path.exists():
            raise ArtifactNotFoundError(f"missing ChatGPT manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_label = f"{self._summary_node_count(run)}_run_{run_idx:03d}"
        entry = next((item for item in manifest if item.get("run_label") == run_label), None)
        if entry is None:
            raise ArtifactNotFoundError(f"missing ChatGPT rollout for {run_label}")
        return entry

    def _summary_node_count(self, run: ClosedLoopRun) -> int:
        if run.id == "graph-48":
            return 48
        if run.id.startswith("graph-512"):
            return 512
        startup = _read_json_if_present(run.path / "startup.json")
        return int(startup.get("num_nodes") or 0)

    def _load_policy_payload(self, run_path: Path, policy: str) -> dict[str, Any]:
        manifest = self._manifest(run_path)
        entry = next((item for item in manifest if item.get("policy") == policy), None)
        if entry is None:
            raise ArtifactNotFoundError(f"missing {policy} manifest entry")
        return torch.load(run_path / str(entry["path"]), map_location="cpu", weights_only=False)


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _dedupe_run_ids(runs: list[ClosedLoopRun]) -> list[ClosedLoopRun]:
    counts: dict[str, int] = {}
    deduped = []
    for run in runs:
        count = counts.get(run.id, 0)
        counts[run.id] = count + 1
        if count == 0:
            deduped.append(run)
        else:
            deduped.append(
                ClosedLoopRun(
                    id=f"{run.id}-{count + 1}",
                    path=run.path,
                    label=f"{run.label} {count + 1}",
                    default_run_idx=run.default_run_idx,
                )
            )
    return deduped


def _run_sort_key(run: ClosedLoopRun) -> tuple[int, str]:
    if run.id == "graph-48":
        return (0, run.label)
    if run.id == "graph-512":
        return (1, run.label)
    return (2, run.label)


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _round(value: float, digits: int = 5) -> float:
    return round(value, digits)


def _improvement_pct(baseline: float, contender: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return _round(100.0 * (baseline - contender) / abs(baseline), 3)


def _headline_metrics(deltas: dict[str, float]) -> list[dict[str, Any]]:
    candidates = [
        (
            "Error AUC Reduced",
            "mean_error_auc_reduction_pct",
            "Mean trajectory error over the recurrent sensing horizon.",
        ),
        (
            "MSE Reduced",
            "mean_error_mse_reduction_pct",
            "Squared node-error reduction over the closed-loop rollout.",
        ),
        (
            "Terminal Error Reduced",
            "final_error_reduction_pct",
            "Final-step mean error reduction at mission end.",
        ),
    ]
    metrics = []
    for label, key, description in candidates:
        value = deltas.get(key)
        if value is not None and value > 0.0:
            metrics.append(
                {
                    "label": label,
                    "value": value,
                    "unit": "%",
                    "description": description,
                    "polarity": "lower_is_better",
                }
            )
    return sorted(metrics, key=lambda item: item["value"], reverse=True)


def _tensor_list(tensor: torch.Tensor) -> list[float]:
    return [_round(float(value)) for value in tensor.detach().cpu().flatten().tolist()]


def _tensor_matrix(tensor: torch.Tensor) -> list[list[float]]:
    return [[_round(float(value)) for value in row] for row in tensor.detach().cpu().tolist()]


def _projection(tensor: torch.Tensor) -> list[list[list[float]]]:
    projected = tensor[..., :3]
    if projected.shape[-1] < 3:
        pad = torch.zeros(*projected.shape[:-1], 3 - projected.shape[-1])
        projected = torch.cat([projected, pad], dim=-1)
    return [
        [[_round(float(component)) for component in node] for node in step]
        for step in projected.detach().cpu().tolist()
    ]


def _per_step_norm(tensor: torch.Tensor) -> list[float]:
    if tensor.numel() == 0:
        return []
    values = torch.linalg.vector_norm(tensor, dim=-1).mean(dim=-1)
    return _tensor_list(values)


def _per_step_delta(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    if left.numel() == 0 or right.numel() == 0:
        return []
    size = min(left.shape[0], right.shape[0])
    values = torch.linalg.vector_norm(left[:size] - right[:size], dim=-1).mean(dim=-1)
    return _tensor_list(values)


def _int_rows(tensor: torch.Tensor) -> list[list[int]]:
    if tensor.numel() == 0:
        return []
    return [[int(value) for value in row] for row in tensor.detach().cpu().tolist()]


def _mask_rows(tensor: torch.Tensor) -> list[list[int]]:
    if tensor.numel() == 0:
        return []
    rows = []
    for row in tensor.detach().cpu():
        rows.append([int(idx) for idx in torch.nonzero(row, as_tuple=False).flatten().tolist()])
    return rows
