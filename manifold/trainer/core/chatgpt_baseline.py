"""ChatGPT 5.5 High baseline export helpers."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from tqdm.auto import tqdm

from manifold.data.types import GraphSignalSample
from manifold.trainer.support.policies import BaselinePolicy
from manifold.trainer.support.utils import write_json_file


@dataclass(frozen=True)
class ChatGPTDecision:
    selected_nodes: list[int]
    damping_gain: float
    laplacian_gain: float
    center_pull_gain: float
    confidence: float
    rationale: str
    valid: bool = True


class ChatGPTRunner(Protocol):
    def __call__(self, prompt: str) -> str:
        ...


class CodexExecChatGPTRunner:
    def __init__(self, *, model: str, reasoning: str, cwd: Path) -> None:
        self.model = model
        self.reasoning = reasoning
        self.cwd = cwd

    def __call__(self, prompt: str) -> str:
        completed = subprocess.run(
            [
                "codex",
                "exec",
                "-m",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning}"',
                "--sandbox",
                "read-only",
                "--cd",
                str(self.cwd),
                "-",
            ],
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout)
        return completed.stdout


class FakeChatGPTRunner:
    def __call__(self, prompt: str) -> str:
        del prompt
        return json.dumps(
            {
                "selected_nodes": [0, 1, 2, 999999],
                "damping_gain": 0.04,
                "laplacian_gain": 0.02,
                "center_pull_gain": 0.03,
                "confidence": 0.5,
                "rationale": "deterministic fake response",
            }
        )


class RawResponseReplayRunner:
    def __init__(self, *, raw_dir: Path, run_label: str) -> None:
        self.raw_dir = raw_dir
        self.run_label = run_label
        self.step_idx = 0

    def __call__(self, prompt: str) -> str:
        del prompt
        path = self.raw_dir / f"{self.run_label}_step_{self.step_idx:02d}.txt"
        self.step_idx += 1
        return path.read_text(encoding="utf-8")


def parse_chatgpt_decision(raw: str) -> ChatGPTDecision:
    try:
        payload = json.loads(_extract_json_object(raw))
        return ChatGPTDecision(
            selected_nodes=[int(node) for node in payload.get("selected_nodes", [])],
            damping_gain=float(payload.get("damping_gain", 0.0)),
            laplacian_gain=float(payload.get("laplacian_gain", 0.0)),
            center_pull_gain=float(payload.get("center_pull_gain", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            rationale=str(payload.get("rationale", ""))[:500],
            valid=True,
        )
    except Exception:
        return zero_chatgpt_decision(valid=False, rationale="invalid JSON response")


def zero_chatgpt_decision(*, valid: bool, rationale: str) -> ChatGPTDecision:
    return ChatGPTDecision(
        selected_nodes=[],
        damping_gain=0.0,
        laplacian_gain=0.0,
        center_pull_gain=0.0,
        confidence=0.0,
        rationale=rationale,
        valid=valid,
    )


def sanitize_chatgpt_decision(
    decision: ChatGPTDecision,
    *,
    observed_mask: torch.Tensor,
    budget: int,
    max_gain: float = 0.15,
) -> ChatGPTDecision:
    visible = set(torch.nonzero(observed_mask.detach().cpu(), as_tuple=False).flatten().tolist())
    selected: list[int] = []
    for node in decision.selected_nodes:
        if node in visible and node not in selected:
            selected.append(node)
        if len(selected) >= budget:
            break
    return ChatGPTDecision(
        selected_nodes=selected,
        damping_gain=_clamp(decision.damping_gain, 0.0, max_gain),
        laplacian_gain=_clamp(decision.laplacian_gain, 0.0, max_gain),
        center_pull_gain=_clamp(decision.center_pull_gain, 0.0, max_gain),
        confidence=_clamp(decision.confidence, 0.0, 1.0),
        rationale=decision.rationale,
        valid=decision.valid,
    )


def chatgpt_decision_to_control(
    trainer,
    h: torch.Tensor,
    decision: ChatGPTDecision,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    control = torch.zeros(h.size(0), trainer.config.model.control_dim, dtype=h.dtype, device=h.device)
    if not decision.selected_nodes:
        return control
    selected = torch.tensor(decision.selected_nodes, dtype=torch.long, device=h.device)
    selected = selected[observed_mask[selected]]
    if selected.numel() == 0:
        return control
    edge_index = trainer.dataset.edge_index.to(h.device)
    edge_attr = trainer.dataset.edge_attr.to(h.device)
    lap, _ = trainer.model.sheaf.apply_laplacian(h, edge_index, edge_attr)
    centers = trainer.dataset.community_centers.to(h.device, h.dtype)
    labels = trainer.dataset.labels.to(h.device)
    center_pull = centers[labels[selected]] - h[selected]
    shared_dim = min(h.size(1), control.size(1))
    raw = (
        -decision.damping_gain * h[selected, :shared_dim]
        - decision.laplacian_gain * lap[selected, :shared_dim]
        + decision.center_pull_gain * center_pull[:, :shared_dim]
    )
    control[selected, :shared_dim] = raw.clamp(-trainer.config.training.source_scale, trainer.config.training.source_scale)
    return control


def build_chatgpt_prompt(
    *,
    run_label: str,
    step_idx: int,
    h: torch.Tensor,
    observed_mask: torch.Tensor,
    probe_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    budget: int,
    previous_control: torch.Tensor | None,
) -> str:
    visible_nodes = torch.nonzero(observed_mask.detach().cpu(), as_tuple=False).flatten()
    h_cpu = h.detach().cpu()
    labels_cpu = labels.detach().cpu()
    edge_cpu = edge_index.detach().cpu()
    node_energy = torch.linalg.vector_norm(h_cpu, dim=-1)
    visible_energy = node_energy[visible_nodes] if visible_nodes.numel() else torch.empty(0)
    top_count = min(32, visible_nodes.numel())
    if top_count:
        top_local = torch.topk(visible_energy, k=top_count).indices
        top_nodes = visible_nodes[top_local]
    else:
        top_nodes = torch.empty(0, dtype=torch.long)
    visible_set = set(visible_nodes.tolist())
    visible_edges = 0
    cross_edges = 0
    for src, dst in edge_cpu.t().tolist():
        if src in visible_set and dst in visible_set:
            visible_edges += 1
            if int(labels_cpu[src]) != int(labels_cpu[dst]):
                cross_edges += 1
    prev_norm = 0.0 if previous_control is None else float(torch.linalg.vector_norm(previous_control.detach().cpu(), dim=-1).mean())
    node_rows = [
        {
            "node": int(node),
            "community": int(labels_cpu[node]),
            "belief_norm": round(float(node_energy[node]), 6),
            "belief_head": [round(float(v), 5) for v in h_cpu[node, :4]],
        }
        for node in top_nodes.tolist()
    ]
    prompt_payload = {
        "run": run_label,
        "step_idx": step_idx,
        "visibility": {
            "budget": budget,
            "visible_node_count": int(visible_nodes.numel()),
            "probe_nodes": [int(node) for node in probe_nodes.detach().cpu().tolist()],
            "visible_edges": visible_edges,
            "visible_cross_community_edges": cross_edges,
        },
        "visible_top_energy_nodes": node_rows,
        "previous_control_mean_norm": round(prev_norm, 6),
        "allowed_response_schema": {
            "selected_nodes": "visible node ids only, max budget",
            "damping_gain": "float in [0, 0.15]",
            "laplacian_gain": "float in [0, 0.15]",
            "center_pull_gain": "float in [0, 0.15]",
            "confidence": "float in [0, 1]",
            "rationale": "short text",
        },
    }
    return (
        "You are ChatGPT 5.5 High controlling a graph signal recovery system under partial visibility.\n"
        "You must not assume access to clean targets, future states, unobserved node errors, or other policies.\n"
        "Choose controls only from the visible state summary. Return JSON only, no markdown.\n"
        f"{json.dumps(prompt_payload, sort_keys=True)}"
    )


def export_chatgpt_baseline_rollout(
    trainer,
    *,
    output_dir: Path,
    source_rollout_path: Path,
    run_label: str,
    runner: ChatGPTRunner,
    progress: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = output_dir / "prompts"
    raw_dir = output_dir / "raw_responses"
    rollout_dir = output_dir / "chatgpt_rollouts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    source = torch.load(source_rollout_path, map_location=trainer.device, weights_only=False)
    sample = GraphSignalSample(
        h0=source["initial_clean_state"].to(trainer.device),
        target_trajectory=source["target_trajectory"].to(trainer.device),
        labels=source["labels"].to(trainer.device),
        times=source["times"].to(trainer.device),
    )
    current = source["initial_belief"].to(trainer.device)
    states = [current]
    controls = []
    corrected_states = []
    probe_nodes = []
    observed_masks = []
    decisions: list[dict[str, object]] = []
    raw_paths: list[str] = []
    prompt_paths: list[str] = []
    step_times: list[float] = []
    previous_control = None
    rollout_start = time.perf_counter()

    step_iter = tqdm(
        range(sample.times.numel() - 1),
        desc=f"chatgpt:{run_label}",
        unit="step",
        disable=not progress,
        leave=False,
    )
    for step_idx in step_iter:
        probes = trainer.sensing.select_probe_nodes(current)
        observed = trainer.sensing.observation_mask(
            trainer.dataset.edge_index.to(trainer.device),
            num_nodes=current.size(0),
            probe_nodes=probes,
        )
        prompt = build_chatgpt_prompt(
            run_label=run_label,
            step_idx=step_idx,
            h=current,
            observed_mask=observed,
            probe_nodes=probes,
            edge_index=trainer.dataset.edge_index,
            labels=trainer.dataset.labels,
            budget=trainer.config.sensing.budget,
            previous_control=previous_control,
        )
        prompt_path = prompt_dir / f"{run_label}_step_{step_idx:02d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        start = time.perf_counter()
        try:
            raw = runner(prompt)
            decision = parse_chatgpt_decision(raw)
            if not decision.valid:
                repair_prompt = prompt + "\nYour previous response was invalid. Return only valid JSON."
                raw = runner(repair_prompt)
                decision = parse_chatgpt_decision(raw)
        except Exception as exc:
            raw = str(exc)
            decision = zero_chatgpt_decision(valid=False, rationale="runner failure")
        elapsed = time.perf_counter() - start
        raw_path = raw_dir / f"{run_label}_step_{step_idx:02d}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        decision = sanitize_chatgpt_decision(
            decision,
            observed_mask=observed,
            budget=trainer.config.sensing.budget,
            max_gain=0.15,
        )
        control = chatgpt_decision_to_control(trainer, current, decision, observed)
        local_times = sample.times[step_idx : step_idx + 2] - sample.times[step_idx]
        pred_window = trainer.model.rollout(
            current,
            local_times,
            edge_index=trainer.dataset.edge_index,
            edge_attr=trainer.dataset.edge_attr,
            control=control,
            step_size=trainer.config.training.dt,
        )
        belief = trainer.sensing(
            pred_window[-1],
            sample.target_trajectory[step_idx + 1],
            trainer.dataset.edge_index,
        )
        current = belief.h_corr
        previous_control = control
        states.append(current)
        controls.append(control)
        corrected_states.append(belief.h_corr)
        probe_nodes.append(probes)
        observed_masks.append(observed)
        decisions.append({**decision.__dict__, "step_idx": step_idx})
        raw_paths.append(str(raw_path.relative_to(output_dir)))
        prompt_paths.append(str(prompt_path.relative_to(output_dir)))
        step_times.append(elapsed)
        if progress:
            step_iter.set_postfix(
                valid=int(decision.valid),
                visible=int(observed.sum().detach().cpu()),
                selected=len(decision.selected_nodes),
                sec=f"{elapsed:.1f}",
            )

    pred = torch.stack(states, dim=0)
    node_error = torch.linalg.vector_norm(pred - sample.target_trajectory, dim=-1)
    mean_error = node_error.mean(dim=-1)
    controls_tensor = torch.stack(controls, dim=0)
    payload = {
        "policy": "ChatGPT 5.5 High",
        "mode": "closed_loop",
        "run_label": run_label,
        "pred_trajectory": pred.detach().cpu(),
        "target_trajectory": sample.target_trajectory.detach().cpu(),
        "initial_clean_state": sample.h0.detach().cpu(),
        "initial_belief": source["initial_belief"].detach().cpu(),
        "controls": controls_tensor.detach().cpu(),
        "corrected_states": torch.stack(corrected_states, dim=0).detach().cpu(),
        "probe_nodes": torch.stack(probe_nodes, dim=0).detach().cpu(),
        "observed_masks": torch.stack(observed_masks, dim=0).detach().cpu(),
        "labels": sample.labels.detach().cpu(),
        "times": sample.times.detach().cpu(),
        "node_error": node_error.detach().cpu(),
        "mean_error": mean_error.detach().cpu(),
        "chatgpt_decisions": decisions,
        "chatgpt_raw_response_paths": raw_paths,
        "chatgpt_prompt_paths": prompt_paths,
        "chatgpt_step_wall_time_sec": torch.tensor(step_times),
    }
    rollout_path = rollout_dir / f"chatgpt55_high_{run_label}.pt"
    torch.save(payload, rollout_path)
    total_time = time.perf_counter() - rollout_start
    return {
        "policy": "ChatGPT 5.5 High",
        "run_label": run_label,
        "path": str(rollout_path.relative_to(output_dir)),
        "mean_error_auc": float(torch.trapz(mean_error.detach().cpu(), sample.times.detach().cpu())),
        "trajectory_mse": float(torch.mean((pred.detach().cpu() - sample.target_trajectory.detach().cpu()) ** 2)),
        "final_mean_error": float(mean_error[-1].detach().cpu()),
        "chatgpt_rollout_wall_time_sec": total_time,
        "chatgpt_step_wall_time_sec": step_times,
        "valid_decisions": sum(1 for item in decisions if item["valid"]),
        "total_decisions": len(decisions),
    }


def _extract_json_object(raw: str) -> str:
    decoder = json.JSONDecoder()
    matches: list[tuple[int, dict[str, object]]] = []
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and _looks_like_decision(payload):
            matches.append((idx, payload))
    if not matches:
        raise ValueError("no decision JSON object found")
    return json.dumps(matches[-1][1])


def _looks_like_decision(payload: dict[str, object]) -> bool:
    return (
        "selected_nodes" in payload
        and "damping_gain" in payload
        and "laplacian_gain" in payload
        and "center_pull_gain" in payload
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
