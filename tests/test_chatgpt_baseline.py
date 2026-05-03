import json
from pathlib import Path

import torch

from manifold.cli import main
from manifold.trainer.core.chatgpt_baseline import (
    build_chatgpt_prompt,
    parse_chatgpt_decision,
    sanitize_chatgpt_decision,
)


def test_chatgpt_decision_sanitization_enforces_visibility_budget_and_clamps() -> None:
    decision = parse_chatgpt_decision(
        json.dumps(
            {
                "selected_nodes": [1, 2, 4, 2, 9],
                "damping_gain": 10.0,
                "laplacian_gain": -1.0,
                "center_pull_gain": 0.2,
                "confidence": 2.0,
                "rationale": "x",
            }
        )
    )
    observed = torch.tensor([False, True, False, False, True])
    sanitized = sanitize_chatgpt_decision(
        decision, observed_mask=observed, budget=1, max_gain=0.15
    )
    assert sanitized.selected_nodes == [1]
    assert sanitized.damping_gain == 0.15
    assert sanitized.laplacian_gain == 0.0
    assert sanitized.center_pull_gain == 0.15
    assert sanitized.confidence == 1.0


def test_chatgpt_decision_parser_extracts_final_json_from_codex_transcript() -> None:
    raw = """OpenAI Codex v0.128.0
user
{"prompt_payload": {"selected_nodes": "schema description"}}
codex
{"center_pull_gain":0.08,"confidence":0.74,"damping_gain":0.11,"laplacian_gain":0.09,"rationale":"ok","selected_nodes":[33,34,35]}
tokens used
12,345
{"center_pull_gain":0.08,"confidence":0.74,"damping_gain":0.11,"laplacian_gain":0.09,"rationale":"ok","selected_nodes":[33,34,35]}
"""
    decision = parse_chatgpt_decision(raw)
    assert decision.valid
    assert decision.selected_nodes == [33, 34, 35]
    assert decision.damping_gain == 0.11
    assert decision.laplacian_gain == 0.09
    assert decision.center_pull_gain == 0.08


def test_chatgpt_prompt_excludes_target_language() -> None:
    prompt = build_chatgpt_prompt(
        run_label="48_run_014",
        step_idx=0,
        h=torch.randn(4, 3),
        observed_mask=torch.tensor([True, False, True, False]),
        probe_nodes=torch.tensor([0]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        labels=torch.tensor([0, 0, 1, 1]),
        budget=1,
        previous_control=None,
    )
    assert "target_trajectory" not in prompt
    assert "future" in prompt
    assert "visible_top_energy_nodes" in prompt


def test_export_chatgpt_baseline_fake_runner(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "outputs" / "smoke"
    train_status = main(
        [
            "compare",
            "--config",
            "configs/default.yaml",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--num-nodes",
            "10",
            "--trajectory-steps",
            "2",
            "--eval-runs",
            "1",
            "--output-dir",
            str(smoke_dir),
            "--policies",
            "random",
            "greedy",
            "neural",
            "--early-stopping",
            "--run-name",
            "chatgpt-source",
            "--no-progress",
        ]
    )
    assert train_status == 0
    closed_status = main(
        [
            "export-closed-loop",
            "--config",
            "configs/default.yaml",
            "--device",
            "cpu",
            "--num-nodes",
            "10",
            "--trajectory-steps",
            "2",
            "--eval-runs",
            "1",
            "--policies",
            "random",
            "greedy",
            "neural",
            "--source-run-dir",
            str(smoke_dir / "chatgpt-source"),
            "--closed-loop-output-name",
            "closed-loop",
            "--fresh-eval-set",
            "--no-progress",
        ]
    )
    assert closed_status == 0
    export_status = main(
        [
            "export-chatgpt-baseline",
            "--config",
            "configs/default.yaml",
            "--device",
            "cpu",
            "--source-run-dir",
            str(smoke_dir / "chatgpt-source"),
            "--rollout-spec",
            "tiny:closed-loop:0",
            "--chatgpt-fake-runner",
            "--no-progress",
        ]
    )
    assert export_status == 0
    output_dir = smoke_dir / "chatgpt-source" / "chatgpt55-high-comparison"
    assert (output_dir / "chatgpt_rollouts_manifest.json").is_file()
    assert (output_dir / "comparison_summary.json").is_file()
    assert (output_dir / "timing.json").is_file()
    assert (output_dir / "chatgpt_rollouts" / "chatgpt55_high_tiny_run_000.pt").is_file()
    payload = torch.load(
        output_dir / "chatgpt_rollouts" / "chatgpt55_high_tiny_run_000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["policy"] == "ChatGPT 5.5 High"
    assert payload["controls"].shape[0] == 2
    assert payload["observed_masks"].shape == (2, 10)
