import json
from pathlib import Path

from manifold.cli import main


def test_compare_cli_outputs_policies(capsys, tmp_path: Path) -> None:
    smoke_dir = tmp_path / "outputs" / "smoke"
    status = main(
        [
            "compare",
            "--config",
            "configs/default.yaml",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--eval-runs",
            "1",
            "--output-dir",
            str(smoke_dir),
            "--policies",
            "random",
            "greedy",
            "neural",
        ]
    )
    assert status == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    comparison = json.loads(lines[-1])["comparison"]
    assert [item["policy"] for item in comparison] == [
        "random",
        "greedy",
        "neural",
        "neural_minus_greedy",
    ]


def test_compare_cli_writes_run_artifacts(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "outputs" / "smoke"
    status = main(
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
            "1",
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
            "artifact-test",
            "--no-progress",
        ]
    )
    assert status == 0
    run_dir = smoke_dir / "artifact-test"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "startup.json").is_file()
    assert (run_dir / "graph.pt").is_file()
    assert (run_dir / "graph_layout.json").is_file()
    assert (run_dir / "eval_set.pt").is_file()
    assert (run_dir / "best.json").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "logs" / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "eval.jsonl").is_file()


def test_compare_cli_exports_rollouts(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "outputs" / "smoke"
    status = main(
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
            "1",
            "--eval-runs",
            "1",
            "--output-dir",
            str(smoke_dir),
            "--policies",
            "random",
            "greedy",
            "neural",
            "--export-rollouts",
            "--run-name",
            "rollout-test",
            "--no-progress",
        ]
    )
    assert status == 0
    run_dir = smoke_dir / "rollout-test"
    assert (run_dir / "rollouts_manifest.json").is_file()
    assert (run_dir / "rollouts" / "random_run_000.pt").is_file()
    assert (run_dir / "rollouts" / "greedy_run_000.pt").is_file()
    assert (run_dir / "rollouts" / "neural_run_000.pt").is_file()


def test_export_closed_loop_cli_uses_saved_run(tmp_path: Path) -> None:
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
            "closed-source",
            "--no-progress",
        ]
    )
    assert train_status == 0

    export_status = main(
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
            "2",
            "--policies",
            "random",
            "greedy",
            "neural",
            "--source-run-dir",
            str(smoke_dir / "closed-source"),
            "--closed-loop-output-name",
            "closed-loop",
            "--fresh-eval-set",
            "--no-progress",
        ]
    )
    assert export_status == 0
    output_dir = smoke_dir / "closed-source" / "closed-loop"
    assert (output_dir / "closed_loop_rollouts_manifest.json").is_file()
    assert (output_dir / "closed_loop_run_rankings.json").is_file()
    assert (
        output_dir / "closed_loop_rollouts" / "random_closed_loop_run_000.pt"
    ).is_file()
    assert (
        output_dir / "closed_loop_rollouts" / "greedy_closed_loop_run_000.pt"
    ).is_file()
    assert (
        output_dir / "closed_loop_rollouts" / "neural_closed_loop_run_000.pt"
    ).is_file()
    assert (
        output_dir / "closed_loop_rollouts" / "neural_closed_loop_run_001.pt"
    ).is_file()
