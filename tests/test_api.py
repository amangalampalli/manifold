from pathlib import Path

from fastapi.testclient import TestClient

from manifold.api.artifacts import ClosedLoopArtifactStore
from manifold.api.server import create_app
from manifold.cli import main


def make_closed_loop_artifacts(tmp_path: Path) -> Path:
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
            "1",
            "--policies",
            "random",
            "greedy",
            "neural",
            "--source-run-dir",
            str(smoke_dir / "closed-source"),
            "--closed-loop-output-name",
            "closed-loop-eval",
            "--no-progress",
        ]
    )
    assert export_status == 0
    return smoke_dir / "closed-source" / "closed-loop-eval"


def test_closed_loop_store_discovers_and_summarizes_run(tmp_path: Path) -> None:
    artifact_dir = make_closed_loop_artifacts(tmp_path)
    store = ClosedLoopArtifactStore(artifact_dir)

    runs = store.list_runs()

    assert len(runs) == 1
    assert runs[0]["id"] == "closed-source"
    assert set(runs[0]["policies"]) == {"random", "greedy", "neural"}
    assert "headlineMetrics" in runs[0]


def test_closed_loop_store_serializes_graph_and_rollout(tmp_path: Path) -> None:
    artifact_dir = make_closed_loop_artifacts(tmp_path)
    store = ClosedLoopArtifactStore(artifact_dir)

    graph = store.graph("closed-source")
    rollout = store.rollout("closed-source", "neural", 0)

    assert graph["numNodes"] == 10
    assert graph["nodes"][0].keys() >= {"id", "community", "x", "y"}
    assert graph["edges"]
    assert rollout["policy"] == "neural"
    assert len(rollout["times"]) == 3
    assert len(rollout["meanError"]) == 3
    assert len(rollout["nodeError"]) == 3
    assert len(rollout["stateProjection"]) == 3
    assert len(rollout["probeNodes"]) == 2
    assert len(rollout["observedNodes"]) == 2


def test_closed_loop_api_endpoints(tmp_path: Path) -> None:
    artifact_dir = make_closed_loop_artifacts(tmp_path)
    client = TestClient(create_app(artifact_dir))

    runs_response = client.get("/api/runs")
    assert runs_response.status_code == 200
    run_id = runs_response.json()["runs"][0]["id"]

    assert client.get(f"/api/runs/{run_id}/summary").status_code == 200
    assert client.get(f"/api/runs/{run_id}/graph").status_code == 200
    rollout_response = client.get(
        f"/api/runs/{run_id}/rollout", params={"policy": "neural", "run_idx": 0}
    )
    assert rollout_response.status_code == 200
    assert rollout_response.json()["policy"] == "neural"
    telemetry_response = client.get(f"/api/runs/{run_id}/telemetry")
    assert telemetry_response.status_code == 200
    assert telemetry_response.json()["lines"]
