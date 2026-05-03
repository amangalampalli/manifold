import pytest
import torch

from manifold.utils.config import (
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    SensingConfig,
    TrainingConfig,
)
from manifold.trainer.core.trainer import MinimaxTrainer
from manifold.trainer.support.policies import ControlPolicy


def tiny_config(device: str = "cpu") -> ExperimentConfig:
    return ExperimentConfig(
        device=device,
        seed=3,
        graph=GraphConfig(
            num_nodes=10, num_communities=2, p_in=0.4, p_out=0.05, seed=3
        ),
        model=ModelConfig(latent_dim=4, hidden_dim=8, control_dim=4, num_classes=2),
        sensing=SensingConfig(k_hop=2, budget=2, noise_std=0.0),
        training=TrainingConfig(
            epochs=1, trajectory_steps=2, dt=0.05, source_steps=1, controller_steps=1
        ),
    )


def test_tiny_minimax_epoch_cpu() -> None:
    trainer = MinimaxTrainer(tiny_config("cpu"))
    history = trainer.train()
    assert len(history.metrics) == 1
    final = history.metrics[-1]
    assert final["perturbation_auc"] >= 0.0
    assert final["post_control_perturbation_auc"] >= 0.0
    assert final["trajectory_mse"] >= 0.0
    assert final["final_belief_mse"] >= 0.0
    assert final["control_energy"] >= 0.0
    assert final["control_target_mse"] >= 0.0
    assert final["source_frozen"] == 0.0
    assert 0.0 <= final["observed_fraction"] <= 1.0


def test_compare_random_greedy_neural_cpu() -> None:
    trainer = MinimaxTrainer(tiny_config("cpu"))
    trainer.train()
    comparison = trainer.compare_policies(("random", "greedy", "neural"), runs=1)
    assert [item["policy"] for item in comparison] == [
        "random",
        "greedy",
        "neural",
        "neural_minus_greedy",
    ]
    for item in comparison:
        assert item["runs"] == 1.0
        if item["policy"] != "neural_minus_greedy":
            assert item["perturbation_auc"] >= 0.0
            assert item["post_control_perturbation_auc"] >= 0.0
            assert item["trajectory_mse"] >= 0.0
        else:
            assert "neural_minus_greedy_auc" in item
            assert "neural_minus_greedy_post_control_auc" in item
            assert "neural_minus_greedy_mse" in item
            assert "neural_vs_greedy_auc_improvement_pct" in item
            assert "neural_vs_greedy_post_control_auc_improvement_pct" in item
            assert "neural_vs_greedy_mse_improvement_pct" in item


def test_boundary_nodes_are_nonempty() -> None:
    trainer = MinimaxTrainer(tiny_config("cpu"))
    nodes = trainer._boundary_nodes()
    assert nodes.numel() > 0
    assert nodes.dtype == torch.long


def test_boundary_perturbation_affects_multiple_nodes() -> None:
    config = tiny_config("cpu")
    config = ExperimentConfig(
        device=config.device,
        seed=config.seed,
        graph=config.graph,
        model=config.model,
        sensing=config.sensing,
        training=TrainingConfig(
            boundary_perturbation_fraction=0.5, boundary_perturbation_scale=0.5
        ),
    )
    trainer = MinimaxTrainer(config)
    sample = trainer.dataset.sample(steps=2, dt=0.05)
    perturbation = trainer._boundary_perturbation(sample.h0, run_idx=0)
    affected = perturbation.abs().sum(dim=-1) > 0
    assert int(affected.sum().item()) >= 2
    assert torch.isfinite(perturbation).all()


def test_source_disturbance_honors_learned_vs_boundary_mode() -> None:
    learned_config = tiny_config("cpu")
    learned_config = ExperimentConfig(
        device=learned_config.device,
        seed=learned_config.seed,
        graph=learned_config.graph,
        model=learned_config.model,
        sensing=learned_config.sensing,
        training=TrainingConfig(perturbation_mode="learned"),
    )
    boundary_config = tiny_config("cpu")
    boundary_config = ExperimentConfig(
        device=boundary_config.device,
        seed=boundary_config.seed,
        graph=boundary_config.graph,
        model=boundary_config.model,
        sensing=boundary_config.sensing,
        training=TrainingConfig(
            perturbation_mode="boundary", boundary_perturbation_fraction=0.5
        ),
    )
    learned_trainer = MinimaxTrainer(learned_config)
    boundary_trainer = MinimaxTrainer(boundary_config)
    sample = learned_trainer.dataset.sample(steps=2, dt=0.05)
    learned, learned_boundary_count, _ = learned_trainer._source_disturbance(
        sample.h0, run_idx=0
    )
    boundary, boundary_count, _ = boundary_trainer._source_disturbance(
        sample.h0, run_idx=0
    )
    assert learned.shape == boundary.shape
    assert learned_boundary_count.item() == 0.0
    assert boundary_count.item() > 0.0
    assert not torch.allclose(learned, boundary)


def test_boundary_recovery_control_targets_selected_boundary_nodes() -> None:
    config = tiny_config("cpu")
    config = ExperimentConfig(
        device=config.device,
        seed=config.seed,
        graph=config.graph,
        model=config.model,
        sensing=config.sensing,
        training=TrainingConfig(
            perturbation_mode="boundary",
            boundary_perturbation_fraction=0.5,
            boundary_perturbation_scale=0.5,
            boundary_recovery_gain=1.0,
        ),
    )
    trainer = MinimaxTrainer(config)
    sample = trainer.dataset.sample(steps=2, dt=0.05)
    perturbation = trainer._boundary_perturbation(sample.h0, run_idx=0)
    control = trainer._boundary_recovery_control(sample.h0 + perturbation)
    assert control.shape == sample.h0.shape
    assert torch.isfinite(control).all()
    assert int((control.abs().sum(dim=-1) > 0).sum().item()) >= 2


def test_structured_control_shape_and_residual_bound() -> None:
    torch.manual_seed(0)
    policy = ControlPolicy(
        latent_dim=4,
        hidden_dim=8,
        control_dim=4,
        residual_scale=0.03,
        analytic_gain=0.08,
        community_gain=0.12,
    )
    h = torch.randn(5, 4)
    features = torch.randn(5, 18)
    control = policy(features, h)
    assert control.shape == h.shape
    assert torch.isfinite(control).all()

    trunk = policy.trunk(features)
    residual = policy.residual_scale * torch.tanh(policy.residual_head(trunk))
    assert residual.abs().max() <= policy.residual_scale + 1e-6


def test_structured_control_initial_prior_uses_laplacian_not_origin_damping() -> None:
    policy = ControlPolicy(
        latent_dim=4,
        hidden_dim=8,
        control_dim=4,
        residual_scale=0.03,
        analytic_gain=0.08,
        community_gain=0.12,
    )
    h = torch.randn(5, 4)
    zero_lap_features = torch.cat(
        [
            h,
            torch.zeros_like(h),
            h,
            torch.zeros_like(h),
            torch.ones(5, 1),
            torch.zeros(5, 1),
        ],
        dim=-1,
    )
    zero_lap_control = policy(zero_lap_features, h)
    assert torch.allclose(
        zero_lap_control, torch.zeros_like(zero_lap_control), atol=1e-6
    )

    nonzero_lap = torch.randn_like(h)
    lap_features = torch.cat(
        [h, nonzero_lap, h, nonzero_lap, torch.ones(5, 1), torch.ones(5, 1)], dim=-1
    )
    lap_control = policy(lap_features, h, analytic_control=-nonzero_lap)
    assert torch.isfinite(lap_control).all()
    assert not torch.allclose(lap_control, torch.zeros_like(lap_control), atol=1e-6)


def test_analytic_sheaf_control_is_topk_and_normalized() -> None:
    config = tiny_config("cpu")
    config = ExperimentConfig(
        device=config.device,
        seed=config.seed,
        graph=config.graph,
        model=config.model,
        sensing=config.sensing,
        training=TrainingConfig(analytic_sheaf_gain=0.08, analytic_community_gain=0.0),
    )
    trainer = MinimaxTrainer(config)
    lap = torch.tensor(
        [
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ]
    )
    lap_energy = torch.linalg.vector_norm(lap, dim=-1, keepdim=True)
    control = trainer._analytic_control(lap, lap_energy, torch.zeros_like(lap))
    active = control.abs().sum(dim=-1) > 0
    expected_topk = min(
        lap.size(0),
        max(
            1,
            int(
                round(config.sensing.budget * config.training.analytic_topk_multiplier)
            ),
        ),
    )
    assert int(active.sum().item()) == expected_topk
    assert control.abs().max() <= 1.0 + 1e-6


def test_analytic_control_uses_community_deviation() -> None:
    config = tiny_config("cpu")
    config = ExperimentConfig(
        device=config.device,
        seed=config.seed,
        graph=config.graph,
        model=config.model,
        sensing=config.sensing,
        training=TrainingConfig(analytic_sheaf_gain=0.0, analytic_community_gain=0.12),
    )
    trainer = MinimaxTrainer(config)
    lap = torch.zeros(3, 4)
    lap_energy = torch.zeros(3, 1)
    h_deviation = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ]
    )
    control = trainer._analytic_control(lap, lap_energy, h_deviation)
    assert control[0, 0] < 0
    assert control[2, 1] < 0


def test_community_centers_for_nodes_match_labels() -> None:
    trainer = MinimaxTrainer(tiny_config("cpu"))
    values = torch.zeros(trainer.dataset.num_nodes, trainer.config.model.latent_dim)
    centers = trainer._community_centers_for_nodes(values)
    expected = trainer.dataset.community_centers[trainer.dataset.labels]
    assert torch.allclose(centers, expected)


def test_control_target_points_toward_final_state() -> None:
    trainer = MinimaxTrainer(tiny_config("cpu"))
    initial = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    times = torch.tensor([0.0, 0.5])
    control_target = trainer._control_target(initial, target, times)
    assert control_target.shape == initial.shape
    assert torch.all(control_target > 0)
    assert control_target.max() <= trainer.config.training.source_scale


def test_source_freeze_leaves_source_parameters_unchanged() -> None:
    config = tiny_config("cpu")
    config = ExperimentConfig(
        device=config.device,
        seed=config.seed,
        graph=config.graph,
        model=config.model,
        sensing=config.sensing,
        training=TrainingConfig(
            epochs=2,
            trajectory_steps=config.training.trajectory_steps,
            dt=config.training.dt,
            source_steps=1,
            controller_steps=1,
            source_freeze_after=1,
            source_freeze_for=2,
        ),
    )
    trainer = MinimaxTrainer(config)
    before = [param.detach().clone() for param in trainer.source_policy.parameters()]
    metrics = trainer.train_epoch(2)
    after = list(trainer.source_policy.parameters())
    assert metrics["source_frozen"] == 1.0
    for left, right in zip(before, after):
        assert torch.allclose(left, right)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_tiny_minimax_epoch_mps() -> None:
    trainer = MinimaxTrainer(tiny_config("mps"))
    history = trainer.train()
    assert len(history.metrics) == 1
