import torch

from manifold.trainer.game.sensing import (
    ActiveSensingBeliefUpdater,
    k_hop_observation_mask,
)


def test_k_hop_observation_mask_respects_two_hop_visibility() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    mask = k_hop_observation_mask(
        edge_index,
        num_nodes=5,
        probe_nodes=torch.tensor([0]),
        k_hop=2,
        device="cpu",
    )
    assert mask.tolist() == [True, True, True, False, False]


def test_active_sensing_budget_and_correction_shape() -> None:
    torch.manual_seed(0)
    updater = ActiveSensingBeliefUpdater(
        latent_dim=4, hidden_dim=8, k_hop=1, budget=2, noise_std=0.0
    )
    h_pred = torch.zeros(4, 4)
    y_true = torch.ones(4, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    update = updater(h_pred, y_true, edge_index)
    assert update.probe_nodes.numel() == 2
    assert update.h_corr.shape == h_pred.shape
    assert update.observed_mask.dtype == torch.bool
    assert update.observed_mask.any()
