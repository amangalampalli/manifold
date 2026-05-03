import torch

from manifold.math.dynamics import NeuralSheafODE


def test_fixed_step_rollout_is_finite_and_stable_shape() -> None:
    torch.manual_seed(0)
    model = NeuralSheafODE(latent_dim=4, edge_attr_dim=4, hidden_dim=8, control_dim=4)
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]], dtype=torch.long)
    edge_attr = torch.zeros(edge_index.size(1), 4)
    h0 = torch.randn(3, 4)
    control = torch.zeros(3, 4)
    times = torch.linspace(0, 0.1, 3)
    out = model.rollout(h0, times, edge_index=edge_index, edge_attr=edge_attr, control=control, step_size=0.05)
    assert out.shape == (3, 3, 4)
    assert torch.isfinite(out).all()


def test_control_projection_initializes_as_identity_when_square() -> None:
    model = NeuralSheafODE(latent_dim=4, edge_attr_dim=4, hidden_dim=8, control_dim=4)
    assert torch.allclose(model.control_projection.weight, torch.eye(4), atol=1e-6)
    assert torch.allclose(model.control_projection.bias, torch.zeros(4), atol=1e-6)
