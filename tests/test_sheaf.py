import pytest
import torch

from manifold.math.sheaf import SparseSheafOperator


def _tiny_graph(device: torch.device):
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    edge_attr = torch.zeros(2, 4, dtype=torch.float32, device=device)
    return edge_index, edge_attr


def test_sheaf_coboundary_and_laplacian_cpu() -> None:
    device = torch.device("cpu")
    op = SparseSheafOperator(latent_dim=3, edge_attr_dim=4).to(device)
    edge_index, edge_attr = _tiny_graph(device)
    delta = op.build_coboundary(edge_index, edge_attr, num_nodes=2)
    assert delta.shape == (6, 6)
    assert delta.device.type == "cpu"

    h = torch.randn(2, 3, device=device)
    lap, residual = op.apply_laplacian(h, edge_index, edge_attr)
    assert lap.shape == h.shape
    assert residual.shape == (2, 3)
    assert torch.isfinite(lap).all()

    quadratic = (h * lap).sum()
    assert quadratic.item() >= -1e-5


def test_identity_restrictions_have_zero_residual_for_constant_signal() -> None:
    op = SparseSheafOperator(latent_dim=3, edge_attr_dim=4)
    edge_index, edge_attr = _tiny_graph(torch.device("cpu"))
    h = torch.ones(2, 3)
    _, residual = op.apply_laplacian(h, edge_index, edge_attr)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_sheaf_sparse_product_mps() -> None:
    device = torch.device("mps")
    op = SparseSheafOperator(latent_dim=3, edge_attr_dim=4).to(device)
    edge_index, edge_attr = _tiny_graph(device)
    h = torch.randn(2, 3, device=device)
    lap, residual = op.apply_laplacian(h, edge_index, edge_attr)
    assert lap.device.type == "mps"
    assert residual.device.type == "mps"
    assert torch.isfinite(lap.cpu()).all()
