import torch

from manifold.utils.devices import resolve_device, sparse_backend_status


def test_resolve_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_auto_returns_supported_device() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_explicit_mps_resolution_matches_availability() -> None:
    if torch.backends.mps.is_available():
        assert resolve_device("mps").type == "mps"
    else:
        try:
            resolve_device("mps")
        except RuntimeError:
            return
        raise AssertionError("explicit mps should fail when unavailable")


def test_sparse_backend_status_cpu() -> None:
    status = sparse_backend_status(torch.device("cpu"))
    assert status.sparse_mm_supported
