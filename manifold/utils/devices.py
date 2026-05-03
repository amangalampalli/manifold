"""Device selection and sparse operation fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseBackendStatus:
    device: str
    sparse_mm_supported: bool
    message: str


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve `auto|mps|cuda|cpu` into a torch device.

    `auto` prefers Apple Metal/MPS, then CUDA, then CPU. Explicit accelerator
    requests fail fast when unavailable so callers do not silently benchmark the
    wrong backend.
    """
    normalized = requested.lower()
    if normalized == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but torch.backends.mps.is_available() is false"
            )
        return torch.device("mps")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{requested}'. Use auto, mps, cuda, or cpu.")


def sparse_backend_status(device: torch.device) -> SparseBackendStatus:
    """Probe sparse matrix multiplication support for a device."""
    try:
        indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long, device=device)
        values = torch.ones(3, dtype=torch.float32, device=device)
        with torch.sparse.check_sparse_tensor_invariants(False):
            matrix = torch.sparse_coo_tensor(
                indices, values, (4, 4), device=device
            ).coalesce()
        dense = torch.ones(4, 2, dtype=torch.float32, device=device)
        out = torch.sparse.mm(matrix, dense)
        ok = out.shape == (4, 2) and out.device.type == device.type
        return SparseBackendStatus(
            str(device), ok, "native sparse.mm ok" if ok else "unexpected output"
        )
    except Exception as exc:  # pragma: no cover - backend-specific
        return SparseBackendStatus(str(device), False, f"{type(exc).__name__}: {exc}")


def sparse_mm(matrix: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """Sparse-dense matmul with CPU fallback for accelerator sparse gaps."""
    try:
        return torch.sparse.mm(matrix, dense)
    except Exception:  # pragma: no cover - backend-specific fallback
        result = torch.sparse.mm(matrix.coalesce().cpu(), dense.cpu())
        return result.to(dense.device)
