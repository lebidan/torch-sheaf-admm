"""Matmul precision for the differentiable inner solvers."""

from __future__ import annotations

import torch


def set_high_precision() -> None:
    """Use full float32 precision for matmuls on CUDA."""
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
