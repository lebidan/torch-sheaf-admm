"""Solver protocols and shared helpers for the ADMM inner updates.

The ADMM loop is polymorphic over two solver families:

* **x-solvers** implement the local proximal step
  ``x_i = argmin_{x} f_i(x) + (rho/2)||x - (z_i - y_i)||^2``,
  where the convex ``f_i`` is parameterized by the encoder. ``y`` is the
  *scaled* dual (``u = lambda / rho``), so the prox center is ``v = z - y``.
* **z-solvers** implement the consensus step over the sheaf (see
  :mod:`sheaf_admm.solvers.z_solvers`).

Both take a small frozen ``*Params`` dataclass and are looked up by name in a
registry, so a config string selects the update rule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch

from sheaf_admm.geometry.base import SheafGeometry


@dataclass
class XSolverParams:
    """Base config for an x-solver. Subclasses add solver-specific fields."""

    def replace(self, **changes):
        return replace(self, **changes)


@dataclass
class ZSolverParams:
    """Base config for a z-solver.

    ``mode`` selects the consensus objective:

    * ``"project"`` — hard-consensus update toward ``ker(F)`` (``F z = 0``).
    * ``"prox"`` — soft consensus, minimize ``gamma * E_sheaf(z) + rho/2 ||z - z_target||^2``.
    """

    mode: str = "prox"  # "project" | "prox"
    gamma: float = 1.0  # weight on sheaf energy (prox mode); the paper's gamma

    def replace(self, **changes):
        return replace(self, **changes)


class XSolver(Protocol):
    """x-update: local proximal solve, ``v = z - y`` is the prox center."""

    @staticmethod
    def solve(
        z: torch.Tensor,  # [N, B, d_v]
        y: torch.Tensor,  # [N, B, d_v] scaled dual
        rho: torch.Tensor,  # scalar or [N, B]
        encoder_output: dict[str, Any],
        params: XSolverParams,
    ) -> torch.Tensor:  # x_new [N, B, d_v]
        ...


class ZSolver(Protocol):
    """z-update: sheaf consensus. ``z_target = x + y``."""

    @staticmethod
    def solve(
        z_target: torch.Tensor,  # [N, B, d_v]
        z_prev: torch.Tensor,  # [N, B, d_v] previous z (warm-start)
        geometry: SheafGeometry,
        params: ZSolverParams,
        rho: torch.Tensor,  # scalar or [N, B]; used only in prox mode
    ) -> torch.Tensor:  # z_new [N, B, d_v]
        ...


def rho_as_vec(rho: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast ``rho`` against node states ``like`` of shape ``[N, B, d_v]``.

    A scalar stays scalar; a per-agent ``[N, B]`` rho becomes ``[N, B, 1]``.
    """
    rho = torch.as_tensor(rho, dtype=like.dtype, device=like.device)
    return rho[..., None] if rho.ndim == 2 else rho


def soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Proximal operator of the L1 norm: ``sign(x) * max(|x| - threshold, 0)``."""
    return torch.sign(x) * torch.clamp_min(torch.abs(x) - threshold, 0.0)
