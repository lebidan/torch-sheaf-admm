"""Fixed-step gradient descent with Optax-style momentum/Nesterov semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sheaf_admm.geometry.base import SheafGeometry
from sheaf_admm.solvers.base import ZSolverParams, rho_as_vec


@dataclass
class GDParams(ZSolverParams):
    num_steps: int = 10
    momentum: float = 0.9
    nesterov: bool = True
    eta: float | torch.Tensor = 0.01


class GDZSolver:
    @staticmethod
    def solve(z_target, z_prev, geometry: SheafGeometry, params: GDParams, rho):
        gamma = torch.as_tensor(params.gamma, dtype=z_target.dtype, device=z_target.device)
        rho_v = rho_as_vec(rho, z_target)
        z = z_target
        trace = torch.zeros_like(z)
        eta = torch.as_tensor(params.eta, dtype=z.dtype, device=z.device)
        for _ in range(params.num_steps):
            if params.mode == "project":
                grad = geometry.laplacian_apply(z)
            elif params.mode == "prox":
                grad = gamma * geometry.laplacian_apply(z) + rho_v * (z - z_target)
            else:
                raise ValueError(f"unknown z mode {params.mode!r} (project|prox)")
            if params.momentum is not None:
                trace = grad + params.momentum * trace
                update = grad + params.momentum * trace if params.nesterov else trace
            else:
                update = grad
            z = z - eta * update
        return z
