"""The unrolled ADMM loop — the coordination core.

Each agent carries three states: the local proposal ``x``, the consensus iterate
``z``, and the scaled dual accumulator ``y`` (``u = lambda / rho``). One ADMM
iteration is::

    z_prev = z
    x       = prox_f(z - y; rho)                       # local x-update (x-solver)
    x_relax = alpha * x + (1 - alpha) * z_prev          # optional over-relaxation
    z       = consensus(x_relax + y; geometry, rho)     # sheaf z-update (z-solver)
    y       = y + (x_relax - z)                          # dual ascent

We unroll ``K`` iterations and backpropagate through the whole trajectory. The
loss is computed from the final ``loss_window`` x-iterates (decoded and
averaged), so the loop returns those. Memory options:

* full unroll (``grad_window=None``): gradients flow through all ``K`` steps.
* truncated BPTT (``grad_window=g``): the first ``K-g`` steps run detached
  (``stop_gradient``); gradients flow only through the last ``g``. Memory is
  ``O(g)`` instead of ``O(K)``.

Either way only the last ``loss_window`` x-iterates are materialized.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from sheaf_admm.geometry.base import SheafGeometry
from sheaf_admm.solvers.base import XSolver, XSolverParams, ZSolver, ZSolverParams


@dataclass
class ADMMState:
    """Per-agent ADMM variables, each ``[N, B, d_v]``."""

    x: torch.Tensor  # local proposal (primal)
    z: torch.Tensor  # consensus iterate
    y: torch.Tensor  # scaled dual accumulator (u = lambda / rho)


def _admm_step(
    state: ADMMState,
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: torch.Tensor,
    alpha: float,
) -> ADMMState:
    z_prev = state.z
    x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
    x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
    z_target = x_relaxed + state.y
    z = z_solver.solve(z_target, z_prev, geometry, z_params, rho)
    y = state.y + (x_relaxed - z)
    return ADMMState(x=x, z=z, y=y)


_compiled_admm_step = torch.compile(_admm_step, mode="reduce-overhead", fullgraph=True)


def inverse_softplus(x: float) -> float:
    """Numerically safe inverse of softplus, for initializing softplus-parameterized scalars.

    Guards ``log(expm1(x))`` against underflow (clamp to >=1e-7) and overflow
    (identity for large x), so any positive init value maps to a finite raw param.
    """
    x = max(x, 1e-7)
    return x if x > 20.0 else math.log(math.expm1(x))


def run_admm(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: torch.Tensor,
    z_init: torch.Tensor,  # [N, B, d_v] initial consensus state (encoder h, or zeros)
    num_iters: int,
    *,
    relaxation_alpha: float = 1.0,
    loss_window: int = 1,
    grad_window: int | None = None,
    compile_step: bool = False,
) -> tuple[ADMMState, torch.Tensor]:
    """Run ``num_iters`` ADMM steps.

    Returns the final :class:`ADMMState` and the stacked last ``loss_window``
    local proposals ``x`` of shape ``[W, N, B, d_v]`` (oldest first).
    """
    alpha = relaxation_alpha
    use_compile = compile_step and z_init.is_cuda and os.environ.get("TORCH_COMPILE_DISABLE") != "1"

    def step(state: ADMMState) -> ADMMState:
        operation = _compiled_admm_step if use_compile else _admm_step
        next_state = operation(
            state, encoder_output, geometry, x_solver, x_params, z_solver, z_params, rho, alpha
        )
        if use_compile and not torch.is_grad_enabled():
            return ADMMState(next_state.x.clone(), next_state.z.clone(), next_state.y.clone())
        return next_state

    state = ADMMState(x=z_init, z=z_init, y=torch.zeros_like(z_init))

    K = num_iters
    n_detached = 0 if grad_window is None else max(0, K - grad_window)
    if n_detached:
        with torch.no_grad():
            for _ in range(n_detached):
                state = step(state)
        state = ADMMState(state.x.detach(), state.z.detach(), state.y.detach())

    n_grad = K - n_detached
    window = min(loss_window, n_grad)
    collected = []
    for i in range(n_grad):
        state = step(state)
        if i >= n_grad - window:
            collected.append(state.x)
    x_window = torch.stack(collected) if collected else z_init.new_empty((0, *z_init.shape))
    return state, x_window


@dataclass
class ADMMHistory:
    """Per-iteration ADMM trajectory, for visualization / convergence diagnostics.

    All arrays stack the ``K`` iterations on a leading axis (oldest first). The
    per-agent state arrays are ``[K, N, B, d_v]``; the residual arrays are the
    standard ADMM stopping diagnostics:

    * ``primal_res = ||x_relaxed - z||`` per agent (``[K, N, B]``) — how far the
      local proposal sits from the consensus it agreed to.
    * ``dual_res = rho * ||z - z_prev||`` per agent (``[K, N, B]``) — the change
      in the consensus iterate, scaled by the penalty (the dual feasibility
      residual of scaled-dual ADMM).
    * ``consistency_rms`` (``[K, B]``) — the geometry's sheaf-disagreement RMS,
      ``sqrt(mean over edges and stalk dims of r^2 + eps)``, the quantity the
      model is trained to drive to zero.
    """

    x: torch.Tensor  # [K, N, B, d_v] local proposals
    z: torch.Tensor  # [K, N, B, d_v] consensus iterates
    y: torch.Tensor  # [K, N, B, d_v] scaled duals
    primal_res: torch.Tensor  # [K, N, B] ||x_relaxed - z|| per agent
    dual_res: torch.Tensor  # [K, N, B] rho * ||z - z_prev|| per agent
    consistency_rms: torch.Tensor  # [K, B] sheaf disagreement RMS


def run_admm_history(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: torch.Tensor,
    z_init: torch.Tensor,
    num_iters: int,
    *,
    relaxation_alpha: float = 1.0,
) -> tuple[ADMMState, ADMMHistory]:
    """Run ``num_iters`` ADMM steps, recording the full per-iteration trajectory.

    Same fixed-point iteration as :func:`run_admm`, but every iterate and its
    primal/dual residuals are stacked and returned in an :class:`ADMMHistory`
    (no gradient window — this is a forward-only diagnostic path). Use it to
    visualize coordination dynamics and x-vs-z trajectories on a single example.
    """
    alpha = relaxation_alpha
    rho_s = torch.as_tensor(rho, dtype=z_init.dtype, device=z_init.device)
    state = ADMMState(x=z_init, z=z_init, y=torch.zeros_like(z_init))
    xs, zs, ys, primals, duals, crms = [], [], [], [], [], []
    for _ in range(num_iters):
        z_prev = state.z
        x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
        x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        z = z_solver.solve(x_relaxed + state.y, z_prev, geometry, z_params, rho)
        y = state.y + x_relaxed - z
        state = ADMMState(x=x, z=z, y=y)
        xs.append(x)
        zs.append(z)
        ys.append(y)
        primals.append(torch.linalg.vector_norm(x_relaxed - z, dim=-1))
        duals.append(rho_s * torch.linalg.vector_norm(z - z_prev, dim=-1))
        crms.append(geometry.consistency_rms(z))
    history = ADMMHistory(*(torch.stack(seq) for seq in (xs, zs, ys, primals, duals, crms)))
    return state, history
