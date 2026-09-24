"""Differentiable fixed-step consensus updates."""

import torch

from sheaf_admm.geometry import FixedGeometry
from sheaf_admm.solvers.z_solvers import GDParams, GDZSolver, UnrolledCGParams, UnrolledCGZSolver


def _case():
    torch.manual_seed(3)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    maps = torch.randn(5, 2, 2, 3, requires_grad=True)
    target = torch.randn(4, 2, 3)
    return FixedGeometry(edges, maps), target, maps


def test_cg_prox_residual_and_map_gradient():
    g, target, maps = _case()
    z = UnrolledCGZSolver.solve(
        target, target, g, UnrolledCGParams(mode="prox", gamma=2.0, num_iters=30), 0.5
    )
    resid = 2 * g.laplacian_apply(z) + 0.5 * (z - target)
    assert resid.norm() / (0.5 * target).norm() < 1e-4
    z.square().sum().backward()
    assert torch.isfinite(maps.grad).all() and maps.grad.abs().sum() > 0


def test_cg_project_reduces_energy_and_warm_prox():
    g, target, _ = _case()
    z = UnrolledCGZSolver.solve(
        target, torch.zeros_like(target), g, UnrolledCGParams(mode="project", num_iters=30), 1.0
    )
    assert g.energy(z) < 0.1 * g.energy(target)
    warm = UnrolledCGZSolver.solve(
        target, torch.zeros_like(target), g, UnrolledCGParams(mode="prox", prox_init="warm"), 0.5
    )
    assert torch.isfinite(warm).all()


def test_gd_decreases_objective():
    g, target, _ = _case()

    def op(x):
        return 2 * g.laplacian_apply(x) + 0.5 * x

    v = torch.randn_like(target)
    for _ in range(30):
        v = op(v)
        v = v / v.norm()
    eta = 1.0 / ((v * op(v)).sum() / (v * v).sum())
    z = GDZSolver.solve(
        target, target, g, GDParams(mode="prox", gamma=2.0, num_steps=80, eta=eta), 0.5
    )

    def obj(x):
        return 2 * g.energy(x) + 0.25 * (x - target).square().sum()

    assert torch.isfinite(z).all() and obj(z) < obj(target)
