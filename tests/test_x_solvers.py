"""Closed form local proximal updates."""

import torch

from sheaf_admm.solvers.x_solvers import (
    DenseQuadraticParams,
    DenseQuadraticXSolver,
    DiagonalProxParams,
    DiagonalProxXSolver,
    SimpleParams,
    SimpleXSolver,
)


def test_diagonal_cases_and_rho_broadcast():
    z = torch.tensor([[[100.0, -100.0, 0.5]], [[1.0, -1.0, 0.05]]])
    q = torch.zeros_like(z)
    enc = {"q_diag": torch.ones_like(z), "q": q, "lower": 0.0, "upper": torch.ones_like(z)}
    rho = torch.tensor([[1.0], [2.0]])
    x = DiagonalProxXSolver.solve(z, torch.zeros_like(z), rho, enc, DiagonalProxParams())
    torch.testing.assert_close(x[0, 0], torch.tensor([1.0, 0.0, 0.25]))
    assert ((x >= 0) & (x <= 1)).all()
    enc = {"q_diag": torch.zeros_like(z), "q": q, "l1_weight": 0.1}
    x = DiagonalProxXSolver.solve(z, torch.zeros_like(z), 1.0, enc, DiagonalProxParams())
    torch.testing.assert_close(x, torch.sign(z) * (z.abs() - 0.1).clamp_min(0))


def test_simple_and_dense_normal_equations():
    torch.manual_seed(2)
    z, y, h = [torch.randn(2, 2, 3) for _ in range(3)]
    rho = 0.7
    x = SimpleXSolver.solve(z, y, rho, {"h": h, "beta": 2.0}, SimpleParams())
    torch.testing.assert_close(x, (2 * h + rho * (z - y)) / (2 + rho))
    M = torch.randn(2, 2, 3, 3)
    Q = M @ M.transpose(-1, -2) + 0.1 * torch.eye(3)
    q = torch.randn_like(z)
    dense = DenseQuadraticXSolver.solve(z, y, rho, {"Q": Q, "q": q}, DenseQuadraticParams())
    torch.testing.assert_close(
        (Q + rho * torch.eye(3)) @ dense[..., None],
        (rho * (z - y) - q)[..., None],
        atol=1e-5,
        rtol=1e-5,
    )
