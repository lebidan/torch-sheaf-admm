"""ADMM state, windowing, history, and truncated gradients."""

import torch

from sheaf_admm.admm import ADMMState, run_admm, run_admm_history
from sheaf_admm.geometry import FixedGeometry
from sheaf_admm.solvers.x_solvers import DiagonalProxParams, DiagonalProxXSolver
from sheaf_admm.solvers.z_solvers import UnrolledCGParams, UnrolledCGZSolver


def _setup():
    torch.manual_seed(4)
    z = torch.randn(4, 2, 3)
    maps = torch.randn(5, 2, 2, 3, requires_grad=True)
    geom = FixedGeometry(torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]), maps)
    enc = {"q_diag": torch.ones_like(z), "q": torch.randn_like(z)}
    return geom, enc, z, maps


def _run(geom, enc, z, **kw):
    return run_admm(
        enc,
        geom,
        DiagonalProxXSolver,
        DiagonalProxParams(),
        UnrolledCGZSolver,
        UnrolledCGParams(mode="prox", gamma=2.0, num_iters=4),
        0.3,
        z,
        **kw,
    )


def test_shapes_windows_and_history():
    geom, enc, z, _ = _setup()
    state, window = _run(geom, enc, z, num_iters=5, loss_window=3)
    assert isinstance(state, ADMMState) and window.shape == (3, 4, 2, 3)
    torch.testing.assert_close(window[-1], state.x)
    _, h = run_admm_history(
        enc,
        geom,
        DiagonalProxXSolver,
        DiagonalProxParams(),
        UnrolledCGZSolver,
        UnrolledCGParams(num_iters=4),
        0.3,
        z,
        5,
    )
    assert h.x.shape == (5, 4, 2, 3) and h.consistency_rms.shape == (5, 2)
    assert torch.isfinite(h.primal_res).all()


def test_truncated_forward_matches_full_and_changes_gradient():
    geom, enc, z, maps = _setup()
    full, wfull = _run(geom, enc, z, num_iters=8, loss_window=3)
    loss = wfull.square().mean()
    gfull = torch.autograd.grad(loss, maps, retain_graph=True)[0]
    trunc, wtrunc = _run(geom, enc, z, num_iters=8, loss_window=3, grad_window=4)
    torch.testing.assert_close(full.z, trunc.z)
    torch.testing.assert_close(wfull, wtrunc)
    gtrunc = torch.autograd.grad(wtrunc.square().mean(), maps)[0]
    assert torch.isfinite(gtrunc).all() and not torch.allclose(gfull, gtrunc, atol=1e-6)
    _, clamped = _run(geom, enc, z, num_iters=2, loss_window=5)
    assert clamped.shape[0] == 2
