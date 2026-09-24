"""Torch sheaf operators, adjoints, masks, and gradients."""

import numpy as np
import torch

from sheaf_admm.geometry import FixedGeometry, create_lora_geometry


def _fixture(dtype=torch.float64):
    torch.manual_seed(1)
    edges = torch.tensor([[0, 1], [0, 1], [1, 2], [2, 0]])
    maps = torch.randn(4, 2, 2, 3, dtype=dtype, requires_grad=True)
    z = torch.randn(3, 2, 3, dtype=dtype, requires_grad=True)
    return edges, maps, z


def test_coboundary_laplacian_dense_and_adjoint():
    edges, maps, z = _fixture()
    g = FixedGeometry(edges, maps)
    F = np.zeros((8, 9))
    for e, (u, v) in enumerate(edges.tolist()):
        F[e * 2 : (e + 1) * 2, u * 3 : (u + 1) * 3] += maps.detach().numpy()[e, 0]
        F[e * 2 : (e + 1) * 2, v * 3 : (v + 1) * 3] -= maps.detach().numpy()[e, 1]
    got = g.laplacian_apply(z)
    want = np.stack(
        [(F.T @ F @ z.detach().numpy()[:, b].reshape(-1)).reshape(3, 3) for b in range(2)], 1
    )
    np.testing.assert_allclose(got.detach().numpy(), want, atol=1e-10)
    r = torch.randn_like(g.edge_residuals(z))
    adj = torch.autograd.grad((g.edge_residuals(z) * r).sum(), z, retain_graph=True)[0]
    direct = (
        torch.zeros_like(z)
        .index_add(0, edges[:, 0], torch.einsum("eij,ebi->ebj", maps[:, 0], r))
        .index_add(0, edges[:, 1], -torch.einsum("eij,ebi->ebj", maps[:, 1], r))
    )
    torch.testing.assert_close(adj, direct)
    torch.testing.assert_close(g.energy(z), 0.5 * g.edge_residuals(z).square().sum())
    assert (z * got).sum() >= -1e-10


def test_mask_empty_edges_and_gradients():
    edges, maps, z = _fixture()
    g = FixedGeometry(edges, maps, torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=z.dtype))
    g.energy(z).backward()
    assert maps.grad is not None and torch.isfinite(maps.grad).all()
    assert maps.grad[1].abs().sum() == 0
    empty = FixedGeometry(edges[:0], maps[:0])
    torch.testing.assert_close(empty.laplacian_apply(z), torch.zeros_like(z))
    torch.testing.assert_close(empty.consistency_rms(z), z.new_full((2,), 1e-3))


def test_zero_lora_equals_fixed_and_factor_gradients():
    edges, maps, z = _fixture(torch.float32)
    positions = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    A = torch.randn(3, 2, 4, 2, 2, requires_grad=True)
    B = torch.zeros(3, 2, 4, 3, 2, requires_grad=True)
    lora = create_lora_geometry(edges, positions, maps, A, B, 1.0, 4)
    fixed = FixedGeometry(edges, maps)
    torch.testing.assert_close(lora.edge_residuals(z), fixed.edge_residuals(z))
    torch.testing.assert_close(lora.laplacian_apply(z), fixed.laplacian_apply(z))
    lora.energy(z).backward()
    assert B.grad is not None and B.grad.abs().sum() > 0
    assert maps.grad is not None and maps.grad.abs().sum() > 0
