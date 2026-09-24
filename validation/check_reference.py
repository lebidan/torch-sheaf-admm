"""Compare the Torch core against the frozen pre-port JAX CPU outputs.

Run ``.venv/bin/python validation/check_reference.py`` after the core port.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sheaf_admm.admm import run_admm
from sheaf_admm.geometry import FixedGeometry
from sheaf_admm.solvers.x_solvers import (
    DenseQuadraticParams,
    DenseQuadraticXSolver,
    DiagonalProxParams,
    DiagonalProxXSolver,
    SimpleParams,
    SimpleXSolver,
)
from sheaf_admm.solvers.z_solvers import GDParams, GDZSolver, UnrolledCGParams, UnrolledCGZSolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = torch.device(args.device)
    ref = np.load(Path(__file__).with_name("jax_reference.npz"))
    t = lambda name: torch.from_numpy(ref[name].copy()).to(device)  # noqa: E731
    edges, maps, z, y, target, rho = (t(k) for k in ("edges", "maps", "z", "y", "target", "rho"))
    geom = FixedGeometry(edges, maps)
    diag = {
        "q_diag": t("q_diag"),
        "q": t("q"),
        "l1_weight": t("l1"),
        "lower": t("lower"),
        "upper": t("upper"),
    }
    actual = {
        "residual": geom.edge_residuals(z),
        "laplacian": geom.laplacian_apply(z),
        "energy": geom.energy(z),
        "rms": geom.consistency_rms(z),
        "x_simple": SimpleXSolver.solve(z, y, rho, {"h": target, "beta": 1.2}, SimpleParams()),
        "x_diagonal": DiagonalProxXSolver.solve(z, y, rho, diag, DiagonalProxParams()),
        "x_dense": DenseQuadraticXSolver.solve(
            z, y, rho, {"Q": t("dense_Q"), "q": diag["q"]}, DenseQuadraticParams()
        ),
    }
    for mode in ("prox", "project"):
        actual[f"cg_{mode}"] = UnrolledCGZSolver.solve(
            target, z, geom, UnrolledCGParams(mode=mode, gamma=2.0, num_iters=4), rho
        )
        actual[f"gd_{mode}"] = GDZSolver.solve(
            target,
            z,
            geom,
            GDParams(mode=mode, gamma=2.0, num_steps=4, eta=torch.tensor(0.01, device=device)),
            rho,
        )

    state, window = run_admm(
        diag,
        geom,
        DiagonalProxXSolver,
        DiagonalProxParams(),
        UnrolledCGZSolver,
        UnrolledCGParams(mode="prox", gamma=2.0, num_iters=4),
        rho,
        z,
        5,
        relaxation_alpha=1.2,
        loss_window=2,
    )
    actual.update(admm_x=state.x, admm_z=state.z, admm_y=state.y, admm_window=window)
    maps_grad = maps.clone().requires_grad_()
    loss = FixedGeometry(edges, maps_grad).laplacian_apply(z).square().sum()
    actual["laplacian_map_grad"] = torch.autograd.grad(loss, maps_grad)[0]

    for key, value in actual.items():
        got = value.detach().cpu().numpy()
        expected = ref[key]
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-5, err_msg=key)
        print(f"{key:>20}: max_abs={np.max(np.abs(got - expected)):.3g}")
    print(f"Passed {len(actual)} JAX-to-Torch parity checks on {device}")


if __name__ == "__main__":
    main()
