"""Compare active-LoRA geometry and factor gradients to frozen JAX outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sheaf_admm.geometry import create_lora_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    ref = np.load(Path(__file__).with_name("jax_lora_reference.npz"))

    def t(name: str) -> torch.Tensor:
        return torch.from_numpy(ref[name].copy()).to(device)

    edges, positions, maps, z, gate = (t(k) for k in ("edges", "positions", "maps", "z", "gate"))
    a, b = t("A").requires_grad_(), t("B").requires_grad_()
    geometry = create_lora_geometry(edges, positions, maps, a, b, 1.5, 8, gate=gate)
    actual = {
        "residual": geometry.edge_residuals(z),
        "laplacian": geometry.laplacian_apply(z),
        "energy": geometry.energy(z),
        "rms": geometry.consistency_rms(z),
    }
    actual["grad_A"], actual["grad_B"] = torch.autograd.grad(actual["energy"], (a, b))
    for key, value in actual.items():
        got = value.detach().cpu().numpy()
        want = ref[key]
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5, err_msg=key)
        print(f"{key:>10}: max_abs={np.max(np.abs(got - want)):.3g}")
    print(f"Passed {len(actual)} active-LoRA parity checks on {device}")


if __name__ == "__main__":
    main()
