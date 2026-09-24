"""Fixed geometry: learned but input-independent restriction maps."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class FixedGeometry:
    """Sheaf geometry with restriction maps that do not depend on the input.

    The maps in ``restriction_maps`` are learned parameters (built once per
    forward pass from the shared base maps, see
    :mod:`sheaf_admm.geometry.restriction_maps`) but are the same for every
    batch element.
    """

    edge_indices: torch.Tensor  # [E, 2]
    restriction_maps: torch.Tensor  # [E, 2, d_e, d_v]
    edge_mask: torch.Tensor | None = None  # [E] float mask (0=dropped, 1=kept)

    def replace(self, **changes):
        return replace(self, **changes)

    def edge_residuals(self, z: torch.Tensor) -> torch.Tensor:
        u, v = self.edge_indices[:, 0].long(), self.edge_indices[:, 1].long()
        F_uv, F_vu = self.restriction_maps[:, 0], self.restriction_maps[:, 1]
        Fz_u = torch.einsum("eij,ebj->ebi", F_uv, z[u])
        Fz_v = torch.einsum("eij,ebj->ebi", F_vu, z[v])
        r = Fz_u - Fz_v
        if self.edge_mask is not None:
            r = r * self.edge_mask[:, None, None]
        return r

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(self.edge_residuals(z) ** 2)

    def laplacian_apply(self, z: torch.Tensor) -> torch.Tensor:
        r = self.edge_residuals(z)  # [E, B, d_e]
        u, v = self.edge_indices[:, 0].long(), self.edge_indices[:, 1].long()
        F_uv, F_vu = self.restriction_maps[:, 0], self.restriction_maps[:, 1]
        # F^T r, scattered back to the two endpoints (note the orientation sign on v).
        contrib_u = torch.einsum("eij,ebi->ebj", F_uv, r)
        contrib_v = torch.einsum("eij,ebi->ebj", F_vu, r)
        out = torch.zeros_like(z)
        out = out.index_add(0, u, contrib_u)
        out = out.index_add(0, v, -contrib_v)
        return out

    def consistency_rms(self, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        r = self.edge_residuals(z)
        return (
            torch.sqrt(torch.mean(r**2, dim=(0, 2)) + eps)
            if r.shape[0]
            else z.new_full((z.shape[1],), eps**0.5)
        )
