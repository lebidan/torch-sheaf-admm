"""LoRA geometry: input-modulated restriction maps ``F = R + (alpha/r) A B^T``.

The shared base map ``R`` is modulated per agent and per edge-direction by a
low-rank update produced by the encoder. To keep the Laplacian matvec free of
the expensive per-iteration multi-index gather, we precompute *edge-indexed*
factors once at construction time (``create_lora_geometry`` /
``create_sudoku_lora_geometry``): each edge already knows the A/B factors of its
two endpoints in the relevant slot/direction. The matvec is then pure einsums.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class LoRAGeometry:
    """Sheaf geometry with per-agent LoRA-modulated restriction maps.

    Effective edge map ``F = R + (alpha/r) A B^T``. The ``*_edge`` tensors are
    the endpoints' A/B factors already gathered for each edge (see the
    ``create_*`` factories below), so ``laplacian_apply`` is gather-free.
    """

    edge_indices: torch.Tensor  # [E, 2]
    restriction_maps: torch.Tensor  # [E, 2, d_e, d_v] base maps R
    lora_alpha: float
    A_u_edge: torch.Tensor  # [E, B, d_e, r]
    A_v_edge: torch.Tensor  # [E, B, d_e, r]
    B_u_edge: torch.Tensor  # [E, B, d_v, r]
    B_v_edge: torch.Tensor  # [E, B, d_v, r]
    gate_u_edge: torch.Tensor | None = None  # [E, B]
    gate_v_edge: torch.Tensor | None = None  # [E, B]
    edge_mask: torch.Tensor | None = None  # [E] float mask

    def replace(self, **changes):
        return replace(self, **changes)

    @property
    def _scale(self) -> float:
        return self.lora_alpha / self.A_u_edge.shape[-1]

    def _apply_endpoint(self, z_e, R, A_edge, B_edge, gate_edge):
        """Effective map applied to one endpoint: ``(R + scale A B^T) z_e``."""
        Rz = torch.einsum("eij,ebj->ebi", R, z_e)  # [E, B, d_e]
        Btz = torch.einsum("ebjr,ebj->ebr", B_edge, z_e)  # [E, B, r]
        ABtz = torch.einsum("ebir,ebr->ebi", A_edge, Btz)  # [E, B, d_e]
        if gate_edge is not None:
            ABtz = ABtz * gate_edge[:, :, None]
        return Rz + self._scale * ABtz

    def _adjoint_endpoint(self, r, R, A_edge, B_edge, gate_edge):
        """Adjoint of ``_apply_endpoint``: ``(R + scale A B^T)^T r``."""
        contrib = torch.einsum("eij,ebi->ebj", R, r)  # [E, B, d_v]
        Atr = torch.einsum("ebir,ebi->ebr", A_edge, r)  # [E, B, r]
        if gate_edge is not None:
            Atr = Atr * gate_edge[:, :, None]
        lora = torch.einsum("ebjr,ebr->ebj", B_edge, Atr)  # [E, B, d_v]
        return contrib + self._scale * lora

    def edge_residuals(self, z: torch.Tensor) -> torch.Tensor:
        u, v = self.edge_indices[:, 0].long(), self.edge_indices[:, 1].long()
        Fz_u = self._apply_endpoint(
            z[u], self.restriction_maps[:, 0], self.A_u_edge, self.B_u_edge, self.gate_u_edge
        )
        Fz_v = self._apply_endpoint(
            z[v], self.restriction_maps[:, 1], self.A_v_edge, self.B_v_edge, self.gate_v_edge
        )
        r = Fz_u - Fz_v
        if self.edge_mask is not None:
            r = r * self.edge_mask[:, None, None]
        return r

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(self.edge_residuals(z) ** 2)

    def laplacian_apply(self, z: torch.Tensor) -> torch.Tensor:
        r = self.edge_residuals(z)  # [E, B, d_e]
        u, v = self.edge_indices[:, 0].long(), self.edge_indices[:, 1].long()
        contrib_u = self._adjoint_endpoint(
            r, self.restriction_maps[:, 0], self.A_u_edge, self.B_u_edge, self.gate_u_edge
        )
        contrib_v = self._adjoint_endpoint(
            r, self.restriction_maps[:, 1], self.A_v_edge, self.B_v_edge, self.gate_v_edge
        )
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


def _gather_edge_factors(edge_indices, A, B, gate, sel_u, sel_v):
    """Gather endpoint LoRA factors into edge-indexed tensors.

    ``A``/``B``: ``[N, B, K, d_*, r]`` per-agent factors (K = directions or
    cell-slots). ``sel_u``/``sel_v``: ``[E]`` slot index per endpoint.
    """
    u, v = edge_indices[:, 0].long(), edge_indices[:, 1].long()
    A_u_edge = A[u, :, sel_u.long()]  # [E, B, d_e, r]
    A_v_edge = A[v, :, sel_v.long()]
    B_u_edge = B[u, :, sel_u.long()]
    B_v_edge = B[v, :, sel_v.long()]
    gate_u_edge = gate_v_edge = None
    if gate is not None:
        gate_u_edge = gate[u, :, sel_u.long()]  # [E, B]
        gate_v_edge = gate[v, :, sel_v.long()]
    return A_u_edge, A_v_edge, B_u_edge, B_v_edge, gate_u_edge, gate_v_edge


def create_lora_geometry(
    edge_indices: torch.Tensor,
    node_positions: torch.Tensor,
    restriction_maps: torch.Tensor,
    A: torch.Tensor,  # [N, B, K, d_e, r]
    B: torch.Tensor,  # [N, B, K, d_v, r]
    lora_alpha: float,
    num_directions: int,
    gate: torch.Tensor | None = None,  # [N, B, K]
    edge_mask: torch.Tensor | None = None,
) -> LoRAGeometry:
    """Directional (grid) LoRA geometry: select factors by edge direction."""
    from .restriction_maps import compute_direction_index

    u, v = edge_indices[:, 0].long(), edge_indices[:, 1].long()
    dy = node_positions[v, 0] - node_positions[u, 0]
    dx = node_positions[v, 1] - node_positions[u, 1]
    dir_uv = compute_direction_index(dy, dx, num_directions)
    dir_vu = compute_direction_index(-dy, -dx, num_directions)
    A_u, A_v, B_u, B_v, g_u, g_v = _gather_edge_factors(edge_indices, A, B, gate, dir_uv, dir_vu)
    return LoRAGeometry(
        edge_indices=edge_indices,
        restriction_maps=restriction_maps,
        lora_alpha=lora_alpha,
        A_u_edge=A_u,
        A_v_edge=A_v,
        B_u_edge=B_u,
        B_v_edge=B_v,
        gate_u_edge=g_u,
        gate_v_edge=g_v,
        edge_mask=edge_mask,
    )


def create_sudoku_lora_geometry(
    edge_indices: torch.Tensor,
    map_u: torch.Tensor,  # [E] cell-slot 0-8
    map_v: torch.Tensor,  # [E] cell-slot 0-8
    restriction_maps: torch.Tensor,
    A: torch.Tensor,  # [N, B, 9, d_e, r]
    B: torch.Tensor,  # [N, B, 9, d_v, r]
    lora_alpha: float,
    gate: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
) -> LoRAGeometry:
    """Sudoku LoRA geometry: select factors by shared-cell slot (replaces direction)."""
    A_u, A_v, B_u, B_v, g_u, g_v = _gather_edge_factors(edge_indices, A, B, gate, map_u, map_v)
    return LoRAGeometry(
        edge_indices=edge_indices,
        restriction_maps=restriction_maps,
        lora_alpha=lora_alpha,
        A_u_edge=A_u,
        A_v_edge=A_v,
        B_u_edge=B_u,
        B_v_edge=B_v,
        gate_u_edge=g_u,
        gate_v_edge=g_v,
        edge_mask=edge_mask,
    )
