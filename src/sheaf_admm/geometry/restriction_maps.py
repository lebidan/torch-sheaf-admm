"""Trainable shared restriction maps and edge endpoint selection."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

Initializer = Callable[[object, tuple[int, ...], torch.dtype], torch.Tensor]


def get_direction_names(num_directions: int) -> tuple[str, ...]:
    if num_directions == 4:
        return ("N", "E", "S", "W")
    if num_directions == 8:
        return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    raise ValueError("num_directions must be 4 or 8")


def normalize_rm_sharing(rm_sharing: str) -> str:
    rm_sharing = rm_sharing.lower()
    return "directional" if rm_sharing == "nesw" else rm_sharing


def compute_direction_index(
    dy: torch.Tensor, dx: torch.Tensor, num_directions: int = 4
) -> torch.Tensor:
    if num_directions not in (4, 8):
        raise ValueError("num_directions must be 4 or 8")
    north, south, east, west = dy < 0, dy > 0, dx > 0, dx < 0
    if num_directions == 4:
        return torch.where(north, 0, torch.where(south, 2, torch.where(east, 1, 3))).long()
    return torch.where(
        north & east,
        1,
        torch.where(
            north & west,
            7,
            torch.where(
                north,
                0,
                torch.where(
                    south & east,
                    3,
                    torch.where(south & west, 5, torch.where(south, 4, torch.where(east, 2, 6))),
                ),
            ),
        ),
    ).long()


def _init_one(method: str, d_e: int, d_v: int, dtype=torch.float32) -> torch.Tensor:
    mat = torch.empty(d_e, d_v, dtype=dtype)
    if method == "orthonormal":
        nn.init.orthogonal_(mat)
    elif method == "default":
        nn.init.xavier_normal_(mat)
    elif method == "identity":
        mat = torch.eye(d_e, d_v, dtype=dtype)
    else:
        raise ValueError(f"unknown rm_init={method!r} (orthonormal|default|identity)")
    return mat


def make_rm_initializer(init_method: str, d_e: int, d_v: int) -> Initializer:
    def init(_key, shape, dtype=torch.float32):
        if shape[-2:] != (d_e, d_v) or len(shape) not in (2, 4):
            raise ValueError(f"rm init expects (d_e,d_v) or (E,2,d_e,d_v), got {shape}")
        if len(shape) == 2:
            return _init_one(init_method, d_e, d_v, dtype)
        return torch.stack(
            [_init_one(init_method, d_e, d_v, dtype) for _ in range(shape[0] * shape[1])]
        ).reshape(shape)

    return init


def make_sudoku_rm_initializer(init_method: str, d_e: int, d_v: int) -> Initializer:
    if init_method == "soft_slice" and 9 * d_e > d_v:
        raise ValueError(f"soft_slice needs 9*d_e <= d_v (got 9*{d_e}={9 * d_e} > {d_v})")

    def init(_key, shape, dtype=torch.float32):
        if shape != (9, d_e, d_v):
            raise ValueError(f"Sudoku map shape must be {(9, d_e, d_v)}, got {shape}")
        if init_method == "soft_slice":
            mats = 0.01 * torch.randn(shape, dtype=dtype)
            for k in range(9):
                mats[k, :, k * d_e : (k + 1) * d_e] += torch.eye(d_e, dtype=dtype)
            return mats
        return torch.stack([_init_one(init_method, d_e, d_v, dtype) for _ in range(9)])

    return init


class DirectionalRestrictionMaps(nn.Module):
    def __init__(
        self,
        stalk_dim: int,
        edge_stalk_dim: int | None = None,
        init_method: str = "orthonormal",
        num_directions: int = 4,
    ):
        super().__init__()
        d_e = edge_stalk_dim or stalk_dim
        init = make_rm_initializer(init_method, d_e, stalk_dim)
        self.maps = nn.ParameterDict(
            {
                name: nn.Parameter(init(None, (d_e, stalk_dim)))
                for name in get_direction_names(num_directions)
            }
        )
        self.num_directions = num_directions

    def forward(self) -> dict[str, torch.Tensor]:
        return {name: self.maps[name] for name in get_direction_names(self.num_directions)}


class SharedRestrictionMap(nn.Module):
    def __init__(
        self, stalk_dim: int, edge_stalk_dim: int | None = None, init_method: str = "orthonormal"
    ):
        super().__init__()
        d_e = edge_stalk_dim or stalk_dim
        self.R_shared = nn.Parameter(
            make_rm_initializer(init_method, d_e, stalk_dim)(None, (d_e, stalk_dim))
        )

    def forward(self) -> torch.Tensor:
        return self.R_shared


class SudokuRestrictionMaps(nn.Module):
    def __init__(self, stalk_dim: int, edge_stalk_dim: int, init_method: str = "soft_slice"):
        super().__init__()
        self.R_indices = nn.Parameter(
            make_sudoku_rm_initializer(init_method, edge_stalk_dim, stalk_dim)(
                None, (9, edge_stalk_dim, stalk_dim)
            )
        )

    def forward(self) -> torch.Tensor:
        return self.R_indices


def build_directional_restriction_maps(
    R_dict: dict[str, torch.Tensor],
    edge_indices: torch.Tensor,
    node_positions: torch.Tensor,
    num_directions: int = 4,
) -> torch.Tensor:
    R_stack = torch.stack([R_dict[n] for n in get_direction_names(num_directions)])
    u, v = edge_indices[:, 0].long(), edge_indices[:, 1].long()
    dy = node_positions[v, 0] - node_positions[u, 0]
    dx = node_positions[v, 1] - node_positions[u, 1]
    dir_uv = compute_direction_index(dy, dx, num_directions)
    dir_vu = compute_direction_index(-dy, -dx, num_directions)
    return torch.stack([R_stack[dir_uv], R_stack[dir_vu]], dim=1)


def build_shared_restriction_maps(R_shared: torch.Tensor, num_edges: int) -> torch.Tensor:
    return R_shared[None, None].expand(num_edges, 2, *R_shared.shape)


def build_sudoku_restriction_maps(
    R_stack: torch.Tensor, map_u: torch.Tensor, map_v: torch.Tensor
) -> torch.Tensor:
    return torch.stack([R_stack[map_u.long()], R_stack[map_v.long()]], dim=1)
