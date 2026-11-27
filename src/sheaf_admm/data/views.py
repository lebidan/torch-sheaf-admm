"""Views: the map between a global grid and the per-agent local representation.

Two task families share this module:

* **Grid tasks (maze / MNIST).** Agents sit on a regular grid of centers
  (:func:`grid_agent_centers`), each owning a square ``patch_size`` patch
  (:func:`patchify_batch_jax`). The agent graph is the grid adjacency
  (:func:`build_grid_edge_indices`). After ADMM the per-agent logit patches are
  stitched back to a global grid by averaging overlaps
  (:func:`reassemble_logits`). Maze borders are handled by a wall-token pre-pad
  *before* patch extraction (see :func:`prepare_maze_patches`), so boundary
  agents see walls rather than zeros.

* **Sudoku.** The 81 cells are covered by 27 constraint agents — 9 rows, 9
  columns, 9 boxes — each a 9-cell view (:func:`sudoku_slice_batch_jax`). The
  agent graph is the constraint multigraph (:func:`build_sudoku_multigraph`):
  each cell is a 3-clique over its (row, col, box) agents, with per-edge
  shared-cell slots. Reassembly averages the three views that cover each cell
  (:func:`reassemble_sudoku_logits`).
* **N-Queens.** A square board is covered by one agent per row and column
  (:func:`nqueens_slice_batch_jax`). Each cell is shared by its row and column
  agents; the graph records those intersections
  (:func:`build_nqueens_multigraph`).

Tensor conventions: images are ``[B, H, W, C]``, agent patches are
``[N, B, ph, pw, C]``, node states are ``[N, B, d_v]``, edges are ``[E, 2]``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

# =============================================================================
# Grid agents: centers, patches, edges
# =============================================================================


def grid_agent_centers(image_hw: Sequence[int], stride: int, patch_size: int) -> np.ndarray:
    """Place agent centers on a regular grid.

    The first center sits at ``patch_size // 2`` along each axis and centers
    step by ``stride`` thereafter (``range(patch_size // 2, dim, stride)``).
    Returns ``[N, 2]`` integer ``(y, x)`` coordinates in row-major order.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    if stride <= 0:
        raise ValueError("stride must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    center = patch_size // 2
    coords = [(y, x) for y in range(center, h, stride) for x in range(center, w, stride)]
    return np.asarray(coords, dtype=np.int64)


def patchify_batch_jax(
    batch_images: torch.Tensor, centers_yx: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """Zero-padded [N,B,ps,ps,C] patches from NHWC images."""
    images = torch.as_tensor(batch_images)
    centers = torch.as_tensor(centers_yx, dtype=torch.long, device=images.device)
    if images.ndim != 4:
        raise ValueError(f"expected images of shape (B,H,W,C); got {images.shape}")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_yx must have shape (N, 2)")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    pad = patch_size // 2
    padded = F.pad(images.permute(0, 3, 1, 2), (pad, pad, pad, pad)).permute(0, 2, 3, 1)
    offsets = torch.arange(patch_size, device=images.device)
    ys = centers[:, 0, None, None] + offsets[None, :, None]
    xs = centers[:, 1, None, None] + offsets[None, None, :]
    return padded[:, ys, xs, :].permute(1, 0, 2, 3, 4)


def build_grid_edge_indices(centers: np.ndarray, stride: int, connectivity: int = 4) -> np.ndarray:
    """Undirected grid edges between adjacent agent centers, each stored once.

    Orientation is fixed so every pair appears a single time:

    * 4-way: right, down.
    * 8-way: right, down, down-right, down-left.

    Returns ``[E, 2]`` int32 ``(u, v)`` indices into ``centers``.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if centers.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    center_to_idx = {tuple(map(int, c)): i for i, c in enumerate(centers)}
    offsets = [(0, stride), (stride, 0)]
    if connectivity == 8:
        offsets += [(stride, stride), (stride, -stride)]

    edges = []
    for i, (cy, cx) in enumerate(centers):
        for dy, dx in offsets:
            j = center_to_idx.get((int(cy + dy), int(cx + dx)))
            if j is not None:
                edges.append([i, j])

    return np.asarray(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)


def node_positions(centers: np.ndarray) -> torch.Tensor:
    """Float32 ``[N, 2]`` node positions for direction-indexed restriction maps."""
    return torch.as_tensor(centers, dtype=torch.float32)


# =============================================================================
# Grid reassembly: stitch agent logit patches back to a global grid
# =============================================================================


def _overlap_slices(
    center_yx: Sequence[int], patch_size: int, image_hw: tuple[int, int]
) -> tuple[slice, slice, slice, slice]:
    """In-bounds image slices and the matching local-patch slices for one agent."""
    h, w = image_hw
    pad = patch_size // 2
    cy, cx = int(center_yx[0]), int(center_yx[1])
    y0, x0 = cy - pad, cx - pad

    y_start, x_start = max(y0, 0), max(x0, 0)
    y_end, x_end = min(y0 + patch_size, h), min(x0 + patch_size, w)

    py_start = y_start - y0
    px_start = x_start - x0
    return (
        slice(y_start, y_end),
        slice(x_start, x_end),
        slice(py_start, py_start + (y_end - y_start)),
        slice(px_start, px_start + (x_end - x_start)),
    )


def reassemble_logits(
    patches: np.ndarray,
    centers_yx: np.ndarray,
    image_hw: Sequence[int],
    num_classes: int,
    mode: str = "mean",
) -> np.ndarray:
    """Stitch per-agent logit patches ``[N, B, ph, pw, C]`` into a global grid.

    * ``"mean"``: average overlapping agent logits — ``logits_sum / max(counts, 1)``.
    * ``"winner_conf"``: per pixel, keep the logits of the most confident agent
      (max softmax probability).

    Returns ``[B, H, W, num_classes]``.
    """
    if isinstance(patches, torch.Tensor):
        return _reassemble_logits_torch(patches, centers_yx, image_hw, num_classes, mode)
    patch_arr = np.asarray(patches)
    centers = np.asarray(centers_yx)
    if patch_arr.ndim != 5:
        raise ValueError("patches must have shape (N, B, ph, pw, num_classes)")
    if centers.ndim != 2 or centers.shape[0] != patch_arr.shape[0]:
        raise ValueError("centers_yx must align with patches on the agent axis")
    if patch_arr.shape[-1] != num_classes:
        raise ValueError("num_classes must match the patch channel dimension")
    mode = str(mode).lower()
    if mode not in {"mean", "winner_conf"}:
        raise ValueError(f"unknown reassembly mode {mode!r} (mean|winner_conf)")

    num_agents, batch, ph, pw, _ = patch_arr.shape
    h, w = int(image_hw[0]), int(image_hw[1])
    if ph != pw:
        raise ValueError("patches must be square")

    if mode == "mean":
        logits_sum = np.zeros((batch, h, w, num_classes), dtype=np.float32)
        counts = np.zeros((batch, h, w, 1), dtype=np.float32)
        for a in range(num_agents):
            ys, xs, pys, pxs = _overlap_slices(centers[a], ph, (h, w))
            if ys.start == ys.stop or xs.start == xs.stop:
                continue
            logits_sum[:, ys, xs, :] += patch_arr[a, :, pys, pxs, :]
            counts[:, ys, xs, :] += 1.0
        return logits_sum / np.maximum(counts, 1.0)

    best_logits = np.zeros((batch, h, w, num_classes), dtype=np.float32)
    best_conf = np.full((batch, h, w), -np.inf, dtype=np.float32)
    for a in range(num_agents):
        ys, xs, pys, pxs = _overlap_slices(centers[a], ph, (h, w))
        if ys.start == ys.stop or xs.start == xs.stop:
            continue
        chunk = patch_arr[a, :, pys, pxs, :].astype(np.float32, copy=False)
        probs = np.exp(chunk - np.max(chunk, axis=-1, keepdims=True))
        probs /= np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-12)
        conf = np.max(probs, axis=-1)  # [B, h', w']
        cur = best_conf[:, ys, xs]
        better = conf > cur
        best_conf[:, ys, xs] = np.where(better, conf, cur)
        best_logits[:, ys, xs, :] = np.where(better[..., None], chunk, best_logits[:, ys, xs, :])
    return best_logits


def _reassemble_logits_torch(
    patches: torch.Tensor, centers_yx, image_hw, num_classes: int, mode: str
):
    if patches.ndim != 5 or patches.shape[-1] != num_classes:
        raise ValueError("patches must have shape (N, B, ph, pw, num_classes)")
    n, batch, ph, pw, _ = patches.shape
    if ph != pw:
        raise ValueError("patches must be square")
    centers = torch.as_tensor(centers_yx).cpu().tolist()
    if len(centers) != n:
        raise ValueError("centers_yx must align with patches on the agent axis")
    h, w = int(image_hw[0]), int(image_hw[1])
    mode = mode.lower()
    if mode not in ("mean", "winner_conf"):
        raise ValueError(f"unknown reassembly mode {mode!r} (mean|winner_conf)")
    if mode == "mean":
        acc = patches.new_zeros((batch, h * w, num_classes))
        counts = patches.new_zeros((h * w,))
        for a, center in enumerate(centers):
            ys, xs, pys, pxs = _overlap_slices(center, ph, (h, w))
            if ys.start == ys.stop or xs.start == xs.stop:
                continue
            yi = torch.arange(ys.start, ys.stop, device=patches.device)
            xi = torch.arange(xs.start, xs.stop, device=patches.device)
            ids = (yi[:, None] * w + xi[None, :]).reshape(-1)
            chunk = patches[a, :, pys, pxs, :].reshape(batch, -1, num_classes)
            acc = acc.index_add(1, ids, chunk)
            counts = counts.index_add(0, ids, torch.ones_like(ids, dtype=patches.dtype))
        return (acc / counts.clamp_min(1)[None, :, None]).reshape(batch, h, w, num_classes)
    best = patches.new_zeros((batch, h, w, num_classes))
    confidence = patches.new_full((batch, h, w), -float("inf"))
    for a, center in enumerate(centers):
        ys, xs, pys, pxs = _overlap_slices(center, ph, (h, w))
        if ys.start == ys.stop or xs.start == xs.stop:
            continue
        chunk = patches[a, :, pys, pxs, :]
        conf = chunk.softmax(dim=-1).amax(dim=-1)
        better = conf > confidence[:, ys, xs]
        updated = best.clone()
        updated[:, ys, xs] = torch.where(better[..., None], chunk, best[:, ys, xs])
        best = updated
        updated_conf = confidence.clone()
        updated_conf[:, ys, xs] = torch.where(better, conf, confidence[:, ys, xs])
        confidence = updated_conf
    return best


# =============================================================================
# Label patches (per-agent supervision)
# =============================================================================


def extract_patch_jax(
    image: torch.Tensor, center_yx: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.as_tensor(image)
    center = torch.as_tensor(center_yx, dtype=torch.long, device=image.device).reshape(1, 2)
    data = image[None] if image.ndim == 3 else image[None, :, :, None]
    patch = patchify_batch_jax(data, center, patch_size)[0, 0]
    mask = patchify_batch_jax(
        torch.ones_like(data[..., :1], dtype=torch.float32), center, patch_size
    )[0, 0, ..., 0]
    return (patch if image.ndim == 3 else patch[..., 0]), mask


def get_agent_label_patches_jax(
    labels: torch.Tensor, centers: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.as_tensor(labels)
    centers = torch.as_tensor(centers, dtype=torch.long, device=labels.device)
    patches = patchify_batch_jax(labels[..., None], centers, patch_size)[..., 0]
    masks = patchify_batch_jax(
        torch.ones_like(labels[..., None], dtype=torch.float32), centers, patch_size
    )[..., 0]
    return patches, masks


# =============================================================================
# Sudoku views: 27-agent constraint slices and reassembly
# =============================================================================


def sudoku_slice_batch_jax(batch_grids: torch.Tensor) -> torch.Tensor:
    grids = torch.as_tensor(batch_grids)
    if grids.ndim != 4 or grids.shape[1:3] != (9, 9):
        raise ValueError(f"expected grids of shape (B,9,9,C); got {grids.shape}")
    batch, _, _, channels = grids.shape
    rows = grids
    cols = grids.transpose(1, 2)
    boxes = (
        grids.reshape(batch, 3, 3, 3, 3, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, 9, 9, channels)
    )
    return torch.cat([rows, cols, boxes], dim=1)


def reassemble_sudoku_logits(batch_logits: torch.Tensor) -> torch.Tensor:
    if batch_logits.ndim != 4 or batch_logits.shape[1:3] != (27, 9):
        raise ValueError(f"expected logits of shape (B,27,9,C); got {batch_logits.shape}")
    batch, _, _, channels = batch_logits.shape
    rows = batch_logits[:, 0:9]
    cols = batch_logits[:, 9:18].transpose(1, 2)
    boxes = (
        batch_logits[:, 18:27]
        .reshape(batch, 3, 3, 3, 3, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, 9, 9, channels)
    )
    return (rows + cols + boxes) / 3.0


def build_sudoku_multigraph(grid_size: int = 9) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grid_size != 9:
        raise ValueError("Sudoku views require a 9x9 grid")
    slices = build_sudoku_cell_indices(grid_size)
    coverage: list[list[tuple[int, int]]] = [[] for _ in range(81)]
    for agent_id in range(27):
        for local_idx in range(9):
            coverage[int(slices[agent_id, local_idx])].append((agent_id, local_idx))
    edges, map_u, map_v = [], [], []
    for agents in coverage:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                (u, u_local), (v, v_local) = agents[i], agents[j]
                edges.append((u, v))
                map_u.append(u_local)
                map_v.append(v_local)
    return (
        torch.tensor(edges, dtype=torch.long),
        torch.tensor(map_u, dtype=torch.long),
        torch.tensor(map_v, dtype=torch.long),
    )


@functools.lru_cache(maxsize=1)
def get_cached_sudoku_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return build_sudoku_multigraph(9)


def build_sudoku_cell_indices(grid_size: int = 9) -> torch.Tensor:
    if grid_size != 9:
        raise ValueError("Sudoku views require a 9x9 grid")
    global_ids = torch.arange(81).reshape(1, 9, 9, 1)
    return sudoku_slice_batch_jax(global_ids)[0, :, :, 0].long()


def sudoku_num_nodes(grid_size: int = 9) -> int:
    return 3 * grid_size


def sudoku_node_positions(num_nodes: int) -> torch.Tensor:
    ids = torch.arange(num_nodes)
    return torch.stack([ids // 9, ids % 9], dim=1).float()


# =============================================================================
# N-Queens views: one agent per row and column (+ optional diagonal agents)
# =============================================================================
#
# Agent layout (``A`` agents, each holding a line of ``N`` slots):
#
# * rows ``0..N-1`` then columns ``N..2N-1`` (always);
# * with ``diagonals=True``: ``2N-1`` main diagonals (``c - r`` constant, ordered
#   by ``c - r + N - 1``) then ``2N-1`` anti-diagonals (``r + c`` constant).
#   Diagonals shorter than ``N`` are right-padded; padded slots have cell id -1.
#
# Row/column-only boards keep the original layout bit-for-bit.

NQUEENS_PAD_CELL = -1


@functools.lru_cache(maxsize=16)
def _nqueens_layout(grid_size: int, diagonals: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(cell_ids [A, N], agent_type [A])``; type 0=row, 1=col, 2=diag, 3=anti-diag."""
    n = int(grid_size)
    if n <= 0:
        raise ValueError("grid_size must be positive")
    cells = np.arange(n * n).reshape(n, n)
    lines = [cells[r] for r in range(n)] + [cells[:, c] for c in range(n)]
    types = [0] * n + [1] * n
    if diagonals:
        for off in range(-(n - 1), n):
            lines.append(np.diagonal(cells, offset=off))
            types.append(2)
        flipped = cells[:, ::-1]
        for s in range(2 * n - 1):
            # anti-diagonal r + c = s, ordered by increasing row
            lines.append(np.diagonal(flipped, offset=(n - 1) - s))
            types.append(3)
    ids = np.full((len(lines), n), NQUEENS_PAD_CELL, dtype=np.int64)
    for a, line in enumerate(lines):
        ids[a, : len(line)] = line
    return ids, np.asarray(types, dtype=np.int64)


def nqueens_num_nodes(grid_size: int = 10, diagonals: bool = False) -> int:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    return 2 * grid_size + (2 * (2 * grid_size - 1) if diagonals else 0)


def build_nqueens_cell_indices(grid_size: int = 10, diagonals: bool = False) -> torch.Tensor:
    """Map each agent's local slots to global cell IDs; shape ``[A, N]`` (pad = -1)."""
    ids, _ = _nqueens_layout(grid_size, diagonals)
    return torch.as_tensor(ids.copy(), dtype=torch.long)


def nqueens_slice_batch_jax(
    batch_grids: torch.Tensor, *, diagonals: bool = False, pad_value=0
) -> torch.Tensor:
    """Return per-agent line views ``[B, A, N, C]`` from ``[B, N, N, C]``.

    Without diagonals, ``A = 2N`` (rows then columns). With diagonals, padded
    slots of short diagonals are filled with ``pad_value``.
    """
    grids = torch.as_tensor(batch_grids)
    if grids.ndim != 4 or grids.shape[1] != grids.shape[2]:
        raise ValueError(f"expected square grids of shape (B,N,N,C); got {grids.shape}")
    if grids.shape[1] == 0:
        raise ValueError("N-Queens grids must have a positive size")
    if not diagonals:
        return torch.cat((grids, grids.transpose(1, 2)), dim=1)
    b, n, _, c = grids.shape
    ids = torch.as_tensor(_nqueens_layout(n, True)[0], device=grids.device)
    flat = grids.reshape(b, n * n, c)
    pad = torch.full((b, 1, c), pad_value, dtype=grids.dtype, device=grids.device)
    flat = torch.cat((flat, pad), dim=1)  # index n*n is the pad cell
    gather = torch.where(ids < 0, torch.full_like(ids, n * n), ids)
    return flat[:, gather.reshape(-1)].reshape(b, ids.shape[0], n, c)


def _infer_nqueens_layout(num_agents: int, line_length: int) -> bool:
    if num_agents == nqueens_num_nodes(line_length, False):
        return False
    if num_agents == nqueens_num_nodes(line_length, True):
        return True
    raise ValueError(
        f"expected 2N or 2N+2(2N-1) agents for N={line_length}; got {num_agents}"
    )


def reassemble_nqueens_logits(batch_logits: torch.Tensor) -> torch.Tensor:
    """Average every agent's prediction for each cell back to ``[B, N, N, C]``.

    Accepts ``[B, A, N, C]`` with either the row/column layout or the layout with
    diagonals (detected from ``A``); padded diagonal slots are ignored.
    """
    logits = torch.as_tensor(batch_logits)
    if logits.ndim != 4:
        raise ValueError(f"expected logits of shape (B,A,N,C); got {logits.shape}")
    b, num_agents, n, c = logits.shape
    if n == 0:
        raise ValueError(f"expected shape (B,A,N,C); got {logits.shape}")
    diagonals = _infer_nqueens_layout(num_agents, n)
    if not diagonals:
        rows = logits[:, :n]
        cols = logits[:, n:].transpose(1, 2)
        return (rows + cols) / 2.0
    ids = torch.as_tensor(_nqueens_layout(n, True)[0], device=logits.device).reshape(-1)
    valid = ids >= 0
    vals = logits.reshape(b, -1, c)[:, valid]
    out = torch.zeros(b, n * n, c, dtype=logits.dtype, device=logits.device)
    out.index_add_(1, ids[valid], vals)
    counts = torch.bincount(ids[valid], minlength=n * n).to(logits.dtype)
    return (out / counts[None, :, None]).reshape(b, n, n, c)


def build_nqueens_multigraph(
    grid_size: int = 10, diagonals: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cell-sharing edges and the matching restriction-map slot of each endpoint.

    Without diagonals: one (row, column) edge per board cell, ordered by cell,
    with ``map_u``/``map_v`` the cell's position inside the row/column (``0..N-1``).

    With diagonals: every pair of the (up to 4) agents covering a cell gets an
    edge. Slots are *typed*, ``type * N + position`` (``0..4N-1``), so rows,
    columns and both diagonal families get their own per-position maps; set
    ``rm_num_slots = 4N`` in the model config.
    """
    n = int(grid_size)
    if n <= 0:
        raise ValueError("grid_size must be positive")
    if not diagonals:
        cells = torch.arange(n * n, dtype=torch.long)
        rows = torch.div(cells, n, rounding_mode="floor")
        cols = cells % n
        edges = torch.stack((rows, n + cols), dim=1)
        return edges, cols, rows

    ids, types = _nqueens_layout(n, True)
    covering: list[list[tuple[int, int]]] = [[] for _ in range(n * n)]
    for a in range(ids.shape[0]):
        for pos in range(n):
            cell = ids[a, pos]
            if cell >= 0:
                covering[cell].append((a, int(types[a]) * n + pos))
    edges, map_u, map_v = [], [], []
    for agents in covering:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                (u, su), (v, sv) = agents[i], agents[j]
                edges.append((u, v))
                map_u.append(su)
                map_v.append(sv)
    return (
        torch.as_tensor(edges, dtype=torch.long),
        torch.as_tensor(map_u, dtype=torch.long),
        torch.as_tensor(map_v, dtype=torch.long),
    )


# =============================================================================
# Minesweeper views: overlapping square patches of cells
# =============================================================================
#
# Agent ``a`` owns a ``patch x patch`` block of cells whose top-left corners sit
# every ``stride`` cells (the last block is shifted to touch the border). Every
# cell shared by two agents gives one edge, with the cell's position inside each
# patch as restriction-map slot (``0 .. patch*patch - 1``).
#
# A revealed number constrains its 3x3 neighbourhood, so every 3x3 window of the
# board must lie inside at least one agent; otherwise no agent sees the whole
# constraint. The layout builder checks this. On a 9x9 board, 5x5 patches with
# stride 2 give 9 agents and satisfy it; 3x3 patches with stride 2 do not.


def _patch_starts(size: int, patch: int, stride: int) -> list[int]:
    if not 0 < patch <= size:
        raise ValueError(f"patch size must be in [1, {size}], got {patch}")
    if stride <= 0:
        raise ValueError("stride must be positive")
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@functools.lru_cache(maxsize=16)
def _minesweeper_layout(size: int, patch: int, stride: int):
    """``(cell_ids [A, patch*patch], edges [E, 2], map_u [E], map_v [E])`` as numpy arrays."""
    starts = _patch_starts(size, patch, stride)
    cell_ids = np.asarray(
        [
            [(r0 + i) * size + (c0 + j) for i in range(patch) for j in range(patch)]
            for r0 in starts
            for c0 in starts
        ],
        dtype=np.int64,
    )
    window = min(3, size)  # neighbourhood of a revealed number
    for r in range(size - window + 1):
        for c in range(size - window + 1):
            if not any(r0 <= r and r + window <= r0 + patch for r0 in starts) or not any(
                c0 <= c and c + window <= c0 + patch for c0 in starts
            ):
                raise ValueError(
                    f"patch={patch}, stride={stride} splits the 3x3 neighbourhood at "
                    f"({r + 1}, {c + 1}) across agents; use larger patches or a smaller stride"
                )
    # covering[cell] = every (agent, slot) that sees this cell
    covering: list[list[tuple[int, int]]] = [[] for _ in range(size * size)]
    for agent in range(cell_ids.shape[0]):
        for slot, cell in enumerate(cell_ids[agent]):
            covering[cell].append((agent, slot))
    edges, map_u, map_v = [], [], []
    for agents_on_cell in covering:
        for i in range(len(agents_on_cell)):
            for j in range(i + 1, len(agents_on_cell)):
                (agent_u, slot_u), (agent_v, slot_v) = agents_on_cell[i], agents_on_cell[j]
                edges.append((agent_u, agent_v))
                map_u.append(slot_u)
                map_v.append(slot_v)
    return (
        cell_ids,
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(map_u, dtype=np.int64),
        np.asarray(map_v, dtype=np.int64),
    )


def minesweeper_num_nodes(size: int = 9, patch: int = 5, stride: int = 2) -> int:
    return len(_patch_starts(size, patch, stride)) ** 2


def build_minesweeper_graph(
    size: int = 9, patch: int = 5, stride: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(cell_ids [A, patch*patch], edges [E, 2], map_u [E], map_v [E])`` tensors."""
    ids, edges, map_u, map_v = _minesweeper_layout(size, patch, stride)
    return tuple(torch.as_tensor(x.copy(), dtype=torch.long) for x in (ids, edges, map_u, map_v))


def minesweeper_slice_batch(batch_grids: torch.Tensor, cell_ids: torch.Tensor) -> torch.Tensor:
    """``[B, S*S, ...]`` per-cell values -> ``[B, A, patch*patch, ...]`` agent views."""
    grids = torch.as_tensor(batch_grids)
    ids = torch.as_tensor(cell_ids, device=grids.device)
    return grids[:, ids.reshape(-1)].reshape(grids.shape[0], *ids.shape, *grids.shape[2:])


def reassemble_minesweeper_logits(batch_logits: torch.Tensor, cell_ids: torch.Tensor) -> torch.Tensor:
    """Average every agent's prediction per cell: ``[B, A, L, C]`` -> ``[B, S*S, C]``."""
    logits = torch.as_tensor(batch_logits)
    if logits.ndim != 4:
        raise ValueError(f"expected logits of shape (B,A,L,C); got {logits.shape}")
    ids = torch.as_tensor(cell_ids, device=logits.device).reshape(-1)
    b, _, _, c = logits.shape
    cells = int(ids.max()) + 1
    out = torch.zeros(b, cells, c, dtype=logits.dtype, device=logits.device)
    out.index_add_(1, ids, logits.reshape(b, -1, c))
    counts = torch.bincount(ids, minlength=cells).to(logits.dtype)
    return out / counts[None, :, None]


# =============================================================================
# Maze wall-border pre-pad for boundary agents.
# =============================================================================


def prepare_maze_patches(
    inputs_img: torch.Tensor,
    centers: torch.Tensor,
    patch_size: int,
    num_input_classes: int,
    *,
    border_fill: float | None = None,
) -> torch.Tensor:
    from .common import TOKEN_IDS

    inputs = torch.as_tensor(inputs_img)
    border = patch_size // 2
    fill = TOKEN_IDS["wall"] if border_fill is None else border_fill
    bordered = F.pad(inputs[:, None].float(), (border, border, border, border), value=float(fill))[
        :, 0
    ].to(inputs.dtype)
    shifted = torch.as_tensor(centers, dtype=torch.long, device=inputs.device) + border
    onehot = F.one_hot(bordered.long(), num_input_classes).float()
    return patchify_batch_jax(onehot, shifted, patch_size)


# =============================================================================
# Misc
# =============================================================================


def infer_image_hw(
    seq_len: int, height: int | None = None, width: int | None = None
) -> tuple[int, int]:
    """Resolve ``(H, W)`` from ``seq_len``; require both dims for non-square grids."""
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("provide both height and width or neither")
        if height * width != seq_len:
            raise ValueError(f"dims {(height, width)} do not match seq_len={seq_len}")
        return int(height), int(width)
    root = int(math.isqrt(seq_len))
    if root * root != seq_len:
        raise ValueError("cannot infer non-square grid from seq_len; pass height/width")
    return root, root
