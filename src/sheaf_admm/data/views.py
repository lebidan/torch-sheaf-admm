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
