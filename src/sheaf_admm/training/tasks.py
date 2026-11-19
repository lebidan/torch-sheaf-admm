"""Task-specific tensor preparation, losses, and evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sheaf_admm.data import views as V

PATH_TOKEN = 5


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.long().reshape(-1), reduction="none"
    ).reshape(labels.shape)


def _grid_graph(image_hw, stride, patch_size, connectivity):
    centers = V.grid_agent_centers(image_hw, stride=stride, patch_size=patch_size)
    edges = torch.as_tensor(
        V.build_grid_edge_indices(centers, stride, connectivity), dtype=torch.long
    )
    positions = torch.as_tensor(V.node_positions(centers), dtype=torch.float32)
    return centers, edges, positions


class MazeTask:
    name = "maze"

    def __init__(self, *, patch_size=3, stride=2, connectivity=8, num_classes=6):
        self.patch_size, self.stride, self.connectivity = patch_size, stride, connectivity
        self.num_classes = num_classes

    def prepare(self, batch):
        inputs = torch.as_tensor(batch["inputs"])
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        B = inputs.shape[0]
        H, W = V.infer_image_hw(inputs.shape[1], batch.get("height"), batch.get("width"))
        inp_img, lab_img = inputs.reshape(B, H, W), labels.reshape(B, H, W)
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.prepare_maze_patches(inp_img, centers, self.patch_size, self.num_classes)
        label_patches, masks = V.get_agent_label_patches_jax(lab_img, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        targets = {"label_patches": label_patches.long(), "masks": masks, "labels_img": lab_img}
        return fwd, targets, {"centers": centers, "image_hw": (H, W)}

    def loss(self, logits_window, targets):
        lp, masks = targets["label_patches"], targets["masks"]
        ce = _ce(logits_window, lp.unsqueeze(0).expand(logits_window.shape[0], *lp.shape))
        return (
            (ce * masks.unsqueeze(0)).sum(dim=tuple(range(1, ce.ndim))) / masks.sum().clamp_min(1)
        ).mean()

    def evaluate(self, logits_final, targets, aux):
        recon = V.reassemble_logits(
            logits_final, aux["centers"], aux["image_hw"], self.num_classes, mode="mean"
        )
        pred = recon.argmax(dim=-1)
        labels = targets["labels_img"]
        solved = ((pred == PATH_TOKEN) == (labels == PATH_TOKEN)).all(dim=(1, 2))
        return {
            "solved": solved.float().mean().item(),
            "cell_acc": (pred == labels).float().mean().item(),
        }


class MNISTTask:
    name = "mnist"

    def __init__(self, *, patch_size=3, stride=3, connectivity=8, num_classes=10):
        self.patch_size, self.stride, self.connectivity, self.num_classes = (
            patch_size,
            stride,
            connectivity,
            num_classes,
        )

    def prepare(self, batch):
        images = torch.as_tensor(batch["images"], dtype=torch.float32)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        H, W = images.shape[1:3]
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.patchify_batch_jax(images, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        return fwd, {"labels": labels}, {}

    def loss(self, logits_window, targets):
        labels = targets["labels"]
        expanded = labels.view(1, 1, -1).expand(logits_window.shape[:-1])
        return _ce(logits_window, expanded).mean()

    def loss_graph(self, logits, targets):
        return _ce(logits, targets["labels"]).mean()

    def evaluate(self, logits_final, targets, aux):
        agg = logits_final.softmax(dim=-1).mean(dim=0)
        return {"acc": (agg.argmax(dim=-1) == targets["labels"]).float().mean().item()}

    def evaluate_graph(self, logits, targets, aux):
        return {"acc": (logits.argmax(dim=-1) == targets["labels"]).float().mean().item()}


class SudokuTask:
    name = "sudoku"
    num_classes = 10

    def prepare(self, batch):
        inputs = torch.as_tensor(batch["inputs"], dtype=torch.long)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        B = inputs.shape[0]
        inp, lab = inputs.reshape(B, 9, 9), labels.reshape(B, 9, 9)
        edges, map_u, map_v = V.build_sudoku_multigraph(9)
        cell_ids = V.build_sudoku_cell_indices(9)
        patches = V.sudoku_slice_batch_jax(F.one_hot(inp, self.num_classes).float()).permute(
            1, 0, 2, 3
        )
        slice_labels = V.sudoku_slice_batch_jax(lab[..., None])[..., 0].permute(1, 0, 2).long()
        fwd = {
            "patches": patches,
            "edge_indices": torch.as_tensor(edges, dtype=torch.long),
            "model_kwargs": {
                "map_u": torch.as_tensor(map_u, dtype=torch.long),
                "map_v": torch.as_tensor(map_v, dtype=torch.long),
                "cell_ids": torch.as_tensor(cell_ids, dtype=torch.long),
            },
        }
        targets = {"slice_labels": slice_labels, "labels_grid": lab, "inputs_grid": inp}
        return fwd, targets, {}

    def loss(self, logits_window, targets):
        labels = targets["slice_labels"]
        return _ce(logits_window, labels.unsqueeze(0).expand(logits_window.shape[:-1])).mean()

    def evaluate(self, logits_final, targets, aux):
        recon = V.reassemble_sudoku_logits(logits_final.permute(1, 0, 2, 3))
        pred = recon.argmax(dim=-1)
        labels = targets["labels_grid"]
        empty = targets["inputs_grid"] == 0
        correct = pred == labels
        return {
            "cell_acc": correct.float().mean().item(),
            "solved": correct.all(dim=(1, 2)).float().mean().item(),
            "completion": (correct & empty).sum().float().div(empty.sum().clamp_min(1)).item(),
        }


def _nqueens_board_metrics(predicted, labels, clues):
    """Row/column/diagonal validity, clue preservation and exact-match for ``[B, N, N]`` bools."""
    size = predicted.shape[1]
    rows_valid = (predicted.sum(dim=2) == 1).all(dim=1)
    columns_valid = (predicted.sum(dim=1) == 1).all(dim=1)
    diagonals_valid = torch.ones(predicted.shape[0], dtype=torch.bool, device=predicted.device)
    flipped = predicted.flip(dims=(2,))
    for offset in range(-(size - 2), size - 1):
        diagonals_valid &= torch.diagonal(predicted, offset, dim1=1, dim2=2).sum(dim=-1) <= 1
        diagonals_valid &= torch.diagonal(flipped, offset, dim1=1, dim2=2).sum(dim=-1) <= 1
    clues_preserved = (predicted | ~clues).all(dim=(1, 2))
    valid = rows_valid & columns_valid & diagonals_valid & clues_preserved
    correct = predicted == labels
    return {
        "cell_acc": correct.float().mean().item(),
        "solved": correct.all(dim=(1, 2)).float().mean().item(),
        "valid": valid.float().mean().item(),
        "clues_preserved": clues_preserved.float().mean().item(),
    }


def max_weight_permutation(weights: torch.Tensor) -> torch.Tensor:
    """Exact max-weight row->column assignment for ``[B, N, N]`` scores (bitmask DP).

    Returns ``[B, N]`` long: the chosen column of each row. Cost is
    ``O(N^2 * 2^N)`` per board, fine up to N ~ 14.
    """
    b, n, _ = weights.shape
    if n > 16:
        raise ValueError(f"bitmask permutation decoding supports N <= 16, got {n}")
    device = weights.device
    weights = weights.float()
    full = 1 << n
    masks = torch.arange(full, device=device)
    popcount = torch.zeros(full, dtype=torch.long, device=device)
    for c in range(n):
        popcount += (masks >> c) & 1
    neg_inf = torch.finfo(torch.float32).min
    dp = torch.full((b, full), neg_inf, device=device)
    dp[:, 0] = 0.0
    choices = []
    for r in range(n):
        new = torch.full((b, full), neg_inf, device=device)
        choice = torch.zeros((b, full), dtype=torch.long, device=device)
        for c in range(n):
            bit = 1 << c
            src = masks[((masks & bit) == 0) & (popcount == r)]
            dst = src | bit
            cand = dp[:, src] + weights[:, r, c, None]
            better = cand > new[:, dst]
            new[:, dst] = torch.where(better, cand, new[:, dst])
            choice[:, dst] = torch.where(better, torch.full_like(choice[:, dst], c), choice[:, dst])
        dp = new
        choices.append(choice)
    cols = torch.zeros((b, n), dtype=torch.long, device=device)
    mask = torch.full((b,), full - 1, dtype=torch.long, device=device)
    for r in range(n - 1, -1, -1):
        c = choices[r].gather(1, mask[:, None])[:, 0]
        cols[:, r] = c
        mask = mask ^ torch.bitwise_left_shift(torch.ones_like(c), c)
    return cols


class NQueensTask:
    """N-Queens with line agents.

    * ``diagonals`` — add one agent per main and anti diagonal (2(2N-1) extra
      agents, padded to length N with input token 0 and ignored in the loss), so
      the diagonal constraint is represented in the sheaf. Requires
      ``rm_sharing: sudoku`` with ``rm_num_slots: 4N`` (typed per-slot maps).
    * ``queen_weight`` — cross-entropy weight on the queen class (~N-1 for
      balanced rows; queens are 1/N of the cells).
    * ``perm_decode`` — also report ``*_perm`` metrics where the board is decoded
      as the max-score permutation (exactly one queen per row and column, given
      queens forced). Diagonals and the clue-implied solution are *not* imposed,
      so a correct ``*_perm`` still reflects what the model learned.
    """

    name = "n_queens"
    input_num_classes = 3
    ignore_index = -100

    def __init__(self, *, diagonals=False, queen_weight=1.0, perm_decode=True):
        self.diagonals = bool(diagonals)
        self.queen_weight = float(queen_weight)
        self.perm_decode = bool(perm_decode)

    def prepare(self, batch):
        inputs = torch.as_tensor(batch["inputs"], dtype=torch.long)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        if inputs.ndim != 2 or labels.shape != inputs.shape:
            raise ValueError("N-Queens inputs and labels must have the same [B, N*N] shape")

        height, width = V.infer_image_hw(
            inputs.shape[1], batch.get("height"), batch.get("width")
        )
        if height != width:
            raise ValueError(f"N-Queens requires square boards, got {(height, width)}")
        if not torch.isin(inputs, torch.tensor((0, 1, 2))).all():
            raise ValueError("N-Queens inputs must use tokens 0, 1, and 2")
        if not torch.isin(labels, torch.tensor((1, 2))).all():
            raise ValueError("N-Queens labels must use tokens 1 (empty) and 2 (queen)")

        batch_size = inputs.shape[0]
        inputs_grid = inputs.reshape(batch_size, height, width)
        labels_grid = (labels.reshape(batch_size, height, width) == 2).long()
        edges, map_u, map_v = V.build_nqueens_multigraph(height, diagonals=self.diagonals)
        cell_ids = V.build_nqueens_cell_indices(height, diagonals=self.diagonals)

        # Slice the integer grid first so padded diagonal slots get input token 0.
        input_tokens = V.nqueens_slice_batch_jax(
            inputs_grid[..., None], diagonals=self.diagonals, pad_value=0
        )[..., 0]
        input_views = F.one_hot(input_tokens, num_classes=self.input_num_classes).float()
        input_views = input_views.permute(1, 0, 2, 3)
        label_views = V.nqueens_slice_batch_jax(
            labels_grid[..., None], diagonals=self.diagonals, pad_value=self.ignore_index
        )[..., 0].permute(1, 0, 2)
        fwd = {
            "patches": input_views,
            "edge_indices": edges,
            "model_kwargs": {
                "map_u": map_u,
                "map_v": map_v,
                "cell_ids": cell_ids,
            },
        }
        targets = {
            "slice_labels": label_views,
            "labels_grid": labels_grid,
            "inputs_grid": inputs_grid,
        }
        return fwd, targets, {}

    def loss(self, logits_window, targets):
        labels = targets["slice_labels"]
        expanded = labels.unsqueeze(0).expand(logits_window.shape[:-1])
        weight = None
        if self.queen_weight != 1.0:
            weight = torch.tensor(
                [1.0, self.queen_weight], dtype=logits_window.dtype, device=logits_window.device
            )
        return F.cross_entropy(
            logits_window.reshape(-1, logits_window.shape[-1]),
            expanded.reshape(-1).long(),
            weight=weight,
            ignore_index=self.ignore_index,
        )

    def evaluate(self, logits_final, targets, aux):
        logits_grid = V.reassemble_nqueens_logits(logits_final.permute(1, 0, 2, 3))
        labels = targets["labels_grid"].bool()
        clues = targets["inputs_grid"] == 2
        metrics = _nqueens_board_metrics(logits_grid.argmax(dim=-1) == 1, labels, clues)
        if self.perm_decode:
            score = logits_grid[..., 1] - logits_grid[..., 0]
            score = torch.where(clues, torch.full_like(score, 1e4), score)
            cols = max_weight_permutation(score)
            perm = torch.zeros_like(clues)
            perm.scatter_(2, cols[..., None], True)
            perm_metrics = _nqueens_board_metrics(perm, labels, clues)
            for key in ("cell_acc", "solved", "valid"):
                metrics[f"{key}_perm"] = perm_metrics[key]
        return metrics


def make_task(name: str, **kwargs):
    tasks = {
        "maze": MazeTask,
        "mnist": MNISTTask,
        "sudoku": SudokuTask,
        "n_queens": NQueensTask,
    }
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; available: {sorted(tasks)}")
    return tasks[name](**kwargs)
