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


def make_task(name: str, **kwargs):
    tasks = {"maze": MazeTask, "mnist": MNISTTask, "sudoku": SudokuTask}
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; available: {sorted(tasks)}")
    return tasks[name](**kwargs)
