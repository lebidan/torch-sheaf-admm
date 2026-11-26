"""Minesweeper task: agent layout, masking, reassembly and top-k decoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sheaf_admm.data import views as V
from sheaf_admm.data.build_minesweeper import generate_board, neighbours
from sheaf_admm.training import make_task


def _boards(count=4, seed=0):
    rng = np.random.default_rng(seed)
    nbrs = neighbours(9)
    out = []
    while len(out) < count:
        board = generate_board(rng, 9, 10, int(rng.integers(20, 72)), nbrs)
        if board is not None:
            out.append(board)
    inputs, labels = map(np.stack, zip(*out, strict=True))
    return {"inputs": inputs, "labels": labels}


def test_layout_covers_every_constraint_and_links_shared_cells():
    cell_ids, edges, map_u, map_v = V.build_minesweeper_graph(9, 5, 2)
    assert cell_ids.shape == (9, 25)
    assert V.minesweeper_num_nodes(9, 5, 2) == 9
    ids = cell_ids.numpy()
    for (u, v), su, sv in zip(edges.numpy(), map_u.numpy(), map_v.numpy(), strict=True):
        assert ids[u, su] == ids[v, sv]
    with pytest.raises(ValueError):
        V.build_minesweeper_graph(9, 3, 2)  # splits 3x3 neighbourhoods


def test_task_masks_revealed_cells_and_decodes_perfect_logits():
    task = make_task("minesweeper", mine_weight=2.0)
    batch = _boards()
    fwd, targets, aux = task.prepare(batch)
    assert fwd["patches"].shape == (9, 4, 25, 11)
    revealed = torch.as_tensor(batch["inputs"]) != 1
    sliced_revealed = V.minesweeper_slice_batch(revealed, targets["cell_ids"]).permute(1, 0, 2)
    assert (targets["slice_labels"][sliced_revealed] == task.ignore_index).all()
    assert (targets["slice_labels"][~sliced_revealed] >= 0).all()

    logits = torch.randn(2, *fwd["patches"].shape[:3], 2)
    assert torch.isfinite(task.loss(logits, targets))

    mines = V.minesweeper_slice_batch(targets["mines"].long(), targets["cell_ids"])
    good = torch.where(mines.permute(1, 0, 2) == 1, 6.0, -6.0)
    metrics = task.evaluate(torch.stack((-good, good), dim=-1), targets, aux)
    assert metrics["solved"] == metrics["solved_topk"] == 1.0
    assert metrics["cell_acc"] == metrics["mine_recall"] == 1.0


def test_topk_fixes_a_miscalibrated_but_well_ranked_board():
    task = make_task("minesweeper")
    fwd, targets, aux = task.prepare(_boards(count=2, seed=3))
    mines = V.minesweeper_slice_batch(targets["mines"].long(), targets["cell_ids"]).permute(1, 0, 2)
    # every mine scores higher than every safe cell, but nothing crosses 0
    margin = torch.where(mines == 1, -0.5, -3.0)
    metrics = task.evaluate(torch.stack((torch.zeros_like(margin), margin), dim=-1), targets, aux)
    assert metrics["solved"] == 0.0
    assert metrics["solved_topk"] == 1.0
