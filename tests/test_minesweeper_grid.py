"""Minesweeper with the paper's Maze layout (task ``minesweeper_grid``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sheaf_admm.data.build_minesweeper import generate_board, neighbours, ood_suite
from sheaf_admm.training import make_task

ROOT = Path(__file__).resolve().parents[1]


def _boards(size=9, mines=10, count=3, seed=0):
    rng = np.random.default_rng(seed)
    nbrs = neighbours(size)
    safe = size * size - mines
    out = []
    while len(out) < count:
        board = generate_board(rng, size, mines, int(rng.integers(int(0.3 * safe), safe + 1)), nbrs)
        if board is not None:
            out.append(board)
    inputs, labels = map(np.stack, zip(*out, strict=True))
    return {"inputs": inputs.astype(np.int64), "labels": labels.astype(np.int64)}


def _perfect_logits(task, batch, fwd, aux):
    """Per-agent logits that are +-6 on the true class of every in-board cell."""
    from sheaf_admm.data import views as V

    size = aux["image_hw"][0]
    b = task.border
    mines = torch.as_tensor(batch["labels"] == 2, dtype=torch.long).reshape(-1, size, size)
    padded = torch.nn.functional.pad(mines.float(), (b, b, b, b)).long()
    patches, _inside = V.get_agent_label_patches_jax(padded, aux["centers"], task.patch_size)
    good = torch.where(patches == 1, 6.0, -6.0)
    return torch.stack((-good, good), dim=-1)  # [N, B, ps, ps, 2]


@pytest.mark.parametrize("size,agents", [(6, 16), (8, 25), (9, 25), (12, 49), (13, 49), (17, 81)])
def test_layout_tiles_odd_boards_with_an_outside_ring(size, agents):
    task = make_task("minesweeper_grid")
    batch = _boards(size, max(1, round(10 / 81 * size * size)), count=2)
    fwd, targets, aux = task.prepare(batch)
    assert fwd["patches"].shape == (agents, 2, 3, 3, 11)
    assert fwd["model_kwargs"]["node_positions"].shape == (agents, 2)
    # channel 0 = "outside the board": exactly the agents of the outer ring see it
    sees_outside = fwd["patches"][..., 0].sum(dim=(1, 2, 3)) > 0
    side = int(round(agents**0.5))
    assert int(sees_outside.sum()) == agents - (side - 2) ** 2
    # 8-neighbour grid: right, down, two diagonals
    assert fwd["edge_indices"].shape == (2 * side * (side - 1) + 2 * (side - 1) ** 2, 2)


def test_labels_ignore_outside_ring_and_revealed_cells():
    task = make_task("minesweeper_grid")
    batch = _boards()
    fwd, targets, aux = task.prepare(batch)
    labels = targets["slice_labels"]
    assert labels.shape == (25, 3, 3, 3)
    outside = fwd["patches"][..., 0] == 1
    assert (labels[outside] == task.ignore_index).all()
    # every hidden cell is supervised at least once
    hidden_cells = int((torch.as_tensor(batch["inputs"]) == 1).sum())
    assert int((labels >= 0).sum()) >= hidden_cells


def test_perfect_logits_solve_every_board_and_loss_is_finite():
    task = make_task("minesweeper_grid")
    batch = _boards(count=4, seed=1)
    fwd, targets, aux = task.prepare(batch)
    logits = _perfect_logits(task, batch, fwd, aux)
    board = task.reassemble(logits, targets, aux)
    assert board.shape == (4, 81, 2)
    metrics = task.evaluate(logits, targets, aux)
    assert metrics["solved"] == metrics["solved_topk"] == 1.0
    assert metrics["cell_acc"] == metrics["mine_recall"] == 1.0
    window = torch.randn(2, *logits.shape)
    assert torch.isfinite(task.loss(window, targets))
    assert task.agent_positions(targets, aux).shape == (25, 2)


def test_even_boards_still_run():
    task = make_task("minesweeper_grid")
    batch = _boards(10, 12, count=2)
    fwd, targets, aux = task.prepare(batch)
    logits = _perfect_logits(task, batch, fwd, aux)
    assert task.evaluate(logits, targets, aux)["solved"] == 1.0


@pytest.mark.parametrize("size,mines,expected", [(9, 10, [13, 17, 33]), (6, 5, [8, 12, 24])])
def test_ood_suite_keeps_parity_and_density(size, mines, expected):
    suite = ood_suite(size, mines)
    assert [s for _n, s, _m in suite] == expected
    for _name, new, count in suite:
        assert abs(count / new**2 - mines / size**2) < 0.01


def _model(name):
    from omegaconf import OmegaConf

    from sheaf_admm.models import model_config_from_dict
    from sheaf_admm.training import build_model

    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "experiment" / f"{name}.yaml"))
    return build_model(model_config_from_dict(cfg["model"]), cfg["model_type"], seed=0)


@pytest.mark.parametrize("size", [6, 9, 13])
def test_paper_configs_run_on_any_size(size):
    task = make_task("minesweeper_grid")
    batch = _boards(size, max(1, round(10 / 81 * size * size)), count=2, seed=2)
    fwd, targets, aux = task.prepare(batch)
    n = fwd["patches"].shape[0]

    sheaf = _model("minesweeper_grid_sheaf")
    window, _final, _geom = sheaf(fwd["patches"], fwd["edge_indices"], num_iters=3, loss_window=2,
                                  **fwd["model_kwargs"], training=False, compile_steps=False)
    assert window.shape == (2, n, 2, 3, 3, 2)
    assert torch.isfinite(task.loss(window, targets))
    assert set(task.evaluate(window[-1], targets, aux)) >= {"solved", "cell_acc"}

    mpnn = _model("minesweeper_grid_mpnn")
    logits, _hidden = mpnn(fwd["patches"], fwd["edge_indices"], num_rounds=3,
                           **fwd["model_kwargs"], training=False, compile_steps=False)
    assert logits.shape == (n, 2, 3, 3, 2)
    history, _ = mpnn(fwd["patches"], fwd["edge_indices"], num_rounds=3,
                      **fwd["model_kwargs"], training=False, compile_steps=False,
                      return_history=True)
    assert history.shape == (3, n, 2, 3, 3, 2)
    torch.testing.assert_close(history[-1], logits)
