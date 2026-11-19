"""Views: grid patchify/reassemble round-trips, Sudoku slice/reassemble round-trips,
and the constraint multigraph has the expected 243 edges over 27 agents."""

from __future__ import annotations

import json

import numpy as np
import torch

from sheaf_admm.data import views as V


def test_grid_patchify_reassemble_roundtrip():
    """Mean-reassembly of patches cut from one image recovers that image on the
    covered pixels (every pixel is covered with stride 2, patch 3)."""
    H, W, C = 8, 8, 3
    stride, ps = 2, 3
    centers = V.grid_agent_centers((H, W), stride=stride, patch_size=ps)
    B = 2
    img = torch.randn(B, H, W, C)
    patches = V.patchify_batch_jax(img, torch.as_tensor(centers), ps)
    assert patches.shape == (centers.shape[0], B, ps, ps, C)

    recon = V.reassemble_logits(np.asarray(patches), centers, (H, W), C, mode="mean")
    assert recon.shape == (B, H, W, C)
    np.testing.assert_allclose(recon, np.asarray(img), atol=1e-5)


def test_grid_agent_centers_and_edges():
    centers = V.grid_agent_centers((8, 8), stride=2, patch_size=3)
    assert centers.shape == (16, 2)  # 4x4 grid of centers
    edges4 = V.build_grid_edge_indices(centers, stride=2, connectivity=4)
    edges8 = V.build_grid_edge_indices(centers, stride=2, connectivity=8)
    # 4-way: right+down on a 4x4 grid = 2*4*3 = 24; 8-way adds the two diagonals.
    assert edges4.shape == (24, 2)
    assert edges8.shape[0] > edges4.shape[0]
    assert int(edges8.max()) < centers.shape[0]


def test_sudoku_slice_reassemble_roundtrip():
    """The 27 row/col/box views reassemble (averaged) back to the 9x9 grid."""
    B, C = 2, 5
    grid = torch.randn(B, 9, 9, C)
    views = V.sudoku_slice_batch_jax(grid)
    assert views.shape == (B, 27, 9, C)
    recon = V.reassemble_sudoku_logits(views)
    assert recon.shape == (B, 9, 9, C)
    np.testing.assert_allclose(np.asarray(recon), np.asarray(grid), atol=1e-5)


def test_sudoku_multigraph_structure():
    """81 cells x 3 edges per cell-clique = 243 edges over 27 agents."""
    edges, map_u, map_v = V.build_sudoku_multigraph(9)
    assert edges.shape == (243, 2)
    assert map_u.shape == (243,) and map_v.shape == (243,)
    assert int(edges.max()) + 1 == 27
    # Cell-slot indices are in 0..8 (the 9-cell local view).
    assert int(map_u.max()) < 9 and int(map_v.max()) < 9
    assert int(map_u.min()) >= 0 and int(map_v.min()) >= 0


def test_sudoku_cell_indices():
    """Each of the 27 agents sees 9 cells; the 81 board cells are each seen 3 times."""
    cell_ids = V.build_sudoku_cell_indices(9)
    assert cell_ids.shape == (27, 9)
    flat = np.asarray(cell_ids).reshape(-1)
    counts = np.bincount(flat, minlength=81)
    assert counts.shape == (81,)
    np.testing.assert_array_equal(counts, np.full(81, 3))


def test_nqueens_row_column_views_roundtrip_and_graph():
    size, batch, channels = 10, 2, 3
    grids = torch.randn(batch, size, size, channels)

    views = V.nqueens_slice_batch_jax(grids)
    assert views.shape == (batch, 2 * size, size, channels)
    np.testing.assert_allclose(
        np.asarray(V.reassemble_nqueens_logits(views)), np.asarray(grids), atol=1e-6
    )

    cell_ids = V.build_nqueens_cell_indices(size)
    assert cell_ids.shape == (2 * size, size)
    counts = np.bincount(np.asarray(cell_ids).reshape(-1), minlength=size * size)
    np.testing.assert_array_equal(counts, np.full(size * size, 2))
    np.testing.assert_array_equal(np.asarray(cell_ids[0]), np.arange(size))
    np.testing.assert_array_equal(np.asarray(cell_ids[size]), np.arange(0, size * size, size))

    edges, map_u, map_v = V.build_nqueens_multigraph(size)
    assert edges.shape == (size * size, 2)
    assert map_u.shape == map_v.shape == (size * size,)
    assert V.nqueens_num_nodes(size) == 2 * size
    np.testing.assert_array_equal(np.asarray(edges[37]), [3, 17])
    assert int(map_u[37]) == 7
    assert int(map_v[37]) == 3


def test_nqueens_views_reject_non_square_grids():
    with np.testing.assert_raises_regex(ValueError, "expected square grids"):
        V.nqueens_slice_batch_jax(torch.zeros(1, 10, 9, 3))


def test_nqueens_task_prepares_loss_and_solution_metrics():
    from sheaf_admm.training import NQueensTask, make_task

    size = 10
    queen_columns = [0, 2, 5, 7, 9, 4, 8, 1, 3, 6]
    labels = np.ones((size, size), dtype=np.int64)
    labels[np.arange(size), queen_columns] = 2
    inputs = labels.copy()
    inputs[3, queen_columns[3]] = 1
    batch = {
        "inputs": inputs.reshape(1, -1),
        "labels": labels.reshape(1, -1),
        "height": size,
        "width": size,
    }

    task = make_task("n_queens")
    assert isinstance(task, NQueensTask)
    fwd, targets, aux = task.prepare(batch)
    assert fwd["patches"].shape == (2 * size, 1, size, 3)
    assert fwd["edge_indices"].shape == (size * size, 2)
    assert fwd["model_kwargs"]["map_u"].shape == (size * size,)
    assert targets["slice_labels"].shape == (2 * size, 1, size)
    assert targets["labels_grid"].shape == (1, size, size)

    logits_window = torch.randn(2, 2 * size, 1, size, 2)
    loss = task.loss(logits_window, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    slice_labels = targets["slice_labels"]
    correct_logits = torch.full((2 * size, 1, size, 2), -8.0)
    correct_logits.scatter_(-1, slice_labels[..., None], 8.0)
    metrics = task.evaluate(correct_logits, targets, aux)
    assert metrics == {
        "cell_acc": 1.0,
        "solved": 1.0,
        "valid": 1.0,
        "clues_preserved": 1.0,
        "cell_acc_perm": 1.0,
        "solved_perm": 1.0,
        "valid_perm": 1.0,
    }

    invalid_grid = targets["labels_grid"].clone()
    invalid_grid[0, 0, 1] = 1
    invalid_views = V.nqueens_slice_batch_jax(invalid_grid[..., None])[..., 0].permute(1, 0, 2)
    invalid_logits = torch.full((2 * size, 1, size, 2), -8.0)
    invalid_logits.scatter_(-1, invalid_views[..., None], 8.0)
    metrics = task.evaluate(invalid_logits, targets, aux)
    assert metrics["solved"] == 0.0
    assert metrics["valid"] == 0.0
    assert metrics["clues_preserved"] == 1.0


def test_nqueens_diagonal_views_graph_and_reassembly():
    size, batch = 6, 2
    num_agents = V.nqueens_num_nodes(size, diagonals=True)
    assert num_agents == 2 * size + 2 * (2 * size - 1)

    cell_ids = V.build_nqueens_cell_indices(size, diagonals=True)
    assert cell_ids.shape == (num_agents, size)
    flat = np.asarray(cell_ids).reshape(-1)
    counts = np.bincount(flat[flat >= 0], minlength=size * size)
    np.testing.assert_array_equal(counts, np.full(size * size, 4))
    # main diagonal (c - r = 0) and anti-diagonal (r + c = N-1) are full length
    main = 2 * size + (size - 1)
    anti = 2 * size + (2 * size - 1) + (size - 1)
    np.testing.assert_array_equal(np.asarray(cell_ids[main]), np.arange(size) * (size + 1))
    np.testing.assert_array_equal(
        np.asarray(cell_ids[anti]), np.arange(size) * (size - 1) + (size - 1)
    )

    grids = torch.randn(batch, size, size, 3)
    views = V.nqueens_slice_batch_jax(grids, diagonals=True, pad_value=0.0)
    assert views.shape == (batch, num_agents, size, 3)
    np.testing.assert_allclose(
        np.asarray(V.reassemble_nqueens_logits(views)), np.asarray(grids), atol=1e-6
    )

    edges, map_u, map_v = V.build_nqueens_multigraph(size, diagonals=True)
    assert edges.shape == (6 * size * size, 2)
    assert int(map_u.max()) < 4 * size and int(map_v.max()) < 4 * size
    # every edge links two agents that really share a cell at the given slots
    ids = np.asarray(cell_ids)
    for (u, v), su, sv in zip(np.asarray(edges), np.asarray(map_u), np.asarray(map_v)):
        assert ids[u, su % size] == ids[v, sv % size] >= 0


def test_nqueens_diagonal_task_masks_padding_and_decodes_permutation():
    from sheaf_admm.training import make_task
    from sheaf_admm.training.tasks import max_weight_permutation

    size = 10
    queen_columns = [0, 2, 5, 7, 9, 4, 8, 1, 3, 6]
    labels = np.ones((size, size), dtype=np.int64)
    labels[np.arange(size), queen_columns] = 2
    inputs = np.ones_like(labels)
    inputs[3, queen_columns[3]] = 2
    batch = {"inputs": inputs.reshape(1, -1), "labels": labels.reshape(1, -1)}

    task = make_task("n_queens", diagonals=True, queen_weight=3.0)
    fwd, targets, aux = task.prepare(batch)
    num_agents = V.nqueens_num_nodes(size, diagonals=True)
    assert fwd["patches"].shape == (num_agents, 1, size, 3)
    pad = targets["slice_labels"] == task.ignore_index
    assert pad.sum() == num_agents * size - 4 * size * size
    # padded slots see input token 0
    assert (fwd["patches"][..., 0][pad] == 1).all()

    logits = torch.randn(2, num_agents, 1, size, 2)
    loss = task.loss(logits, targets)
    assert torch.isfinite(loss)

    # perfect per-cell logits -> everything solved, through both decoders
    good = torch.where(targets["slice_labels"] == 1, 8.0, -8.0)
    good_logits = torch.stack((-good, good), dim=-1)
    metrics = task.evaluate(good_logits, targets, aux)
    assert metrics["solved"] == 1.0 and metrics["solved_perm"] == 1.0

    w = torch.randn(4, 7, 7)
    cols = max_weight_permutation(w)
    import itertools

    for i in range(4):
        best = max(
            itertools.permutations(range(7)),
            key=lambda p, i=i: sum(float(w[i, r, p[r]]) for r in range(7)),
        )
        assert sorted(cols[i].tolist()) == list(range(7))
        np.testing.assert_allclose(
            float(w[i, torch.arange(7), cols[i]].sum()),
            sum(float(w[i, r, best[r]]) for r in range(7)),
            rtol=1e-5,
        )


def test_random_dihedral_batch_keeps_nqueens_solutions_valid():
    from sheaf_admm.data import random_dihedral_batch

    size = 10
    queen_columns = [0, 2, 5, 7, 9, 4, 8, 1, 3, 6]
    labels = np.ones((size, size), dtype=np.uint8)
    labels[np.arange(size), queen_columns] = 2
    inputs = np.ones_like(labels)
    inputs[[1, 4, 7], [queen_columns[1], queen_columns[4], queen_columns[7]]] = 2
    batch_in = np.repeat(inputs.reshape(1, -1), 16, axis=0)
    batch_lab = np.repeat(labels.reshape(1, -1), 16, axis=0)

    rng = np.random.default_rng(0)
    aug_in, aug_lab, tids = random_dihedral_batch(rng, batch_in, batch_lab, size, size)
    assert set(tids.tolist()) <= set(range(8)) and len(set(tids.tolist())) > 1
    for board_in, board_lab in zip(aug_in, aug_lab, strict=True):
        q = board_lab.reshape(size, size) == 2
        r, c = np.nonzero(q)
        assert len(set(r)) == len(set(c)) == size
        assert len(set(r - c)) == len(set(r + c)) == size
        # clues move with the solution
        clue = board_in.reshape(size, size) == 2
        assert clue.sum() == 3 and (q | ~clue).all()

    # non-square boards only use shape-preserving symmetries
    _, _, tids = random_dihedral_batch(
        rng, np.zeros((32, 12), np.int64), np.zeros((32, 12), np.int64), 3, 4
    )
    assert set(tids.tolist()) <= {0, 2, 4, 5}


def test_maze_wall_border_prepad():
    """``prepare_maze_patches`` one-hots tokens and pads the *border* with the wall
    token, so a boundary agent whose patch runs out of bounds sees walls there
    (not the zero-padding ``patchify`` would otherwise insert)."""
    from sheaf_admm.data.common import TOKEN_IDS

    B, H, W, ps, ncls = 1, 6, 6, 3, 6
    inp = torch.zeros((B, H, W), dtype=torch.long)  # interior token 0 everywhere
    centers = V.grid_agent_centers((H, W), stride=2, patch_size=ps)
    patches = V.prepare_maze_patches(inp, torch.as_tensor(centers), ps, ncls)
    assert patches.shape == (centers.shape[0], B, ps, ps, ncls)

    # The agent at center (5, 5) has a 3x3 patch covering rows/cols 4,5,6; row 6
    # and col 6 are out of bounds -> wall token, while in-bounds pixels stay 0.
    cs = np.asarray(centers)
    idx = int(np.where((cs[:, 0] == 5) & (cs[:, 1] == 5))[0][0])
    am = np.argmax(np.asarray(patches[idx, 0]), axis=-1)  # [ps, ps]
    assert am[2, 2] == TOKEN_IDS["wall"]  # bottom-right (out of bounds) -> wall
    assert am[0, 0] == 0  # top-left (in bounds) -> interior token


def test_maze_task_prepare_rectangular_metadata():
    """Rectangular OOD maze splits carry height/width metadata through the task hook."""
    from sheaf_admm.training.tasks import MazeTask

    B, H, W = 1, 19, 37
    batch = {
        "inputs": np.zeros((B, H * W), dtype=np.int32),
        "labels": np.zeros((B, H * W), dtype=np.int32),
        "height": H,
        "width": W,
    }
    fwd, targets, aux = MazeTask().prepare(batch)
    assert aux["image_hw"] == (H, W)
    assert fwd["patches"].shape[1] == B
    assert targets["labels_img"].shape == (B, H, W)


def test_puzzle_dataset_batches_include_grid_metadata(tmp_path):
    """Puzzle batches preserve metadata dimensions for non-square grids."""
    from sheaf_admm.data.common import PuzzleDatasetMetadata
    from sheaf_admm.data.loaders import PuzzleDataset

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    H, W = 19, 37
    np.save(split_dir / "all__inputs.npy", np.zeros((1, H * W), dtype=np.uint8))
    np.save(split_dir / "all__labels.npy", np.zeros((1, H * W), dtype=np.uint8))
    np.save(split_dir / "all__puzzle_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(split_dir / "all__group_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(split_dir / "all__puzzle_identifiers.npy", np.array([0], dtype=np.int32))
    PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=6,
        seq_len=H * W,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        total_puzzles=1,
        sets=["all"],
        height=H,
        width=W,
    ).write(split_dir / "dataset.json")

    _set, batch = next(PuzzleDataset(tmp_path, "train").iter_test_batches(1))
    assert batch["height"] == H
    assert batch["width"] == W


def test_puzzle_metadata_ignores_legacy_format_keys(tmp_path):
    """Old standard-token dataset metadata may include now-unused format keys."""
    from sheaf_admm.data.common import PuzzleDatasetMetadata

    payload = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 6,
        "seq_len": 35,
        "num_puzzle_identifiers": 1,
        "total_groups": 1,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": 1,
        "sets": ["all"],
        "height": 5,
        "width": 7,
        "input_format": "standard",
        "num_input_channels": None,
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload))

    metadata = PuzzleDatasetMetadata.from_json(path)
    assert metadata.seq_len == 35
    assert metadata.height == 5
    assert metadata.width == 7


def test_nqueens_builder_writes_valid_puzzle_dataset(tmp_path):
    from sheaf_admm.data.build_N_Queens import _write_split
    from sheaf_admm.data.loaders import PuzzleDataset

    size = 4
    target = np.ones((size, size), dtype=np.int64)
    queen_columns = [1, 3, 0, 2]
    target[np.arange(size), queen_columns] = 2
    input_board = target.copy()
    input_board[0, queen_columns[0]] = 1
    example = {
        "n": size,
        "num_solutions": 1,
        "input": input_board.reshape(-1).tolist(),
        "target": target.reshape(-1).tolist(),
        "instance_id": "unique-example",
    }

    _write_split(tmp_path, "train", [example], size)
    dataset = PuzzleDataset(tmp_path, "train")
    puzzle_set = dataset.sets["all"]

    assert dataset.metadata.vocab_size == 3
    assert dataset.metadata.height == size
    assert dataset.metadata.width == size
    np.testing.assert_array_equal(puzzle_set.inputs[0].reshape(size, size), input_board)
    np.testing.assert_array_equal(puzzle_set.labels[0].reshape(size, size), target)


def test_nqueens_builder_rejects_invalid_target():
    from sheaf_admm.data.build_N_Queens import _validate_example

    example = {
        "n": 4,
        "num_solutions": 1,
        "input": [1] * 16,
        "target": [2] * 16,
    }
    with np.testing.assert_raises_regex(ValueError, "target must contain 4 queens"):
        _validate_example(example, board_size=4)
