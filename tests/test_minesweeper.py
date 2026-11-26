"""Minesweeper generator: solver correctness and uniqueness of generated boards."""

from __future__ import annotations

import itertools

import numpy as np

from sheaf_admm.data.build_minesweeper import (
    HIDDEN,
    MINE,
    NUMBER_OFFSET,
    canonical_key,
    count_layouts,
    generate_board,
    neighbours,
    numbers_of,
)


def _brute_force(hidden, numbers, nbrs):
    cells = np.flatnonzero(hidden)
    total = 0
    for bits in itertools.product((0, 1), repeat=len(cells)):
        mines = np.zeros(len(hidden), dtype=int)
        mines[cells] = bits
        total += all(hidden[r] or mines[nbrs[r]].sum() == numbers[r] for r in range(len(hidden)))
    return total


def test_count_layouts_matches_brute_force():
    rng = np.random.default_rng(0)
    size = 5
    nbrs = neighbours(size)
    checked = 0
    for _ in range(300):
        mine = np.zeros(size * size, dtype=bool)
        mine[rng.choice(size * size, 4, replace=False)] = True
        numbers = numbers_of(mine, nbrs)
        hidden = mine.copy()
        extra = rng.choice(np.flatnonzero(~mine), rng.integers(0, 8), replace=False)
        hidden[extra] = True
        if not all(not hidden[i] or any(not hidden[n] for n in nbrs[i]) for i in range(25)):
            continue
        assert count_layouts(hidden, numbers, nbrs, limit=2) == min(
            _brute_force(hidden, numbers, nbrs), 2
        )
        checked += 1
    assert checked > 100


def test_generated_boards_are_unique_and_consistent():
    size, mines = 9, 10
    nbrs = neighbours(size)
    rng = np.random.default_rng(1)
    boards = []
    while len(boards) < 30:
        board = generate_board(rng, size, mines, int(rng.integers(20, 72)), nbrs)
        if board is not None:
            boards.append(board)
    for inputs, labels in boards:
        hidden = inputs == HIDDEN
        mine = labels == MINE
        assert mine.sum() == mines
        assert (mine <= hidden).all()  # mines are never revealed
        numbers = numbers_of(mine, nbrs)
        assert (inputs[~hidden] == numbers[~hidden] + NUMBER_OFFSET).all()
        assert all(any(not hidden[n] for n in nbrs[i]) for i in np.flatnonzero(hidden))
        assert count_layouts(hidden, numbers, nbrs, limit=5) == 1


def test_canonical_key_is_symmetry_invariant():
    rng = np.random.default_rng(2)
    board = rng.integers(1, 11, size=81).astype(np.uint8)
    key = canonical_key(board, 9)
    grid = board.reshape(9, 9)
    for k in range(4):
        rot = np.rot90(grid, k)
        assert canonical_key(rot.reshape(-1), 9) == key
        assert canonical_key(np.fliplr(rot).reshape(-1), 9) == key
