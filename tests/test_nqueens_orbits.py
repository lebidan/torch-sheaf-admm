"""Symmetry-disjoint N-Queens dataset builder."""

from __future__ import annotations

import numpy as np

from sheaf_admm.data.build_N_Queens_orbits import (
    enumerate_solutions,
    orbit_key,
    split_by_orbit,
    transform_cells,
    unique_puzzles,
)


def test_enumerate_solutions_counts():
    assert [len(enumerate_solutions(n)) for n in (4, 5, 6, 8)] == [2, 10, 4, 92]
    for sol in enumerate_solutions(8):
        rows = np.arange(8)
        assert len(set(sol)) == len(set(rows - sol)) == len(set(rows + sol)) == 8


def test_unique_puzzles_have_one_completion():
    solutions = enumerate_solutions(8)
    puzzles = unique_puzzles(solutions, [2, 3])
    assert puzzles
    for cells, s in list(puzzles.items())[:200]:
        matches = [
            i for i, sol in enumerate(solutions) if all(sol[r] == c for r, c in cells)
        ]
        assert matches == [s]


def test_split_by_orbit_never_leaks_symmetric_puzzles():
    n = 8
    puzzles = unique_puzzles(enumerate_solutions(n), [2, 3])
    train, test = split_by_orbit(puzzles, n, test_size=100, seed=0)
    assert len(train) + len(test) == len(puzzles)
    assert len(test) >= 100
    train_images = {transform_cells(p, t, n) for p in train for t in range(8)}
    assert not train_images & set(test)
    assert all(orbit_key(transform_cells(p, 5, n), n) == orbit_key(p, n) for p in test)
