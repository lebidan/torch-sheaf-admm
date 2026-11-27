"""Build an N-Queens dataset whose train/test split is closed under board symmetries.

The ``brozonoyer/gram-n-queens`` splits are disjoint as puzzles, but every 10x10
test puzzle is a rotation or reflection of a training puzzle. With
``data.augment: dihedral`` the model then trains on (transformed) test puzzles.

This builder enumerates the puzzles itself instead:

1. all N-Queens solutions (backtracking; 724 for N=10);
2. every puzzle = a solution with ``k`` given queens (``k`` in ``--clues``) whose
   completion is unique;
3. puzzles are grouped into dihedral orbits (a puzzle and its 7 rotations /
   reflections), and whole orbits are assigned to train or test.

So no test puzzle, nor any symmetric image of one, is ever seen in training.
Output format matches :mod:`sheaf_admm.data.build_N_Queens` (tokens: 1 = empty,
2 = queen), so the N-Queens task and configs work unchanged.

Usage:
    python -m sheaf_admm.data.build_N_Queens_orbits --output-dir datasets/n_queens_10x10_orbits
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .common import PuzzleDatasetMetadata, save_npy

Puzzle = tuple[tuple[int, int], ...]  # sorted (row, col) of the given queens


@dataclass
class NQueensOrbitsConfig:
    board_size: int = 10
    clues: list[int] = field(default_factory=lambda: [2, 3])
    test_size: int = 1000
    train_size: int | None = None
    seed: int = 0
    output_dir: Path = Path("datasets/n_queens_10x10_orbits")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def enumerate_solutions(n: int) -> np.ndarray:
    """All N-Queens solutions as ``[S, n]`` arrays: column of the queen in each row."""
    solutions: list[list[int]] = []
    cols = [0] * n

    def place(row: int, used: int, diag: int, anti: int) -> None:
        if row == n:
            solutions.append(cols.copy())
            return
        for c in range(n):
            d, a = row - c + n - 1, row + c
            if used >> c & 1 or diag >> d & 1 or anti >> a & 1:
                continue
            cols[row] = c
            place(row + 1, used | 1 << c, diag | 1 << d, anti | 1 << a)

    place(0, 0, 0, 0)
    return np.asarray(solutions, dtype=np.int64).reshape(-1, n)


def unique_puzzles(solutions: np.ndarray, clue_counts) -> dict[Puzzle, int]:
    """Map every unique-completion puzzle to the index of its solution."""
    n = solutions.shape[1]
    puzzles: dict[Puzzle, int] = {}
    for k in clue_counts:
        for rows in itertools.combinations(range(n), k):
            rows_arr = np.asarray(rows)
            keys = solutions[:, rows_arr]  # [S, k] given columns per solution
            _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            for s in np.nonzero(counts[inverse.reshape(-1)] == 1)[0]:
                puzzles[tuple((int(r), int(solutions[s, r])) for r in rows)] = int(s)
    return puzzles


def transform_cells(cells: Puzzle, t: int, n: int) -> Puzzle:
    """Apply dihedral symmetry ``t`` (0..7): ``t % 4`` quarter turns, then a flip if ``t >= 4``."""
    out = []
    for r, c in cells:
        for _ in range(t % 4):
            r, c = c, n - 1 - r
        if t >= 4:
            c = n - 1 - c
        out.append((r, c))
    return tuple(sorted(out))


def orbit_key(puzzle: Puzzle, n: int) -> Puzzle:
    return min(transform_cells(puzzle, t, n) for t in range(8))


def split_by_orbit(
    puzzles, n: int, test_size: int, seed: int
) -> tuple[list[Puzzle], list[Puzzle]]:
    """Assign whole symmetry orbits to test (until ``test_size`` puzzles) or train."""
    orbits: dict[Puzzle, list[Puzzle]] = {}
    for p in sorted(puzzles):
        orbits.setdefault(orbit_key(p, n), []).append(p)
    keys = sorted(orbits)
    order = np.random.default_rng(seed).permutation(len(keys))
    train, test = [], []
    for i in order:
        members = orbits[keys[i]]
        (test if len(test) < test_size else train).extend(members)
    return train, test


def _boards(puzzles, solutions, solution_of, n):
    inputs = np.ones((len(puzzles), n, n), dtype=np.uint8)
    labels = np.ones((len(puzzles), n, n), dtype=np.uint8)
    for i, p in enumerate(puzzles):
        for r, c in p:
            inputs[i, r, c] = 2
        labels[i, np.arange(n), solutions[solution_of[p]]] = 2
    return inputs.reshape(len(puzzles), -1), labels.reshape(len(puzzles), -1)


def _write_split(output_dir: Path, split: str, inputs: np.ndarray, labels: np.ndarray, n: int):
    split_dir = output_dir / split
    count = len(inputs)
    save_npy(split_dir / "all__inputs.npy", inputs)
    save_npy(split_dir / "all__labels.npy", labels)
    save_npy(split_dir / "all__puzzle_indices.npy", np.arange(count + 1, dtype=np.int32))
    save_npy(split_dir / "all__group_indices.npy", np.arange(count + 1, dtype=np.int32))
    save_npy(split_dir / "all__puzzle_identifiers.npy", np.zeros(count, dtype=np.int32))
    PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=3,
        seq_len=n * n,
        num_puzzle_identifiers=1,
        total_groups=count,
        mean_puzzle_examples=1.0,
        total_puzzles=count,
        sets=["all"],
        height=n,
        width=n,
    ).write(split_dir / "dataset.json")
    print(f"{split}: {count} puzzles, inputs={inputs.shape}, labels={labels.shape}")


def build(cfg: NQueensOrbitsConfig) -> None:
    n = cfg.board_size
    solutions = enumerate_solutions(n)
    solution_of = unique_puzzles(solutions, cfg.clues)
    print(f"{len(solutions)} solutions, {len(solution_of)} unique-completion puzzles "
          f"with {cfg.clues} given queens")
    train, test = split_by_orbit(solution_of, n, cfg.test_size, cfg.seed)
    if cfg.train_size is not None:
        rng = np.random.default_rng(cfg.seed + 1)
        train = [train[i] for i in sorted(rng.permutation(len(train))[: cfg.train_size])]

    train_orbits = {orbit_key(p, n) for p in train}
    leaked = [p for p in test if orbit_key(p, n) in train_orbits]
    if leaked:
        raise AssertionError(f"{len(leaked)} test puzzles share a symmetry orbit with train")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for split, puzzles in (("train", train), ("test", test)):
        rng = np.random.default_rng(cfg.seed + (2 if split == "train" else 3))
        puzzles = [puzzles[i] for i in rng.permutation(len(puzzles))]
        inputs, labels = _boards(puzzles, solutions, solution_of, n)
        _write_split(cfg.output_dir, split, inputs, labels, n)
    (cfg.output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"No test puzzle shares a symmetry orbit with train. Saved to {cfg.output_dir}")


def parse_args() -> NQueensOrbitsConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument("--clues", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=NQueensOrbitsConfig.output_dir)
    args = parser.parse_args()
    return NQueensOrbitsConfig(
        board_size=args.board_size,
        clues=args.clues,
        test_size=args.test_size,
        train_size=args.train_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    build(parse_args())
