"""Generate Minesweeper puzzles whose mine layout is uniquely determined.

A puzzle is a partially revealed board: revealed cells show their number (how
many of their 8 neighbours are mines), the other cells are hidden. The target is
the mine / safe status of every cell. Only boards where exactly one mine layout
is consistent with the visible numbers are kept, so every board can be solved
by pure deduction (no 50/50 guesses).

Generation of one board (see :func:`generate_board`):

1. place ``--mines`` mines uniformly at random;
2. start from the fully revealed board (every safe cell shows its number, only
   the mines are hidden): the layout is trivially unique;
3. visit the safe cells in random order and try to hide each one; a hide is
   kept only if (a) every hidden cell still touches a revealed number and
   (b) the solver still finds exactly one consistent layout;
4. stop once the board has a target number of hidden safe cells, drawn per board
   in ``[--hidden-safe-min, --hidden-safe-max]`` (difficulty knob; defaults: 30%
   of the safe cells .. all of them), or once no further cell can be hidden (a
   *minimal* puzzle, the hardest kind).

The total number of mines is not given to the model, hence rule (a): a hidden
cell touching no number could only be deduced from that total.

Train and test never share a board up to the 8 symmetries of the square
(rotations / reflections), so dihedral augmentation cannot leak test boards.

Tokens (same convention as the other puzzle datasets):
    inputs: 1 = hidden, 2 + n = revealed cell showing n (n = 0..8)  -> vocab 11
    labels: 1 = safe, 2 = mine

Usage:
    python -m sheaf_admm.data.build_minesweeper                       # 9x9, 10 mines
    python -m sheaf_admm.data.build_minesweeper --size 6 --mines 5    # 6x6, 5 mines

Test-only sets for size generalisation (same mine density as 6x6 / 5 mines):
    python -m sheaf_admm.data.build_minesweeper --size 9 --mines 11 --train-size 0 \
        --output-dir datasets/minesweeper_ood_9x9
    python -m sheaf_admm.data.build_minesweeper --size 12 --mines 20 --train-size 0 \
        --output-dir datasets/minesweeper_ood_12x12
(give an explicit --output-dir: the default would overwrite datasets/minesweeper_9x9)

Paper-style sets (task minesweeper_grid): 10k boards like the Maze sets, plus the
size-generalisation splits test_ood_1.5x / 2x / 4x (13x13, 17x17, 33x33 at the
9x9 density; 33x33 takes a few seconds per board, use several workers):
    python -m sheaf_admm.data.build_minesweeper --size 9 --mines 10 --train-size 10000 \
        --test-size 1000 --ood-sizes --workers 8 --output-dir datasets/minesweeper_grid_9x9
    python -m sheaf_admm.data.build_minesweeper --size 6 --mines 5 --train-size 10000 \
        --test-size 1000 --ood-sizes --workers 8 --output-dir datasets/minesweeper_grid_6x6
    (6x6 -> test_ood_1.5x / 2x / 4x = 8x8, 12x12, 24x24 at the 6x6 density)

Small boards have few distinct puzzles: 4x4 already repeats about half of its
boards after 30k draws (risk of memorisation), 6x6 with 5 mines does not.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from .common import PuzzleDatasetMetadata, save_npy

HIDDEN, SAFE, MINE = 1, 1, 2
NUMBER_OFFSET = 2  # revealed cell showing n -> token n + 2


@dataclass
class MinesweeperConfig:
    size: int = 9
    mines: int = 10
    train_size: int = 100_000
    test_size: int = 1_000
    hidden_safe_min: int | None = None  # default: 30% of the safe cells
    hidden_safe_max: int | None = None  # default: every safe cell (minimal puzzles)
    seed: int = 0
    workers: int = 1
    output_dir: Path | None = None  # default: datasets/minesweeper_<size>x<size>
    ood_sizes: bool = False  # also write the size-generalisation test splits (see ood_suite)
    ood_test_size: int | None = None  # boards per OOD split (default: test_size)

    def __post_init__(self) -> None:
        safe = self.size * self.size - self.mines
        if self.hidden_safe_min is None:
            self.hidden_safe_min = max(1, int(0.3 * safe))
        if self.hidden_safe_max is None:
            self.hidden_safe_max = safe
        if not 0 <= self.hidden_safe_min <= self.hidden_safe_max:
            raise ValueError("need 0 <= hidden_safe_min <= hidden_safe_max")
        if self.output_dir is None:
            self.output_dir = Path(f"datasets/minesweeper_{self.size}x{self.size}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


# ----------------------------------------------------------------------------
# Board geometry
# ----------------------------------------------------------------------------


def neighbours(size: int) -> list[list[int]]:
    """8-neighbourhood of every cell (flat indices)."""
    out = []
    for r in range(size):
        for c in range(size):
            out.append(
                [
                    rr * size + cc
                    for rr in range(max(0, r - 1), min(size, r + 2))
                    for cc in range(max(0, c - 1), min(size, c + 2))
                    if (rr, cc) != (r, c)
                ]
            )
    return out


def numbers_of(mines: np.ndarray, nbrs: list[list[int]]) -> np.ndarray:
    """Number shown by every cell (count of neighbouring mines)."""
    return np.asarray([int(mines[n].sum()) for n in nbrs], dtype=np.int64)


# ----------------------------------------------------------------------------
# Solver: count consistent mine layouts, stopping at ``limit``
# ----------------------------------------------------------------------------


def count_layouts(hidden: np.ndarray, numbers: np.ndarray, nbrs, limit: int = 2) -> int:
    """Number of mine layouts on the hidden cells consistent with the revealed numbers.

    Constraint per revealed cell ``r``: the hidden neighbours of ``r`` contain
    exactly ``numbers[r]`` mines (revealed cells are safe by definition).
    Depth-first search with unit propagation; returns as soon as ``limit``
    layouts are found (``limit=2`` answers "is it unique?").
    """
    n_cells = len(hidden)
    var_cells = [i for i in range(n_cells) if hidden[i]]
    constraints: list[tuple[list[int], int]] = []
    for r in range(n_cells):
        if hidden[r]:
            continue
        scope = [n for n in nbrs[r] if hidden[n]]
        if scope:
            constraints.append((scope, int(numbers[r])))
        elif numbers[r] != 0:
            return 0
    cons_of: dict[int, list[int]] = {v: [] for v in var_cells}
    for k, (scope, _target) in enumerate(constraints):
        for v in scope:
            cons_of[v].append(k)
    # unconstrained hidden cells are free: >= 2 layouts unless there are none
    if any(not cons_of[v] for v in var_cells):
        return limit

    value = dict.fromkeys(var_cells, -1)
    order = sorted(var_cells, key=lambda v: -len(cons_of[v]))

    def propagate(trail: list[int], queue: list[int]) -> bool:
        while queue:
            k = queue.pop()
            scope, target = constraints[k]
            ones = unknown = 0
            for v in scope:
                x = value[v]
                if x == 1:
                    ones += 1
                elif x == -1:
                    unknown += 1
            if ones > target or ones + unknown < target:
                return False
            if unknown and (ones == target or ones + unknown == target):
                fill = 0 if ones == target else 1
                for v in scope:
                    if value[v] == -1:
                        value[v] = fill
                        trail.append(v)
                        queue.extend(cons_of[v])
        return True

    found = 0

    def search() -> None:
        nonlocal found
        var = next((v for v in order if value[v] == -1), None)
        if var is None:
            found += 1
            return
        for guess in (0, 1):
            trail = [var]
            value[var] = guess
            if propagate(trail, list(cons_of[var])):
                search()
            for v in trail:
                value[v] = -1
            if found >= limit:
                return

    trail: list[int] = []
    if propagate(trail, list(range(len(constraints)))):
        search()
    return found


# ----------------------------------------------------------------------------
# Board generation
# ----------------------------------------------------------------------------


def _every_hidden_touches_a_number(hidden: np.ndarray, nbrs) -> bool:
    return all(not hidden[i] or any(not hidden[n] for n in nbrs[i]) for i in range(len(hidden)))


def generate_board(rng: np.random.Generator, size: int, mines: int, hidden_safe_target: int, nbrs):
    """One uniquely solvable board: ``(inputs, labels)`` token vectors, or ``None``."""
    cells = size * size
    mine = np.zeros(cells, dtype=bool)
    mine[rng.choice(cells, size=mines, replace=False)] = True
    numbers = numbers_of(mine, nbrs)
    hidden = mine.copy()
    if not _every_hidden_touches_a_number(hidden, nbrs):
        return None  # a mine walled in by mines: no clue can ever reach it

    hidden_safe = 0
    for s in rng.permutation(np.flatnonzero(~mine)):
        if hidden_safe >= hidden_safe_target:
            break
        hidden[s] = True
        if _every_hidden_touches_a_number(hidden, nbrs) and count_layouts(
            hidden, numbers, nbrs, limit=2
        ) == 1:
            hidden_safe += 1
        else:
            hidden[s] = False

    inputs = np.where(hidden, HIDDEN, numbers + NUMBER_OFFSET).astype(np.uint8)
    labels = np.where(mine, MINE, SAFE).astype(np.uint8)
    return inputs, labels


def _worker(args):
    seed, count, size, mines, lo, hi = args
    rng = np.random.default_rng(seed)
    nbrs = neighbours(size)
    out = []
    while len(out) < count:
        board = generate_board(rng, size, mines, int(rng.integers(lo, hi + 1)), nbrs)
        if board is not None:
            out.append(board)
    return out


# ----------------------------------------------------------------------------
# Symmetry-aware deduplication and split
# ----------------------------------------------------------------------------


def canonical_key(inputs: np.ndarray, size: int) -> bytes:
    """Smallest byte string among the 8 rotations / reflections of the input board."""
    board = inputs.reshape(size, size)
    variants = []
    for k in range(4):
        rot = np.rot90(board, k)
        variants.append(rot.tobytes())
        variants.append(np.fliplr(rot).tobytes())
    return min(variants)


def _write_split(output_dir: Path, split: str, inputs, labels, size: int) -> None:
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
        vocab_size=NUMBER_OFFSET + 9,
        seq_len=size * size,
        num_puzzle_identifiers=1,
        total_groups=count,
        mean_puzzle_examples=1.0,
        total_puzzles=count,
        sets=["all"],
        height=size,
        width=size,
    ).write(split_dir / "dataset.json")
    hidden_safe = ((inputs == HIDDEN) & (labels == SAFE)).sum(1)
    print(
        f"{split}: {count} boards, hidden safe cells per board "
        f"mean={hidden_safe.mean():.1f} min={hidden_safe.min()} max={hidden_safe.max()}"
    )


def generate(cfg: MinesweeperConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate enough distinct boards (up to symmetry) for both splits."""
    if not 0 < cfg.mines < cfg.size * cfg.size:
        raise ValueError("mines must be between 1 and size*size - 1")
    wanted = cfg.train_size + cfg.test_size
    seen: set[bytes] = set()
    inputs, labels = [], []
    seed_seq = np.random.SeedSequence(cfg.seed)
    start = time.time()
    with Pool(cfg.workers) if cfg.workers > 1 else _NoPool() as pool:
        while len(inputs) < wanted:
            missing = wanted - len(inputs)
            chunk = max(1, -(-missing // cfg.workers))
            jobs = [
                (int(s.generate_state(1)[0]), chunk, cfg.size, cfg.mines,
                 cfg.hidden_safe_min, cfg.hidden_safe_max)
                for s in seed_seq.spawn(cfg.workers)
            ]
            for batch in pool.map(_worker, jobs):
                for x, y in batch:
                    key = canonical_key(x, cfg.size)
                    if key not in seen and len(inputs) < wanted:
                        seen.add(key)
                        inputs.append(x)
                        labels.append(y)
            print(f"  {len(inputs)}/{wanted} boards ({time.time() - start:.0f}s)")
    return np.stack(inputs), np.stack(labels)


class _NoPool:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def map(fn, jobs):
        return [fn(j) for j in jobs]


def ood_suite(size: int, mines: int) -> list[tuple[str, int, int]]:
    """``(split, size, mines)`` of the size-generalisation tests, as for the paper's Maze.

    The Maze sets go from 19x19 to 29 (1.5x), 37 (2x) and 73 (4x). The sizes keep
    the parity of the training size, because with 3x3 agents every 2 cells all
    boards of one parity are cut the same way (odd: exact tiling; even: a double
    ring of padding on the right / bottom side). 9x9 -> 13, 17, 33; 6x6 -> 8, 12, 24.
    The mine density of the training board is kept.
    """
    density = mines / size**2
    odd = size % 2
    one_and_half = (3 * size - 1) // 2
    if one_and_half % 2 != odd:
        one_and_half += 1
    out = []
    for name, new in (("test_ood_1.5x", one_and_half), ("test_ood_2x", 2 * size - odd),
                      ("test_ood_4x", 4 * size - 3 * odd)):
        out.append((name, new, max(1, round(density * new * new))))
    return out


def build(cfg: MinesweeperConfig) -> None:
    inputs, labels = generate(cfg)
    # boards are distinct up to symmetry, so any split is symmetry-disjoint
    perm = np.random.default_rng(cfg.seed + 1).permutation(len(inputs))
    test_idx, train_idx = perm[: cfg.test_size], perm[cfg.test_size :]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train_size > 0:  # --train-size 0: test set only (e.g. larger boards for OOD tests)
        _write_split(cfg.output_dir, "train", inputs[train_idx], labels[train_idx], cfg.size)
    _write_split(cfg.output_dir, "test", inputs[test_idx], labels[test_idx], cfg.size)
    info = cfg.to_dict()
    if cfg.ood_sizes:
        info["ood"] = {}
        count = cfg.ood_test_size or cfg.test_size
        for split, size, mines in ood_suite(cfg.size, cfg.mines):
            print(f"{split}: {size}x{size}, {mines} mines, {count} boards")
            sub = MinesweeperConfig(size=size, mines=mines, train_size=0, test_size=count,
                                    seed=cfg.seed + 1000 * size, workers=cfg.workers,
                                    output_dir=cfg.output_dir)
            x, y = generate(sub)
            _write_split(cfg.output_dir, split, x, y, size)
            info["ood"][split] = {"size": size, "mines": mines, "boards": count}
    (cfg.output_dir / "config.json").write_text(json.dumps(info, indent=2))
    print(f"Done. Dataset saved to {cfg.output_dir}")


def parse_args() -> MinesweeperConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = MinesweeperConfig()  # defaults for size / mines / counts
    p.add_argument("--size", type=int, default=d.size)
    p.add_argument("--mines", type=int, default=d.mines)
    p.add_argument("--train-size", type=int, default=d.train_size)
    p.add_argument("--test-size", type=int, default=d.test_size)
    p.add_argument("--hidden-safe-min", type=int, default=None)
    p.add_argument("--hidden-safe-max", type=int, default=None)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--ood-sizes", action="store_true",
                   help="also write test_ood_1.5x / 2x / 4x splits (Maze-style size generalisation)")
    p.add_argument("--ood-test-size", type=int, default=None, help="boards per OOD split")
    a = p.parse_args()
    return MinesweeperConfig(
        size=a.size, mines=a.mines, train_size=a.train_size, test_size=a.test_size,
        hidden_safe_min=a.hidden_safe_min, hidden_safe_max=a.hidden_safe_max,
        seed=a.seed, workers=a.workers, output_dir=a.output_dir,
        ood_sizes=a.ood_sizes, ood_test_size=a.ood_test_size,
    )


if __name__ == "__main__":
    build(parse_args())
