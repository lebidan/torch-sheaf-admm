"""Build N-Queens puzzle caches from ``brozonoyer/gram-n-queens``.

Only 10x10 boards with exactly one valid completion are retained. The official
train and test splits stay separate, and each split is saved in the format read
by :class:`~sheaf_admm.data.loaders.PuzzleDataset`.

Usage:
    python -m sheaf_admm.data.build_N_Queens --output-dir datasets/n_queens_10x10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import PuzzleDatasetMetadata, save_npy


@dataclass
class NQueensConfig:
    dataset: str = "brozonoyer/gram-n-queens"
    board_size: int = 10
    unique_solutions: int = 1
    train_size: int | None = None
    test_size: int | None = None
    seed: int = 0
    output_dir: Path = Path("datasets/n_queens_10x10")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _validate_example(example: dict[str, Any], board_size: int) -> tuple[np.ndarray, np.ndarray]:
    if example["n"] != board_size:
        raise ValueError(f"expected n={board_size}, got n={example['n']}")
    if example["num_solutions"] != 1:
        raise ValueError(f"expected one solution, got {example['num_solutions']}")

    boards = []
    for field in ("input", "target"):
        values = np.asarray(example[field])
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{field} must contain integers, got {values.dtype}")
        if values.size != board_size * board_size:
            raise ValueError(
                f"{field} has {values.size} cells; expected {board_size * board_size}"
            )
        board = values.reshape(board_size, board_size)
        if not np.isin(board, (0, 1, 2)).all():
            raise ValueError(f"{field} contains values outside the N-Queens vocabulary {{0, 1, 2}}")
        boards.append(board.astype(np.uint8, copy=False))

    input_board, target_board = boards
    target_queens = target_board == 2
    queen_rows, queen_cols = np.where(target_queens)
    if queen_rows.size != board_size:
        raise ValueError(f"target must contain {board_size} queens, got {queen_rows.size}")
    if (
        np.any(np.bincount(queen_rows, minlength=board_size) != 1)
        or np.any(np.bincount(queen_cols, minlength=board_size) != 1)
        or len(set(queen_rows - queen_cols)) != board_size
        or len(set(queen_rows + queen_cols)) != board_size
    ):
        raise ValueError("target is not a valid N-Queens solution")
    if np.any((input_board == 2) & ~target_queens):
        raise ValueError("input contains a queen that is absent from target")

    return input_board, target_board


def _write_split(
    output_dir: Path,
    split: str,
    examples: list[dict[str, Any]],
    board_size: int,
) -> set[str]:
    if not examples:
        raise ValueError(f"no qualifying N-Queens examples found for split {split!r}")

    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    instance_ids: set[str] = set()
    for example in examples:
        input_board, target_board = _validate_example(example, board_size)
        inputs.append(input_board.reshape(-1))
        labels.append(target_board.reshape(-1))
        instance_ids.add(example["instance_id"])
    if len(instance_ids) != len(examples):
        raise ValueError(f"duplicate instance IDs found in split {split!r}")

    inputs_arr = np.stack(inputs)
    labels_arr = np.stack(labels)
    split_dir = output_dir / split
    save_npy(split_dir / "all__inputs.npy", inputs_arr)
    save_npy(split_dir / "all__labels.npy", labels_arr)
    save_npy(split_dir / "all__puzzle_indices.npy", np.arange(len(examples) + 1, dtype=np.int32))
    save_npy(split_dir / "all__group_indices.npy", np.arange(len(examples) + 1, dtype=np.int32))
    save_npy(
        split_dir / "all__puzzle_identifiers.npy",
        np.zeros(len(examples), dtype=np.int32),
    )

    PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=3,
        seq_len=board_size * board_size,
        num_puzzle_identifiers=1,
        total_groups=len(examples),
        mean_puzzle_examples=1.0,
        total_puzzles=len(examples),
        sets=["all"],
        height=board_size,
        width=board_size,
    ).write(split_dir / "dataset.json")
    print(f"{split}: {len(examples)} puzzles, inputs={inputs_arr.shape}, labels={labels_arr.shape}")
    return instance_ids


def _select_examples(source: Any, cfg: NQueensConfig, split: str) -> list[dict[str, Any]]:
    required_columns = {"input", "target", "n", "num_solutions", "instance_id"}
    missing_columns = required_columns - set(source.column_names)
    if missing_columns:
        raise ValueError(f"dataset split {split!r} is missing columns: {sorted(missing_columns)}")

    filtered = source.filter(
        lambda example: example["n"] == cfg.board_size
        and example["num_solutions"] == cfg.unique_solutions,
        desc=f"Filtering {split} for {cfg.board_size}x{cfg.board_size} unique-solution boards",
    )
    requested_size = getattr(cfg, f"{split}_size")
    available = len(filtered)
    if requested_size is not None and requested_size <= 0:
        raise ValueError(f"{split}_size must be positive or None, got {requested_size}")
    if requested_size is not None and requested_size > available:
        raise ValueError(
            f"requested {requested_size} examples for {split}, but only {available} qualify"
        )

    count = available if requested_size is None else requested_size
    selected = filtered.shuffle(seed=cfg.seed + (0 if split == "train" else 1)).select(
        range(count)
    )
    return [selected[index] for index in range(count)]


def build(cfg: NQueensConfig) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("the 'datasets' library is required: pip install datasets") from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {}
    for split in ("train", "test"):
        source = load_dataset(cfg.dataset, split=split)
        examples = _select_examples(source, cfg, split)
        split_ids[split] = _write_split(cfg.output_dir, split, examples, cfg.board_size)

    overlap = split_ids["train"] & split_ids["test"]
    if overlap:
        raise ValueError(f"train/test contain {len(overlap)} duplicate instance IDs")

    (cfg.output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"Done. Dataset saved to {cfg.output_dir}")


def parse_args() -> NQueensConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=NQueensConfig.dataset)
    parser.add_argument("--board-size", type=int, default=NQueensConfig.board_size)
    parser.add_argument("--train-size", type=int, default=NQueensConfig.train_size)
    parser.add_argument("--test-size", type=int, default=NQueensConfig.test_size)
    parser.add_argument("--seed", type=int, default=NQueensConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=NQueensConfig.output_dir)
    args = parser.parse_args()
    return NQueensConfig(
        dataset=args.dataset,
        board_size=args.board_size,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    build(parse_args())
