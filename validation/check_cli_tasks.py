"""Exercise MNIST and Sudoku data-loader, trainer and visualization CLIs.

Builds tiny synthetic on-disk datasets in a temporary directory. The labels are
arbitrary; this checks operational wiring, not task accuracy. Run with
``.venv/bin/python validation/check_cli_tasks.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from sheaf_admm.data.build_sudoku import SudokuConfig, _build_split_from_puzzles

ROOT = Path(__file__).resolve().parents[1]


def _mnist_data(path: Path) -> None:
    rng = np.random.default_rng(12)
    for split in ("train", "test"):
        directory = path / split
        directory.mkdir(parents=True)
        np.save(directory / "images.npy", rng.random((2, 28, 28, 1), dtype=np.float32))
        np.save(directory / "labels.npy", np.array([1, 2], dtype=np.int64))
    (path / "metadata.json").write_text(
        json.dumps({"image_shape": [28, 28, 1], "splits": {"train": 2, "test": 2}})
    )


def _sudoku_data(path: Path) -> None:
    cfg = SudokuConfig(output_dir=path)
    empty = np.zeros((9, 9), dtype=np.uint8)
    solution = (np.arange(81).reshape(9, 9) % 9 + 1).astype(np.uint8)
    for split in ("train", "test_hard"):
        _build_split_from_puzzles(split, [(empty, solution)], cfg, apply_augmentation=False)


def _run(name: str, data_dir: Path, temp: Path, batch_size: int) -> None:
    run_dir = temp / f"{name}_run"
    command = [
        sys.executable,
        "scripts/train.py",
        f"+experiment={name}",
        f"data.dir={data_dir}",
        f"training.batch_size={batch_size}",
        "training.epochs=1",
        "training.K_train=2",
        "training.K_eval=2",
        "training.loss_window=1",
        "training.warmup_steps=0",
        f"hydra.run.dir={run_dir}",
        "wandb.mode=disabled",
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.is_file():
        raise AssertionError(f"{name}: checkpoint missing")
    viz_dir = temp / f"{name}_viz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.visualize",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(viz_dir),
            "--num-iters",
            "2",
            "--ks",
            "1",
            "2",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    expected = ("prediction_evolution.pdf", "coordination_dynamics.pdf", "xz_trajectories.pdf")
    if not all((viz_dir / filename).is_file() for filename in expected):
        raise AssertionError(f"{name}: visualization output missing")
    print(f"{name}: training checkpoint and three visualization PDFs passed")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sheaf_cli_smoke_") as temp_dir:
        temp = Path(temp_dir)
        mnist = temp / "mnist"
        sudoku = temp / "sudoku"
        _mnist_data(mnist)
        _sudoku_data(sudoku)
        _run("mnist_sheaf", mnist, temp, batch_size=2)
        _run("sudoku_sheaf", sudoku, temp, batch_size=1)


if __name__ == "__main__":
    main()
