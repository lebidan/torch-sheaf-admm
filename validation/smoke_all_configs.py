"""Run one train and evaluation step for every shipped config.

Uses tiny synthetic batches to exercise wiring, backward, optimizer, EMA and
metrics. Run ``.venv/bin/python validation/smoke_all_configs.py --device cuda``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from sheaf_admm.models import model_config_from_dict
from sheaf_admm.training import (
    build_model,
    create_train_state,
    evaluate,
    make_task,
    make_train_step,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    "maze_sheaf",
    "mnist_sheaf",
    "sudoku_sheaf",
    "sudoku_sheaf_lora",
    "maze_mpnn",
    "mnist_mpnn",
    "sudoku_mpnn",
)


def synthetic_batch(task_name: str) -> dict:
    rng = np.random.default_rng(17)
    if task_name == "maze":
        return {
            "inputs": rng.integers(0, 5, (2, 7 * 7), dtype=np.int64),
            "labels": rng.integers(0, 6, (2, 7 * 7), dtype=np.int64),
            "height": 7,
            "width": 7,
        }
    if task_name == "mnist":
        return {
            "images": rng.random((2, 9, 9, 1), dtype=np.float32),
            "labels": np.array([1, 2], dtype=np.int64),
        }
    return {
        "inputs": np.zeros((2, 81), dtype=np.int64),
        "labels": rng.integers(1, 10, (2, 81), dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    for name in EXPERIMENTS:
        exp = yaml.safe_load((ROOT / "configs" / "experiment" / f"{name}.yaml").read_text())
        task = make_task(exp["task"], **exp.get("task_cfg", {}))
        model_cfg = model_config_from_dict(exp["model"])
        model_type = exp["model_type"]
        batch = synthetic_batch(exp["task"])
        fwd, _, _ = task.prepare(batch)
        model = build_model(model_cfg, model_type, seed=17)
        state = create_train_state(
            model,
            fwd,
            model_type=model_type,
            lr=3e-4,
            weight_decay=1e-6,
            warmup_steps=0,
            grad_clip=1.0,
            ema_decay=0.9,
            k_init=2,
            loss_window=2,
            seed=17,
            device=device,
        )
        step = make_train_step(task, model_type, model_cfg.mpnn_graph_readout)
        _, targets, _ = task.prepare(batch)
        state, loss = step(state, fwd, targets, n_iter=2, loss_window=2)
        if not torch.isfinite(loss):
            raise AssertionError(f"{name}: non-finite loss")
        metrics = evaluate(
            state,
            task,
            [batch],
            model_type=model_type,
            graph_readout=model_cfg.mpnn_graph_readout,
            k_eval=2,
        )
        if not all(np.isfinite(v) for v in metrics.values()):
            raise AssertionError(f"{name}: non-finite metrics {metrics}")
        print(f"{name:20} {device} loss={loss.item():.5f} metrics={metrics}")
    print(f"Passed {len(EXPERIMENTS)} config train/eval smoke checks on {device}")


if __name__ == "__main__":
    main()
