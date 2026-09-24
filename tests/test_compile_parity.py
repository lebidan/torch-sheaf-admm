"""Check that compiled CUDA training preserves the eager parameter update."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sheaf_admm.models import model_config_from_dict
from sheaf_admm.training import (
    build_model,
    create_train_state,
    make_task,
    make_train_step,
)
from sheaf_admm.training.loop import _forward, move_to_device


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("name", ["maze_sheaf", "maze_mpnn"])
def test_compiled_training_matches_eager(name, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    exp = yaml.safe_load((root / "configs" / "experiment" / f"{name}.yaml").read_text())
    cfg = model_config_from_dict(exp["model"])
    task = make_task("maze", **exp["task_cfg"])
    rng = np.random.default_rng(17)
    batch = {
        "inputs": rng.integers(0, 5, (2, 7 * 7), dtype=np.int64),
        "labels": rng.integers(0, 6, (2, 7 * 7), dtype=np.int64),
        "height": 7,
        "width": 7,
    }
    fwd, targets, _ = task.prepare(batch)

    def make_state():
        model = build_model(cfg, exp["model_type"], seed=17)
        return create_train_state(
            model,
            fwd,
            model_type=exp["model_type"],
            lr=3e-4,
            weight_decay=1e-6,
            warmup_steps=0,
            grad_clip=1.0,
            ema_decay=0.0,
            k_init=2,
            loss_window=2,
            seed=17,
            device="cuda",
        )

    eager, compiled = make_state(), make_state()
    step = make_train_step(task, exp["model_type"], cfg.mpnn_graph_readout)
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    eager, eager_loss = step(eager, fwd, targets, n_iter=2, loss_window=2)
    monkeypatch.delenv("TORCH_COMPILE_DISABLE")
    compiled, compiled_loss = step(compiled, fwd, targets, n_iter=2, loss_window=2)
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-4, atol=1e-5)
    for key, value in eager.model.state_dict().items():
        torch.testing.assert_close(
            compiled.model.state_dict()[key], value, rtol=1e-4, atol=1e-5, msg=key
        )

    inputs = move_to_device(fwd, "cuda")
    compiled.model.eval()
    with torch.no_grad():
        monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
        eager_output = _forward(
            compiled.model,
            inputs,
            n_iter=5,
            loss_window=1,
            model_type=exp["model_type"],
            training=False,
        )
        monkeypatch.delenv("TORCH_COMPILE_DISABLE")
        for _ in range(2):
            compiled_output = _forward(
                compiled.model,
                inputs,
                n_iter=5,
                loss_window=1,
                model_type=exp["model_type"],
                training=False,
            )
            torch.testing.assert_close(compiled_output, eager_output, rtol=1e-4, atol=1e-5)
