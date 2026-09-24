"""Benchmark a matched synthetic training or evaluation step from either checkout.

Run from the repository root. For ``--framework jax``, run this script from a
checkout of the original JAX revision with its CUDA environment active.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml


def batch_for(task: str, batch_size: int, side: int) -> dict:
    rng = np.random.default_rng(54321)
    if task == "maze":
        return {
            "inputs": rng.integers(0, 5, (batch_size, side * side), dtype=np.int64),
            "labels": rng.integers(0, 6, (batch_size, side * side), dtype=np.int64),
            "height": side,
            "width": side,
        }
    if task == "mnist":
        return {
            "images": rng.random((batch_size, 28, 28, 1), dtype=np.float32),
            "labels": rng.integers(0, 10, batch_size, dtype=np.int64),
        }
    return {
        "inputs": np.zeros((batch_size, 81), dtype=np.int64),
        "labels": rng.integers(1, 10, (batch_size, 81), dtype=np.int64),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["jax", "torch"], required=True)
    p.add_argument("--phase", choices=["train", "eval"], default="train")
    p.add_argument("--experiment", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--maze-side", type=int, default=19)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--include-prep", action="store_true")
    a = p.parse_args()
    root = Path.cwd()
    exp = yaml.safe_load((root / "configs" / "experiment" / f"{a.experiment}.yaml").read_text())
    model_type = exp["model_type"]
    if a.iters is not None:
        iters = a.iters
    elif model_type == "sheaf":
        iters = exp["training"]["K_train" if a.phase == "train" else "K_eval"]
    else:
        iters = exp["training"][
            "mpnn_train_rounds" if a.phase == "train" else "mpnn_eval_rounds"
        ]
    loss_window = min(iters, exp["training"].get("loss_window", 1))
    from sheaf_admm.models import model_config_from_dict
    from sheaf_admm.training import (
        build_model,
        create_train_state,
        make_task,
        make_train_step,
    )

    task = make_task(exp["task"], **exp.get("task_cfg", {}))
    raw = batch_for(exp["task"], a.batch, a.maze_side)
    fwd, targets, _ = task.prepare(raw)
    cfg = model_config_from_dict(exp["model"])

    if a.framework == "jax":
        import jax

        if jax.default_backend() != "gpu":
            raise RuntimeError("JAX is not using the GPU")
        model = build_model(cfg, model_type)
        state = create_train_state(
            model,
            fwd,
            model_type=model_type,
            lr=exp["training"].get("lr", 3e-4),
            weight_decay=exp["training"].get("weight_decay", 1e-6),
            warmup_steps=0,
            grad_clip=1.0,
            ema_decay=0.0,
            k_init=iters,
            loss_window=loss_window,
            seed=17,
        )
        if a.phase == "train":
            step = make_train_step(task, model_type, cfg.mpnn_graph_readout)
            rng = jax.random.PRNGKey(37)

            def run():
                nonlocal state, rng
                rng, sub = jax.random.split(rng)
                step_fwd, step_targets = (
                    task.prepare(raw)[:2] if a.include_prep else (fwd, targets)
                )
                state, loss = step(
                    state, step_fwd, step_targets, sub, n_iter=iters, loss_window=loss_window
                )
                jax.block_until_ready((state, loss))
                return float(loss)

        else:
            import jax.numpy as jnp

            from sheaf_admm.training.loop import _forward

            @jax.jit
            def eval_forward(params, inputs):
                return _forward(
                    model.apply,
                    params,
                    inputs,
                    n_iter=iters,
                    loss_window=1,
                    model_type=model_type,
                    training=False,
                    rng=None,
                )

            def run():
                inputs = task.prepare(raw)[0] if a.include_prep else fwd
                output = eval_forward(state.params, inputs)
                jax.block_until_ready(output)
                return float(jnp.mean(output))

        framework_version = jax.__version__

        def synchronize():
            jax.block_until_ready(state.params)
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA unavailable")
        model = build_model(cfg, model_type, seed=17)
        state = create_train_state(
            model,
            fwd,
            model_type=model_type,
            lr=exp["training"].get("lr", 3e-4),
            weight_decay=exp["training"].get("weight_decay", 1e-6),
            warmup_steps=0,
            grad_clip=1.0,
            ema_decay=0.0,
            k_init=iters,
            loss_window=loss_window,
            seed=17,
            device="cuda",
        )
        if a.phase == "train":
            step = make_train_step(task, model_type, cfg.mpnn_graph_readout)
            from sheaf_admm.training.loop import move_to_device

            if not a.include_prep:
                fwd = move_to_device(fwd, "cuda")
                targets = move_to_device(targets, "cuda")

            def run():
                nonlocal state
                step_fwd, step_targets = (
                    task.prepare(raw)[:2] if a.include_prep else (fwd, targets)
                )
                state, loss = step(
                    state, step_fwd, step_targets, n_iter=iters, loss_window=loss_window
                )
                torch.cuda.synchronize()
                return float(loss)

        else:
            from sheaf_admm.training.loop import _forward, move_to_device

            state.model.eval()
            if not a.include_prep:
                fwd = move_to_device(fwd, "cuda")

            def run():
                inputs = task.prepare(raw)[0] if a.include_prep else fwd
                inputs = move_to_device(inputs, "cuda")
                with torch.no_grad():
                    output = _forward(
                        state.model,
                        inputs,
                        n_iter=iters,
                        loss_window=1,
                        model_type=model_type,
                        training=False,
                    )
                torch.cuda.synchronize()
                return float(output.mean())

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        framework_version = torch.__version__
        synchronize = torch.cuda.synchronize

    synchronize()
    t0 = time.perf_counter()
    first_loss = run()
    first = time.perf_counter() - t0
    for _ in range(a.warmup):
        run()
    times = []
    losses = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        losses.append(run())
        times.append(time.perf_counter() - t0)
    result = {
        "framework": a.framework,
        "phase": a.phase,
        "include_prep": a.include_prep,
        "version": framework_version,
        "experiment": a.experiment,
        "batch": a.batch,
        "maze_side": a.maze_side if exp["task"] == "maze" else None,
        "iters": iters,
        "first_sec": first,
        "median_steady_sec": statistics.median(times),
        "steady_sec": times,
        "first_value": first_loss,
        "last_value": losses[-1],
    }
    if a.framework == "torch":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
