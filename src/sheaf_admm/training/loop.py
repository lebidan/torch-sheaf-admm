"""PyTorch training and evaluation driver for both model families."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from sheaf_admm.models import ModelConfig, MPNNModel, SheafADMMModel

from .optim import TrainState, eval_params, init_ema, make_optimizer, make_schedule


def build_model(config: ModelConfig, model_type: str, *, seed: int | None = None):
    """Construct a model, optionally seeding eager parameter initialization."""
    if seed is not None:
        torch.manual_seed(int(seed))
    if model_type == "sheaf":
        return SheafADMMModel(config=config)
    if model_type == "mpnn":
        return MPNNModel(config=config)
    raise ValueError(f"unknown model_type={model_type!r} (sheaf|mpnn)")


def move_to_device(tree, device):
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if isinstance(tree, dict):
        return {k: move_to_device(v, device) for k, v in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(move_to_device(v, device) for v in tree)
    return tree


def _forward(model, fwd, *, n_iter, loss_window, model_type, training):
    if model_type == "sheaf":
        logits_window, _state, _geom = model(
            fwd["patches"],
            fwd["edge_indices"],
            num_iters=n_iter,
            loss_window=loss_window,
            **fwd["model_kwargs"],
            training=training,
        )
        return logits_window
    logits, _hidden = model(
        fwd["patches"],
        fwd["edge_indices"],
        num_rounds=n_iter,
        **fwd["model_kwargs"],
        training=training,
    )
    return logits


def create_train_state(
    model,
    sample_fwd,
    *,
    model_type,
    lr,
    weight_decay,
    warmup_steps,
    grad_clip,
    ema_decay,
    k_init,
    loss_window,
    seed,
    device=None,
):
    torch.manual_seed(int(seed))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    sample_fwd = move_to_device(sample_fwd, device)
    with torch.no_grad():
        _forward(model, sample_fwd, n_iter=1, loss_window=1, model_type=model_type, training=False)
    optimizer = make_optimizer(model.parameters(), lr, weight_decay, warmup_steps, grad_clip)
    state = TrainState(
        model, optimizer, make_schedule(lr, warmup_steps), float(grad_clip), float(ema_decay)
    )
    return init_ema(state)


def make_train_step(task, model_type, graph_readout):
    def train_step(state: TrainState, fwd, targets, dropout_rng=None, *, n_iter, loss_window):
        device = next(state.model.parameters()).device
        fwd, targets = move_to_device(fwd, device), move_to_device(targets, device)
        state.model.train()
        logits = _forward(
            state.model,
            fwd,
            n_iter=n_iter,
            loss_window=loss_window,
            model_type=model_type,
            training=True,
        )
        if model_type == "mpnn" and graph_readout == "graph":
            loss = task.loss_graph(logits, targets)
        elif model_type == "mpnn":
            loss = task.loss(logits.unsqueeze(0), targets)
        else:
            loss = task.loss(logits, targets)
        return state, state.update(loss)

    return train_step


@contextmanager
def _ema_model(state: TrainState):
    shadow = eval_params(state)
    if state.ema_params is None:
        yield state.model
        return
    live = {k: v.detach().clone() for k, v in state.model.state_dict().items()}
    state.model.load_state_dict(shadow)
    try:
        yield state.model
    finally:
        state.model.load_state_dict(live)


def evaluate(state, task, batches, *, model_type, graph_readout, k_eval, max_batches=None):
    agg: dict[str, float] = {}
    n = 0
    device = next(state.model.parameters()).device
    with _ema_model(state), torch.no_grad():
        state.model.eval()
        for batch in batches:
            fwd, targets, aux = task.prepare(batch)
            fwd, targets = move_to_device(fwd, device), move_to_device(targets, device)
            logits = _forward(
                state.model,
                fwd,
                n_iter=k_eval,
                loss_window=1,
                model_type=model_type,
                training=False,
            )
            final = logits if model_type == "mpnn" else logits[-1]
            metrics = (
                task.evaluate_graph(final, targets, aux)
                if model_type == "mpnn" and graph_readout == "graph"
                else task.evaluate(final, targets, aux)
            )
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
            if max_batches is not None and n >= max_batches:
                break
    return {k: v / max(n, 1) for k, v in agg.items()}


def sample_k(rng_np: np.random.Generator, dist: str, k_min: int, k_max: int) -> int:
    if dist != "uniform":
        return k_max
    return int(rng_np.integers(min(k_min, k_max), k_max + 1))
