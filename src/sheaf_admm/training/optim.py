"""PyTorch AdamW, stepwise warmup, and EMA training state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def make_schedule(learning_rate: float, warmup_steps: int):
    """Optax-compatible schedule: update 0 has LR zero during warmup."""
    return lambda step: (
        float(learning_rate) * min(1.0, step / warmup_steps)
        if warmup_steps > 0
        else float(learning_rate)
    )


def make_optimizer(
    params,
    learning_rate: float,
    weight_decay: float = 0.0,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
):
    optimizer = torch.optim.AdamW(
        params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
    )
    return optimizer


@dataclass
class TrainState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_schedule: object
    grad_clip: float
    ema_decay: float
    ema_params: dict[str, torch.Tensor] | None = None
    step: int = 0

    @property
    def params(self):
        return self.model.state_dict()

    def update(self, loss: torch.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        lr = self.lr_schedule(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.step += 1
        ema_update(self)
        return loss.detach()


def init_ema(state: TrainState) -> TrainState:
    if state.ema_decay > 0:
        state.ema_params = {k: v.detach().clone() for k, v in state.model.state_dict().items()}
    return state


def ema_update(state: TrainState) -> TrainState:
    if state.ema_decay <= 0 or state.ema_params is None:
        return state
    d = state.ema_decay
    with torch.no_grad():
        for k, v in state.model.state_dict().items():
            if v.is_floating_point():
                state.ema_params[k].lerp_(v, 1.0 - d)
            else:
                state.ema_params[k].copy_(v)
    return state


def eval_params(state: TrainState):
    return (
        state.ema_params
        if state.ema_decay > 0 and state.ema_params is not None
        else state.model.state_dict()
    )
