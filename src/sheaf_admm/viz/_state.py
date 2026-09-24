"""Temporary selection of raw or EMA Torch weights for artifact helpers."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def use_model_state(model, state_dict):
    if state_dict is None:
        yield
        return
    current = {k: value.detach().clone() for k, value in model.state_dict().items()}
    try:
        model.load_state_dict(state_dict, strict=True)
        yield
    finally:
        model.load_state_dict(current, strict=True)
