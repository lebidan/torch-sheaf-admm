"""Learning-rate schedules."""

from __future__ import annotations

import math

import pytest

from sheaf_admm.training.optim import make_schedule


def test_constant_schedule_matches_original_warmup():
    sched = make_schedule(1e-3, 100)
    assert sched(0) == 0.0
    assert sched(50) == pytest.approx(5e-4)
    assert sched(100) == pytest.approx(1e-3)
    assert sched(10_000) == pytest.approx(1e-3)


def test_cosine_schedule_decays_to_floor():
    sched = make_schedule(1e-3, 100, kind="cosine", total_steps=1100, min_lr_ratio=0.05)
    assert sched(50) == pytest.approx(5e-4)
    assert sched(100) == pytest.approx(1e-3)
    mid = 5e-5 + (1e-3 - 5e-5) * 0.5 * (1 + math.cos(math.pi * 0.5))
    assert sched(600) == pytest.approx(mid)
    assert sched(1100) == pytest.approx(5e-5)
    assert sched(5000) == pytest.approx(5e-5)


def test_cosine_schedule_requires_total_steps():
    with pytest.raises(ValueError):
        make_schedule(1e-3, 100, kind="cosine")
