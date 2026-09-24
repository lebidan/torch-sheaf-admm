"""Artifact helpers honor the supplied checkpoint state."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sheaf_admm.models import ModelConfig, SheafADMMModel
from sheaf_admm.training.tasks import MazeTask
from sheaf_admm.viz.prediction import _final_logits
from sheaf_admm.viz.trajectory import run_trajectory


def test_artifact_helpers_use_requested_weights_and_restore_model():
    task = MazeTask()
    zeros = np.zeros((1, 81), dtype=np.int64)
    fwd, _, aux = task.prepare({"inputs": zeros, "labels": zeros})
    cfg = ModelConfig(d_v=4, d_e=2, enc_hidden_dim=8, dec_hidden_dim=8, num_directions=8)
    model = SheafADMMModel(cfg)
    with torch.no_grad():
        _final_logits(model, None, fwd, 1)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    alternate = {k: v.clone() for k, v in original.items()}
    alternate["decoder.output_dense.bias"] += 1
    alternate["rho_raw"] += 1
    with torch.no_grad():
        reference = _final_logits(model, original, fwd, 1)
        changed = _final_logits(model, alternate, fwd, 1)
        baseline_rho = run_trajectory(model, original, fwd, num_iters=1, centers=aux["centers"]).rho
        changed_rho = run_trajectory(model, alternate, fwd, num_iters=1, centers=aux["centers"]).rho
    assert not torch.allclose(reference, changed)
    assert changed_rho > baseline_rho
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])

    malformed = {key: value.clone() for key, value in original.items()}
    malformed["rho_raw"] += 2
    malformed["decoder.output_dense.weight"] = torch.zeros(1, 1)
    with pytest.raises(RuntimeError), torch.no_grad():
        _final_logits(model, malformed, fwd, 1)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
