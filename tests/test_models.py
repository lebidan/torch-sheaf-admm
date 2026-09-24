"""PyTorch model wiring, parameter budgets and forward shape contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from sheaf_admm.data import views as V
from sheaf_admm.models import MPNNModel, SheafADMMModel, model_config_from_dict
from sheaf_admm.models.mpnn import DirectionalGGNN
from sheaf_admm.training.loop import build_model

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "experiment"


def _load(name):
    data = yaml.safe_load((_CONFIGS / f"{name}.yaml").read_text())
    return model_config_from_dict(data["model"]), data


def _inputs(cfg, task):
    if task == "sudoku":
        edges, map_u, map_v = V.build_sudoku_multigraph(9)
        patches = torch.zeros(27, 2, 9, cfg.num_classes)
        kwargs = dict(map_u=map_u, map_v=map_v, cell_ids=V.build_sudoku_cell_indices(9))
    else:
        centers = V.grid_agent_centers((7, 7), stride=2, patch_size=3)
        edges = V.build_grid_edge_indices(centers, 2, 8)
        channels = 1 if task == "mnist" else cfg.num_classes
        patches = torch.zeros(len(centers), 2, 3, 3, channels)
        kwargs = dict(node_positions=V.node_positions(centers))
    return patches, edges, kwargs


def _count(model):
    return sum(p.numel() for p in model.parameters())


def test_model_config_ignores_legacy_noop_keys():
    cfg = model_config_from_dict({"num_classes": 6, "task": "grid", "mpnn_num_rounds": 40})
    assert cfg.num_classes == 6


def test_build_model_seed_controls_eager_parameters():
    cfg, _ = _load("maze_sheaf")
    first = build_model(cfg, "sheaf", seed=123).state_dict()
    torch.manual_seed(999)
    second = build_model(cfg, "sheaf", seed=123).state_dict()
    for key in ("rm.maps.N", "encoder.comm_head.comm_dense.weight"):
        torch.testing.assert_close(first[key], second[key])


def test_dense_defaults_and_special_head_biases():
    cfg, _ = _load("maze_sheaf")
    model = SheafADMMModel(cfg, input_shape=(3, 3, 6))
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or module.bias is None:
            continue
        if name.endswith("l1_weight_dense"):
            expected = torch.full_like(module.bias, -4.600166)
        elif name.endswith("upper_bound_dense"):
            expected = torch.full_like(module.bias, 0.541325)
        else:
            expected = torch.zeros_like(module.bias)
        torch.testing.assert_close(module.bias, expected, atol=2e-5, rtol=0)


def test_lazy_dense_bias_zero_after_materialization():
    cfg, _ = _load("maze_sheaf")
    patches, edges, kwargs = _inputs(cfg, "maze")
    model = SheafADMMModel(cfg)
    with torch.no_grad():
        model(patches, edges, num_iters=1, loss_window=1, **kwargs, training=False)
    torch.testing.assert_close(model.encoder.dense.bias, torch.zeros_like(model.encoder.dense.bias))


def test_direction_kernel_flax_xavier_scale():
    torch.manual_seed(7)
    directions, messages, hidden = 8, 42, 84
    model = DirectionalGGNN(hidden, messages, directions)
    bound = (6 / (directions * (messages + hidden))) ** 0.5
    kernel = model.cell.dir_kernel.detach()
    assert kernel.abs().max() <= bound
    assert abs(kernel.var().item() - bound**2 / 3) < 0.02 * bound**2


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("maze_sheaf", 181859),
        ("maze_mpnn", 182007),
        ("sudoku_sheaf", 543025),
        ("sudoku_sheaf_lora", 2029233),
    ],
)
def test_locked_parameter_budgets(name, expected):
    torch.manual_seed(0)
    cfg, data = _load(name)
    patches, edges, kwargs = _inputs(cfg, data["task"])
    model_cls = SheafADMMModel if data["model_type"] == "sheaf" else MPNNModel
    model = model_cls(cfg, input_shape=tuple(patches.shape[2:]))
    assert _count(model) == expected
    call = dict(num_iters=2, loss_window=1) if data["model_type"] == "sheaf" else dict(num_rounds=2)
    (
        logits,
        _state,
    ) = model(patches, edges, **call, **kwargs, training=False)[:2]
    assert torch.isfinite(logits).all()
    assert _count(model) == expected


def test_maze_compute_matched_mpnn_budget():
    cfg, _ = _load("maze_mpnn")
    cfg = cfg.__class__(**{**cfg.__dict__, "d_v": 10, "d_e": 5, "mpnn_message_dim": 5})
    assert _count(MPNNModel(cfg, input_shape=(3, 3, cfg.num_classes))) == 48807


@pytest.mark.parametrize(
    "name",
    [
        "maze_sheaf",
        "mnist_sheaf",
        "sudoku_sheaf",
        "sudoku_sheaf_lora",
        "maze_mpnn",
        "mnist_mpnn",
        "sudoku_mpnn",
    ],
)
def test_all_shipped_configs_forward(name):
    torch.manual_seed(0)
    cfg, data = _load(name)
    patches, edges, kwargs = _inputs(cfg, data["task"])
    if data["model_type"] == "sheaf":
        model = SheafADMMModel(cfg, input_shape=tuple(patches.shape[2:]))
        logits, state, geometry = model(
            patches, edges, num_iters=2, loss_window=2, **kwargs, training=False
        )
        assert logits.shape[:3] == (2, patches.shape[0], patches.shape[1])
        assert state.z.shape == (patches.shape[0], patches.shape[1], cfg.d_v)
        assert geometry is not None
    else:
        model = MPNNModel(cfg, input_shape=tuple(patches.shape[2:]))
        logits, hidden = model(patches, edges, num_rounds=2, **kwargs, training=False)
        assert hidden.shape == (patches.shape[0], patches.shape[1], cfg.d_v)
        if cfg.mpnn_graph_readout == "graph":
            assert logits.shape == (patches.shape[1], cfg.num_classes)
        else:
            assert logits.shape[:2] == patches.shape[:2]
    assert torch.isfinite(logits).all()


def test_sudoku_encoder_cell_block_layout():
    from sheaf_admm.models import create_encoder

    cfg, _ = _load("sudoku_sheaf")
    encoder = create_encoder(
        "sudoku",
        comm_dim=cfg.d_v,
        edge_stalk_dim=cfg.d_e,
        d_model=cfg.enc_d_model,
        num_blocks=cfg.enc_num_blocks,
        comm_norm_type=cfg.comm_norm_type,
        objective_mode=cfg.objective_mode,
    )
    out = encoder(torch.zeros(5, 9, cfg.num_classes))
    assert out["h"].shape == (5, cfg.d_v)
    assert out["q_diag"].shape == (5, cfg.d_v)
    assert out["q"].shape == (5, cfg.d_v)
    assert "lower" in out


def test_coordinate_history_reuses_parameters():
    cfg, _ = _load("maze_sheaf")
    patches, edges, kwargs = _inputs(cfg, "maze")
    model = SheafADMMModel(cfg, input_shape=tuple(patches.shape[2:]))
    count = _count(model)
    history, logits, final, geom, rho = model.coordinate_history(
        patches, edges, num_iters=2, **kwargs
    )
    assert history.x.shape[:3] == (2, patches.shape[0], patches.shape[1])
    assert logits.shape[0] == 2
    assert final.z.shape[-1] == cfg.d_v
    assert rho.item() > 0 and geom is not None
    assert _count(model) == count
