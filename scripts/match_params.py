"""Count model parameters, and find the MPNN state size matching a target count.

The paper's parameter-matched (PM) MPNN changes the hidden size until the MPNN
has as many parameters as Sheaf-ADMM; the communication-matched (CM) one keeps
the same d_v. The count is taken from the real model (built and materialised on
one dummy batch), so it follows any change in the code.

    # parameters of each experiment
    python -m scripts.match_params minesweeper_sheaf minesweeper_mpnn

    # state size giving the MPNN as many parameters as the sheaf model
    python -m scripts.match_params minesweeper_mpnn --match minesweeper_sheaf

    # paper protocol (Maze-style): message sizes tied to d_v / 2, as in maze_mpnn.yaml
    python -m scripts.match_params minesweeper_grid_mpnn --match minesweeper_grid_sheaf --half-message

    # a bigger MPNN: 4x the sheaf model's parameter count
    python -m scripts.match_params minesweeper_grid_mpnn --match minesweeper_grid_sheaf --half-message --scale 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_experiment(name: str) -> dict:
    """Merge configs/config.yaml with configs/experiment/<name>.yaml (Hydra-like)."""
    from omegaconf import OmegaConf

    base = OmegaConf.load(PROJECT_ROOT / "configs" / "config.yaml")
    exp = OmegaConf.load(PROJECT_ROOT / "configs" / "experiment" / f"{name}.yaml")
    base = {k: v for k, v in OmegaConf.to_container(base).items() if k != "defaults"}
    return OmegaConf.to_container(OmegaConf.merge(base, exp), resolve=True)


def dummy_batch(cfg: dict):
    """One small batch in the task's format (only shapes matter)."""
    from sheaf_admm.data import PuzzleDataset

    data_dir = cfg["data"]["dir"]
    try:
        ds = PuzzleDataset(data_dir, cfg["data"]["val_splits"][0])
        for _set, b in ds.iter_test_batches(batch_size=2):
            return {"inputs": np.asarray(b["inputs"]), "labels": np.asarray(b["labels"])}
    except Exception:
        pass
    if cfg["task"] not in ("minesweeper", "minesweeper_grid"):
        raise FileNotFoundError(f"no dataset found in {data_dir}")
    from sheaf_admm.data.build_minesweeper import generate_board, neighbours

    size = 9
    rng = np.random.default_rng(0)
    board = None
    while board is None:
        board = generate_board(rng, size, 10, 30, neighbours(size))
    return {"inputs": board[0][None].astype(np.int64), "labels": board[1][None].astype(np.int64)}


def count_params(cfg: dict, model_overrides: dict | None = None) -> int:
    import torch

    from sheaf_admm.models import model_config_from_dict
    from sheaf_admm.training import build_model, make_task
    from sheaf_admm.training.loop import _forward

    model_cfg = dict(cfg["model"])
    for key, value in (model_overrides or {}).items():
        model_cfg.pop("agent_state_dim" if key == "d_v" else "message_dim" if key == "d_e" else "", None)
        model_cfg[key] = value
    task = make_task(cfg["task"], **cfg.get("task_cfg", {}))
    model = build_model(model_config_from_dict(model_cfg), cfg["model_type"])
    fwd, _targets, _aux = task.prepare(dummy_batch(cfg))
    with torch.no_grad():  # materialise lazy layers
        _forward(model, fwd, n_iter=1, loss_window=1, model_type=cfg["model_type"],
                 training=False, compile_steps=False)
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+", help="experiment names (configs/experiment/<name>.yaml)")
    p.add_argument("--match", default=None,
                   help="experiment whose parameter count the first (MPNN) experiment should match")
    p.add_argument("--min-dv", type=int, default=8)
    p.add_argument("--max-dv", type=int, default=1024)
    p.add_argument("--scale", type=float, default=1.0,
                   help="target = scale x the parameter count of --match (e.g. 4 for a bigger MPNN)")
    p.add_argument("--half-message", action="store_true",
                   help="tie d_e and the MPNN message size to d_v // 2 (paper's Maze PM: 84 / 42 / 42)")
    args = p.parse_args()

    def overrides(dv: int) -> dict:
        out = {"d_v": dv}
        if args.half_message:
            half = max(1, dv // 2)
            out.update(d_e=half, mpnn_message_dim=half)
        return out


    for name in args.experiments:
        print(f"{name}: {count_params(load_experiment(name)):,} parameters".replace(",", " "))
    if args.match is None:
        return
    target = int(round(args.scale * count_params(load_experiment(args.match))))
    cfg = load_experiment(args.experiments[0])
    lo, hi = args.min_dv, args.max_dv
    while hi - lo > 1:  # the count grows with d_v: bisection
        mid = (lo + hi) // 2
        if count_params(cfg, overrides(mid)) < target:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda d: abs(count_params(cfg, overrides(d)) - target))
    got = count_params(cfg, overrides(best))
    scale = f"{args.scale:g} x " if args.scale != 1 else ""
    print(f"\ntarget ({scale}{args.match}): {target:,} parameters".replace(",", " "))
    print(f"{args.experiments[0]} with agent_state_dim={best}: {got:,} parameters "
          f"({(got - target) / target * 100:+.1f}%)".replace(",", " "))
    extra = (f" model.message_dim={max(1, best // 2)} model.mpnn_message_dim={max(1, best // 2)}"
             if args.half_message else "")
    print(f"-> training override: model.agent_state_dim={best}{extra}")


if __name__ == "__main__":
    main()
