"""Evaluate a Sheaf-ADMM checkpoint on a full split for several ADMM iteration counts.

If metrics keep improving past the ``K_eval`` used during training, the model
can still reason further and simply needs more iterations at test time.

    python -m scripts.eval_k --checkpoint outputs/.../checkpoint_best.pt --ks 40 80 120 160

``--data-dir`` / ``--split`` evaluate on another dataset (e.g. the symmetry-
disjoint N-Queens split) with the same checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import sheaf_admm as _sheaf_admm  # noqa: F401,E402
from sheaf_admm.data import PuzzleDataset  # noqa: E402
from sheaf_admm.models import model_config_from_dict  # noqa: E402
from sheaf_admm.training import build_model, make_task  # noqa: E402
from sheaf_admm.training.loop import _forward, move_to_device  # noqa: E402


def _batches(data_dir: str, split: str, batch_size: int):
    ds = PuzzleDataset(data_dir, split)
    for _set, b in ds.iter_test_batches(batch_size=batch_size):
        batch = {"inputs": np.asarray(b["inputs"]), "labels": np.asarray(b["labels"])}
        for key in ("height", "width"):
            if key in b:
                batch[key] = b[key]
        yield batch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[20, 40, 80, 120, 160])
    p.add_argument("--data-dir", default=None, help="default: the checkpoint's data.dir")
    p.add_argument("--split", default=None, help="default: the checkpoint's first val split")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--params", choices=("ema", "raw"), default="ema")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("model_type") != "sheaf":
        raise ValueError("eval_k supports Sheaf-ADMM checkpoints only")
    if cfg["data"].get("loader", "puzzle") != "puzzle":
        raise ValueError("eval_k supports puzzle datasets only")
    params = ckpt.get("ema_state") if args.params == "ema" else ckpt["model_state"]
    if params is None:
        params = ckpt["model_state"]

    data_dir = args.data_dir or cfg["data"]["dir"]
    split = args.split or cfg["data"]["val_splits"][0]
    task = make_task(cfg["task"], **cfg.get("task_cfg", {}))
    model = build_model(model_config_from_dict(cfg["model"]), cfg["model_type"]).to(device)

    batches = [
        (move_to_device(fwd, device), move_to_device(targets, device), aux, len(b["inputs"]))
        for b in _batches(data_dir, split, args.batch_size)
        for fwd, targets, aux in [task.prepare(b)]
    ]
    with torch.no_grad():  # materialize lazy layers before loading weights
        _forward(model, batches[0][0], n_iter=1, loss_window=1, model_type="sheaf", training=False)
    model.load_state_dict(params)
    model.eval()

    total = sum(n for *_, n in batches)
    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {data_dir} [{split}], {total} puzzles, params={args.params}, "
          f"trained with K_train={cfg['training'].get('K_train')} "
          f"K_eval={cfg['training'].get('K_eval')}")
    for k in args.ks:
        start = time.time()
        agg: dict[str, float] = {}
        with torch.no_grad():
            for fwd, targets, aux, n in batches:
                logits = _forward(
                    model, fwd, n_iter=k, loss_window=1, model_type="sheaf", training=False
                )
                for name, value in task.evaluate(logits[-1], targets, aux).items():
                    agg[name] = agg.get(name, 0.0) + value * n
        line = "  ".join(f"{name}={agg[name] / total * 100:.2f}%" for name in agg)
        print(f"[K={k:>4}] {line}  ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
