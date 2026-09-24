"""Convert a trusted legacy Flax Maze Sheaf checkpoint to a Torch checkpoint.

This offline command requires the optional ``reference`` dependencies because
unpickling legacy JAX arrays imports JAX. Never unpickle an untrusted file.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sheaf_admm.models import model_config_from_dict  # noqa: E402
from sheaf_admm.training import build_model, make_task  # noqa: E402


def _flatten(tree, prefix=()):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from _flatten(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _target_name(path):
    root, *parts = path
    if root == "rho_raw":
        return "rho_raw"
    if root == "rm":
        return "rm.maps." + parts[0][2:]
    if root.startswith("MLPEncoderV2_"):
        parts = ["encoder", *parts]
        parts = ["lora_pre_norm" if x == "lora_pre_ln" else x for x in parts]
        if "lora_A_dense" in parts or "lora_B_dense" in parts:
            parts.insert(1, "lora_heads")
        return ".".join(
            "weight"
            if x == "kernel" or x == "scale" and parts[-2] in {"comm_norm", "lora_pre_norm"}
            else x
            for x in parts
        )
    if root.startswith("ConcatMLPDecoderV2_"):
        parts = ["decoder", *parts]
        return ".".join("weight" if x == "kernel" else x for x in parts)
    raise ValueError(f"unsupported legacy parameter path: {'/'.join(path)}")


def _materialize(model, config):
    task = make_task("maze", **config.get("task_cfg", {}))
    side = 9
    zeros = np.zeros((1, side * side), dtype=np.int64)
    fwd, _, _ = task.prepare({"inputs": zeros, "labels": zeros, "height": side, "width": side})
    with torch.no_grad():
        model(
            fwd["patches"],
            fwd["edge_indices"],
            num_iters=1,
            loss_window=1,
            **fwd["model_kwargs"],
            training=False,
        )


def convert(source: Path, destination: Path):
    with source.open("rb") as file:
        old = pickle.load(file)
    config = old["config"]
    if (
        config.get("task") != "maze"
        or config.get("model_type") != "sheaf"
        or config["model"].get("encoder_arch") != "mlp_v2"
    ):
        raise ValueError("converter currently supports legacy Maze Sheaf MLP-v2 checkpoints")
    model = build_model(model_config_from_dict(config["model"]), "sheaf")
    _materialize(model, config)
    expected = model.state_dict()

    def map_tree(tree, label):
        tree = tree.get("params", tree)
        converted = {}
        for path, value in _flatten(tree):
            key = _target_name(path)
            if key in converted:
                raise ValueError(f"duplicate target key: {key}")
            if key not in expected:
                raise ValueError(f"unexpected target key: {key}")
            array = np.asarray(value)
            if path[-1] == "kernel":
                array = array.T
            tensor = torch.as_tensor(array.copy(), dtype=expected[key].dtype)
            if tensor.shape != expected[key].shape:
                raise ValueError(
                    f"{label}: {key} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected[key].shape)}"
                )
            converted[key] = tensor
        missing = set(expected) - set(converted)
        if missing:
            raise ValueError(f"{label}: missing Torch parameters: {sorted(missing)}")
        return converted

    raw = map_tree(old["params"], "params")
    ema = map_tree(old["ema_params"], "ema_params") if old.get("ema_params") is not None else None
    model.load_state_dict(raw, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": raw,
            "ema_state": ema,
            "optimizer_state": None,
            "step": None,
            "config": config,
            "converted_from": str(source),
        },
        destination,
    )
    return len(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = convert(args.source, args.destination)
    print(f"Converted {count} parameters to {args.destination}")


if __name__ == "__main__":
    main()
