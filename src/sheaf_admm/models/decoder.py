"""Shared per-agent PyTorch decoders."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Dense, LazyDense, MLPBlock, RMSNorm


class ConcatMLPDecoderV2(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        output_shape=(5, 5, 6),
        dropout_rate=0.0,
        comm_dim=None,
        input_shape=None,
    ):
        super().__init__()
        self.output_shape, self.dropout_rate = tuple(output_shape), dropout_rate
        width = (
            math.prod(input_shape) + comm_dim
            if input_shape is not None and comm_dim is not None
            else None
        )
        self.input_norm = RMSNorm(width) if width is not None else None
        self.dense = Dense(width, hidden_dim) if width is not None else LazyDense(hidden_dim)
        self.output_dense = Dense(hidden_dim, math.prod(output_shape))

    def forward(self, x, patches, training=True):
        h = torch.cat([patches.flatten(1), x], dim=-1)
        if self.input_norm is None:
            self.input_norm = RMSNorm(h.shape[-1]).to(h.device)
        h = F.gelu(self.dense(self.input_norm(h)), approximate="tanh")
        h = F.dropout(h, self.dropout_rate, training=training)
        return self.output_dense(h).reshape(x.shape[0], *self.output_shape)


class SudokuDecoder(nn.Module):
    def __init__(self, hidden_dims=(128,), output_channels=10, norm_type="rmsnorm", comm_dim=288):
        super().__init__()
        if comm_dim % 9:
            raise ValueError("comm_dim must be divisible by 9")
        widths = [comm_dim // 9, *hidden_dims]
        self.blocks = nn.ModuleList(
            [
                MLPBlock(widths[i], widths[i + 1], norm_type=norm_type)
                for i in range(len(hidden_dims))
            ]
        )
        self.output_dense = Dense(widths[-1], output_channels)

    def forward(self, x, patches=None, training=True):
        h = x.reshape(x.shape[0], 9, -1)
        for block in self.blocks:
            h = block(h)
        return self.output_dense(h)


class ClassificationDecoder(nn.Module):
    def __init__(
        self,
        hidden_dims=(128,),
        output_channels=10,
        norm_type="rmsnorm",
        linear_head=False,
        readout_mode="concat",
        comm_dim=None,
        input_shape=None,
    ):
        super().__init__()
        if readout_mode not in ("concat", "x_only"):
            raise ValueError(f"unknown readout_mode={readout_mode!r}")
        self.readout_mode = readout_mode
        width = (
            comm_dim + (math.prod(input_shape) if readout_mode == "concat" else 0)
            if comm_dim is not None and (readout_mode == "x_only" or input_shape is not None)
            else None
        )
        self.input_proj = (
            None
            if linear_head
            else (Dense(width, hidden_dims[0]) if width is not None else LazyDense(hidden_dims[0]))
        )
        self.blocks = (
            nn.ModuleList(
                [
                    MLPBlock(hidden_dims[i - 1] if i else hidden_dims[0], d, norm_type=norm_type)
                    for i, d in enumerate(hidden_dims)
                ]
            )
            if not linear_head
            else nn.ModuleList()
        )
        last = width if linear_head else hidden_dims[-1]
        self.cls_output = (
            Dense(last, output_channels) if last is not None else LazyDense(output_channels)
        )

    def forward(self, x, patches, training=True):
        h = torch.cat([x, patches.flatten(1)], dim=-1) if self.readout_mode == "concat" else x
        if self.input_proj is not None:
            h = self.input_proj(h)
            for block in self.blocks:
                h = block(h)
        return self.cls_output(h)


def create_decoder(arch="mlp_concat_v2", **kwargs):
    classes = {
        "mlp_concat_v2": ConcatMLPDecoderV2,
        "sudoku": SudokuDecoder,
        "classification": ClassificationDecoder,
    }
    if arch not in classes:
        raise ValueError(f"unknown decoder arch={arch!r}")
    return classes[arch](**kwargs)
