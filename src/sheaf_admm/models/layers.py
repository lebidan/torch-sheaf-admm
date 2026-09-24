"""Shared PyTorch network layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter


def _flax_dense_init(module):
    """Flax Dense default: LeCun truncated normal kernel and zero bias."""
    if isinstance(module.weight, UninitializedParameter) or module.weight.shape[1] == 0:
        return
    fan_in = module.weight.shape[1]
    std = (fan_in**-0.5) / 0.87962566103423978
    nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class Dense(nn.Linear):
    def reset_parameters(self):
        _flax_dense_init(self)


class LazyDense(nn.LazyLinear):
    def reset_parameters(self):
        _flax_dense_init(self)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return rms_norm(x, self.eps) * self.scale


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        norm_type: str = "rmsnorm",
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.norm = (
            RMSNorm(in_dim, eps) if norm_type == "rmsnorm" else nn.LayerNorm(in_dim, eps=eps)
        )
        self.dense1 = Dense(in_dim, hidden_dim)
        self.dense2 = Dense(hidden_dim, out_dim)
        self.residual_proj = Dense(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual_proj(x) + self.dense2(
            F.gelu(self.dense1(self.norm(x)), approximate="tanh")
        )


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gate_up = Dense(in_dim, 2 * hidden_dim)
        self.down = Dense(hidden_dim, out_dim)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class GeLUMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.up = Dense(in_dim, hidden_dim)
        self.down = Dense(hidden_dim, out_dim)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class MLPMixerBlock(nn.Module):
    def __init__(
        self,
        tokens: int,
        channels: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        mlp_type: str = "swiglu",
        rms_eps: float = 1e-6,
    ):
        super().__init__()
        if mlp_type not in ("swiglu", "gelu"):
            raise ValueError(f"unknown mlp_type={mlp_type!r}")
        cls = SwiGLU if mlp_type == "swiglu" else GeLUMLP
        self.token_mlp = cls(tokens, token_mlp_dim, tokens)
        self.channel_mlp = cls(channels, channel_mlp_dim, channels)
        self.token_norm = RMSNorm(channels, rms_eps)
        self.channel_norm = RMSNorm(channels, rms_eps)

    def forward(self, x):
        x = self.token_norm(x + self.token_mlp(x.transpose(1, 2)).transpose(1, 2))
        return self.channel_norm(x + self.channel_mlp(x))


class CommHead(nn.Module):
    def __init__(self, in_dim: int, communication_dim: int, comm_norm_type: str = "identity"):
        super().__init__()
        self.comm_dense = Dense(in_dim, communication_dim)
        self.comm_norm_type = comm_norm_type
        self.comm_norm = (
            nn.LayerNorm(communication_dim, eps=1e-6)
            if comm_norm_type in ("layernorm", "layernorm_zeros")
            else nn.Identity()
        )
        if comm_norm_type == "layernorm_zeros":
            nn.init.constant_(self.comm_norm.weight, 1e-4)
        if comm_norm_type not in ("identity", "tanh", "layernorm", "layernorm_zeros"):
            raise ValueError(f"unknown comm_norm_type={comm_norm_type!r}")

    def forward(self, x):
        x = self.comm_dense(x)
        return torch.tanh(x) if self.comm_norm_type == "tanh" else self.comm_norm(x)
