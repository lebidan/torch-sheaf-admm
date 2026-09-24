"""Shared per-agent encoders and convex objective heads."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import CommHead, Dense, LazyDense, MLPBlock, MLPMixerBlock, RMSNorm


def inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class ObjectiveHeads(nn.Module):
    def __init__(
        self,
        feat_dim,
        comm_dim,
        objective_mode="simple",
        q_epsilon=1e-4,
        q_diag_softplus=True,
        l1_weight=0.0,
        l1_init=0.01,
        upper_init=1.0,
        beta_init=1.0,
    ):
        super().__init__()
        self.objective_mode, self.q_epsilon = objective_mode, q_epsilon
        self.q_diag_softplus, self.l1_weight = q_diag_softplus, l1_weight
        if objective_mode == "simple":
            self.beta_raw = nn.Parameter(torch.tensor(inverse_softplus(beta_init)))
        elif objective_mode in ("quadratic", "lasso", "non_negative", "l1box_diag"):
            self.q_diag_dense = Dense(feat_dim, comm_dim)
            self.q_dense = Dense(feat_dim, comm_dim)
            if objective_mode == "l1box_diag":
                self.l1_weight_dense = Dense(feat_dim, comm_dim)
                self.upper_bound_dense = Dense(feat_dim, comm_dim)
                for layer, init in (
                    (self.l1_weight_dense, l1_init),
                    (self.upper_bound_dense, upper_init),
                ):
                    nn.init.zeros_(layer.weight)
                    nn.init.constant_(layer.bias, inverse_softplus(init))
        else:
            raise ValueError(f"unknown objective_mode={objective_mode!r}")

    def forward(self, h, feats):
        out = {"h": h}
        if self.objective_mode == "simple":
            out["beta"] = F.softplus(self.beta_raw)
            return out
        raw = self.q_diag_dense(feats)
        out["q_diag"] = (F.softplus(raw) if self.q_diag_softplus else raw) + self.q_epsilon
        out["q"] = self.q_dense(feats)
        if self.objective_mode == "lasso":
            out["l1_weight"] = feats.new_tensor(self.l1_weight)
        elif self.objective_mode == "non_negative":
            out["lower"] = 0.0
        elif self.objective_mode == "l1box_diag":
            out["l1_weight"] = F.softplus(self.l1_weight_dense(feats))
            out["upper"] = F.softplus(self.upper_bound_dense(feats))
            out["lower"] = 0.0
        return out


class LoRAHeads(nn.Module):
    def __init__(self, feat_dim, num_slots, d_e, d_v, rank, init_style="legacy", use_gate=False):
        super().__init__()
        if init_style not in ("legacy", "standard"):
            raise ValueError(f"unknown lora_init_style={init_style!r}")
        self.num_slots, self.d_e, self.d_v, self.rank = num_slots, d_e, d_v, rank
        self.lora_A_dense = Dense(feat_dim, num_slots * d_e * rank)
        self.lora_B_dense = Dense(feat_dim, num_slots * d_v * rank)
        zero = self.lora_B_dense if init_style == "legacy" else self.lora_A_dense
        nn.init.zeros_(zero.weight)
        nn.init.zeros_(zero.bias)
        self.lora_gate_dense = Dense(feat_dim, num_slots) if use_gate else None
        if use_gate:
            nn.init.constant_(self.lora_gate_dense.bias, -2.0)

    def forward(self, x):
        n = x.shape[0]
        out = {
            "A": self.lora_A_dense(x).reshape(n, self.num_slots, self.d_e, self.rank),
            "B": self.lora_B_dense(x).reshape(n, self.num_slots, self.d_v, self.rank),
        }
        if self.lora_gate_dense is not None:
            out["gate"] = torch.sigmoid(self.lora_gate_dense(x))
        return out


def _heads(owner, feat_dim, kw, lora_norm):
    owner.comm_head = CommHead(
        feat_dim if not hasattr(owner, "cell_dim") else owner.comm_dim,
        owner.comm_dim,
        owner.comm_norm_type,
    )
    owner.objective_heads = ObjectiveHeads(
        feat_dim,
        owner.comm_dim,
        owner.objective_mode,
        owner.q_epsilon,
        owner.q_diag_softplus,
        owner.l1_weight,
        owner.l1_init,
        owner.upper_init,
        owner.beta_init,
    )
    owner.lora_pre_norm = lora_norm if owner.rm_mode == "context" else None
    owner.lora_heads = (
        LoRAHeads(
            feat_dim,
            owner.num_directions,
            owner.edge_stalk_dim or owner.comm_dim,
            owner.comm_dim,
            owner.lora_rank,
            owner.lora_init_style,
            owner.lora_use_gate,
        )
        if owner.rm_mode == "context"
        else None
    )


class _EncoderBase(nn.Module):
    def _set_common(
        self,
        comm_dim,
        edge_stalk_dim,
        comm_norm_type,
        objective_mode,
        q_epsilon,
        q_diag_softplus,
        l1_weight,
        l1_init,
        upper_init,
        beta_init,
        rm_mode,
        lora_rank,
        lora_use_gate,
        lora_init_style,
        num_directions,
    ):
        self.comm_dim, self.edge_stalk_dim, self.comm_norm_type = (
            comm_dim,
            edge_stalk_dim,
            comm_norm_type,
        )
        self.objective_mode, self.q_epsilon, self.q_diag_softplus = (
            objective_mode,
            q_epsilon,
            q_diag_softplus,
        )
        self.l1_weight, self.l1_init, self.upper_init, self.beta_init = (
            l1_weight,
            l1_init,
            upper_init,
            beta_init,
        )
        self.rm_mode, self.lora_rank, self.lora_use_gate = rm_mode, lora_rank, lora_use_gate
        self.lora_init_style, self.num_directions = lora_init_style, num_directions

    def _finish(self, h_source, objective_source):
        h = self.comm_head(h_source)
        out = self.objective_heads(h, objective_source)
        if self.lora_heads is not None:
            out.update(self.lora_heads(self.lora_pre_norm(objective_source)))
        return out


class MLPEncoderV2(_EncoderBase):
    def __init__(
        self,
        hidden_dim=256,
        comm_dim=64,
        edge_stalk_dim=None,
        comm_norm_type="layernorm_zeros",
        dropout_rate=0.0,
        objective_mode="simple",
        q_epsilon=1e-4,
        q_diag_softplus=True,
        l1_weight=0.0,
        l1_init=0.01,
        upper_init=1.0,
        beta_init=1.0,
        rm_mode="fixed",
        lora_rank=4,
        lora_use_gate=False,
        lora_init_style="legacy",
        num_directions=4,
        input_shape=None,
    ):
        super().__init__()
        self._set_common(
            comm_dim,
            edge_stalk_dim,
            comm_norm_type,
            objective_mode,
            q_epsilon,
            q_diag_softplus,
            l1_weight,
            l1_init,
            upper_init,
            beta_init,
            rm_mode,
            lora_rank,
            lora_use_gate,
            lora_init_style,
            num_directions,
        )
        width = math.prod(input_shape) if input_shape is not None else None
        self.input_norm = RMSNorm(width) if width is not None else None
        self.dense = Dense(width, hidden_dim) if width is not None else LazyDense(hidden_dim)
        self.dropout_rate = dropout_rate
        _heads(self, hidden_dim, None, nn.LayerNorm(hidden_dim, eps=1e-6))

    def forward(self, x, training=True):
        x = x.flatten(1)
        if self.input_norm is None:
            self.input_norm = RMSNorm(x.shape[-1]).to(x.device)
        x = F.gelu(self.dense(self.input_norm(x)), approximate="tanh")
        x = F.dropout(x, self.dropout_rate, training=training)
        return self._finish(x, x)


class MLPEncoder(_EncoderBase):
    def __init__(
        self,
        hidden_dims=(256,),
        comm_dim=32,
        edge_stalk_dim=None,
        comm_norm_type="layernorm_zeros",
        norm_type="rmsnorm",
        dropout_rate=0.0,
        objective_mode="simple",
        q_epsilon=1e-4,
        q_diag_softplus=True,
        l1_weight=0.0,
        l1_init=0.01,
        upper_init=1.0,
        beta_init=1.0,
        rm_mode="fixed",
        lora_rank=4,
        lora_use_gate=False,
        lora_init_style="legacy",
        num_directions=4,
        input_shape=None,
    ):
        super().__init__()
        self._set_common(
            comm_dim,
            edge_stalk_dim,
            comm_norm_type,
            objective_mode,
            q_epsilon,
            q_diag_softplus,
            l1_weight,
            l1_init,
            upper_init,
            beta_init,
            rm_mode,
            lora_rank,
            lora_use_gate,
            lora_init_style,
            num_directions,
        )
        width = math.prod(input_shape) if input_shape is not None else None
        self.input_proj = (
            Dense(width, hidden_dims[0]) if width is not None else LazyDense(hidden_dims[0])
        )
        self.blocks = nn.ModuleList(
            [
                MLPBlock(hidden_dims[i - 1] if i else hidden_dims[0], d, norm_type=norm_type)
                for i, d in enumerate(hidden_dims)
            ]
        )
        self.dropout_rate = dropout_rate
        _heads(self, hidden_dims[-1], None, nn.LayerNorm(hidden_dims[-1], eps=1e-6))

    def forward(self, x, training=True):
        x = self.input_proj(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        x = F.dropout(x, self.dropout_rate, training=training)
        return self._finish(x, x)


class SudokuEncoder(_EncoderBase):
    def __init__(
        self,
        d_model=128,
        num_blocks=2,
        mlp_ratio=2.0,
        mlp_type="swiglu",
        comm_dim=288,
        edge_stalk_dim=None,
        comm_norm_type="layernorm_zeros",
        objective_mode="simple",
        q_epsilon=1e-4,
        q_diag_softplus=True,
        l1_weight=0.0,
        l1_init=0.01,
        upper_init=1.0,
        beta_init=1.0,
        rm_mode="fixed",
        lora_rank=4,
        lora_use_gate=False,
        lora_init_style="legacy",
        num_directions=4,
        input_shape=None,
    ):
        super().__init__()
        self._set_common(
            comm_dim,
            edge_stalk_dim,
            comm_norm_type,
            objective_mode,
            q_epsilon,
            q_diag_softplus,
            l1_weight,
            l1_init,
            upper_init,
            beta_init,
            rm_mode,
            lora_rank,
            lora_use_gate,
            lora_init_style,
            num_directions,
        )
        if comm_dim % 9:
            raise ValueError("comm_dim must be divisible by 9")
        self.cell_dim = comm_dim // 9
        in_dim = input_shape[-1] if input_shape is not None else 10
        self.token_embed = Dense(in_dim, d_model)
        nn.init.normal_(self.token_embed.weight, std=1 / math.sqrt(d_model))
        self.global_pos_embed = nn.Embedding(81, d_model)
        nn.init.normal_(self.global_pos_embed.weight, std=1 / math.sqrt(d_model))
        self.pos_embed = nn.Parameter(torch.empty(1, 9, d_model))
        nn.init.normal_(self.pos_embed, std=1 / math.sqrt(d_model))
        self.blocks = nn.ModuleList(
            [
                MLPMixerBlock(9, d_model, int(9 * mlp_ratio), int(d_model * mlp_ratio), mlp_type)
                for _ in range(num_blocks)
            ]
        )
        self.pre_flat_norm = RMSNorm(d_model)
        self.cell_proj = Dense(d_model, self.cell_dim)
        self.cell_norm = RMSNorm(self.cell_dim)
        _heads(self, d_model, None, RMSNorm(d_model))
        self.embed_scale = math.sqrt(d_model / 2)

    def forward(self, x, cell_ids=None, training=True):
        y = self.token_embed(x)
        if cell_ids is not None:
            y = y + self.global_pos_embed(cell_ids.long())
        y = (y + self.pos_embed) * self.embed_scale
        for block in self.blocks:
            y = block(y)
        y = self.pre_flat_norm(y)
        cells = self.cell_norm(self.cell_proj(y))
        return self._finish(cells.flatten(1), y.mean(dim=1))


def create_encoder(arch="mlp_v2", **kwargs):
    classes = {"mlp_v2": MLPEncoderV2, "mlp": MLPEncoder, "sudoku": SudokuEncoder}
    if arch not in classes:
        raise ValueError(f"unknown encoder arch={arch!r}")
    return classes[arch](**kwargs)
