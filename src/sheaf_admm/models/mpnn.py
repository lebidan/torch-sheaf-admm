"""Direction-typed recurrent gated graph network."""

from __future__ import annotations

import torch
from torch import nn

from .layers import Dense, LazyDense, MLPBlock, RMSNorm


class GraphClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_dims=(128,),
        output_channels=10,
        norm_type="rmsnorm",
        linear_head=False,
        input_dim=None,
    ):
        super().__init__()
        self.input_proj = (
            None
            if linear_head
            else (
                Dense(input_dim, hidden_dims[0])
                if input_dim is not None
                else LazyDense(hidden_dims[0])
            )
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
        width = input_dim if linear_head else hidden_dims[-1]
        self.cls_output = (
            Dense(width, output_channels) if width is not None else LazyDense(output_channels)
        )

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
        return self.cls_output(x)


class DirectionalGGNNCell(nn.Module):
    def __init__(
        self,
        hidden_dim,
        message_dim,
        num_directions,
        aggregation="add",
        norm_type="rmsnorm",
        eps=1e-6,
        context_dim=None,
    ):
        super().__init__()
        if aggregation not in ("add", "mean", "symnorm", "max"):
            raise ValueError(f"unknown aggregation={aggregation!r}")
        self.hidden_dim, self.message_dim, self.aggregation = hidden_dim, message_dim, aggregation
        self.dir_kernel = nn.Parameter(torch.empty(num_directions, message_dim, hidden_dim))
        # Flax Xavier treats every direction as part of both fan dimensions.
        bound = (6.0 / (num_directions * (message_dim + hidden_dim))) ** 0.5
        nn.init.uniform_(self.dir_kernel, -bound, bound)
        self.dir_bias = nn.Parameter(torch.zeros(num_directions, message_dim))
        self.self_message_to_edge = (
            Dense(hidden_dim, message_dim, bias=False)
            if aggregation == "symnorm" and message_dim != hidden_dim
            else None
        )
        self.message_to_hidden = (
            Dense(message_dim, hidden_dim, bias=False) if message_dim != hidden_dim else None
        )

        def norm(w):
            return RMSNorm(w, eps) if norm_type == "rmsnorm" else nn.LayerNorm(w, eps=eps)

        context_dim = context_dim or hidden_dim
        self.aggregated_norm = norm(hidden_dim)
        self.context_norm = norm(context_dim)
        self.update_input = Dense(hidden_dim + context_dim, hidden_dim)
        self.gate_norm = norm(hidden_dim * 2)
        self.reset_gate = Dense(hidden_dim * 2, hidden_dim)
        self.update_gate = Dense(hidden_dim * 2, hidden_dim)
        self.candidate_norm = norm(hidden_dim * 2)
        self.candidate = Dense(hidden_dim * 2, hidden_dim)

    def _aggregate(self, messages, src, dst, num_nodes, self_messages=None):
        shape = (num_nodes, *messages.shape[1:])
        if self.aggregation == "max":
            if messages.shape[0] == 0:
                return messages.new_zeros(shape)
            result = messages.new_full(shape, -torch.inf)
            index = dst[:, None, None].expand_as(messages)
            result = result.scatter_reduce(0, index, messages, reduce="amax", include_self=True)
            return torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        result = messages.new_zeros(shape)
        result = result.index_add(0, dst, messages)
        if self.aggregation == "mean":
            counts = messages.new_zeros(num_nodes).index_add(
                0, dst, messages.new_ones(dst.shape[0])
            )
            return result / counts.clamp_min(1)[:, None, None]
        if self.aggregation == "symnorm":
            deg_src = messages.new_ones(num_nodes).index_add(
                0, src, messages.new_ones(src.shape[0])
            )
            deg_dst = messages.new_ones(num_nodes).index_add(
                0, dst, messages.new_ones(dst.shape[0])
            )
            scales = (deg_src[src] * deg_dst[dst]).rsqrt()[:, None, None]
            result = messages.new_zeros(shape).index_add(0, dst, messages * scales)
            return result + self_messages / deg_dst[:, None, None]
        return result

    def forward(self, hidden, context, edge_indices, direction_ids):
        src, dst = edge_indices[:, 0].long(), edge_indices[:, 1].long()
        kernel, bias = self.dir_kernel[direction_ids.long()], self.dir_bias[direction_ids.long()]
        messages = torch.einsum("emd,ebd->ebm", kernel, hidden[src]) + bias[:, None, :]
        self_messages = None
        if self.aggregation == "symnorm":
            self_messages = (
                hidden if self.self_message_to_edge is None else self.self_message_to_edge(hidden)
            )
        agg = self._aggregate(messages, src, dst, hidden.shape[0], self_messages)
        if self.message_to_hidden is not None:
            agg = self.message_to_hidden(agg)
        update = self.update_input(
            torch.cat((self.aggregated_norm(agg), self.context_norm(context)), -1)
        )
        gates = self.gate_norm(torch.cat((hidden, update), -1))
        reset = torch.sigmoid(self.reset_gate(gates))
        gate = torch.sigmoid(self.update_gate(gates))
        candidate = torch.tanh(
            self.candidate(self.candidate_norm(torch.cat((reset * hidden, update), -1)))
        )
        return (1 - gate) * candidate + gate * hidden


class DirectionalGGNN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        message_dim,
        num_directions,
        aggregation="add",
        norm_type="rmsnorm",
        eps=1e-6,
        context_dim=None,
    ):
        super().__init__()
        self.cell = DirectionalGGNNCell(
            hidden_dim, message_dim, num_directions, aggregation, norm_type, eps, context_dim
        )

    def forward(
        self, hidden0, context, edge_indices, direction_ids, num_rounds, return_history=False
    ):
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        hidden, history = hidden0, []
        for _ in range(num_rounds):
            hidden = self.cell(hidden, context, edge_indices, direction_ids)
            if return_history:
                history.append(hidden)
        return (hidden, torch.stack(history)) if return_history else hidden
