"""Recurrent MPNN baseline with shared encoder and decoder."""

from __future__ import annotations

import torch
from torch import nn

from sheaf_admm.geometry import compute_direction_index

from .config import ModelConfig
from .decoder import create_decoder
from .encoder import create_encoder
from .layers import Dense
from .mpnn import DirectionalGGNN, GraphClassificationHead


class MPNNModel(nn.Module):
    def __init__(self, config: ModelConfig, input_shape=None):
        super().__init__()
        self.config = config
        c = config
        enc_kwargs = dict(
            comm_dim=c.d_v,
            edge_stalk_dim=c.d_e,
            comm_norm_type=c.comm_norm_type,
            objective_mode="simple",
            rm_mode="fixed",
            beta_init=c.beta_init,
            input_shape=input_shape,
        )
        if c.encoder_arch == "mlp_v2":
            enc_kwargs.update(hidden_dim=c.enc_hidden_dim, dropout_rate=c.dropout_rate)
        elif c.encoder_arch == "mlp":
            enc_kwargs.update(hidden_dims=(c.enc_hidden_dim,), dropout_rate=c.dropout_rate)
        elif c.encoder_arch == "sudoku":
            enc_kwargs.update(d_model=c.enc_d_model, num_blocks=c.enc_num_blocks)
        self.encoder = create_encoder(c.encoder_arch, **enc_kwargs)
        self.init_hidden = Dense(c.d_v, c.d_v)
        num_types = {"shared": 1, "spatial": c.num_directions, "slot": 9}[c.mpnn_edge_type_mode]
        self.ggnn = DirectionalGGNN(
            c.d_v, c.mpnn_message_dim or c.d_e, num_types, c.mpnn_aggregation, context_dim=c.d_v
        )
        if c.mpnn_graph_readout == "graph":
            self.graph_head = GraphClassificationHead(
                c.dec_hidden_dims, c.num_classes, c.dec_norm_type, c.dec_linear_head, c.d_v
            )
            self.decoder = None
        else:
            kwargs = {"comm_dim": c.d_v, "input_shape": input_shape}
            if c.decoder_arch == "mlp_concat_v2":
                shape = (
                    (*input_shape[:-1], c.num_classes)
                    if input_shape is not None
                    else (3, 3, c.num_classes)
                )
                kwargs.update(
                    hidden_dim=c.dec_hidden_dim, output_shape=shape, dropout_rate=c.dropout_rate
                )
            elif c.decoder_arch == "sudoku":
                kwargs = dict(
                    hidden_dims=c.dec_hidden_dims,
                    output_channels=c.num_classes,
                    norm_type=c.dec_norm_type,
                    comm_dim=c.d_v,
                )
            else:
                kwargs.update(
                    hidden_dims=c.dec_hidden_dims,
                    output_channels=c.num_classes,
                    norm_type=c.dec_norm_type,
                    linear_head=c.dec_linear_head,
                    readout_mode=c.dec_readout_mode,
                )
            self.decoder = create_decoder(c.decoder_arch, **kwargs)
            self.graph_head = None

    def _directed_edges(self, edge_indices, node_positions, map_u, map_v):
        c = self.config
        u, v = edge_indices[:, 0].long(), edge_indices[:, 1].long()
        directed = torch.cat((edge_indices, torch.stack((v, u), 1)), 0).long()
        if c.mpnn_edge_type_mode == "shared":
            ids = torch.zeros(directed.shape[0], dtype=torch.long, device=directed.device)
        elif c.mpnn_edge_type_mode == "spatial":
            if node_positions is None:
                raise ValueError("node_positions required for spatial edge types")
            dy = node_positions[v, 0] - node_positions[u, 0]
            dx = node_positions[v, 1] - node_positions[u, 1]
            ids = torch.cat(
                (
                    compute_direction_index(dy, dx, c.num_directions),
                    compute_direction_index(-dy, -dx, c.num_directions),
                )
            ).long()
        else:
            if map_u is None or map_v is None:
                raise ValueError("map_u/map_v required for slot edge types")
            ids = torch.cat((map_u, map_v)).long()
        return directed, ids

    def forward(
        self,
        patches,
        edge_indices,
        *,
        num_rounds,
        node_positions=None,
        map_u=None,
        map_v=None,
        cell_ids=None,
        training=True,
    ):
        c = self.config
        edge_indices = torch.as_tensor(edge_indices, device=patches.device, dtype=torch.long)
        node_positions = (
            torch.as_tensor(node_positions, device=patches.device)
            if node_positions is not None
            else None
        )
        map_u = (
            torch.as_tensor(map_u, device=patches.device, dtype=torch.long)
            if map_u is not None
            else None
        )
        map_v = (
            torch.as_tensor(map_v, device=patches.device, dtype=torch.long)
            if map_v is not None
            else None
        )
        cell_ids = (
            torch.as_tensor(cell_ids, device=patches.device, dtype=torch.long)
            if cell_ids is not None
            else None
        )
        n, b = patches.shape[:2]
        flat = patches.reshape(n * b, *patches.shape[2:])
        if c.encoder_arch == "sudoku":
            ids = (
                cell_ids[:, None, :].expand(n, b, -1).reshape(n * b, -1)
                if cell_ids is not None
                else None
            )
            enc = self.encoder(flat, ids, training=training)
        else:
            enc = self.encoder(flat, training=training)
        context = enc["h"].reshape(n, b, c.d_v)
        directed, types = self._directed_edges(edge_indices, node_positions, map_u, map_v)
        hidden = self.ggnn(self.init_hidden(context), context, directed, types, num_rounds)
        if self.graph_head is not None:
            return self.graph_head(hidden.mean(dim=0)), hidden
        logits = self.decoder(hidden.reshape(n * b, c.d_v), flat, training=training)
        return logits.reshape(n, b, *logits.shape[1:]), hidden
