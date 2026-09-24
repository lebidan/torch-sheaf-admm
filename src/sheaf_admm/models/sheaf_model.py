"""End-to-end PyTorch Sheaf-ADMM model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from sheaf_admm.admm import run_admm, run_admm_history
from sheaf_admm.geometry import (
    DirectionalRestrictionMaps,
    FixedGeometry,
    SharedRestrictionMap,
    SudokuRestrictionMaps,
    build_directional_restriction_maps,
    build_shared_restriction_maps,
    build_sudoku_restriction_maps,
    create_lora_geometry,
    create_sudoku_lora_geometry,
)
from sheaf_admm.solvers.x_solvers import make_x_solver
from sheaf_admm.solvers.z_solvers import GDParams, UnrolledCGParams, make_z_solver

from .config import ModelConfig
from .decoder import create_decoder
from .encoder import create_encoder, inverse_softplus

_ARRAY_KEYS = ("h", "q_diag", "q", "l1_weight", "upper", "A", "B", "gate")


class SheafADMMModel(nn.Module):
    def __init__(self, config: ModelConfig, input_shape=None):
        super().__init__()
        self.config = c = config
        enc_kwargs = dict(
            comm_dim=c.d_v,
            edge_stalk_dim=c.d_e,
            comm_norm_type=c.comm_norm_type,
            objective_mode=c.objective_mode,
            q_epsilon=c.q_epsilon,
            l1_weight=c.l1_weight,
            l1_init=c.l1_init,
            upper_init=c.upper_init,
            beta_init=c.beta_init,
            rm_mode=c.rm_mode,
            lora_rank=c.lora_rank,
            lora_use_gate=c.lora_use_gate,
            lora_init_style=c.lora_init_style,
            num_directions=c.num_directions,
            input_shape=input_shape,
        )
        if c.encoder_arch == "mlp_v2":
            enc_kwargs.update(hidden_dim=c.enc_hidden_dim, dropout_rate=c.dropout_rate)
        elif c.encoder_arch == "mlp":
            enc_kwargs.update(hidden_dims=(c.enc_hidden_dim,), dropout_rate=c.dropout_rate)
        elif c.encoder_arch == "sudoku":
            enc_kwargs.update(d_model=c.enc_d_model, num_blocks=c.enc_num_blocks)
        self.encoder = create_encoder(c.encoder_arch, **enc_kwargs)
        if c.rm_sharing == "directional":
            self.rm = DirectionalRestrictionMaps(c.d_v, c.d_e, c.rm_init, c.num_directions)
        elif c.rm_sharing == "global":
            self.rm = SharedRestrictionMap(c.d_v, c.d_e, c.rm_init)
        elif c.rm_sharing == "sudoku":
            self.rm = SudokuRestrictionMaps(c.d_v, c.d_e, c.rm_init)
        else:
            raise ValueError(f"unknown rm_sharing={c.rm_sharing!r}")
        self.rho_raw = nn.Parameter(torch.zeros(()), requires_grad=c.rho_learnable)
        self.eta_raw = (
            nn.Parameter(torch.zeros(()), requires_grad=c.eta_learnable)
            if c.z_solver == "gd"
            else None
        )
        dec_kwargs = {"comm_dim": c.d_v, "input_shape": input_shape}
        if c.decoder_arch == "mlp_concat_v2":
            shape = (
                (*input_shape[:-1], c.num_classes)
                if input_shape is not None
                else (3, 3, c.num_classes)
            )
            dec_kwargs.update(
                hidden_dim=c.dec_hidden_dim, output_shape=shape, dropout_rate=c.dropout_rate
            )
        elif c.decoder_arch == "sudoku":
            dec_kwargs = dict(
                hidden_dims=c.dec_hidden_dims,
                output_channels=c.num_classes,
                norm_type=c.dec_norm_type,
                comm_dim=c.d_v,
            )
        elif c.decoder_arch == "classification":
            dec_kwargs.update(
                hidden_dims=c.dec_hidden_dims,
                output_channels=c.num_classes,
                norm_type=c.dec_norm_type,
                linear_head=c.dec_linear_head,
                readout_mode=c.dec_readout_mode,
            )
        self.decoder = create_decoder(c.decoder_arch, **dec_kwargs)

    def _learned_scalar(self, raw, init):
        return F.softplus(raw + inverse_softplus(init))

    def _learned_eta(self):
        c = self.config
        if c.eta_min is not None and c.eta_max is not None:
            lo, hi = math.log(c.eta_min), math.log(c.eta_max)
            t0 = min(max((math.log(c.eta_init) - lo) / (hi - lo), 1e-6), 1 - 1e-6)
            raw = self.eta_raw + math.log(t0 / (1 - t0))
            return torch.exp(raw.sigmoid() * (hi - lo) + lo)
        return self._learned_scalar(self.eta_raw, c.eta_init)

    def _encode(self, patches, cell_ids, training):
        c = self.config
        n, b = patches.shape[:2]
        flat = patches.reshape(n * b, *patches.shape[2:])
        if c.encoder_arch == "sudoku":
            ids = (
                cell_ids[:, None, :].expand(n, b, -1).reshape(n * b, -1)
                if cell_ids is not None
                else None
            )
            out = dict(self.encoder(flat, ids, training=training))
        else:
            out = dict(self.encoder(flat, training=training))
        for key in _ARRAY_KEYS:
            val = out.get(key)
            if isinstance(val, torch.Tensor) and val.ndim and val.shape[0] == n * b:
                out[key] = val.reshape(n, b, *val.shape[1:])
        return out

    def _build_geometry(self, enc_out, edge_indices, node_positions, map_u, map_v):
        c = self.config
        if c.rm_sharing == "directional":
            base = build_directional_restriction_maps(
                self.rm(), edge_indices, node_positions, c.num_directions
            )
        elif c.rm_sharing == "global":
            base = build_shared_restriction_maps(self.rm(), edge_indices.shape[0])
        else:
            base = build_sudoku_restriction_maps(self.rm(), map_u, map_v)
        if c.rm_mode == "fixed":
            return FixedGeometry(edge_indices=edge_indices, restriction_maps=base)
        A, B, gate = enc_out["A"], enc_out["B"], enc_out.get("gate")
        if c.rm_sharing == "sudoku":
            return create_sudoku_lora_geometry(
                edge_indices, map_u, map_v, base, A, B, c.lora_alpha, gate=gate
            )
        return create_lora_geometry(
            edge_indices, node_positions, base, A, B, c.lora_alpha, c.num_directions, gate=gate
        )

    def _z_params(self):
        c = self.config
        if c.z_solver == "unrolled_cg":
            return UnrolledCGParams(
                mode=c.z_mode,
                gamma=c.gamma,
                num_iters=c.cg_iters,
                tikhonov_eps=c.tikhonov_eps,
                prox_init=c.prox_init,
            )
        if c.z_solver == "gd":
            return GDParams(
                mode=c.z_mode,
                gamma=c.gamma,
                num_steps=c.gd_steps,
                momentum=c.gd_momentum,
                eta=self._learned_eta(),
            )
        raise ValueError(f"unknown z_solver={c.z_solver!r}")

    def _setup_admm(self, patches, edge_indices, node_positions, map_u, map_v, cell_ids, training):
        c = self.config
        enc_out = self._encode(patches, cell_ids, training)
        geometry = self._build_geometry(enc_out, edge_indices, node_positions, map_u, map_v)
        rho = self._learned_scalar(self.rho_raw, c.rho_init)
        x_solver, x_params = make_x_solver(c.x_solver)
        z_solver, _ = make_z_solver(c.z_solver)
        z_init = enc_out["h"] if c.z_init == "h" else torch.zeros_like(enc_out["h"])
        return enc_out, geometry, rho, x_solver, x_params, z_solver, self._z_params(), z_init

    def _decode_window(self, x_window, patches, training):
        w, n, b = x_window.shape[:3]
        flat = patches.reshape(n * b, *patches.shape[2:])
        outs = []
        for step in range(w):
            logits = self.decoder(x_window[step].reshape(n * b, -1), flat, training=training)
            outs.append(logits.reshape(n, b, *logits.shape[1:]))
        return torch.stack(outs)

    def forward(
        self,
        patches,
        edge_indices,
        *,
        num_iters,
        loss_window=1,
        grad_window=None,
        node_positions=None,
        map_u=None,
        map_v=None,
        cell_ids=None,
        training=True,
        compile_steps=True,
    ):
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
        args = self._setup_admm(
            patches, edge_indices, node_positions, map_u, map_v, cell_ids, training
        )
        enc_out, geometry, rho, x_solver, x_params, z_solver, z_params, z_init = args
        final, window = run_admm(
            enc_out,
            geometry,
            x_solver,
            x_params,
            z_solver,
            z_params,
            rho,
            z_init,
            num_iters,
            relaxation_alpha=self.config.relaxation_alpha,
            loss_window=loss_window,
            grad_window=grad_window,
            compile_step=compile_steps,
        )
        return self._decode_window(window, patches, training), final, geometry

    def coordinate_history(
        self,
        patches,
        edge_indices,
        *,
        num_iters,
        node_positions=None,
        map_u=None,
        map_v=None,
        cell_ids=None,
        training=False,
    ):
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
        args = self._setup_admm(
            patches, edge_indices, node_positions, map_u, map_v, cell_ids, training
        )
        enc_out, geometry, rho, x_solver, x_params, z_solver, z_params, z_init = args
        final, history = run_admm_history(
            enc_out,
            geometry,
            x_solver,
            x_params,
            z_solver,
            z_params,
            rho,
            z_init,
            num_iters,
            relaxation_alpha=self.config.relaxation_alpha,
        )
        return history, self._decode_window(history.x, patches, training), final, geometry, rho
