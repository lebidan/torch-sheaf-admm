"""Compare Sheaf-ADMM and MPNN on Minesweeper, with the paper's criteria and figures.

Evaluates one or more Sheaf-ADMM and MPNN checkpoints (several checkpoints per
model = several seeds, reported as mean +/- std) on the training board size and
zero-shot on smaller / larger boards, for k = 1 .. --kmax iterations, and writes:

* ``comparison.pdf`` with
  - Table A  exact solve rate and per-cell accuracy per board size
    (paper Table 2 style), at each model's own evaluation K and at K_max;
  - Table B  K ablation on the training size (paper Table 3: K = 1, 2, 5, 10, 20, 30);
  - Fig. 6   size generalisation: solve rate and mean iterations-to-solve vs
    board size, dashed line at K_max;
  - Fig. 3   intermediate predictions, Sheaf-ADMM vs MPNN, one board per size
    (red = violated number, blue frame = cell changed since the previous panel);
  - Fig. 4   coordination dynamics (Sheaf-ADMM): per-agent primal ||x_i - z_i|| and
    dual rho ||dz_i|| residuals, their log-ratio, mean / max over agents;
  - Fig. 8   x_i (blue) vs z_i (red) trajectories, one panel per agent laid out by
    agent position, star = final state;
* GIFs (blog style): ``solve_<S>x<S>.gif`` (Sheaf-ADMM | MPNN side by side, one
  frame per iteration) for every size, and ``sheaf_xz_<S>x<S>.gif`` (board +
  per-agent x/z trajectories growing with k) for the training size and the
  largest size;
* ``summary.md`` / ``summary.json`` with every number of the tables.

Criteria (paper): a board counts as solved only if every hidden cell is
classified correctly (argmax mine / safe); cell accuracy is measured on hidden
cells (revealed cells are safe by definition). ``solved_topk`` (the m most likely
cells are mines, m = mine count) is reported as an extra column.

    python -m scripts.compare_minesweeper \\
        --sheaf outputs/2026-11-26/14-47-05/checkpoint_best.pt \\
        --mpnn  outputs/2026-11-26/15-23-39/checkpoint_best.pt

Paper protocol (``task: minesweeper_grid``, 3x3 agents every 2 cells around an
"outside the board" ring, as for the paper's Maze): every size with the parity of
the training board is cut like it (6x6 -> 8, 10, 12, ... ; 9x9 -> 11, 13, ...), and
the OOD test splits of the dataset (test_ood_1.5x / 2x / 4x) are used when their
size is asked for:

    python -m scripts.compare_minesweeper \
        --sheaf outputs/benchmark_grid_9x9/sheaf/seed42/checkpoint_best.pt ... \
        --mpnn  outputs/benchmark_grid_9x9/mpnn_pm_max/seed42/checkpoint_best.pt ... \
        --sizes 9 13 17 25 33 --kmax 100 --out-dir comparisons/grid_9x9

Sizes (``task: minesweeper``): agents are patch x patch blocks every ``stride`` cells, so only sizes
patch + stride*n give the same overlaps as in training (5x5 agents every 2
cells: 9, 11, 13, 15, 17, 19...; 4x4 agents: 4, 6, 8, 10, 12...). Other sizes
shift the last agent by one cell, an overlap never seen in training, and are
skipped unless --allow-irregular is given.

Boards: on the training size the dataset's test split is used; other sizes are
generated with a fixed seed at the training mine density (6x6 / 5 mines ->
4x4: 2, 5x5: 3, 9x9: 11, 12x12: 20 mines). A 4x4 board with 4x4 agents has a
single agent (no communication at all), which is reported as is.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

HIDDEN, MINE, NUMBER_OFFSET = 1, 2, 2
ABLATION_KS = (1, 2, 5, 10, 20, 30, 40, 50, 100)  # paper Table 3 (1..30) + longer horizons
COLORS = {"Sheaf-ADMM": "#1d4ed8", "MPNN": "#ea580c"}
NUMBER_COLORS = {
    1: "#1d4ed8", 2: "#15803d", 3: "#dc2626", 4: "#1e3a8a",
    5: "#7f1d1d", 6: "#0e7490", 7: "#111827", 8: "#6b7280",
}


# =============================================================================
# Metrics (numpy)
# =============================================================================


def neighbour_lists(size: int) -> list[list[int]]:
    return [
        [rr * size + cc
         for rr in range(max(0, r - 1), min(size, r + 2))
         for cc in range(max(0, c - 1), min(size, c + 2))
         if (rr, cc) != (r, c)]
        for r in range(size) for c in range(size)
    ]


def topk_prediction(prob: np.ndarray, hidden: np.ndarray, mines: np.ndarray) -> np.ndarray:
    """``[..., B, C]``: the m most likely hidden cells are mines (m = board's mine count)."""
    score = np.where(hidden, prob, -np.inf)
    order = np.argsort(-score, axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[-1]), axis=-1)
    return (rank < mines.sum(-1, keepdims=True)) & hidden


def board_metrics(probs: np.ndarray, inputs: np.ndarray, labels: np.ndarray) -> dict:
    """``probs``: ``[K, B, C]`` mine probability after k = 1..K iterations."""
    hidden = inputs == HIDDEN
    mines = labels == MINE
    pred = (probs > 0.5) & hidden
    right = (pred == mines) | ~hidden
    solved = right.all(-1)  # [K, B]
    cell_acc = ((pred == mines) & hidden).sum((1, 2)) / hidden.sum()
    topk = topk_prediction(probs, hidden[None], mines[None])
    solved_topk = ((topk == mines) | ~hidden).all(-1)
    # iterations-to-solve: first k from which the board stays solved up to K
    stays = np.flip(np.logical_and.accumulate(np.flip(solved, 0), 0), 0)
    first = np.where(stays.any(0), stays.argmax(0) + 1, 0)
    return {
        "solved": solved.mean(1),          # [K]
        "cell_acc": cell_acc,              # [K]
        "solved_topk": solved_topk.mean(1),  # [K]
        "iters_to_solve": first,           # [B], 0 = not solved at K
    }


# =============================================================================
# Model side (torch): runs on the user's machine
# =============================================================================


class LoadedModel:
    def __init__(self, path: str, device):
        import torch

        from sheaf_admm.models import model_config_from_dict
        from sheaf_admm.training import build_model, make_task

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.path = path
        self.cfg = ckpt["config"]
        if self.cfg.get("task") not in ("minesweeper", "minesweeper_grid"):
            raise ValueError(f"{path}: not a minesweeper checkpoint")
        self.model_type = self.cfg["model_type"]
        self.name = "Sheaf-ADMM" if self.model_type == "sheaf" else "MPNN"
        self.device = device
        self.task = make_task(self.cfg["task"], **self.cfg.get("task_cfg", {}))
        self.model = build_model(model_config_from_dict(self.cfg["model"]), self.model_type)
        self.model.to(device)
        self.params = ckpt.get("ema_state") or ckpt["model_state"]
        self._loaded = False
        t = self.cfg["training"]
        self.eval_k = int(t["K_eval"] if self.model_type == "sheaf" else t["mpnn_eval_rounds"])
        self.train_k = (
            f"K in [{t.get('train_iters_min')}, {t.get('K_train')}]"
            if self.model_type == "sheaf" and t.get("train_iters_dist") == "uniform"
            else f"K = {t['K_train'] if self.model_type == 'sheaf' else t['mpnn_train_rounds']}"
        )
        self.seed = t.get("seed")

    def _prepare(self, inputs, labels):
        from sheaf_admm.training.loop import move_to_device

        fwd, targets, aux = self.task.prepare({"inputs": inputs, "labels": labels})
        return move_to_device(fwd, self.device), move_to_device(targets, self.device), aux

    def num_agents(self, inputs, labels) -> int:
        fwd, _targets, _aux = self.task.prepare({"inputs": inputs[:1], "labels": labels[:1]})
        return int(fwd["patches"].shape[0])

    def _ensure_loaded(self, fwd):
        import torch

        from sheaf_admm.training.loop import _forward

        if self._loaded:
            return
        with torch.no_grad():  # materialize lazy layers, then load the weights
            _forward(self.model, fwd, n_iter=1, loss_window=1, model_type=self.model_type,
                     training=False, compile_steps=False)
        self.model.load_state_dict(self.params)
        self.model.eval()
        self._loaded = True
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def mine_probs(self, inputs, labels, kmax: int, chunk: int = 128) -> np.ndarray:
        """``[kmax, B, cells]`` mine probability after k = 1..kmax iterations."""
        import torch

        out = []
        for start in range(0, len(inputs), chunk):
            fwd, targets, aux = self._prepare(inputs[start:start + chunk], labels[start:start + chunk])
            self._ensure_loaded(fwd)
            per_k = []
            with torch.no_grad():
                if self.model_type == "sheaf":
                    _hist, logits, *_ = self.model.coordinate_history(
                        fwd["patches"], fwd["edge_indices"], num_iters=kmax,
                        **fwd["model_kwargs"], training=False)
                else:  # one pass, logits after every round
                    logits, _ = self.model(fwd["patches"], fwd["edge_indices"], num_rounds=kmax,
                                           **fwd["model_kwargs"], training=False,
                                           compile_steps=False, return_history=True)
                for k in range(kmax):  # logits: [K, agents, B, ..., 2]
                    grid = self.task.reassemble(logits[k], targets, aux)
                    per_k.append(grid.softmax(-1)[..., 1].cpu().numpy())
            out.append(np.stack(per_k))
        return np.concatenate(out, axis=1)

    def trajectory(self, inputs, labels, kmax: int) -> dict:
        """ADMM internals for one board (Sheaf-ADMM only)."""
        import torch

        fwd, targets, aux = self._prepare(inputs[None], labels[None])
        self._ensure_loaded(fwd)
        with torch.no_grad():
            hist, *_ = self.model.coordinate_history(
                fwd["patches"], fwd["edge_indices"], num_iters=kmax,
                **fwd["model_kwargs"], training=False)
        return {
            "x": hist.x[:, :, 0].cpu().numpy(),            # [K, N, d_v]
            "z": hist.z[:, :, 0].cpu().numpy(),
            "primal": hist.primal_res[:, :, 0].cpu().numpy(),  # [K, N]
            "dual": hist.dual_res[:, :, 0].cpu().numpy(),
            "agent_rc": np.asarray(self.task.agent_positions(targets, aux)),
        }


def load_boards(size, mines, count, seed, test_dir=None):
    """Boards of a dataset split of this size when there is one (``test`` first, then
    the OOD splits ``test_ood_*``), else boards generated with a fixed seed."""
    if test_dir is not None:
        root = Path(test_dir)
        splits = ["test"] + sorted(p.name for p in root.glob("test_ood_*") if p.is_dir())
        for split in splits:
            if not (root / split / "all__inputs.npy").exists():
                continue
            x = np.load(root / split / "all__inputs.npy", mmap_mode="r")
            if x.shape[1] != size * size:
                continue
            y = np.load(root / split / "all__labels.npy", mmap_mode="r")
            x, y = np.asarray(x[:count]), np.asarray(y[:count])
            return x.astype(np.int64), y.astype(np.int64), f"split {split} of {test_dir}"
    from sheaf_admm.data.build_minesweeper import generate_board, neighbours

    rng = np.random.default_rng(seed + 1000 * size)
    nbrs = neighbours(size)
    safe = size * size - mines
    boards = []
    while len(boards) < count:
        board = generate_board(rng, size, mines, int(rng.integers(max(1, int(0.3 * safe)), safe + 1)), nbrs)
        if board is not None:
            boards.append(board)
    x, y = map(np.stack, zip(*boards, strict=True))
    return x.astype(np.int64), y.astype(np.int64), f"generated (seed {seed})"


# =============================================================================
# Drawing
# =============================================================================


def draw_board(ax, inputs, size, prob=None, prev_pred=None, title="", show_mines=None):
    """Paper Fig. 3 style for Minesweeper.

    Hidden cells: white (safe) -> red (mine) by predicted probability, black dot
    = predicted mine (argmax). Revealed numbers with a red frame are violated
    (predicted mines around them != number). Blue frame = cell whose prediction
    changed since ``prev_pred``.
    """
    from matplotlib.patches import Circle, Rectangle

    ax.clear()
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    hidden = inputs == HIDDEN
    pred = None if prob is None else (prob > 0.5) & hidden
    nbrs = neighbour_lists(size)
    lw = max(0.3, 2.4 - size / 8)
    fs = max(4, int(70 / size))
    red = np.array([0.86, 0.15, 0.15])
    for i in range(size * size):
        r, c = divmod(i, size)
        if hidden[i]:
            if prob is None:
                face = "#64748b"
            else:  # float32 rounding can push a channel a hair above 1
                p_i = float(np.clip(prob[i], 0.0, 1.0))
                face = tuple(np.clip((1 - p_i) * np.ones(3) + p_i * red, 0.0, 1.0))
        else:
            face = "#cbd5e1"
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face, edgecolor="#374151", lw=lw * 0.4))
        if not hidden[i] and inputs[i] > NUMBER_OFFSET:
            n = int(inputs[i]) - NUMBER_OFFSET
            ax.text(c + .5, r + .54, str(n), ha="center", va="center", fontsize=fs,
                    fontweight="bold", color=NUMBER_COLORS[n])
        if pred is not None and not hidden[i]:
            n = int(inputs[i]) - NUMBER_OFFSET
            if int(pred[nbrs[i]].sum()) != n:
                ax.add_patch(Rectangle((c + .06, r + .06), .88, .88, fill=False,
                                       edgecolor="#dc2626", lw=lw * 1.1))
        if pred is not None and pred[i]:
            ax.add_patch(Circle((c + .5, r + .5), .2, color="#111827"))
        if show_mines is not None and show_mines[i]:
            ax.add_patch(Circle((c + .5, r + .5), .2, color="#111827"))
        if prev_pred is not None and pred is not None and hidden[i] and pred[i] != prev_pred[i]:
            ax.add_patch(Rectangle((c + .04, r + .04), .92, .92, fill=False,
                                   edgecolor="#2563eb", lw=lw * 1.3))
    ax.set_title(title, fontsize=8)


def board_status(prob, inputs, labels) -> str:
    hidden = inputs == HIDDEN
    mines = labels == MINE
    errors = int((((prob > 0.5) & hidden) != mines)[hidden].sum())
    return "résolu" if errors == 0 else f"{errors} erreur{'s' if errors > 1 else ''}"


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std()) if len(v) > 1 else 0.0


def fmt(values, pct=True):
    m, s = mean_std(values)
    scale = 100 if pct else 1
    return f"{m * scale:.1f} ± {s * scale:.1f}" if len(values) > 1 else f"{m * scale:.1f}"


def pca2(points):
    mean = points.mean(0)
    _u, _s, vt = np.linalg.svd(points - mean, full_matrices=False)
    return mean, vt[:2].T


def project_xz(traj):
    k, n, d = traj["x"].shape
    stacked = np.concatenate([traj["x"].reshape(-1, d), traj["z"].reshape(-1, d)])
    mean, comps = pca2(stacked)
    x2 = ((traj["x"].reshape(-1, d) - mean) @ comps).reshape(k, n, 2)
    z2 = ((traj["z"].reshape(-1, d) - mean) @ comps).reshape(k, n, 2)
    return x2, z2


def draw_xz_grid(fig, gs, traj, upto, size):
    """Paper Fig. 8: one panel per agent laid out by agent position."""
    x2, z2 = traj["xz2"]
    rows = sorted(set(traj["agent_rc"][:, 0].tolist()))
    cols = sorted(set(traj["agent_rc"][:, 1].tolist()))
    allpts = np.concatenate([x2.reshape(-1, 2), z2.reshape(-1, 2)])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.06 * np.maximum(hi - lo, 1e-6)
    sub = gs.subgridspec(len(rows), len(cols), wspace=0.08, hspace=0.08)
    for a, (r0, c0) in enumerate(traj["agent_rc"]):
        ax = fig.add_subplot(sub[rows.index(r0), cols.index(c0)])
        ax.plot(x2[:upto, a, 0], x2[:upto, a, 1], color="#1d4ed8", lw=0.9, marker="o", ms=1.5)
        ax.plot(z2[:upto, a, 0], z2[:upto, a, 1], color="#dc2626", lw=0.9, marker="o", ms=1.5)
        ax.scatter(*x2[upto - 1, a], marker="*", s=40, color="#1d4ed8", zorder=4)
        ax.scatter(*z2[upto - 1, a], marker="*", s=40, color="#dc2626", zorder=4)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"agent ({r0},{c0})", fontsize=max(5, 11 - len(traj["agent_rc"]) // 3), pad=1)


# =============================================================================
# Report
# =============================================================================


def build_report(res, meta, out_dir: Path, fps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm

    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = meta["sizes"]
    names = meta["models"]
    kmax = meta["kmax"]
    train = meta["train_size"]
    ks = [k for k in ABLATION_KS if k <= kmax]
    summary = {"meta": meta, "table_A": {}, "table_B": {}, "size_generalisation": {}}

    def per_seed(name, size, key, k):
        return [run["metrics"][size][key][k - 1] for run in res[name] if size in run["metrics"]]

    with PdfPages(out_dir / "comparison.pdf") as pdf:
        # ---- page 1: setup ---------------------------------------------------
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(.5, .93, "Démineur : " + " vs ".join(names), ha="center", fontsize=20, fontweight="bold")
        lines = [f"Entraînement : plateaux {train}×{train}, {meta['train_mines']} mines.  "
                 f"Tests : {', '.join(f'{s}×{s}' for s in sizes)} ({meta['boards']} plateaux par taille, "
                 f"même densité de mines).", ""]
        for name in names:
            r0 = res[name][0]
            lines.append(f"{name} : {len(res[name])} seed(s) {[r['seed'] for r in res[name]]}, "
                         f"{r0['num_params']:,} paramètres, entraîné avec {r0['train_k']}, "
                         f"évalué à K = {r0['eval_k']}".replace(",", " "))
        lines += ["", f"Découpage : {meta.get('layout', '')}.  Agents par taille : "
                  + ", ".join(f"{s}×{s} : {meta['agents'].get(str(s), '?')}" for s in sizes) + "."]
        lines += ["", "Critères (article) : un plateau est résolu si toutes les cases cachées sont bien",
                  "classées (argmax mine / sûre) ; précision par case sur les cases cachées.",
                  "Moyenne ± écart-type sur les seeds.  solved_topk : les m cases les plus probables sont",
                  f"des mines (m = nombre de mines).  K_max = {kmax}.", "",
                  "Plateaux : " + "; ".join(f"{s}×{s} : {meta['mines'][str(s)]} mines, {meta['sources'][str(s)]}"
                                           for s in sizes)]
        fig.text(.07, .84, "\n".join(lines), va="top", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Table A -----------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
        for ax, mode in zip(axes, ("eval", "kmax"), strict=True):
            ax.axis("off")
            header = ["taille"]
            for name in names:
                header += [f"{name}\nrésolus %", f"{name}\nprécision %", f"{name}\nsolved_topk %"]
            rows = []
            for s in sizes:
                row = [f"{s}×{s}" + (" (entr.)" if s == train else "")]
                for name in names:
                    k = res[name][0]["eval_k"] if mode == "eval" else kmax
                    k = min(k, kmax)
                    vals = {key: per_seed(name, s, key, k) for key in ("solved", "cell_acc", "solved_topk")}
                    if not vals["solved"]:
                        row += ["n/a"] * 3
                        continue
                    row += [fmt(vals["solved"]), fmt(vals["cell_acc"]), fmt(vals["solved_topk"])]
                    summary["table_A"].setdefault(mode, {}).setdefault(name, {})[f"{s}x{s}"] = {
                        "k": k, **{key: mean_std(v) for key, v in vals.items()}}
                rows.append(row)
            table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.6)
            ax.set_title("Tableau A — " + ("à la valeur de K d'évaluation de chaque modèle"
                                           if mode == "eval" else f"à K = K_max = {kmax}"), fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Table B + curve: K ablation on the training size ----------------
        fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(11.69, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
        ax_t.axis("off")
        rows = []
        for k in ks:
            row = [str(k)]
            for name in names:
                v = per_seed(name, train, "solved", k)
                row.append(fmt(v) if v else "n/a")
                if v:
                    summary["table_B"].setdefault(name, {})[str(k)] = mean_std(v)
            rows.append(row)
        table = ax_t.table(cellText=rows, colLabels=["K"] + [f"{n}\nrésolus %" for n in names],
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        ax_t.set_title(f"Tableau B — ablation sur K ({train}×{train})", fontsize=12)
        kk = np.arange(1, kmax + 1)
        for name in names:
            curves = np.array([run["metrics"][train]["solved"] for run in res[name]]) * 100
            m, s = curves.mean(0), curves.std(0)
            color = COLORS.get(name, COLORS["MPNN"])
            ax_c.plot(kk, m, color=color, label=name)
            ax_c.fill_between(kk, m - s, m + s, color=color, alpha=0.2)
            ax_c.axvline(res[name][0]["eval_k"], color=color, ls=":", lw=1)
        ax_c.set_xlabel("itérations k (ADMM / tours de messages)")
        ax_c.set_ylabel("plateaux résolus (%)")
        ax_c.set_ylim(0, 102)
        ax_c.grid(alpha=0.3)
        ax_c.legend(fontsize=9)
        ax_c.set_title("pointillés : K d'évaluation de chaque modèle", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Fig 6: size generalisation ------------------------------------------
        fig, (ax_s, ax_i) = plt.subplots(1, 2, figsize=(11.69, 5.2))
        for name in names:
            sol, sol_eval, its = [], [], []
            k_eval = min(res[name][0]["eval_k"], kmax)
            for s in sizes:
                v = per_seed(name, s, "solved", kmax)
                sol.append(mean_std(v) if v else (np.nan, 0))
                v = per_seed(name, s, "solved", k_eval)
                sol_eval.append(mean_std(v) if v else (np.nan, 0))
                firsts = np.concatenate([run["metrics"][s]["iters_to_solve"] for run in res[name]
                                         if s in run["metrics"]] or [np.array([0])])
                solved_ones = firsts[firsts > 0]
                its.append((solved_ones.mean(), solved_ones.std()) if solved_ones.size else (np.nan, 0))
                summary["size_generalisation"].setdefault(name, {})[f"{s}x{s}"] = {
                    "solved_at_eval_k": sol_eval[-1], "solved_at_kmax": sol[-1],
                    "iters_to_solve": its[-1]}
            sol, sol_eval, its = np.array(sol), np.array(sol_eval), np.array(its)
            color = COLORS.get(name, COLORS["MPNN"])
            ax_s.errorbar(sizes, sol_eval[:, 0] * 100, yerr=sol_eval[:, 1] * 100, marker="o", capsize=3,
                          color=color, label=f"{name}, K = {k_eval}")
            if k_eval != kmax:
                ax_s.errorbar(sizes, sol[:, 0] * 100, yerr=sol[:, 1] * 100, marker="s", capsize=3,
                              color=color, ls="--", alpha=0.6, label=f"{name}, K_max = {kmax}")
            ax_i.errorbar(sizes, its[:, 0], yerr=its[:, 1], marker="o", capsize=3,
                          color=color, label=name)
        for ax in (ax_s, ax_i):
            ax.axvline(train, color="#6b7280", ls=":", lw=1)
            ax.set_xlabel("taille du plateau")
            ax.set_xticks(sizes)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        ax_s.set_ylabel("plateaux résolus (%)")
        ax_s.set_ylim(0, 102)
        ax_i.axhline(kmax, color="#111827", ls="--", lw=1)
        ax_i.set_ylabel("itérations pour résoudre (plateaux résolus)")
        fig.suptitle("Figure 6 — généralisation en taille. Gauche : trait plein = K d'évaluation de chaque "
                     "modèle, tirets = K_max ; pointillé vertical = taille d'entraînement", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Fig 3: intermediate predictions, one page per size ----------------
        shown = [k for k in (1, 3, 5, 10, 30, 100) if k <= kmax] or [kmax]
        for s in sizes:
            ex = meta["examples"][str(s)]
            inputs, labels = np.asarray(ex["inputs"]), np.asarray(ex["labels"])
            fig, axes = plt.subplots(len(names), len(shown) + 1,
                                     figsize=(2.3 * (len(shown) + 1), 2.6 * len(names) + 0.8), squeeze=False)
            for r, name in enumerate(names):
                probs = res[name][0]["examples"].get(s)
                draw_board(axes[r][0], inputs, s, show_mines=labels == MINE, title=f"{name}\nsolution")
                prev = None
                for c, k in enumerate(shown, start=1):
                    if probs is None:
                        axes[r][c].axis("off")
                        continue
                    p = probs[k - 1]
                    draw_board(axes[r][c], inputs, s, prob=p, prev_pred=prev,
                               title=f"k = {k} : {board_status(p, inputs, labels)}")
                    prev = (p > 0.5) & (inputs == HIDDEN)
            fig.suptitle(f"Figure 3 — prédictions intermédiaires, plateau {s}×{s}"
                         + (" (taille d'entraînement)" if s == train else " (hors distribution)")
                         + "\nrouge : chiffre violé ; cadre bleu : case changée depuis le panneau précédent ;"
                           " ● : mine prédite", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            pdf.savefig(fig)
            plt.close(fig)

        # ---- Fig 4 / Fig 8: Sheaf-ADMM internals -----------------------------------
        for s, traj in meta.get("trajectories", {}).items():
            primal, dual = traj["primal"], traj["dual"]
            fig = plt.figure(figsize=(11.69, 7.5))
            gs = fig.add_gridspec(2, 3)
            eps = 1e-12
            frac_p = primal / (primal.sum(1, keepdims=True) + eps)
            frac_d = dual / (dual.sum(1, keepdims=True) + eps)
            for i, (data, ttl, cmap) in enumerate([
                (frac_p, "résidu primal ||x_i − z_i|| (part du total)", "viridis"),
                (frac_d, "résidu dual ρ||Δz_i|| (part du total)", "magma"),
            ]):
                ax = fig.add_subplot(gs[0, i])
                im = ax.imshow(data.T, aspect="auto", cmap=cmap, origin="lower", interpolation="nearest")
                ax.set_xlabel("itération k")
                ax.set_ylabel("agent i")
                ax.set_title(ttl, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
            ax = fig.add_subplot(gs[0, 2])
            ratio = np.log10((primal + eps) / (dual + eps))
            lim = np.nanmax(np.abs(ratio[np.isfinite(ratio)])) if np.isfinite(ratio).any() else 1
            im = ax.imshow(ratio.T, aspect="auto", cmap="coolwarm", origin="lower",
                           vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_title("log10(primal / dual)", fontsize=9)
            ax.set_xlabel("itération k")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax = fig.add_subplot(gs[1, :])
            kk = np.arange(1, primal.shape[0] + 1)
            for data, color, label in ((primal, "#1d4ed8", "primal"), (dual, "#dc2626", "dual")):
                ax.plot(kk, data.mean(1), color=color, label=f"{label} (moyenne)")
                ax.plot(kk, data.max(1), color=color, ls="--", alpha=0.6, label=f"{label} (max)")
            pos = np.concatenate([primal.ravel(), dual.ravel()])
            if (pos > 0).any():
                ax.set_yscale("log")
            ax.set_xlabel("itération k")
            ax.set_ylabel("résidu")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8, ncol=4)
            fig.suptitle(f"Figure 4 — dynamique de coordination Sheaf-ADMM, plateau {s}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(figsize=(8.27, 8.27))
            gs = fig.add_gridspec(1, 1, left=0.04, right=0.96, bottom=0.07, top=0.9)
            draw_xz_grid(fig, gs[0, 0], traj, primal.shape[0], int(s.split("x")[0]))
            fig.suptitle(f"Figure 8 — trajectoires x_i (bleu, décision locale) et z_i (rouge, consensus),\n"
                         f"plateau {s}, projection ACP commune, ★ = état final", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

    # ---- GIFs -----------------------------------------------------------------
    for s in sizes:
        ex = meta["examples"][str(s)]
        inputs, labels = np.asarray(ex["inputs"]), np.asarray(ex["labels"])
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.4 * (len(names) + 1), 3.8))
        draw_board(axes[0], inputs, s, show_mines=labels == MINE, title="solution")

        def frame(i, axes=axes, inputs=inputs, labels=labels, s=s):
            for ax, name in zip(axes[1:], names, strict=True):
                probs = res[name][0]["examples"].get(s)
                if probs is None:
                    ax.axis("off")
                    continue
                prev = (probs[i - 1] > 0.5) & (inputs == HIDDEN) if i > 0 else None
                draw_board(ax, inputs, s, prob=probs[i], prev_pred=prev,
                           title=f"{name}, k = {i + 1} : {board_status(probs[i], inputs, labels)}")
            fig.suptitle(f"Démineur {s}×{s}" + (" (taille d'entraînement)" if s == train else " (hors distribution)"),
                         fontsize=11)
            return []

        anim = FuncAnimation(fig, frame, frames=kmax, blit=False)
        anim.save(out_dir / f"solve_{s}x{s}.gif", writer=PillowWriter(fps=fps))
        plt.close(fig)

    for s, traj in meta.get("trajectories", {}).items():
        size = int(s.split("x")[0])
        ex = meta["examples"][str(size)]
        inputs, labels = np.asarray(ex["inputs"]), np.asarray(ex["labels"])
        probs = res["Sheaf-ADMM"][0]["examples"][size]
        k_traj = traj["primal"].shape[0]
        fig = plt.figure(figsize=(11, 5.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.03, right=0.98, top=0.84, bottom=0.07)

        def xz_frame(i, fig=fig, gs=gs, inputs=inputs, labels=labels, probs=probs, traj=traj, size=size):
            fig.clf()
            ax = fig.add_subplot(gs[0, 0])
            draw_board(ax, inputs, size, prob=probs[i],
                       title=f"k = {i + 1} : {board_status(probs[i], inputs, labels)}")
            draw_xz_grid(fig, gs[0, 1], traj, i + 1, size)
            fig.suptitle(f"Sheaf-ADMM {size}×{size} : x_i (bleu) et z_i (rouge) de chaque agent se rejoignent "
                         "au consensus", fontsize=10)
            return []

        anim = FuncAnimation(fig, xz_frame, frames=min(k_traj, kmax), blit=False)
        anim.save(out_dir / f"sheaf_xz_{s}.gif", writer=PillowWriter(fps=fps))
        plt.close(fig)

    # ---- summary files --------------------------------------------------------------
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in ("examples", "trajectories")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and math.isnan(o):
            return None
        return o

    (out_dir / "summary.json").write_text(json.dumps(clean(summary), indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    md = ["# Démineur : " + " vs ".join(names), "", "## Tableau A (K d'évaluation de chaque modèle)", "",
          "| taille | " + " | ".join(f"{n} résolus % | {n} précision %" for n in names) + " |",
          "|---|" + "---|---|" * len(names)]
    for s in sizes:
        row = [f"{s}×{s}"]
        for name in names:
            k = min(res[name][0]["eval_k"], kmax)
            v1, v2 = per_seed(name, s, "solved", k), per_seed(name, s, "cell_acc", k)
            row += [fmt(v1) if v1 else "n/a", fmt(v2) if v2 else "n/a"]
        md.append("| " + " | ".join(row) + " |")
    md += ["", f"## Tableau B (ablation sur K, {train}×{train})", "",
           "| K | " + " | ".join(f"{n} résolus %" for n in names) + " |", "|---|" + "---|" * len(names)]
    for k in ks:
        md.append("| " + " | ".join([str(k)] + [fmt(per_seed(n, train, "solved", k)) if per_seed(n, train, "solved", k)
                                               else "n/a" for n in names]) + " |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheaf", nargs="+", required=True, help="Sheaf-ADMM checkpoint(s), one per seed")
    p.add_argument("--mpnn-label", default="MPNN",
                   help='name of the MPNN variant in tables / figures, e.g. "MPNN (PM, mean)"')
    p.add_argument("--mpnn", nargs="*", default=[],
                   help="MPNN checkpoint(s), one per seed (optional: Sheaf-ADMM alone otherwise)")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 9, 12])
    p.add_argument("--boards", type=int, default=500, help="boards per size")
    p.add_argument("--kmax", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="seed of the generated boards")
    p.add_argument("--example", type=int, default=0, help="index of the board shown in figures / GIFs")
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--allow-irregular", action="store_true",
                   help="also test sizes where the last agent overlaps its neighbour more than usual")
    p.add_argument("--max-traj-agents", type=int, default=100,
                   help="Fig. 4 / 8 and sheaf_xz GIF: largest board with at most this many agents")
    p.add_argument("--out-dir", default="comparisons/minesweeper")
    args = p.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {"Sheaf-ADMM": [LoadedModel(c, device) for c in args.sheaf],
            args.mpnn_label: [LoadedModel(c, device) for c in args.mpnn]}
    runs = {name: models for name, models in runs.items() if models}
    ref = runs["Sheaf-ADMM"][0]
    data_dir = Path(ref.cfg["data"]["dir"])
    stored = json.loads((data_dir / "config.json").read_text()) if (data_dir / "config.json").exists() else {}
    train = int(stored.get("size", 6))
    train_mines = int(stored.get("mines", 5))
    density = train_mines / train**2
    sizes = sorted(set(args.sizes) | {train})
    task_cfg = ref.cfg.get("task_cfg", {})
    grid_layout = ref.cfg["task"] == "minesweeper_grid"
    patch = int(task_cfg.get("patch_size", 3 if grid_layout else 5))
    stride = int(task_cfg.get("stride", 2))
    border = int(task_cfg.get("border", 1)) if grid_layout else 0
    if not grid_layout:
        sizes = [s for s in sizes if s >= patch]
    if not args.allow_irregular:
        # Slot layout: (size - patch) not a multiple of stride -> the last agent is shifted
        # by less than the stride (unseen overlap). Grid layout: a size of the other parity
        # than the training board changes how the border agents are cut (exact tiling for
        # odd sizes, a double ring of padding on two sides for even sizes).
        if grid_layout:  # same parity as the training board -> same cut of the border agents
            irregular = [s for s in sizes if (s - train) % stride]
        else:
            irregular = [s for s in sizes if (s - patch) % stride]
        if irregular:
            print(f"skipping sizes {irregular}: they are not cut like the training board "
                  f"({patch}x{patch} agents every {stride} cells"
                  + (f", {border}-cell outside ring: keep the parity of {train}" if grid_layout else "")
                  + "). Use --allow-irregular to keep them.")
        sizes = [s for s in sizes if s not in irregular]

    meta = {"sizes": sizes, "models": list(runs), "kmax": args.kmax, "train_size": train,
            "train_mines": train_mines, "boards": args.boards, "mines": {}, "sources": {},
            "examples": {}, "trajectories": {}, "agents": {},
            "layout": (f"agents {patch}×{patch} tous les {stride} (protocole labyrinthe, anneau hors plateau)"
                       if grid_layout else f"agents {patch}×{patch} tous les {stride}")}
    res = {name: [{"seed": m.seed, "train_k": m.train_k, "eval_k": m.eval_k, "metrics": {}, "examples": {}}
                  for m in models] for name, models in runs.items()}
    for s in sizes:
        mines = train_mines if s == train else max(1, round(density * s * s))
        x, y, source = load_boards(s, mines, args.boards, args.seed, data_dir)
        meta["mines"][str(s)] = mines
        meta["sources"][str(s)] = source
        meta["agents"][str(s)] = ref.num_agents(x, y)
        ex = min(args.example, len(x) - 1)
        meta["examples"][str(s)] = {"inputs": x[ex].tolist(), "labels": y[ex].tolist()}
        for name, models in runs.items():
            for i, m in enumerate(models):
                t0 = time.time()
                try:
                    probs = m.mine_probs(x, y, args.kmax)
                except Exception as err:  # e.g. a single-agent board without edges
                    print(f"[{name} seed {m.seed}] {s}x{s}: skipped ({type(err).__name__}: {err})")
                    continue
                res[name][i]["metrics"][s] = board_metrics(probs, x, y)
                res[name][i]["examples"][s] = probs[:, ex]
                res[name][i]["num_params"] = m.num_params
                k = min(m.eval_k, args.kmax)
                print(f"[{name} seed {m.seed}] {s}x{s}: solved@K={k} "
                      f"{res[name][i]['metrics'][s]['solved'][k - 1] * 100:.1f}%  "
                      f"solved@{args.kmax} {res[name][i]['metrics'][s]['solved'][-1] * 100:.1f}%  "
                      f"({time.time() - t0:.1f}s)")
        # Fig. 4 / 8: training size + the largest size with a readable number of agents
        largest = max((t for t in sizes if t <= s and meta["agents"].get(str(t), 10**9) <= args.max_traj_agents),
                      default=train)
        if s == train or (s == max(sizes) and largest != train):
            if s != train:
                # same deterministic boards as the ones evaluated for that size
                sx, sy = (x, y) if largest == s else load_boards(
                    largest, meta["mines"][str(largest)], args.example + 1, args.seed, data_dir)[:2]
                s_traj = largest
            else:
                sx, sy, s_traj = x, y, s
            exi = min(args.example, len(sx) - 1)
            traj = ref.trajectory(sx[exi], sy[exi], args.kmax)
            traj["xz2"] = project_xz(traj)
            meta["trajectories"][f"{s_traj}x{s_traj}"] = traj
    for name in res:
        for r in res[name]:
            r.setdefault("num_params", 0)

    out_dir = Path(args.out_dir)
    build_report(res, meta, out_dir, args.fps)
    print(f"Written to {out_dir}: comparison.pdf, summary.md, summary.json, "
          + ", ".join(f"solve_{s}x{s}.gif" for s in sizes) + ", "
          + ", ".join(f"sheaf_xz_{k}.gif" for k in meta["trajectories"]))


if __name__ == "__main__":
    main()
