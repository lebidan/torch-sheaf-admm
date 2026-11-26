"""Watch a trained Minesweeper model solve boards, iteration by iteration.

Works for Sheaf-ADMM checkpoints (k = ADMM iterations) and MPNN baselines
(k = message-passing rounds).

For every board, the model is run with k = 1, 2, ... ADMM iterations. Each panel
shows, on the hidden cells, the model's mine probability (white = safe, red =
mine) and the decoded answer (the ``m`` most likely cells are mines, ``m`` being
the board's mine count, as in the ``*_topk`` metrics).

    # PDF: one page per board + a summary curve
    python -m scripts.watch_minesweeper --checkpoint outputs/.../checkpoint_best.pt

    # live window, iteration by iteration (close it to stop)
    python -m scripts.watch_minesweeper --checkpoint outputs/.../checkpoint_best.pt --live

    # boards never seen before, freshly generated
    python -m scripts.watch_minesweeper --checkpoint ... --new --count 12

Markers: black dot = predicted mine (correct); red cross = predicted mine on a
safe cell; red dashed square = mine the model missed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

HIDDEN, MINE, NUMBER_OFFSET = 1, 2, 2
NUMBER_COLORS = {
    1: "#1d4ed8", 2: "#15803d", 3: "#dc2626", 4: "#1e3a8a",
    5: "#7f1d1d", 6: "#0e7490", 7: "#111827", 8: "#6b7280",
}


# ----------------------------------------------------------------------------
# Decoding and drawing (numpy / matplotlib only)
# ----------------------------------------------------------------------------


def decode_topk(prob: np.ndarray, hidden: np.ndarray, num_mines: int) -> np.ndarray:
    """The ``num_mines`` hidden cells with the highest mine probability."""
    score = np.where(hidden, prob, -np.inf)
    pred = np.zeros_like(hidden)
    pred[np.argsort(-score, kind="stable")[:num_mines]] = True
    return pred & hidden


def draw_board(ax, inputs, size, *, prob=None, pred=None, mines=None, title=""):
    """One board: numbers on revealed cells, mine probability on hidden cells."""
    from matplotlib.patches import Circle, Rectangle

    hidden = inputs == HIDDEN
    ax.clear()
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    red = np.array([0.86, 0.15, 0.15])
    for i in range(size * size):
        r, c = divmod(i, size)
        if hidden[i]:
            if prob is None:
                face = "#64748b"
            else:  # white (safe) -> red (mine)
                p = float(np.clip(prob[i], 0.0, 1.0))
                face = tuple((1 - p) * np.ones(3) + p * red)
        else:
            face = "#cbd5e1"  # revealed cells: slate, distinct from white (safe) hidden cells
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face, edgecolor="#374151", lw=0.8))
        if not hidden[i] and inputs[i] > NUMBER_OFFSET:
            n = int(inputs[i]) - NUMBER_OFFSET
            ax.text(c + 0.5, r + 0.53, str(n), ha="center", va="center",
                    fontsize=max(6, 90 // size), fontweight="bold", color=NUMBER_COLORS[n])
        if pred is not None and pred[i]:
            if mines is None or mines[i]:
                ax.add_patch(Circle((c + 0.5, r + 0.5), 0.22, color="#111827"))
            else:
                ax.plot([c + 0.25, c + 0.75], [r + 0.25, r + 0.75], color="#b91c1c", lw=2.5)
                ax.plot([c + 0.25, c + 0.75], [r + 0.75, r + 0.25], color="#b91c1c", lw=2.5)
        if mines is not None and mines[i] and pred is not None and not pred[i]:
            ax.add_patch(Rectangle((c + 0.1, r + 0.1), 0.8, 0.8, fill=False,
                                   edgecolor="#b91c1c", lw=2, ls="--"))
        if mines is not None and mines[i] and pred is None:
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.22, color="#111827"))
    ax.set_title(title, fontsize=9)


def board_errors(pred, mines, hidden) -> int:
    return int(((pred != mines) & hidden).sum())


def render_pdf(path: Path, boards, probs, ks, size, source: str) -> dict:
    """``boards``: (inputs, labels) arrays; ``probs``: [B, len(ks), cells]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    inputs_all, labels_all = boards
    num_boards = len(inputs_all)
    solved = np.zeros((num_boards, len(ks)), dtype=bool)
    for b in range(num_boards):
        hidden = inputs_all[b] == HIDDEN
        mines = labels_all[b] == MINE
        for j in range(len(ks)):
            pred = decode_topk(probs[b, j], hidden, int(mines.sum()))
            solved[b, j] = board_errors(pred, mines, hidden) == 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 5.5))
        ax.plot(ks, solved.mean(0) * 100, marker="o", color="#1d4ed8")
        ax.set_xlabel("itérations ADMM (k)")
        ax.set_ylabel("plateaux résolus (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        ax.set_title(f"{num_boards} plateaux {size}×{size} ({source}) : résolus en fonction de k")
        pdf.savefig(fig)
        plt.close(fig)

        ncols = 4
        panels = 2 + len(ks)
        nrows = -(-panels // ncols)
        for b in range(num_boards):
            inputs = inputs_all[b]
            hidden = inputs == HIDDEN
            mines = labels_all[b] == MINE
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 3.1 * nrows + 0.8))
            axes = np.atleast_1d(axes).reshape(-1)
            draw_board(axes[0], inputs, size, title=f"entrée ({int(hidden.sum())} cases cachées)")
            draw_board(axes[1], inputs, size, mines=mines, title=f"solution ({int(mines.sum())} mines)")
            for j, k in enumerate(ks):
                pred = decode_topk(probs[b, j], hidden, int(mines.sum()))
                errors = board_errors(pred, mines, hidden)
                status = "résolu" if errors == 0 else f"{errors} erreur{'s' if errors > 1 else ''}"
                draw_board(axes[2 + j], inputs, size, prob=probs[b, j], pred=pred, mines=mines,
                           title=f"k = {k} : {status}")
            for ax in axes[panels:]:
                ax.axis("off")
            first = next((ks[j] for j in range(len(ks)) if solved[b, j:].all()), None)
            verdict = f"résolu dès k = {first}" if first is not None else "non résolu au dernier k"
            fig.suptitle(f"Plateau {b + 1}/{num_boards} — {verdict}\n"
                         "cases grises = révélées ; couleur = probabilité de mine (blanc → rouge) ;  ● mine trouvée,  "
                         "✕ fausse mine,  carré pointillé = mine manquée", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig)
            plt.close(fig)
    return {"boards": num_boards, "solved_at_last_k": float(solved[:, -1].mean())}


def run_live(boards, probs, ks, size, delay: float) -> None:
    """Animate every board in a window, iteration by iteration."""
    import matplotlib.pyplot as plt

    inputs_all, labels_all = boards
    plt.ion()
    fig, (ax_model, ax_truth) = plt.subplots(1, 2, figsize=(10, 5.4))
    for b in range(len(inputs_all)):
        inputs = inputs_all[b]
        hidden = inputs == HIDDEN
        mines = labels_all[b] == MINE
        draw_board(ax_truth, inputs, size, mines=mines, title="solution")
        for j, k in enumerate(ks):
            if not plt.fignum_exists(fig.number):
                return
            pred = decode_topk(probs[b, j], hidden, int(mines.sum()))
            errors = board_errors(pred, mines, hidden)
            status = "résolu" if errors == 0 else f"{errors} erreur(s)"
            draw_board(ax_model, inputs, size, prob=probs[b, j], pred=pred, mines=mines,
                       title=f"modèle, itération k = {k} : {status}")
            fig.suptitle(f"Plateau {b + 1}/{len(inputs_all)}", fontsize=12)
            plt.pause(delay)
        plt.pause(delay * 2)
    plt.ioff()
    plt.show()


# ----------------------------------------------------------------------------
# Model side (torch)
# ----------------------------------------------------------------------------


def load_boards(args, cfg):
    if not args.new:
        from sheaf_admm.data import PuzzleDataset

        data_dir = args.data_dir or cfg["data"]["dir"]
        split = args.split or cfg["data"]["val_splits"][0]
        ds = PuzzleDataset(data_dir, split)
        rows_in, rows_lab = [], []
        for _set, batch in ds.iter_test_batches(batch_size=256):
            rows_in.append(np.asarray(batch["inputs"]))
            rows_lab.append(np.asarray(batch["labels"]))
            if sum(len(x) for x in rows_in) >= args.start + args.count:
                break
        inputs = np.concatenate(rows_in)[args.start : args.start + args.count]
        labels = np.concatenate(rows_lab)[args.start : args.start + args.count]
        size = int(round(inputs.shape[1] ** 0.5))
        return inputs, labels, size, f"test : {data_dir}"

    from sheaf_admm.data.build_minesweeper import generate_board, neighbours

    data_cfg = Path(args.data_dir or cfg["data"]["dir"]) / "config.json"
    stored = json.loads(data_cfg.read_text()) if data_cfg.exists() else {}
    size = args.size or stored.get("size", 9)
    mines = args.mines or stored.get("mines", 10)
    safe = size * size - mines
    lo = stored.get("hidden_safe_min") or max(1, int(0.3 * safe))
    hi = stored.get("hidden_safe_max") or safe
    rng = np.random.default_rng(args.seed)
    nbrs = neighbours(size)
    boards = []
    while len(boards) < args.count:
        board = generate_board(rng, size, mines, int(rng.integers(lo, hi + 1)), nbrs)
        if board is not None:
            boards.append(board)
    inputs, labels = map(np.stack, zip(*boards, strict=True))
    return inputs, labels, size, f"nouveaux plateaux, graine {args.seed}"


def mine_probabilities(args, ckpt, inputs, labels, ks) -> np.ndarray:
    """``[B, len(ks), cells]`` mine probability after k ADMM iterations."""
    import torch

    from sheaf_admm.data import views as V
    from sheaf_admm.models import model_config_from_dict
    from sheaf_admm.training import build_model, make_task
    from sheaf_admm.training.loop import _forward, move_to_device

    cfg = ckpt["config"]
    model_type = cfg["model_type"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = ckpt.get("ema_state") if args.params == "ema" else ckpt["model_state"]
    if params is None:
        params = ckpt["model_state"]
    task = make_task(cfg["task"], **cfg.get("task_cfg", {}))
    model = build_model(model_config_from_dict(cfg["model"]), cfg["model_type"]).to(device)
    fwd, targets, _aux = task.prepare({"inputs": inputs, "labels": labels})
    fwd, targets = move_to_device(fwd, device), move_to_device(targets, device)
    with torch.no_grad():  # materialize lazy layers before loading weights
        _forward(model, fwd, n_iter=1, loss_window=1, model_type=model_type, training=False)
    model.load_state_dict(params)
    model.eval()
    out = []
    with torch.no_grad():
        for k in ks:
            logits = _forward(model, fwd, n_iter=k, loss_window=1, model_type=model_type,
                              training=False)
            if model_type == "sheaf":
                logits = logits[-1]  # last ADMM iteration of the window
            grid = V.reassemble_minesweeper_logits(logits.permute(1, 0, 2, 3), targets["cell_ids"])
            out.append(grid.softmax(dim=-1)[..., 1].cpu().numpy())
    return np.stack(out, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--count", type=int, default=8, help="number of boards")
    p.add_argument("--start", type=int, default=0, help="first test board (test-set mode)")
    p.add_argument("--new", action="store_true", help="generate fresh boards instead of the test set")
    p.add_argument("--seed", type=int, default=1234, help="seed for --new")
    p.add_argument("--size", type=int, default=None, help="board size for --new (default: dataset)")
    p.add_argument("--mines", type=int, default=None, help="mine count for --new (default: dataset)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--ks", type=int, nargs="+", default=None,
                   help="iterations to show (default: 1 2 3 5 and K_eval)")
    p.add_argument("--params", choices=("ema", "raw"), default="ema")
    p.add_argument("--out", default=None, help="PDF path (default: next to the checkpoint)")
    p.add_argument("--live", action="store_true", help="animate in a window instead of a PDF")
    p.add_argument("--delay", type=float, default=0.6, help="seconds per iteration with --live")
    args = p.parse_args()

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("task") != "minesweeper":
        raise ValueError(f"expected a minesweeper checkpoint, got task={cfg.get('task')!r}")
    key = "K_eval" if cfg["model_type"] == "sheaf" else "mpnn_eval_rounds"
    k_eval = int(cfg["training"].get(key, 8))
    ks = sorted(set(args.ks or [1, 2, 3, 5, k_eval]))
    if args.live and not args.ks:
        ks = list(range(1, k_eval + 1))

    inputs, labels, size, source = load_boards(args, cfg)
    probs = mine_probabilities(args, ckpt, inputs, labels, ks)
    if args.live:
        run_live((inputs, labels), probs, ks, size, args.delay)
        return
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "viz" / "minesweeper_steps.pdf"
    summary = render_pdf(out, (inputs, labels), probs, ks, size, source)
    print(f"{summary['boards']} boards, {summary['solved_at_last_k'] * 100:.0f}% solved at k={ks[-1]}")
    print(f"PDF written to {out}")


if __name__ == "__main__":
    main()
