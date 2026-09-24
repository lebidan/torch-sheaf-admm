# Torch-Sheaf-ADMM: educational PyTorch port of Sheaf-ADMM

This repository is an **unofficial PyTorch port** for
students and educational projects, created with the help of **Codex (GPT-6)**.
Full credit for the
algorithm, paper, and original JAX implementation belongs to **Jeffrey Seely,
Bartłomiej Cupiał, and Llion Jones at Sakana AI**. See the
[original Sheaf-ADMM repository](https://github.com/SakanaAI/sheaf-admm) and
[paper](https://arxiv.org/abs/2605.31005). This port is independently
maintained and is not endorsed by the original authors.

![Sheaf-ADMM overview](assets/fig2.png)

*Figure from the original Sheaf-ADMM repository.*

[![arXiv](https://img.shields.io/badge/arXiv-2605.31005-b31b1b?style=flat-square)](https://arxiv.org/abs/2605.31005)
[![Blog](https://img.shields.io/badge/Blog-Sakana%20AI-1f6feb?style=flat-square)](https://pub.sakana.ai/sheaf-admm/)

Sheaf-ADMM decomposes an input into overlapping local views, each processed by an
agent that solves a small convex subproblem parameterized by a neural encoder.
Agents coordinate through the Alternating Direction Method of Multipliers (ADMM),
with the inter-agent constraints specified by a *cellular sheaf* — which aspects
of neighboring solutions must agree. The optimization is unrolled for a fixed
number of iterations, so the whole pipeline is differentiable and every component
is trained end-to-end.

## Installation

This repository is intended to be run from a source checkout. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
```

On Linux, the lockfile selects the PyTorch CUDA 13.0 wheel; training uses CUDA
when a compatible NVIDIA GPU is available and otherwise runs on CPU. JAX, Flax,
and Optax are available only for reference validation with `uv sync --extra reference`.
When switching an existing `.venv` from the old JAX CUDA 12 setup, use
`uv sync --locked --reinstall` once to restore CUDA libraries shared by the wheels.

## Data

```bash
uv run python -m sheaf_admm.data.build_maze \
  --height 19 \
  --width 19 \
  --train-size 10000 \
  --test-size 1000 \
  --min-path-length 18 \
  --ood-sizes \
  --output-dir datasets/maze_std3_19px_10k

uv run python -m sheaf_admm.data.build_mnist \
  --output-dir datasets/mnist

uv run python -m sheaf_admm.data.build_sudoku \
  --output-dir datasets/sudoku_easy
```

The MNIST and Sudoku builders download their source datasets on first run.

## Training

Each task has a Sheaf-ADMM model and a recurrent-MPNN baseline:

```bash
# Maze
uv run python scripts/train.py +experiment=maze_sheaf
uv run python scripts/train.py +experiment=maze_mpnn

# MNIST
uv run python scripts/train.py +experiment=mnist_sheaf
uv run python scripts/train.py +experiment=mnist_mpnn

# Sudoku
uv run python scripts/train.py +experiment=sudoku_sheaf
uv run python scripts/train.py +experiment=sudoku_sheaf_lora
uv run python scripts/train.py +experiment=sudoku_mpnn
```

Set `training.seed=42`, `123`, or `456` for the paper seeds. Set
`wandb.mode=online` to enable Weights & Biases logging. Checkpoints and
`history.json` are written to Hydra's run directory under `outputs/`.
The checkpoint is `checkpoint.pt` and contains model and EMA state dictionaries,
optimizer state, step, and the resolved config.

For the legacy Maze Sheaf `checkpoint.pkl` files produced by the JAX version,
install the reference extra and convert each trusted file offline:

```bash
uv sync --extra reference
uv run --extra reference python scripts/convert_checkpoint.py \
  path/to/checkpoint.pkl path/to/checkpoint.pt
```

## Visualization

The visualization script expects a Sheaf-ADMM checkpoint:

```bash
uv run python -m scripts.visualize \
  --checkpoint outputs/<date>/<time>/checkpoint.pt \
  --out-dir /tmp/sheaf_admm_viz
```

## Checks

```bash
uv run python -m ruff check .
uv run python -m pytest -q
uv run python validation/check_reference.py --device cpu
uv run python validation/check_lora_reference.py --device cpu
uv run python validation/smoke_all_configs.py --device cpu
uv run python validation/check_cli_tasks.py
```

With an NVIDIA GPU, repeat the three device-selectable validation commands with
`--device cuda`.
Run `uv run python -m pytest -q tests/test_compile_parity.py` to compare one
compiled and eager parameter update for each model family on CUDA.
The frozen JAX reference fixtures are bundled, so these checks use Torch only.

## Runtime on an RTX 4090

CUDA training and evaluation compile one ADMM iteration or one MPNN round with
`torch.compile` and reuse it throughout the loop. CPU runs use eager execution.
Set `TORCH_COMPILE_DISABLE=1` to run eagerly for debugging or a short smoke
test. The first call for a new model shape can spend tens of seconds compiling;
the timings below measure subsequent calls.

**Training step**

| Experiment | Batch | Iterations | JAX | PyTorch | Torch / JAX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Maze Sheaf | 128 | 40 | 67 ms | 73 ms | 1.09× |
| Maze MPNN | 128 | 40 | 71 ms | 40 ms | 0.56× |
| MNIST Sheaf | 128 | 20 | 145 ms | 169 ms | 1.16× |
| MNIST MPNN | 128 | 20 | 17 ms | 20 ms | 1.19× |
| Sudoku Sheaf | 32 | 20 | 71 ms | 73 ms | 1.03× |
| Sudoku Sheaf LoRA | 16 | 20 | 72 ms | 78 ms | 1.09× |
| Sudoku MPNN | 32 | 20 | 39 ms | 28 ms | 0.70× |

**Evaluation forward pass**

| Experiment | Batch | Iterations | JAX | PyTorch | Torch / JAX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Maze Sheaf | 128 | 100 | 50 ms | 85 ms | 1.70× |
| Maze MPNN | 128 | 100 | 41 ms | 30 ms | 0.72× |
| MNIST Sheaf | 128 | 100 | 289 ms | 284 ms | 0.98× |
| MNIST MPNN | 128 | 50 | 10 ms | 13 ms | 1.21× |
| Sudoku Sheaf | 32 | 50 | 46 ms | 60 ms | 1.30× |
| Sudoku Sheaf LoRA | 16 | 50 | 52 ms | 58 ms | 1.13× |
| Sudoku MPNN | 32 | 50 | 21 ms | 25 ms | 1.19× |

These are medians of five GPU-synchronized calls after two warmups on the same
RTX 4090, with JAX 0.10.1 and PyTorch 2.14.0+cu130. Training includes
synthetic batch preparation, device transfer, forward and backward passes, and
the optimizer update. Evaluation includes preparation, transfer, and the model
forward pass; it excludes task metrics and EMA swapping. Both exclude dataset
I/O and first compilation. Starting weights come from each framework's own
RNG, so this measures comparable work, not an identical trajectory. To repeat
a case from this checkout, run:

```bash
uv run python scripts/benchmark_runtime.py --framework torch \
  --experiment maze_sheaf --batch 128 --iters 40 --include-prep
```

The same script accepts `--framework jax` when run from the original JAX
checkout with its CUDA environment. Add `--phase eval --iters 100` for an
evaluation forward pass.

## How this port differs from the original JAX code

The model and ADMM update rules follow the original implementation. This port
uses PyTorch modules, autograd, optimizers, and CUDA tensors in place of JAX,
Flax, and Optax. It keeps the seven shipped experiment configurations and their
Hydra options. The Python import path remains `sheaf_admm`, while the project
and installable distribution are named `torch-sheaf-admm`.

New checkpoints use PyTorch `checkpoint.pt` state dictionaries. A conversion
script is provided for trusted legacy Maze Sheaf checkpoints, as described in
[Training](#training). Other legacy checkpoints need additional conversion work.

The checks above compare fixed JAX reference outputs with PyTorch on CPU and
CUDA, including model outputs, ADMM behavior, and LoRA restriction maps. They
also exercise all seven configurations. These checks establish parity for the
covered cases; separately initialized runs can differ because the frameworks
use different random number generators.

## Citation

If you use this port, please cite the original work:

```bibtex
@inproceedings{sheafadmm2026,
  title     = {Learning Multi-Agent Coordination via Sheaf-ADMM},
  author    = {Seely, Jeffrey and Cupia{\l}, Bart{\l}omiej and Jones, Llion},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

## License

This port retains the original repository's Apache 2.0 license in
[LICENSE](LICENSE). The original authors and Sakana AI retain credit for their
work.
