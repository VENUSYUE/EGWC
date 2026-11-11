# Edge Group Walk Centrality

Python implementation of edge group walk centrality (EGWC) and its greedy optimization algorithms. EGWC measures the expected first traversal time of an edge group for a random walk initialized from its stationary distribution.

The implementation includes:

- **ApproxSingleton:** shared electrical sketches for singleton initialization.
- **Exact-MinEGWC:** sparse factorization and exact local marginal gains.
- **Schur-MinEGWC:** rooted Wilson forests, harmonic coupling updates, Schur recovery, shared-sketch candidate screening, and sparse-solve refinement.

## Installation

Use Python 3.9. The tested dependency versions are NumPy 1.21.5, SciPy 1.9.1, and Numba 0.55.1.

```sh
python -m venv .venv
```

Activate the environment:

```sh
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```sh
python -m pip install -e .
```

Computation runs on the CPU. The paper's experiment configuration uses 16 threads and 32 GB RAM.

## Quick start

Run these commands from the project directory:

```sh
egwc-run --graphs Karate --threads 2 --seeds 0 --out results/karate.jsonl
egwc-summarize --input results/karate.jsonl --output-dir results/karate
```

This runs both greedy algorithms on Karate and generates CSV, Markdown, and LaTeX summaries. Output directories are created at runtime.

To use the algorithms directly:

```python
import numpy as np
import egwc

graph, metadata = egwc.load("data/Karate.txt")
first_edge, initialization, _ = egwc.approx_singleton(graph, q=64, seed=12345)
selected, seconds, rounds = egwc.schur_greedy(
    graph, k=10, e1=first_edge, eps_s=0.1, q=64, seed=0, threads=2
)
mask = np.zeros(graph.m, dtype=bool)
mask[selected] = True
objective = egwc.egwc(graph, mask)
print(objective)
```

The selected edge indices refer to `graph.u` and `graph.v` in the processed graph. The loader canonically relabels vertices to consecutive integers.

## Experiments

```sh
egwc-run --threads 16 --out results/runs.jsonl
egwc-summarize --input results/runs.jsonl --output-dir results/tables
```

The default experiment plan uses 23 networks, an edge budget of 10, and sampling thresholds of 0.2, 0.1, and 0.05. Exact-MinEGWC runs on the first 15 networks in `src/egwc/datasets.json`. Schur-MinEGWC uses seeds 0, 1, and 2 for the first 21 networks and seed 0 for Amazon and com-DBLP. Generated summaries report the repetition counts for each setting.

| Parameter | Default |
| --- | --- |
| Initialization sketch width | 64 |
| Initialization seed | 12345 |
| Terminal count | `ceil(sqrt(n))` |
| Terminal selection | Highest-degree endpoints of unselected edges |
| Harmonic coupling updates | 4 |
| Screening sketch width | 64 |
| Refinement short-list size | 64 |
| Sparse-solve relative residual tolerance | `1e-8` |
| Sampling parameter `delta` | 0.05 |
| Forest batch size | 32 |
| Minimum forest count | `ceil(0.4 / eps_s**2)` |
| Maximum forest count | 4096 |

The practical sampling rule uses the empirical variance of the sampled Schur correction and stops at a batch boundary. The parameter `eps_s` controls this stopping rule. Forest counts are averaged over rounds and independent runs.

Use `--graphs`, `--eps`, `--seeds`, and `--k` to choose datasets, sampling thresholds, repetitions, and the edge budget. Use `--exact no` to skip exact greedy or `--exact yes` to request it on every selected graph. The options `--q`, `--init-q`, `--terminals`, `--smooth`, `--refine`, `--delta`, and `--rtol` expose the method parameters.

```sh
egwc-run --help
```

Resume an interrupted experiment with the same configuration:

```sh
egwc-run --threads 16 --out results/runs.jsonl --resume
```

Completed runs are skipped after checking the configuration and implementation fingerprint. Initialization is reused when resuming a partially completed dataset. Existing output files are protected unless `--resume` is specified. If interruption leaves an incomplete final JSON line, remove that line before resuming.

## Data

Input edge lists are included in `data/`. The manifest `src/egwc/datasets.json` contains their paths, SHA-256 checksums, processed sizes, and experiment settings. Inputs are checked before computation. When invoking the installed command from another directory, pass `--data-root /path/to/egwc` to locate the repository datasets.

Preprocessing reads the first two columns as vertex identifiers, ignores directions and input weights, removes self-loops, merges duplicate and reciprocal edges, and retains the largest connected component. Processed edges have unit weight.

Dataset identifiers and original filenames are listed in `DATASETS.md`. The benchmark collections are [SNAP](https://snap.stanford.edu/data/), [KONECT](http://konect.cc/), and [Network Repository](https://networkrepository.com/). Dataset use remains subject to the original providers' terms.

## Output and timing

The experiment runner writes JSONL records containing the dataset checksum, method configuration, dependency versions, selected edges, objective value, initialization time, selection time, and per-round statistics.

End-to-end time is common initialization plus greedy selection. Data loading and independent final-objective evaluation are outside this interval. Final objectives use sparse direct solves for graphs with at most 5,000 vertices and conjugate gradients with relative residual tolerance `1e-12` otherwise.

Numba compiles kernels on first use and caches them. Timing depends on the machine and compilation-cache state; the runner does not perform a separate warm-up pass.

The summarizer writes `table1.csv`, `table1.md`, and `table1.tex`. CSV values retain full precision. The LaTeX table requires `booktabs`.

## Project structure

```text
pyproject.toml         Package metadata, dependencies, and command-line entry points
README.md              Installation, API usage, and experiment instructions
requirements.txt       Pinned numerical dependencies
src/
  egwc/
    __init__.py        Public Python API
    core.py            Graph operations and complete algorithms
    cli.py             Experiment execution and recording
    reporting.py       Aggregation and table generation
    datasets.json      Dataset manifest and experiment plan
DATASETS.md            Dataset identifiers
data/                  Input edge lists
tests/                 Numerical correctness tests
```

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests compare singleton scores and local marginal gains with independent grounded-system evaluations, verify Schur recovery with all vertices terminal, exercise forest sampling and refinement, and check preprocessing.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and contribution guidelines.
