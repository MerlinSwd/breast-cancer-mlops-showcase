# Breast Cancer MLOps Showcase

An end-to-end **tabular MLOps** project for classifying breast cancer tumors with a
config-driven training pipeline, **MLflow tracking**, **uv-managed environments**,
and swappable model backends for **scikit-learn** and **PyTorch** across both the
built-in Wisconsin diagnostic dataset and a harder Coimbra benchmark.

This repository is meant to feel like a small but real ML platform rather than a
single notebook that accidentally escaped into version control.

## What this project does

- trains reproducible models from YAML configuration
- switches model families through `model.kind` instead of CLI rewrites
- switches datasets through `dataset.kind` instead of pipeline rewrites
- supports `sklearn_logreg`, `sklearn_random_forest`, `sklearn_hist_gradient_boosting`, and `pytorch_mlp` backends
- benchmarks on the built-in sklearn breast-cancer dataset and the Coimbra CSV dataset
- logs runs, metrics, params, and artifacts to **MLflow**
- validates trained models against configurable quality gates
- generates run artifacts and markdown model cards
- supports offline inference from JSON or CSV payloads
- runs CI, smoke training, releases, and docs deployment through GitHub Actions

## Documentation

- Repository: https://github.com/MerlinSwd/breast-cancer-mlops-showcase
- Project docs: https://merlinswd.github.io/breast-cancer-mlops-showcase/

The Sphinx site includes:

- installation
- usage
- how-to guides
- architecture and UML diagrams
- auto-generated API reference

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/MerlinSwd/breast-cancer-mlops-showcase.git
cd breast-cancer-mlops-showcase
```

### 2. Install with uv

```bash
uv sync --extra dev --extra docs
```

### 3. Verify the environment

```bash
uv run ruff check .
uv run python -m pytest
uv run python -m build
uv run python -m sphinx -W -b html docs/source docs/_build/html
```

### 4. Train models

```bash
uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs
uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs
uv run bc-mlops train --config configs/train-coimbra-random-forest.yaml --output-dir artifacts/runs
uv run bc-mlops train --config configs/train-coimbra-hist-gradient-boosting.yaml --output-dir artifacts/runs
```

### 5. Inspect results

```bash
uv run bc-mlops compare --registry artifacts/registry.json
uv run bc-mlops dashboard --registry artifacts/registry.json --run-root artifacts/runs
```

## CLI usage

### Train

```bash
uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs
```

### Compare runs

```bash
uv run bc-mlops compare --registry artifacts/registry.json
```

### Open the branded terminal dashboard

```bash
uv run bc-mlops dashboard \
  --registry artifacts/registry.json \
  --run-root artifacts/runs
```

### Open the interactive command deck

```bash
uv run bc-mlops dashboard \
  --registry artifacts/registry.json \
  --run-root artifacts/runs \
  --interactive
```

The dashboard gives you a quick terminal "bridge view" of:

- the current champion run
- a metric leaderboard across tracked runs
- artifact health checks for model files, metrics, and model cards
- operator hints for the next useful command

The interactive deck adds:

- live filtering by run name or model kind
- keyboard navigation across tracked runs
- an overview pane with champion, visible-run counts, current sort, and search state
- a richer run dossier with timestamp, train/test rows, runtime, dataset, MLflow IDs, and artifact paths
- a detail pane for the selected run with metric deltas vs the champion
- unhealthy-only mode to isolate runs missing artifacts or model cards
- sort cycling with `s`, health filtering with `h`, reload with `r`, filtering with `/`, and quit with `q`

### Validate against quality gates

```bash
uv run bc-mlops validate \
  --metrics artifacts/runs/<run-name>/metrics.json \
  --gates configs/quality_gates.yaml
```

### Run offline inference

```bash
uv run bc-mlops predict \
  --model artifacts/runs/<run-name>/model.joblib \
  --input sample-inputs/sample.json

uv run bc-mlops predict \
  --model artifacts/runs/<run-name>/model.pt \
  --input sample-inputs/sample.json
```

### Generate a model card

```bash
uv run bc-mlops report \
  --run-dir artifacts/runs/<run-name> \
  --output artifacts/runs/<run-name>/MODEL_CARD.md
```

## Configuration model

The key design choice in this repo is that **backend selection happens in config**.
That keeps the CLI stable while making model expansion cheap.

### scikit-learn baseline

```yaml
experiment_name: baseline-logreg
tracking:
  uri: ./mlruns
  experiment_name: bc-mlops-showcase
dataset:
  kind: sklearn_breast_cancer
model:
  kind: sklearn_logreg
  device: cpu
  params:
    c: 1.0
    max_iter: 500
```

### PyTorch baseline

```yaml
experiment_name: baseline-pytorch-mlp
tracking:
  uri: ./mlruns
  experiment_name: bc-mlops-showcase
dataset:
  kind: sklearn_breast_cancer
model:
  kind: pytorch_mlp
  device: auto
  params:
    hidden_dims: [64, 32]
    epochs: 25
    batch_size: 32
    learning_rate: 0.01
    dropout: 0.1
```

### Harder Coimbra benchmark with random forest

```yaml
experiment_name: coimbra-random-forest
tracking:
  uri: ./mlruns
  experiment_name: bc-mlops-showcase
dataset:
  kind: csv_tabular_binary
  path: data/breast-cancer-coimbra.csv
  target_column: Classification
  positive_label: 2.0
model:
  kind: sklearn_random_forest
  device: cpu
  params:
    n_estimators: 200
    max_depth: 6
    min_samples_leaf: 2
```

### Harder Coimbra benchmark with histogram gradient boosting

```yaml
experiment_name: coimbra-hist-gradient-boosting
tracking:
  uri: ./mlruns
  experiment_name: bc-mlops-showcase
dataset:
  kind: csv_tabular_binary
  path: data/breast-cancer-coimbra.csv
  target_column: Classification
  positive_label: 2.0
model:
  kind: sklearn_hist_gradient_boosting
  device: cpu
  params:
    learning_rate: 0.05
    max_iter: 200
    max_depth: 3
    min_samples_leaf: 3
```

## Training outputs

Each run writes a timestamped directory under `artifacts/runs/` with:

- model artifact (`model.joblib` or `model.pt`)
- `metrics.json`
- `metadata.json`
- `config.resolved.yaml`
- optional `feature_importance.csv`
- generated `MODEL_CARD.md`

A lightweight registry is also updated at:

- `artifacts/registry.json`

## MLflow tracking

By default the project resolves `tracking.uri: ./mlruns` into:

- tracking database: `mlruns/mlflow.db`
- artifact store: `mlruns/artifacts/`

This uses **SQLite + filesystem artifacts** rather than the deprecated pure
file-store mode in MLflow 3.x. Less pain, fewer ritual incantations.

## Architecture at a glance

Core modules:

- `src/bc_mlops_showcase/config.py` — config loading and backend resolution
- `src/bc_mlops_showcase/data.py` — dataset loading
- `src/bc_mlops_showcase/modeling.py` — backend-specific training and scoring
- `src/bc_mlops_showcase/tracking.py` — MLflow bootstrap and run lifecycle
- `src/bc_mlops_showcase/pipeline.py` — orchestration, metrics, artifacts, registry
- `src/bc_mlops_showcase/inference.py` — backend-agnostic offline scoring
- `src/bc_mlops_showcase/validation.py` — quality gate enforcement
- `src/bc_mlops_showcase/reporting.py` — model card generation

If you want the fuller diagrams and interaction flow, the Sphinx docs have those.
I know, reading documentation: scandalous.

## Development commands

The `Makefile` wraps the most common flows:

```bash
make install
make test
make train
make compare
make docs
```

Equivalent direct commands use `uv run ...` throughout.

## Repository layout

- `src/bc_mlops_showcase/` — application code
- `tests/` — regression tests
- `configs/` — training and quality gate config
- `sample-inputs/` — sample inference payloads
- `docs/source/` — Sphinx documentation source
- `.github/workflows/` — CI/CD and Pages deployment workflows

## CI/CD

GitHub Actions currently handles:

- code quality checks
- test execution
- package builds
- smoke training
- release workflow
- documentation build and GitHub Pages deployment

## License

MIT
