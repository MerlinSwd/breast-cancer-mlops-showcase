# Breast Cancer MLOps Showcase

An end-to-end **tabular MLOps** project for binary breast-cancer classification.
It combines a config-driven training pipeline, MLflow tracking, offline
inference, validation gates, a terminal dashboard, and interactive TUI designer
workflows.

This repo is meant to feel like a small but real ML platform rather than a
single notebook that wandered into production wearing a fake moustache.

## What it supports

- config-driven training from YAML
- switchable datasets via `dataset.kind`
- switchable model families via `model.kind`
- supported model kinds:
  - `sklearn_logreg`
  - `sklearn_random_forest`
  - `sklearn_hist_gradient_boosting`
  - `pytorch_mlp`
  - `pytorch_cnn`
- supported dataset kinds:
  - `sklearn_breast_cancer`
  - `csv_tabular_binary`
  - `sklearn_digits_binary`
- MLflow logging with local SQLite or remote tracking backends
- quality-gate validation against `metrics.json`
- Markdown model card generation
- offline inference from JSON or CSV
- pull Kaggle datasets or competition files into a local workspace
- static dashboard plus interactive Textual command deck
- in-TUI run designer and model designer workflows

## Documentation

- Repository: https://github.com/MerlinSwd/breast-cancer-mlops-showcase
- Project docs: https://merlinswd.github.io/breast-cancer-mlops-showcase/

The docs site is organized into:

- getting started
- configuration and datasets
- dashboard and TUI workflows
- MLflow/tracking and artifact reference
- how-to guides
- architecture and API reference

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/MerlinSwd/breast-cancer-mlops-showcase.git
cd breast-cancer-mlops-showcase
```

### 2. Install dependencies

```bash
uv sync --extra dev --extra docs
```

### 3. Verify the environment

```bash
uv run ruff check .
uv run python -m pytest tests/ -q
uv run python -m sphinx -W -b html docs/source docs/_build/html
```

### 4. Train a model

```bash
uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs
```

### 5. Compare and inspect runs

```bash
uv run bc-mlops compare --registry artifacts/registry.json --summary
uv run bc-mlops dashboard --registry artifacts/registry.json --run-root artifacts/runs
```

### 6. Open the interactive command deck

```bash
uv run bc-mlops dashboard \
  --registry artifacts/registry.json \
  --run-root artifacts/runs \
  --interactive
```

## Common commands

Train a PyTorch baseline:

```bash
uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs
```

Train the digits vision CNN example:

```bash
uv run bc-mlops train --config configs/train-digits-cnn.yaml --output-dir artifacts/runs
```

Train a Coimbra benchmark config:

```bash
uv run bc-mlops train \
  --config configs/train-coimbra-random-forest.yaml \
  --output-dir artifacts/runs
```

Validate a run:

```bash
uv run bc-mlops validate \
  --metrics artifacts/runs/<run-name>/metrics.json \
  --gates configs/quality_gates.yaml
```

Generate a model card:

```bash
uv run bc-mlops report \
  --run-dir artifacts/runs/<run-name> \
  --output artifacts/runs/<run-name>/MODEL_CARD.md
```

Run offline inference:

```bash
uv run bc-mlops predict \
  --model artifacts/runs/<run-name>/model.joblib \
  --input sample-inputs/sample.json
```

Pull a Kaggle dataset:

```bash
uv run bc-mlops kaggle pull \
  --resource-type dataset \
  --handle merlinswd/breast-cancer-demo \
  --output-dir data/external/breast-cancer-demo
```

Pull a Kaggle competition file:

```bash
uv run bc-mlops kaggle pull \
  --resource-type competition \
  --handle titanic \
  --file-name train.csv \
  --output-dir data/external/titanic
```

Authenticate Kaggle first with:

```bash
uv run kaggle auth login
```

If you need a headless flow, the current Kaggle client also supports
`uv run kaggle auth login --no-launch-browser`.

## What training writes

Each training run creates a directory under `artifacts/runs/<run-name>/` with:

- the serialized model artifact (`model.joblib` or `model.pt`)
- `metrics.json`
- `metadata.json`
- `config.resolved.yaml`
- optional `fold_metrics.json`
- optional `feature_importance.csv`

`MODEL_CARD.md` is created later by `bc-mlops report`, not automatically during
training.

## Key project ideas

- **config first**: the CLI stays stable while datasets and backends change in YAML
- **shared pipeline**: training, validation, reporting, inference, and tracking reuse the same artifact contract
- **operator tooling**: dashboards and in-TUI designers sit on top of the same run artifacts and configs as the CLI

## Where to go next

- Start with the docs site: https://merlinswd.github.io/breast-cancer-mlops-showcase/
- Read `docs/source/usage.rst` for CLI workflows
- Read `docs/source/dashboard.rst` for the interactive deck
- Read `docs/source/architecture.rst` for the system design
