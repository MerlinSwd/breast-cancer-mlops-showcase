# Breast Cancer MLOps Showcase

A private end-to-end Python project demonstrating a realistic ML workflow on the scikit-learn breast cancer dataset, now with **MLflow tracking**, **uv-managed environments**, and **config-swappable model backends** for both scikit-learn and PyTorch.

## What it includes

- Reproducible training pipeline with YAML config
- Backend-driven model selection through `model.kind`
- scikit-learn logistic regression baseline
- PyTorch MLP baseline with automatic CPU/GPU device resolution
- MLflow experiment tracking backed by local SQLite + filesystem artifacts
- Metrics and model artifact generation
- Lightweight experiment registry in JSON
- Offline inference for JSON or CSV payloads
- Quality-gate validation for model metrics
- Markdown model card generation for each run
- Test suite, linting, formatting, and package build
- GitHub Actions CI, smoke-training, nightly retraining, and release workflows
- Dockerfile for containerized execution
- Pre-commit hooks and Make targets

## Quickstart with uv

```bash
uv sync --extra dev
uv run pytest
uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs
uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs
uv run bc-mlops compare --registry artifacts/registry.json
```

## CLI

```bash
uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs
uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs
uv run bc-mlops compare --registry artifacts/registry.json
uv run bc-mlops validate --metrics artifacts/runs/<run>/metrics.json --gates configs/quality_gates.yaml
uv run bc-mlops predict --model artifacts/runs/<run>/model.joblib --input sample-inputs/sample.json
uv run bc-mlops predict --model artifacts/runs/<run>/model.pt --input sample-inputs/sample.json
uv run bc-mlops report --run-dir artifacts/runs/<run> --output artifacts/runs/<run>/MODEL_CARD.md
```

## Configuration shape

The pipeline is designed so model swaps happen in config, not in the CLI contract.

### scikit-learn baseline

```yaml
experiment_name: baseline-logreg
tracking:
  uri: ./mlruns
  experiment_name: bc-mlops-showcase
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

## MLOps workflow

1. Train a backend-specific model and persist reproducible artifacts.
2. Log params, metrics, and artifacts to MLflow.
3. Enforce minimum quality gates before promotion.
4. Generate a model card for review and auditability.
5. Use GitHub Actions for CI, smoke training, scheduled retraining, and tagged releases.

## Architecture summary

- `src/bc_mlops_showcase/config.py` — config loading and backend model resolution
- `src/bc_mlops_showcase/modeling.py` — backend registry/training/prediction logic
- `src/bc_mlops_showcase/tracking.py` — MLflow SQLite tracking bootstrap
- `src/bc_mlops_showcase/pipeline.py` — orchestration, metrics, artifacts, registry
- `src/bc_mlops_showcase/inference.py` — backend-agnostic scoring
- `configs/train.yaml` — sklearn baseline
- `configs/train-pytorch.yaml` — PyTorch baseline

## Deep learning base recommendation

The current PyTorch MLP is the first deep-learning-ready backend. The recommended base solution for expansion is:

- keep the **same CLI contract**
- add new backends under the same `model.kind` pattern
- keep hyperparameters entirely under `model.params`
- log every run through the same MLflow layer
- standardize every backend on `train -> artifact -> predict -> report`

That means adding a deeper MLP, tabular transformer, or Lightning-style trainer later should be a backend module addition, not a pipeline rewrite. Miraculous, I know.

## Repo structure

- `src/bc_mlops_showcase/` — package code
- `tests/` — regression tests
- `configs/` — training and quality-gate config
- `sample-inputs/` — example inference payloads
- `.github/workflows/` — CI/CD and retraining automation
- `artifacts/` — generated outputs (gitignored except placeholders)
- `mlruns/` — local MLflow SQLite database and artifacts
