# Breast Cancer MLOps Showcase

A private end-to-end Python project demonstrating a realistic ML workflow on the scikit-learn breast cancer dataset.

## What it includes

- Reproducible training pipeline with YAML config
- Metrics and model artifact generation
- Lightweight experiment registry in JSON
- Offline inference for JSON or CSV payloads
- Quality-gate validation for model metrics
- Markdown model card generation for each run
- Test suite, linting, formatting, and package build
- GitHub Actions CI, smoke-training, nightly retraining, and release workflows
- Dockerfile for containerized execution
- Pre-commit hooks and Make targets

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
make train
make validate
make report
```

## CLI

```bash
bc-mlops train --config configs/train.yaml --output-dir artifacts/runs
bc-mlops compare --registry artifacts/registry.json
bc-mlops validate --metrics artifacts/runs/<run>/metrics.json --gates configs/quality_gates.yaml
bc-mlops predict --model artifacts/runs/<run>/model.joblib --input sample-inputs/sample.json
bc-mlops report --run-dir artifacts/runs/<run> --output artifacts/runs/<run>/MODEL_CARD.md
```

## MLOps workflow

1. Train a baseline model and persist reproducible artifacts.
2. Enforce minimum quality gates before promotion.
3. Generate a model card for review and auditability.
4. Use GitHub Actions for CI, smoke training, scheduled retraining, and tagged releases.

## Repo structure

- `src/bc_mlops_showcase/` — package code
- `tests/` — regression tests
- `configs/` — training and quality-gate config
- `sample-inputs/` — example inference payloads
- `.github/workflows/` — CI/CD and retraining automation
- `artifacts/` — generated outputs (gitignored except placeholders)
