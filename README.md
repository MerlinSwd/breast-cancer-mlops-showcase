# Breast Cancer MLOps Showcase

A private end-to-end Python project demonstrating a realistic ML workflow on the scikit-learn breast cancer dataset.

## What it includes

- Reproducible training pipeline with YAML config
- Metrics and model artifact generation
- Lightweight experiment registry in JSON
- Test suite, linting, formatting, and package build
- GitHub Actions CI and smoke-training workflow
- Dockerfile for containerized execution
- Pre-commit hooks and Make targets

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
make train
```

## CLI

```bash
bc-mlops train --config configs/train.yaml --output-dir artifacts/runs
bc-mlops compare --registry artifacts/registry.json
```

## Repo structure

- `src/bc_mlops_showcase/` — package code
- `tests/` — regression tests
- `configs/` — training config
- `.github/workflows/` — CI/CD automation
- `artifacts/` — generated outputs (gitignored except placeholders)
