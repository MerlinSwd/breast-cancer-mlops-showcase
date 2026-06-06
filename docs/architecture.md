# Architecture Notes

## Problem

Classify whether a tumor is malignant or benign using the breast cancer dataset shipped with scikit-learn.

## Approach

- Data source: `sklearn.datasets.load_breast_cancer`
- Model: `StandardScaler` + `LogisticRegression`
- Configuration: YAML-based hyperparameters and split controls
- Artifacts: trained model, metrics JSON, metadata JSON, resolved config, feature importance CSV
- Registry: append-only `artifacts/registry.json` with `best_run` summary

## DevOps / MLOps choices

- Package-first Python project with `pyproject.toml`
- Dockerfile for reproducible execution
- GitHub Actions CI and smoke-training workflows
- Dependabot for Python + GitHub Actions dependency hygiene
- Pre-commit hooks for lint/format discipline
