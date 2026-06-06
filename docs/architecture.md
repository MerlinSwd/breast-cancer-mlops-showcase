# Architecture Notes

## Problem

Classify whether a tumor is malignant or benign using the breast cancer dataset shipped with scikit-learn while keeping the project easy to extend across classical ML and deep learning backends.

## Architecture review

### What changed

The original design trained a single hard-coded `LogisticRegression` pipeline directly inside `pipeline.py`.
That was fine for a showcase, but brittle for model family growth: every new model would have forced more branching into the orchestration layer.

The revised design moves model choice behind a backend contract:

- `config.py` resolves **which backend** to use
- `modeling.py` owns backend-specific training and prediction
- `pipeline.py` stays focused on orchestration, metrics, registry, and artifacts
- `tracking.py` centralizes MLflow bootstrap and metadata capture

### Why this is better

This shape makes model expansion cheap:

- CLI stays stable
- run artifact layout stays predictable
- MLflow logging stays centralized
- quality gates stay backend-agnostic
- adding a new model mostly means adding one backend path and one config

In other words: less spaghetti, fewer regrets.

## Current backends

### 1. `sklearn_logreg`

- StandardScaler + LogisticRegression
- artifact: `model.joblib`
- strong interpretable baseline
- fast CI/smoke test candidate

### 2. `pytorch_mlp`

- feed-forward MLP for tabular classification
- artifact: `model.pt`
- device selection: `auto`, `cpu`, or `cuda`
- trained with standardized inputs and checkpoint metadata for portable inference

## MLflow design

For local developer ergonomics, tracking uses:

- SQLite tracking DB: `mlruns/mlflow.db`
- filesystem artifact root: `mlruns/artifacts/`

This avoids the deprecated pure file-store path and gives a cleaner base for future upgrades.

## Deep learning base solution proposal

The recommended deep-learning base for this repo is a **backend registry for tabular models** with one shared contract:

1. **Config contract**
   - `model.kind`
   - `model.device`
   - `model.params`

2. **Backend contract**
   - train from `X_train, y_train`
   - save a portable model artifact
   - expose `predict_probabilities()`
   - emit optional feature importance / attribution summary

3. **Pipeline contract**
   - train/evaluate
   - write metrics + metadata + config snapshot
   - log to MLflow
   - update registry

### Recommended next deep-learning steps

If you push further, the best next shape is:

- keep `pytorch_mlp` as the fast baseline
- add `pytorch_tabular_residual_mlp` for a stronger tabular default
- optionally add a Lightning-based trainer only if experiment complexity grows
- keep inference artifact compatibility stable per backend

### Why not jump straight to a giant framework?

Because this repo is still a showcase, not a wandering cathedral of abstractions. The current backend split is enough structure to scale without turning the codebase into a ritual sacrifice to "future flexibility."

## Future extensions

- richer calibration and threshold tuning
- model promotion rules via MLflow model registry
- drift monitoring / shadow evaluation
- SHAP or Captum for explanations
- FastAPI serving wrapper that loads either backend artifact from metadata
