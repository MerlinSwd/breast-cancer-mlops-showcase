# Architecture Notes

This file is the lightweight Markdown companion to the published Sphinx
architecture page in `docs/source/architecture.rst`.

## Core design

The project keeps orchestration stable and moves variability into configuration:

- `config.py` resolves datasets, model families, devices, defaults, and registry-declared parameter schemas
- `data.py` loads supported datasets into pandas objects, including the built-in digits vision dataset
- `modeling.py` owns backend-specific training plus an artifact-loader registry for serialization and inference
- `pipeline.py` orchestrates evaluation, artifact writing, registry updates, and MLflow logging
- `tracking.py` resolves local-vs-remote MLflow backends and manages run lifecycle
- `tui.py` provides the operator dashboard and interactive command deck
- `designer.py` and `model_designer.py` power the run designer and model designer workflows

## Current backend families

- `sklearn_logreg`
- `sklearn_random_forest`
- `sklearn_hist_gradient_boosting`
- `pytorch_mlp`
- `pytorch_cnn`

## Storage layers

The system uses three storage layers on purpose:

1. **Run directory** under `artifacts/runs/<run-name>/`
   - richest per-run local artifact view
2. **Registry** at `artifacts/registry.json`
   - lightweight compare/dashboard summary layer
3. **MLflow**
   - experiment history, flattened params, scalar metrics, and logged artifacts

## Evaluation modes

Supported evaluation modes:

- `holdout`
- `stratified_k_fold`

Current limitation: `stratified_k_fold` is implemented for scikit-learn backends
only.

## TUI architecture

The interactive command deck has four top-level modes:

- `runs`
- `configs`
- `run-designer`
- `model-designer`

The run designer owns full `TrainingConfig` drafting and launch workflows. The
model designer owns guided editing of the `model` slice and applies the result
back into the run designer draft.

## Why this shape works

This structure keeps model and dataset expansion cheap:

- CLI stays stable
- dataset and model capabilities live in explicit registries instead of scattered conditionals
- artifact layout stays predictable
- MLflow logging stays centralized
- dashboards and designers reuse the same config and artifact contracts
- prediction labels come from dataset metadata rather than a hard-coded domain assumption
- artifact loading can resolve from metadata contracts instead of trusting filename suffixes like a gullible raccoon
- adding a backend is mostly a backend/config/docs/test exercise instead of a repo-wide branch explosion

Run metadata now includes a stable artifact handshake for downstream surfaces:

- `contract.metadata_version`
- `contract.task`
- `model.artifact.filename`
- `model.artifact.format`
- `model.artifact.loader`
- `model.artifact.version`

## Areas to future-proof next

The architecture is in decent shape now, but a few seams still deserve armor plating:

- quality gates are metric-name driven today; if the project expands beyond binary classification, task-aware metric contracts would age better

The model designer itself is no longer a per-backend pile of bespoke draft fields. It now consumes model-parameter schemas from the registry, stores generic parameter text values, and renders registry-defined controls in the TUI. That makes backend growth much less annoying and much less likely to drift across config, TUI, and validation layers.

A rare case where restraint beat framework cosplay.
