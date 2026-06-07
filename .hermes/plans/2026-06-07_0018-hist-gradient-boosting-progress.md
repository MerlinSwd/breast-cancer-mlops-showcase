# 2026-06-07 Night Tick Progress — HistGradientBoosting on Coimbra

## What changed
- Added `sklearn_hist_gradient_boosting` as a supported model kind in `src/bc_mlops_showcase/config.py` with default hyperparameters and default experiment naming.
- Added HistGradientBoosting training support in `src/bc_mlops_showcase/modeling.py` using `HistGradientBoostingClassifier` and standard `.joblib` artifacts.
- Added TDD coverage for config loading and end-to-end CLI train/predict support on the Coimbra dataset.
- Added `configs/train-coimbra-hist-gradient-boosting.yaml` as a ready-to-run benchmark config.
- Updated `README.md`, `docs/source/usage.rst`, and `docs/source/howtos/train-models.rst` to document the new backend and training command.

## RED/GREEN evidence
- RED: `uv run python -m pytest tests/test_config.py::test_load_training_config_supports_hist_gradient_boosting_on_coimbra -v`
- GREEN: same test passed after config support landed.
- RED: `uv run python -m pytest tests/test_training_backends.py::test_cli_train_and_predict_support_hist_gradient_boosting_on_coimbra_dataset -v`
- GREEN: same test passed after backend implementation landed.

## Verification
- `uv run python -m pytest tests/test_config.py::test_load_training_config_supports_hist_gradient_boosting_on_coimbra tests/test_training_backends.py::test_cli_train_and_predict_support_hist_gradient_boosting_on_coimbra_dataset -v`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run python -m pytest -q`
- `uv run python -m sphinx -W -b html docs/source docs/_build/html`

Note: full pytest still emits one external `opentelemetry` deprecation warning from the environment, but all tests pass.

## Next best task
- High-value next slice: expand the TUI with registry/disk drift detection and orphan-run detection so operators can spot mismatches between `artifacts/registry.json` and on-disk run directories immediately.
