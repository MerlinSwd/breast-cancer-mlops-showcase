# Tick note — compare summary leaderboard

## Changed
- added `bc-mlops compare --summary` for a lightweight terminal compare view without opening the dashboard
- surfaced rank, evaluation mode, champion F1 deltas, and `F1 σ` in the compare summary output
- documented the new summary mode in `README.md`, `docs/source/usage.rst`, and `docs/source/howtos/train-models.rst`
- added CLI regression coverage for the new compare summary path while preserving the existing non-Textual compare flow

## Verified
- RED: `uv run python -m pytest tests/test_cli.py::test_cli_compare_summary_surfaces_rank_deltas_and_evaluation_mode -v`
- GREEN: `uv run python -m pytest tests/test_cli.py::test_cli_compare_summary_surfaces_rank_deltas_and_evaluation_mode -v`
- `uv run python -m pytest tests/test_cli.py tests/test_tui_dashboard.py::test_compare_command_imports_without_textual_for_non_dashboard_usage -v`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run python -m pytest -q`
- `uv run python -m sphinx -W -b html docs/source docs/_build/html`

## Next best task
- persist evaluation/stability fields directly into `artifacts/registry.json` during training so `compare --summary` and other registry-only views can surface k-fold metadata even when run directories are moved or unavailable
