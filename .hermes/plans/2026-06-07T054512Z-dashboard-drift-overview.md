# 2026-06-07T054512Z dashboard drift overview

## Changed
- Expanded the interactive dashboard overview to list orphan run directory names and registry entries whose run directories are missing, not just counts.
- Added a regression test covering explicit orphan/missing run name rendering in `build_overview_text()`.

## TDD verification
- RED: `uv run python -m pytest tests/test_interactive_tui.py::test_build_overview_text_lists_orphan_and_missing_run_names -v`
- GREEN: `uv run python -m pytest tests/test_interactive_tui.py::test_build_overview_text_lists_orphan_and_missing_run_names tests/test_interactive_tui.py::test_build_overview_text_reports_visible_runs_sort_and_health -v`

## Full verification
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run python -m pytest -q`

## Notes
- Full pytest currently passes with one third-party `opentelemetry` deprecation warning emitted from the environment.

## Next best task
- Add an interactive drift-focused inspector or filter mode so operators can jump directly to orphaned/missing run discrepancies from the dashboard.
