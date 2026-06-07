# Tick note — 2026-06-07T01:24:20Z

## Changed
- Added registry/disk drift detection to the dashboard summary and static dashboard output.
- Added a dedicated "Registry / Disk Drift" panel showing orphan run directories and stale registry entries.
- Surfaced drift counts in operator hints and overview text.
- Added a TDD regression test covering orphan-run and missing-run detection.
- Updated README and docs usage text to mention drift visibility.

## Verified
- RED: `uv run python -m pytest tests/test_tui_dashboard.py::test_render_dashboard_text_surfaces_registry_disk_drift -v` (failed before implementation)
- GREEN: `uv run python -m pytest tests/test_tui_dashboard.py::test_render_dashboard_text_surfaces_registry_disk_drift -v`
- Targeted regression: `uv run python -m pytest tests/test_tui_dashboard.py tests/test_interactive_tui.py -q`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run python -m pytest -q`
- `uv run python -m sphinx -W -b html docs/source docs/_build/html`

## Next best task
- Expand model evaluation for the Coimbra benchmark with a small-dataset mode such as stratified k-fold cross-validation, including persisted fold metrics and dashboard surfacing.
