# Tick note — dashboard k-fold stability signals

## Changed
- surfaced `fold_metrics.json` summary data into dashboard run enrichment
- added compare-view `F1 σ` output so stratified k-fold runs show cross-validation dispersion at a glance
- added dossier lines for `CV F1` and `CV ROC AUC` mean/std summaries in the interactive run detail pane
- updated README and `docs/source/howtos/train-models.rst` to document the new stability signals

## Verified
- RED: `uv run python -m pytest tests/test_interactive_tui.py::test_build_run_detail_text_surfaces_kfold_stability_summary tests/test_tui_dashboard.py::test_render_dashboard_text_surfaces_kfold_dispersion_in_compare_view -v`
- GREEN: same targeted pytest command
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run python -m pytest -q`
- `uv run python -m sphinx -W -b html docs/source docs/_build/html`

## Next best task
- extend evaluation visibility into registry-level summaries or validation/report flows so unstable small-dataset runs can be gated automatically, not just inspected manually
