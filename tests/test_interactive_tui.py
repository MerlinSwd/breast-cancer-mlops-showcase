import asyncio
import json
from pathlib import Path

from bc_mlops_showcase.tui import (
    build_overview_text,
    build_run_detail_text,
    render_merlin_logo,
    select_run_views,
)


def _seed_run(
    run_root: Path,
    run_name: str,
    *,
    model_artifact: str,
    with_model_card: bool,
    metrics: dict[str, float],
    model_kind: str = "sklearn_logreg",
) -> Path:
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": run_name,
                "timestamp": "20260607T120000Z",
                "model": {
                    "kind": model_kind,
                    "artifact": model_artifact,
                },
            }
        )
    )
    (run_dir / model_artifact).write_text("pretend-model")
    if with_model_card:
        (run_dir / "MODEL_CARD.md").write_text("# Model Card\n")
    return run_dir


def _seed_registry(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "artifacts" / "runs"
    _seed_run(
        run_root,
        "baseline-logreg-20260607T120000Z",
        model_artifact="model.joblib",
        with_model_card=True,
        metrics={"accuracy": 0.9825, "f1": 0.9861, "roc_auc": 0.9954},
    )
    _seed_run(
        run_root,
        "pytorch-mlp-20260607T120500Z",
        model_artifact="model.pt",
        with_model_card=False,
        metrics={"accuracy": 0.9784, "f1": 0.9810, "roc_auc": 0.9921},
        model_kind="pytorch_mlp",
    )
    _seed_run(
        run_root,
        "hi-acc-xgb-20260607T121000Z",
        model_artifact="model.joblib",
        with_model_card=True,
        metrics={"accuracy": 0.9899, "f1": 0.9700, "roc_auc": 0.9910},
        model_kind="xgboost",
    )

    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "baseline-logreg-20260607T120000Z",
                        "accuracy": 0.9825,
                        "f1": 0.9861,
                        "roc_auc": 0.9954,
                        "model_kind": "sklearn_logreg",
                    },
                    {
                        "run_name": "pytorch-mlp-20260607T120500Z",
                        "accuracy": 0.9784,
                        "f1": 0.9810,
                        "roc_auc": 0.9921,
                        "model_kind": "pytorch_mlp",
                    },
                    {
                        "run_name": "hi-acc-xgb-20260607T121000Z",
                        "accuracy": 0.9899,
                        "f1": 0.9700,
                        "roc_auc": 0.9910,
                        "model_kind": "xgboost",
                    },
                ]
            }
        )
    )
    return registry_path, run_root


def test_render_merlin_logo_contains_branding() -> None:
    logo = render_merlin_logo()

    assert "MERLIN" in logo
    assert "ONCO-OPS" in logo
    assert "Command Deck" in logo


def test_select_run_views_supports_sorting_and_unhealthy_filter(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)

    default_views = select_run_views(summary, query="", sort_key="f1", unhealthy_only=False)
    accuracy_views = select_run_views(
        summary,
        query="",
        sort_key="accuracy",
        unhealthy_only=False,
    )
    unhealthy_views = select_run_views(
        summary,
        query="",
        sort_key="f1",
        unhealthy_only=True,
    )

    assert default_views[0].run_name == "baseline-logreg-20260607T120000Z"
    assert accuracy_views[0].run_name == "hi-acc-xgb-20260607T121000Z"
    assert [view.run_name for view in unhealthy_views] == ["pytorch-mlp-20260607T120500Z"]


def test_build_overview_text_reports_visible_runs_sort_and_health(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
    visible_runs = select_run_views(
        summary,
        query="pytorch",
        sort_key="roc_auc",
        unhealthy_only=True,
    )

    overview = build_overview_text(
        summary,
        visible_runs,
        sort_key="roc_auc",
        unhealthy_only=True,
        query="pytorch",
    )

    assert "Tracked runs: 3" in overview
    assert "Visible runs: 1" in overview
    assert "Sort: roc_auc" in overview
    assert "Health filter: unhealthy only" in overview
    assert "Search: pytorch" in overview


def test_build_run_detail_text_surfaces_selected_run_metrics_and_health(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)

    detail_text = build_run_detail_text(summary, selected_run_name="pytorch-mlp-20260607T120500Z")

    assert "pytorch-mlp-20260607T120500Z" in detail_text
    assert "pytorch_mlp" in detail_text
    assert "0.9810" in detail_text
    assert "Artifact issues: 1" in detail_text
    assert "MODEL_CARD missing" in detail_text


def test_build_run_detail_text_counts_multiple_artifact_issues(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    broken_run_dir = run_root / "pytorch-mlp-20260607T120500Z"
    (broken_run_dir / "model.pt").unlink()
    (broken_run_dir / "metrics.json").unlink()
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)

    detail_text = build_run_detail_text(summary, selected_run_name="pytorch-mlp-20260607T120500Z")

    assert "Artifact issues: 3" in detail_text
    assert "model.pt missing" in detail_text
    assert "metrics.json missing" in detail_text
    assert "MODEL_CARD missing" in detail_text


def test_interactive_dashboard_app_supports_sort_cycle_and_health_filter(
    tmp_path: Path,
) -> None:
    from textual.widgets import ListView, Static

    from bc_mlops_showcase.tui import MerlinDashboardApp

    registry_path, run_root = _seed_registry(tmp_path)

    async def scenario() -> None:
        app = MerlinDashboardApp(registry_path=registry_path, run_root=run_root)
        async with app.run_test() as pilot:
            run_list = app.query_one("#run-list", ListView)
            details = app.query_one("#run-details", Static)
            overview = app.query_one("#overview", Static)

            assert run_list.index == 0
            assert "baseline-logreg-20260607T120000Z" in str(details.render())
            assert "Tracked runs: 3" in str(overview.render())

            app.action_cycle_sort()
            await pilot.pause()
            assert "hi-acc-xgb-20260607T121000Z" in str(details.render())
            assert "Sort: accuracy" in str(overview.render())

            app.action_toggle_health_filter()
            await pilot.pause()
            assert len(run_list.children) == 1
            assert "pytorch-mlp-20260607T120500Z" in str(details.render())
            assert "Health filter: unhealthy only" in str(overview.render())

    asyncio.run(scenario())


def test_interactive_dashboard_app_shows_empty_state_for_no_match_filter(
    tmp_path: Path,
) -> None:
    from textual.widgets import Input, ListView, Static

    from bc_mlops_showcase.tui import MerlinDashboardApp

    registry_path, run_root = _seed_registry(tmp_path)

    async def scenario() -> None:
        app = MerlinDashboardApp(registry_path=registry_path, run_root=run_root)
        async with app.run_test() as pilot:
            run_list = app.query_one("#run-list", ListView)
            details = app.query_one("#run-details", Static)
            overview = app.query_one("#overview", Static)
            run_filter = app.query_one("#run-filter", Input)

            run_filter.focus()
            await pilot.press("z", "z", "z")
            await pilot.pause()

            assert len(run_list.children) == 1
            assert run_list.children[0].name == ""
            assert "No runs match that filter." in str(details.render())
            assert "Visible runs: 0" in str(overview.render())
            assert "Search: zzz" in str(overview.render())

    asyncio.run(scenario())
