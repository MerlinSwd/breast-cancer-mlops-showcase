import asyncio
import json
from pathlib import Path

from bc_mlops_showcase.tui import (
    ActionResult,
    build_artifact_detail_text,
    build_compare_text,
    build_config_detail_text,
    build_overview_text,
    build_run_detail_text,
    execute_run_action,
    load_config_views,
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
                "train_rows": 455,
                "test_rows": 114,
                "dataset": {
                    "kind": "csv_tabular_binary",
                    "path": "data/breast-cancer-coimbra.csv",
                    "target_column": "Classification",
                },
                "config": {
                    "tracking": {"experiment_name": "bc-mlops-showcase"},
                    "threshold": 0.5,
                },
                "model": {
                    "kind": model_kind,
                    "artifact": model_artifact,
                    "runtime": {"framework": "sklearn", "device": "cpu"},
                },
                "mlflow": {
                    "run_id": f"{run_name}-run-id",
                    "tracking_uri": "sqlite:///mlruns/mlflow.db",
                },
            }
        )
    )
    (run_dir / model_artifact).write_text("pretend-model")
    (run_dir / "config.resolved.yaml").write_text("model:\n  kind: sklearn_logreg\n")
    (run_dir / "feature_importance.csv").write_text("feature,importance\nradius_mean,0.42\n")
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


def test_build_overview_text_lists_orphan_and_missing_run_names(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    _seed_run(
        run_root,
        "orphan-run-20260607T121500Z",
        model_artifact="model.joblib",
        with_model_card=True,
        metrics={"accuracy": 0.95, "f1": 0.95, "roc_auc": 0.95},
    )

    payload = json.loads(registry_path.read_text())
    payload["runs"].append(
        {
            "run_name": "missing-run-20260607T122000Z",
            "accuracy": 0.9,
            "f1": 0.9,
            "roc_auc": 0.9,
            "model_kind": "sklearn_logreg",
        }
    )
    registry_path.write_text(json.dumps(payload))

    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
    visible_runs = select_run_views(summary, query="", sort_key="f1", unhealthy_only=False)

    overview = build_overview_text(
        summary,
        visible_runs,
        sort_key="f1",
        unhealthy_only=False,
        query="",
    )

    assert "Orphan run dirs: 1" in overview
    assert "Registry entries without run dirs: 1" in overview
    assert "Orphans: orphan-run-20260607T121500Z" in overview
    assert "Missing dirs: missing-run-20260607T122000Z" in overview


def test_build_run_detail_text_surfaces_selected_run_metrics_and_dossier(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)

    detail_text = build_run_detail_text(summary, selected_run_name="pytorch-mlp-20260607T120500Z")

    assert "pytorch-mlp-20260607T120500Z" in detail_text
    assert "pytorch_mlp" in detail_text
    assert "0.9810" in detail_text
    assert "Artifact issues: 1" in detail_text
    assert "Timestamp: 20260607T120000Z" in detail_text
    assert "Rows: train=455 test=114" in detail_text
    assert "Runtime: sklearn on cpu" in detail_text
    assert "MLflow run id: pytorch-mlp-20260607T120500Z-run-id" in detail_text
    assert "Dataset: csv_tabular_binary" in detail_text
    assert "Target column: Classification" in detail_text
    assert "config.resolved.yaml" in detail_text
    assert "feature_importance.csv" in detail_text
    assert "MODEL_CARD missing" in detail_text
    assert "Operator actions:" in detail_text
    assert (
        "bc-mlops validate --metrics artifacts/runs/pytorch-mlp-20260607T120500Z/metrics.json"
        in detail_text
    )
    assert "--gates configs/quality_gates.yaml" in detail_text
    assert "bc-mlops report --run-dir artifacts/runs/pytorch-mlp-20260607T120500Z" in detail_text
    assert "--output artifacts/runs/pytorch-mlp-20260607T120500Z/MODEL_CARD.md" in detail_text


def test_build_run_detail_text_surfaces_evaluation_strategy(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    kfold_run_dir = run_root / "baseline-logreg-20260607T120000Z"
    metadata = json.loads((kfold_run_dir / "metadata.json").read_text())
    metadata["evaluation"] = {"mode": "stratified_k_fold", "folds": 5}
    (kfold_run_dir / "metadata.json").write_text(json.dumps(metadata))

    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
    detail_text = build_run_detail_text(
        summary,
        selected_run_name="baseline-logreg-20260607T120000Z",
    )

    assert "Evaluation: stratified_k_fold (5 folds)" in detail_text


def test_build_run_detail_text_surfaces_kfold_stability_summary(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    kfold_run_dir = run_root / "baseline-logreg-20260607T120000Z"
    metadata = json.loads((kfold_run_dir / "metadata.json").read_text())
    metadata["evaluation"] = {"mode": "stratified_k_fold", "folds": 5}
    (kfold_run_dir / "metadata.json").write_text(json.dumps(metadata))
    (kfold_run_dir / "fold_metrics.json").write_text(
        json.dumps(
            {
                "evaluation_mode": "stratified_k_fold",
                "fold_count": 5,
                "summary": {
                    "f1": {"mean": 0.9861, "std": 0.0215},
                    "roc_auc": {"mean": 0.9954, "std": 0.0082},
                },
            }
        )
    )

    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
    detail_text = build_run_detail_text(
        summary,
        selected_run_name="baseline-logreg-20260607T120000Z",
    )

    assert "CV F1: mean=0.9861 std=0.0215" in detail_text
    assert "CV ROC AUC: mean=0.9954 std=0.0082" in detail_text


def test_build_run_detail_text_counts_multiple_artifact_issues(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    broken_run_dir = run_root / "pytorch-mlp-20260607T120500Z"
    (broken_run_dir / "model.pt").unlink()
    (broken_run_dir / "metrics.json").unlink()
    (broken_run_dir / "config.resolved.yaml").unlink()
    (broken_run_dir / "feature_importance.csv").unlink()
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)

    detail_text = build_run_detail_text(summary, selected_run_name="pytorch-mlp-20260607T120500Z")

    assert "Artifact issues: 3" in detail_text
    assert "model.pt missing" in detail_text
    assert "metrics.json missing" in detail_text
    assert "MODEL_CARD missing" in detail_text
    assert "config.resolved.yaml missing" in detail_text
    assert "feature_importance.csv missing" in detail_text


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
            assert "MLflow run id: pytorch-mlp-20260607T120500Z-run-id" in str(details.render())

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


def test_load_config_views_supports_config_browser(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True)
    config_path = config_root / "train-demo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: demo-run",
                "evaluation:",
                "  mode: stratified_k_fold",
                "  folds: 5",
                "dataset:",
                "  kind: csv_tabular_binary",
                "  path: data/demo.csv",
                "  target_column: Classification",
                "model:",
                "  kind: sklearn_random_forest",
            ]
        )
    )

    config_views = load_config_views(config_root)

    assert len(config_views) == 1
    assert config_views[0].name == "train-demo"
    assert config_views[0].experiment_name == "demo-run"
    assert config_views[0].model_kind == "sklearn_random_forest"
    assert config_views[0].dataset_kind == "csv_tabular_binary"
    assert config_views[0].evaluation_mode == "stratified_k_fold (5 folds)"


def test_build_config_detail_text_surfaces_selected_config_metadata(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True)
    config_path = config_root / "train-demo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: demo-run",
                "tracking:",
                "  experiment_name: bc-mlops-showcase",
                "dataset:",
                "  kind: csv_tabular_binary",
                "  path: data/demo.csv",
                "  target_column: Classification",
                "model:",
                "  kind: sklearn_random_forest",
                "evaluation:",
                "  mode: stratified_k_fold",
                "  folds: 5",
            ]
        )
    )

    config_view = load_config_views(config_root)[0]
    detail_text = build_config_detail_text(config_view)

    assert "Config: train-demo" in detail_text
    assert "Experiment: demo-run" in detail_text
    assert "Model kind: sklearn_random_forest" in detail_text
    assert "Dataset: csv_tabular_binary" in detail_text
    assert "Target column: Classification" in detail_text
    assert "Evaluation: stratified_k_fold (5 folds)" in detail_text


def test_build_artifact_detail_text_shows_run_files_without_leaving_tui(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)

    metrics_text = build_artifact_detail_text(
        summary,
        run_root=run_root,
        selected_run_name="baseline-logreg-20260607T120000Z",
        artifact_key="metrics",
    )
    metadata_text = build_artifact_detail_text(
        summary,
        run_root=run_root,
        selected_run_name="baseline-logreg-20260607T120000Z",
        artifact_key="metadata",
    )

    assert '"accuracy": 0.9825' in metrics_text
    assert '"experiment_name": "baseline-logreg-20260607T120000Z"' in metadata_text
    assert '"mlflow": {' in metadata_text


def test_build_compare_text_surfaces_selected_run_vs_anchor_delta(tmp_path: Path) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)

    compare_text = build_compare_text(
        summary,
        left_run_name="baseline-logreg-20260607T120000Z",
        right_run_name="pytorch-mlp-20260607T120500Z",
    )

    assert "Compare Runs" in compare_text
    assert "baseline-logreg-20260607T120000Z" in compare_text
    assert "pytorch-mlp-20260607T120500Z" in compare_text
    assert "ΔF1" in compare_text
    assert "ΔROC AUC" in compare_text


def test_execute_run_action_runs_validate_report_predict_and_retrain(
    tmp_path: Path, monkeypatch
) -> None:
    from bc_mlops_showcase.tui import load_dashboard_summary

    registry_path, run_root = _seed_registry(tmp_path)
    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
    run_name = "baseline-logreg-20260607T120000Z"
    sample_input = tmp_path / "sample.json"
    sample_input.write_text('{"feature": 1.0}')
    gates_path = tmp_path / "quality_gates.yaml"
    gates_path.write_text("min_accuracy: 0.9\nmin_f1: 0.9\nmin_roc_auc: 0.9\n")

    calls: list[tuple[str, object, object]] = []

    def fake_validate(metrics_path: Path, gates_path_value: Path) -> dict[str, object]:
        calls.append(("validate", metrics_path, gates_path_value))
        return {"passed": True, "checks": []}

    def fake_report(run_dir: Path, output_path: Path) -> Path:
        calls.append(("report", run_dir, output_path))
        output_path.write_text("# Model Card\n")
        return output_path

    def fake_predict(model_path: Path, input_path: Path) -> dict[str, object]:
        calls.append(("predict", model_path, input_path))
        return {"predictions": [{"index": 0, "label": "benign", "probability": 0.1234}]}

    def fake_load_training_config(path: Path) -> object:
        calls.append(("load_training_config", path, path))
        return {"loaded": str(path)}

    def fake_train_and_evaluate(config: object, output_root: Path) -> object:
        calls.append(("retrain", config, output_root))
        return type(
            "Result",
            (),
            {"summary": lambda self: {"run_dir": str(output_root / "demo")}},
        )()

    monkeypatch.setattr("bc_mlops_showcase.validation.validate_metrics", fake_validate)
    monkeypatch.setattr("bc_mlops_showcase.reporting.build_model_card", fake_report)
    monkeypatch.setattr("bc_mlops_showcase.inference.predict_records", fake_predict)
    monkeypatch.setattr("bc_mlops_showcase.config.load_training_config", fake_load_training_config)
    monkeypatch.setattr("bc_mlops_showcase.pipeline.train_and_evaluate", fake_train_and_evaluate)

    validate_result = execute_run_action(
        summary,
        run_root=run_root,
        selected_run_name=run_name,
        action="validate",
        gates_path=gates_path,
    )
    report_result = execute_run_action(
        summary,
        run_root=run_root,
        selected_run_name=run_name,
        action="report",
    )
    predict_result = execute_run_action(
        summary,
        run_root=run_root,
        selected_run_name=run_name,
        action="predict",
        sample_input_path=sample_input,
    )
    retrain_result = execute_run_action(
        summary,
        run_root=run_root,
        selected_run_name=run_name,
        action="retrain",
    )

    assert isinstance(validate_result, ActionResult)
    assert validate_result.ok is True
    assert report_result.ok is True
    assert predict_result.ok is True
    assert retrain_result.ok is True
    assert any(call[0] == "validate" for call in calls)
    assert any(call[0] == "report" for call in calls)
    assert any(call[0] == "predict" for call in calls)
    assert any(call[0] == "retrain" for call in calls)


def test_interactive_dashboard_app_supports_mode_switch_compare_and_help(tmp_path: Path) -> None:
    from textual.widgets import ListView, Static

    from bc_mlops_showcase.tui import MerlinDashboardApp

    registry_path, run_root = _seed_registry(tmp_path)
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True)
    (config_root / "train-demo.yaml").write_text(
        "experiment_name: demo-run\n"
        "model:\n"
        "  kind: sklearn_logreg\n"
        "dataset:\n"
        "  kind: sklearn_breast_cancer\n"
    )

    async def scenario() -> None:
        app = MerlinDashboardApp(
            registry_path=registry_path,
            run_root=run_root,
            config_root=config_root,
        )
        async with app.run_test() as pilot:
            run_list = app.query_one("#run-list", ListView)
            details = app.query_one("#run-details", Static)
            status = app.query_one("#status-bar", Static)

            assert "Run:" in str(details.render())
            await pilot.press("c")
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert "Compare Runs" in str(details.render())

            await pilot.press("tab")
            await pilot.pause()
            assert run_list.children[0].name == "train-demo"
            assert "Mode: configs" in str(status.render())

            await pilot.press("question_mark")
            await pilot.pause()
            assert "Keyboard Help" in str(details.render())

    asyncio.run(scenario())


def test_interactive_dashboard_app_supports_toolbar_buttons_and_select_menus(
    tmp_path: Path, monkeypatch
) -> None:
    from textual.widgets import Button, ListView, Select, Static

    from bc_mlops_showcase.tui import MerlinDashboardApp

    registry_path, run_root = _seed_registry(tmp_path)
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True)
    (config_root / "train-demo.yaml").write_text(
        "experiment_name: demo-run\n"
        "model:\n"
        "  kind: sklearn_logreg\n"
        "dataset:\n"
        "  kind: sklearn_breast_cancer\n"
    )

    calls: list[str] = []

    def fake_execute_selected_action(self: object, action: str) -> None:
        calls.append(action)

    monkeypatch.setattr(
        "bc_mlops_showcase.tui.MerlinDashboardApp._execute_selected_action",
        fake_execute_selected_action,
    )

    async def scenario() -> None:
        app = MerlinDashboardApp(
            registry_path=registry_path,
            run_root=run_root,
            config_root=config_root,
        )
        async with app.run_test() as pilot:
            run_list = app.query_one("#run-list", ListView)
            details = app.query_one("#run-details", Static)
            status = app.query_one("#status-bar", Static)
            mode_select = app.query_one("#mode-select", Select)
            sort_select = app.query_one("#sort-select", Select)
            health_select = app.query_one("#health-select", Select)

            assert app.query_one("#reload-button", Button).label.plain == "Reload"
            assert app.query_one("#actions-button", Button).label.plain == "Actions"
            assert app.query_one("#validate-button", Button).label.plain == "Validate"

            mode_select.value = "configs"
            await pilot.pause()
            assert run_list.children[0].name == "train-demo"
            assert "Mode: configs" in str(status.render())

            mode_select.value = "runs"
            await pilot.pause()
            sort_select.value = "accuracy"
            await pilot.pause()
            assert run_list.children[0].name == "hi-acc-xgb-20260607T121000Z"

            health_select.value = "card_missing"
            await pilot.pause()
            await pilot.pause()
            assert run_list.children[0].name == "pytorch-mlp-20260607T120500Z"
            assert "Health filter: missing model cards" in str(
                details.app.query_one("#overview", Static).render()
            )

            app.query_one("#actions-button", Button).press()
            await pilot.pause()
            assert "Action Catalog" in str(details.render())

            app.query_one("#validate-button", Button).press()
            await pilot.pause()
            assert calls == ["validate"]

    asyncio.run(scenario())
