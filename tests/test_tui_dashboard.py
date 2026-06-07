import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bc_mlops_showcase.cli import main
from bc_mlops_showcase.tui import render_dashboard_text


def _seed_run(run_root: Path, run_name: str, *, model_artifact: str, with_model_card: bool) -> Path:
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.9825,
                "precision": 0.98,
                "recall": 0.99,
                "f1": 0.9861,
                "roc_auc": 0.9954,
            }
        )
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": "baseline-logreg",
                "timestamp": "20260606T155803Z",
                "train_rows": 455,
                "test_rows": 114,
                "dataset": {
                    "kind": "csv_tabular_binary",
                    "path": "data/breast-cancer-coimbra.csv",
                    "target_column": "Classification",
                },
                "model": {
                    "kind": "sklearn_logreg",
                    "artifact": model_artifact,
                    "runtime": {"framework": "sklearn", "device": "cpu"},
                },
                "mlflow": {"run_id": "run-123", "tracking_uri": "sqlite:///mlruns/mlflow.db"},
            }
        )
    )
    (run_dir / model_artifact).write_text("model-bytes-go-here")
    (run_dir / "config.resolved.yaml").write_text("model:\n  kind: sklearn_logreg\n")
    pd.DataFrame(
        [
            {"feature": "radius_mean", "importance": 0.42},
            {"feature": "texture_mean", "importance": 0.25},
        ]
    ).to_csv(run_dir / "feature_importance.csv", index=False)
    if with_model_card:
        (run_dir / "MODEL_CARD.md").write_text("# Model Card\n")
    return run_dir


def test_render_dashboard_text_surfaces_branding_and_artifact_health(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "runs"
    _seed_run(
        run_root,
        "baseline-logreg-20260606T155802Z",
        model_artifact="model.joblib",
        with_model_card=True,
    )

    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "baseline-logreg-20260606T155802Z",
                        "accuracy": 0.9825,
                        "f1": 0.9861,
                        "roc_auc": 0.9954,
                        "experiment_name": "baseline-logreg",
                        "timestamp": "20260606T155803Z",
                        "model_kind": "sklearn_logreg",
                        "mlflow_run_id": "run-123",
                    }
                ],
                "best_run": {
                    "run_name": "baseline-logreg-20260606T155802Z",
                    "accuracy": 0.9825,
                    "f1": 0.9861,
                    "roc_auc": 0.9954,
                    "experiment_name": "baseline-logreg",
                    "timestamp": "20260606T155803Z",
                    "model_kind": "sklearn_logreg",
                    "mlflow_run_id": "run-123",
                },
            }
        )
    )

    output = render_dashboard_text(registry_path=registry_path, run_root=run_root, width=100)

    assert "MERLIN // ONCO-OPS COMMAND DECK" in output
    assert "Champion Run" in output
    assert "baseline-logreg-20260606T155802Z" in output
    assert "Artifact Health" in output
    assert "OK" in output
    assert "sklearn_logreg" in output


def test_dashboard_command_warns_when_a_model_card_is_missing(tmp_path: Path, capsys) -> None:
    run_root = tmp_path / "artifacts" / "runs"
    _seed_run(
        run_root,
        "baseline-logreg-20260606T155802Z",
        model_artifact="model.joblib",
        with_model_card=False,
    )

    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "baseline-logreg-20260606T155802Z",
                        "accuracy": 0.9825,
                        "f1": 0.9861,
                        "roc_auc": 0.9954,
                        "experiment_name": "baseline-logreg",
                        "timestamp": "20260606T155803Z",
                        "model_kind": "sklearn_logreg",
                        "mlflow_run_id": "run-123",
                    }
                ]
            }
        )
    )

    exit_code = main(
        [
            "dashboard",
            "--registry",
            str(registry_path),
            "--run-root",
            str(run_root),
            "--width",
            "100",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "MODEL_CARD missing" in output
    assert "Ready to train" not in output


def test_render_dashboard_text_handles_empty_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"runs": []}))

    output = render_dashboard_text(
        registry_path=registry_path,
        run_root=tmp_path / "artifacts" / "runs",
    )

    assert "No tracked runs yet" in output
    assert "Train with: bc-mlops train --config configs/train.yaml" in output


def test_render_dashboard_text_tolerates_malformed_registry_json(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("{ definitely-not-json")

    output = render_dashboard_text(
        registry_path=registry_path,
        run_root=tmp_path / "artifacts" / "runs",
    )

    assert "No tracked runs yet" in output


def test_render_dashboard_text_tolerates_schema_mismatches_in_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"runs": [None, {}, {"f1": 0.9}], "best_run": None}))

    output = render_dashboard_text(
        registry_path=registry_path,
        run_root=tmp_path / "artifacts" / "runs",
    )

    assert "No tracked runs yet" in output


def test_render_dashboard_text_tolerates_non_list_runs_field(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"runs": None}))

    output = render_dashboard_text(
        registry_path=registry_path,
        run_root=tmp_path / "artifacts" / "runs",
    )

    assert "No tracked runs yet" in output


def test_render_dashboard_text_tolerates_missing_metrics_in_registry(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "runs"
    _seed_run(run_root, "broken-run", model_artifact="model.joblib", with_model_card=True)

    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "broken-run",
                        "experiment_name": "baseline-logreg",
                        "timestamp": "20260606T155803Z",
                    }
                ]
            }
        )
    )

    output = render_dashboard_text(registry_path=registry_path, run_root=run_root, width=100)

    assert "broken-run" in output
    assert "n/a" in output


def test_render_dashboard_text_surfaces_registry_disk_drift(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "runs"
    _seed_run(run_root, "tracked-run", model_artifact="model.joblib", with_model_card=True)
    _seed_run(run_root, "orphan-run", model_artifact="model.joblib", with_model_card=True)

    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "tracked-run",
                        "accuracy": 0.9825,
                        "f1": 0.9861,
                        "roc_auc": 0.9954,
                        "experiment_name": "baseline-logreg",
                        "timestamp": "20260606T155803Z",
                        "model_kind": "sklearn_logreg",
                        "mlflow_run_id": "run-123",
                    },
                    {
                        "run_name": "missing-run",
                        "accuracy": 0.8,
                        "f1": 0.79,
                        "roc_auc": 0.81,
                        "experiment_name": "baseline-logreg",
                        "timestamp": "20260606T155804Z",
                        "model_kind": "sklearn_logreg",
                        "mlflow_run_id": "run-456",
                    },
                ]
            }
        )
    )

    output = render_dashboard_text(registry_path=registry_path, run_root=run_root, width=100)

    assert "Registry / Disk Drift" in output
    assert "Orphan run dirs: 1" in output
    assert "Registry entries without run dirs: 1" in output
    assert "orphan-run" in output
    assert "missing-run" in output


def test_render_dashboard_text_suggests_validate_and_report_commands_for_best_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "artifacts" / "runs"
    _seed_run(
        run_root,
        "champion-run",
        model_artifact="model.joblib",
        with_model_card=False,
    )

    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "champion-run",
                        "accuracy": 0.9825,
                        "f1": 0.9861,
                        "roc_auc": 0.9954,
                        "experiment_name": "baseline-logreg",
                        "timestamp": "20260606T155803Z",
                        "model_kind": "sklearn_logreg",
                        "mlflow_run_id": "run-123",
                    }
                ],
                "best_run": {
                    "run_name": "champion-run",
                    "accuracy": 0.9825,
                    "f1": 0.9861,
                    "roc_auc": 0.9954,
                    "experiment_name": "baseline-logreg",
                    "timestamp": "20260606T155803Z",
                    "model_kind": "sklearn_logreg",
                    "mlflow_run_id": "run-123",
                },
            }
        )
    )

    output = render_dashboard_text(registry_path=registry_path, run_root=run_root, width=120)

    assert "bc-mlops validate --metrics" in output
    assert "champion-run/metrics.json" in output
    assert "configs/quality_gates.yaml" in output
    assert "bc-mlops report --run-dir" in output
    assert "champion-run/MODEL_CARD.md" in output


def test_dashboard_command_imports_without_torch_for_registry_only_usage(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"runs": []}))

    script = f"""
import builtins

real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError('No module named torch')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import

from bc_mlops_showcase.cli import main

raise SystemExit(main([
    'dashboard',
    '--registry', {str(registry_path)!r},
    '--run-root', {str(tmp_path / "artifacts" / "runs")!r},
    '--width', '100',
]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "No tracked runs yet" in result.stdout


def test_dashboard_command_dispatches_to_interactive_app(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"runs": []}))

    captured: dict[str, object] = {}

    def fake_launch_dashboard_app(*, registry_path: Path, run_root: Path) -> None:
        captured["registry_path"] = registry_path
        captured["run_root"] = run_root

    monkeypatch.setattr("bc_mlops_showcase.tui.launch_dashboard_app", fake_launch_dashboard_app)

    exit_code = main(
        [
            "dashboard",
            "--registry",
            str(registry_path),
            "--run-root",
            str(tmp_path / "artifacts" / "runs"),
            "--interactive",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "registry_path": registry_path,
        "run_root": tmp_path / "artifacts" / "runs",
    }


def test_compare_command_imports_without_textual_for_non_dashboard_usage(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"runs": []}))

    script = f"""
import builtins

real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'textual' or name.startswith('textual.'):
        raise ModuleNotFoundError('No module named textual')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import

from bc_mlops_showcase.cli import main

raise SystemExit(main([
    'compare',
    '--registry', {str(registry_path)!r},
]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"runs": []' in result.stdout
