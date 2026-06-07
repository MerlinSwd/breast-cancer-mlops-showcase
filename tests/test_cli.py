import json
from pathlib import Path

from bc_mlops_showcase.cli import main


def test_cli_train_command_generates_run_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment_name: cli-test
""".strip()
    )
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    run_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    assert run_dirs, "expected at least one run directory"
    metrics = json.loads((run_dirs[-1] / "metrics.json").read_text())
    assert metrics["f1"] >= 0.85


def test_cli_compare_summary_surfaces_rank_deltas_and_evaluation_mode(
    tmp_path: Path, capsys
) -> None:
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
                        "model_kind": "sklearn_hist_gradient_boosting",
                        "evaluation_mode": "stratified_k_fold",
                        "evaluation_folds": 5,
                        "cv_f1_std": 0.0123,
                    },
                    {
                        "run_name": "challenger-run",
                        "accuracy": 0.9785,
                        "f1": 0.9801,
                        "roc_auc": 0.9930,
                        "model_kind": "sklearn_random_forest",
                        "evaluation_mode": "holdout",
                        "cv_f1_std": None,
                    },
                ],
                "best_run": {
                    "run_name": "champion-run",
                    "accuracy": 0.9825,
                    "f1": 0.9861,
                    "roc_auc": 0.9954,
                    "model_kind": "sklearn_hist_gradient_boosting",
                    "evaluation_mode": "stratified_k_fold",
                    "evaluation_folds": 5,
                    "cv_f1_std": 0.0123,
                },
            }
        )
    )

    exit_code = main(["compare", "--registry", str(registry_path), "--summary"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Compare Summary" in output
    assert "Champion: champion-run" in output
    assert "Evaluation" in output
    assert "F1 σ" in output
    assert "champion-run" in output
    assert "challenger-run" in output
    assert "+0.0000" in output
    assert "-0.0060" in output
    assert "stratified_k_fold (5 folds)" in output
    assert "holdout" in output
