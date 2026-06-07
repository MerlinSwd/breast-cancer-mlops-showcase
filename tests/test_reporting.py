import json
from pathlib import Path

from bc_mlops_showcase.reporting import build_model_card


def test_build_model_card_includes_kfold_fold_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "runs" / "coimbra-kfold"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.75,
                "precision": 0.74,
                "recall": 0.76,
                "f1": 0.75,
                "roc_auc": 0.81,
            }
        )
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": "coimbra-kfold",
                "timestamp": "20260607T080000Z",
                "train_rows": 116,
                "test_rows": 116,
                "dataset": {
                    "kind": "csv_tabular_binary",
                    "target_column": "Classification",
                },
                "evaluation": {"mode": "stratified_k_fold", "folds": 5},
                "config": {
                    "threshold": 0.5,
                    "tracking": {"experiment_name": "bc-mlops-showcase"},
                },
                "model": {
                    "kind": "sklearn_hist_gradient_boosting",
                    "runtime": {"framework": "sklearn", "device": "cpu"},
                },
                "mlflow": {
                    "run_id": "run-123",
                    "tracking_uri": "sqlite:///mlruns/mlflow.db",
                },
            }
        )
    )
    (run_dir / "fold_metrics.json").write_text(
        json.dumps(
            {
                "evaluation_mode": "stratified_k_fold",
                "summary": {
                    "f1": {"mean": 0.7512, "std": 0.0314},
                    "roc_auc": {"mean": 0.8123, "std": 0.0222},
                },
            }
        )
    )

    destination = build_model_card(run_dir=run_dir, output_path=run_dir / "MODEL_CARD.md")

    text = destination.read_text()
    assert "Evaluation mode: stratified_k_fold (5 folds)" in text
    assert "Cross-validation F1: 0.7512 ± 0.0314" in text
    assert "Cross-validation ROC AUC: 0.8123 ± 0.0222" in text
