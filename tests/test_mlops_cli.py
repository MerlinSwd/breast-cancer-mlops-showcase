import json
from pathlib import Path

from bc_mlops_showcase.cli import main
from bc_mlops_showcase.data import load_dataset


def test_cli_predict_command_scores_a_single_json_record(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("experiment_name: predict-test")
    output_dir = tmp_path / "artifacts"

    assert main(["train", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    capsys.readouterr()
    run_dir = sorted(path for path in output_dir.iterdir() if path.is_dir())[-1]

    record = load_dataset().features.iloc[0].to_dict()
    payload_path = tmp_path / "sample.json"
    payload_path.write_text(json.dumps(record))

    exit_code = main(
        [
            "predict",
            "--model",
            str(run_dir / "model.joblib"),
            "--input",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["predictions"][0]["label"] in {"benign", "malignant"}
    assert 0.0 <= output["predictions"][0]["probability"] <= 1.0


def test_cli_validate_command_enforces_metric_thresholds(tmp_path: Path, capsys) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy": 0.95, "f1": 0.92, "roc_auc": 0.99}))
    gates_path = tmp_path / "gates.yaml"
    gates_path.write_text(
        """
min_accuracy: 0.90
min_f1: 0.90
min_roc_auc: 0.95
""".strip()
    )

    exit_code = main(["validate", "--metrics", str(metrics_path), "--gates", str(gates_path)])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is True
    assert all(check["passed"] for check in output["checks"])


def test_cli_report_command_generates_model_card_markdown(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("experiment_name: report-test")
    output_dir = tmp_path / "artifacts"

    assert main(["train", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    run_dir = sorted(path for path in output_dir.iterdir() if path.is_dir())[-1]
    report_path = tmp_path / "MODEL_CARD.md"

    exit_code = main(
        [
            "report",
            "--run-dir",
            str(run_dir),
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    content = report_path.read_text()
    assert "# Model Card" in content
    assert "report-test" in content
    assert "ROC AUC" in content
