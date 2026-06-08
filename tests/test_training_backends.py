import json
from pathlib import Path

import pandas as pd

from bc_mlops_showcase.cli import main
from bc_mlops_showcase.config import DatasetConfig, load_training_config
from bc_mlops_showcase.data import load_dataset
from bc_mlops_showcase.pipeline import train_and_evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
COIMBRA_DATASET = REPO_ROOT / "data" / "breast-cancer-coimbra.csv"


def test_load_training_config_supports_model_backend_and_tracking(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment_name: backend-config
tracking:
  uri: ./mlruns-tests
  experiment_name: backend-tests
model:
  kind: sklearn_logreg
  params:
    c: 0.5
    max_iter: 250
""".strip()
    )

    config = load_training_config(config_path)

    assert config.tracking.uri == "./mlruns-tests"
    assert config.tracking.experiment_name == "backend-tests"
    assert config.model.kind == "sklearn_logreg"
    assert config.model.params["c"] == 0.5
    assert config.model.params["max_iter"] == 250


def test_train_and_evaluate_logs_mlflow_metadata_for_sklearn_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    tracking_dir = tmp_path / "mlruns"
    config_path.write_text(
        f"""
experiment_name: mlflow-sklearn
tracking:
  uri: {tracking_dir}
  experiment_name: integration-tests
model:
  kind: sklearn_logreg
  params:
    c: 1.0
    max_iter: 200
""".strip()
    )
    config = load_training_config(config_path)

    result = train_and_evaluate(config=config, output_root=tmp_path / "artifacts")

    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["model"]["kind"] == "sklearn_logreg"
    assert metadata["mlflow"]["run_id"]
    assert (tracking_dir / "mlflow.db").exists()
    assert (tracking_dir / "artifacts").exists()


def test_cli_train_and_predict_support_pytorch_backend(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "train.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        f"""
experiment_name: pytorch-smoke
tracking:
  uri: {tmp_path / "mlruns"}
  experiment_name: integration-tests
model:
  kind: pytorch_mlp
  params:
    hidden_dims: [16, 8]
    epochs: 4
    batch_size: 32
    learning_rate: 0.01
""".strip()
    )

    assert main(["train", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    capsys.readouterr()
    run_dir = sorted(path for path in output_dir.iterdir() if path.is_dir())[-1]
    assert (run_dir / "model.pt").exists()

    record = load_dataset().features.iloc[0].to_dict()
    payload_path = tmp_path / "sample.json"
    payload_path.write_text(json.dumps(record))

    exit_code = main(
        [
            "predict",
            "--model",
            str(run_dir / "model.pt"),
            "--input",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["predictions"][0]["label"] in {"benign", "malignant"}
    assert 0.0 <= output["predictions"][0]["probability"] <= 1.0


def test_cli_train_and_predict_support_random_forest_on_coimbra_dataset(
    tmp_path: Path, capsys
) -> None:
    config_path = tmp_path / "train.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        f"""
experiment_name: coimbra-rf
tracking:
  uri: {tmp_path / "mlruns"}
  experiment_name: integration-tests
dataset:
  kind: csv_tabular_binary
  path: {COIMBRA_DATASET}
  target_column: Classification
  positive_label: 2.0
model:
  kind: sklearn_random_forest
  params:
    n_estimators: 24
    max_depth: 4
    min_samples_leaf: 2
""".strip()
    )

    assert main(["train", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    capsys.readouterr()
    run_dir = sorted(path for path in output_dir.iterdir() if path.is_dir())[-1]
    assert (run_dir / "model.joblib").exists()

    coimbra_frame = pd.read_csv(COIMBRA_DATASET)
    record = coimbra_frame.drop(columns=["Classification"]).iloc[0].to_dict()
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
    assert output["predictions"][0]["label"] in {"1.0", "2.0"}
    assert 0.0 <= output["predictions"][0]["probability"] <= 1.0


def test_cli_train_and_predict_support_hist_gradient_boosting_on_coimbra_dataset(
    tmp_path: Path, capsys
) -> None:
    config_path = tmp_path / "train.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        f"""
experiment_name: coimbra-hgb
tracking:
  uri: {tmp_path / "mlruns"}
  experiment_name: integration-tests
dataset:
  kind: csv_tabular_binary
  path: {COIMBRA_DATASET}
  target_column: Classification
  positive_label: 2.0
model:
  kind: sklearn_hist_gradient_boosting
  params:
    learning_rate: 0.05
    max_iter: 80
    max_depth: 3
    min_samples_leaf: 3
""".strip()
    )

    assert main(["train", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    capsys.readouterr()
    run_dir = sorted(path for path in output_dir.iterdir() if path.is_dir())[-1]
    assert (run_dir / "model.joblib").exists()

    coimbra_frame = pd.read_csv(COIMBRA_DATASET)
    record = coimbra_frame.drop(columns=["Classification"]).iloc[0].to_dict()
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
    assert output["predictions"][0]["label"] in {"1.0", "2.0"}
    assert 0.0 <= output["predictions"][0]["probability"] <= 1.0


def test_cli_train_and_predict_support_digits_cnn_backend(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "train.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        f"""
experiment_name: digits-cnn
tracking:
  uri: {tmp_path / "mlruns"}
  experiment_name: integration-tests
dataset:
  kind: sklearn_digits_binary
model:
  kind: pytorch_cnn
  params:
    conv_channels: [8, 16]
    kernel_size: 3
    epochs: 3
    batch_size: 16
    learning_rate: 0.01
    hidden_dim: 32
""".strip()
    )

    assert main(["train", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    capsys.readouterr()
    run_dir = sorted(path for path in output_dir.iterdir() if path.is_dir())[-1]
    assert (run_dir / "model.pt").exists()

    digits = load_dataset(DatasetConfig(kind="sklearn_digits_binary"))
    record = digits.features.iloc[0].to_dict()
    payload_path = tmp_path / "sample.json"
    payload_path.write_text(json.dumps(record))

    exit_code = main(
        [
            "predict",
            "--model",
            str(run_dir / "model.pt"),
            "--input",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["predictions"][0]["label"] in {"digit_0", "digit_1"}
    assert 0.0 <= output["predictions"][0]["probability"] <= 1.0


def test_cli_predict_uses_metadata_contract_when_artifact_suffix_changes(
    tmp_path: Path, capsys
) -> None:
    config_path = tmp_path / "train.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        f"""
experiment_name: artifact-contract
tracking:
  uri: {tmp_path / "mlruns"}
  experiment_name: integration-tests
model:
  kind: sklearn_logreg
  params:
    c: 1.0
    max_iter: 200
""".strip()
    )

    assert main(["train", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    capsys.readouterr()
    run_dir = sorted(path for path in output_dir.iterdir() if path.is_dir())[-1]

    original_model_path = run_dir / "model.joblib"
    renamed_model_path = run_dir / "model.bundle"
    original_model_path.rename(renamed_model_path)

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model"]["artifact"]["filename"] = renamed_model_path.name
    metadata_path.write_text(json.dumps(metadata, indent=2))

    record = load_dataset().features.iloc[0].to_dict()
    payload_path = tmp_path / "sample.json"
    payload_path.write_text(json.dumps(record))

    exit_code = main(
        [
            "predict",
            "--model",
            str(renamed_model_path),
            "--input",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["predictions"][0]["label"] in {"benign", "malignant"}
    assert 0.0 <= output["predictions"][0]["probability"] <= 1.0
