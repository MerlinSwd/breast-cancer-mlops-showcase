import json
from pathlib import Path

from bc_mlops_showcase.config import TrainingConfig, load_training_config
from bc_mlops_showcase.pipeline import train_and_evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
COIMBRA_DATASET = REPO_ROOT / "data" / "breast-cancer-coimbra.csv"


def test_train_and_evaluate_emits_model_and_metrics(tmp_path: Path) -> None:
    config = TrainingConfig()

    result = train_and_evaluate(config=config, output_root=tmp_path)

    assert result.run_dir.exists()
    assert result.model_path.exists()
    assert result.metrics_path.exists()

    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["accuracy"] >= 0.85
    assert metrics["roc_auc"] >= 0.90
    assert metrics["positive_rate"] > 0


def test_train_and_evaluate_supports_csv_dataset_and_random_forest(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    tracking_dir = tmp_path / "mlruns"
    config_path.write_text(
        f"""
experiment_name: coimbra-rf
tracking:
  uri: {tracking_dir}
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
    config = load_training_config(config_path)

    result = train_and_evaluate(config=config, output_root=tmp_path / "artifacts")

    metrics = json.loads(result.metrics_path.read_text())
    metadata = json.loads(result.metadata_path.read_text())
    assert result.model_path.name == "model.joblib"
    assert (result.run_dir / "feature_importance.csv").exists()
    assert metrics["accuracy"] >= 0.60
    assert metrics["roc_auc"] >= 0.60
    assert metadata["dataset"]["kind"] == "csv_tabular_binary"
    assert metadata["dataset"]["path"].endswith("breast-cancer-coimbra.csv")
    assert metadata["dataset"]["target_column"] == "Classification"
    assert metadata["model"]["kind"] == "sklearn_random_forest"


def test_train_and_evaluate_supports_stratified_k_fold_on_coimbra(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    tracking_dir = tmp_path / "mlruns"
    config_path.write_text(
        f"""
experiment_name: coimbra-kfold
tracking:
  uri: {tracking_dir}
  experiment_name: integration-tests
dataset:
  kind: csv_tabular_binary
  path: {COIMBRA_DATASET}
  target_column: Classification
  positive_label: 2.0
evaluation:
  mode: stratified_k_fold
  folds: 5
model:
  kind: sklearn_hist_gradient_boosting
  params:
    learning_rate: 0.05
    max_iter: 80
    max_depth: 3
    min_samples_leaf: 3
""".strip()
    )
    config = load_training_config(config_path)

    result = train_and_evaluate(config=config, output_root=tmp_path / "artifacts")

    metrics = json.loads(result.metrics_path.read_text())
    metadata = json.loads(result.metadata_path.read_text())
    assert result.model_path.name == "model.joblib"
    assert metrics["accuracy"] >= 0.55
    assert metrics["roc_auc"] >= 0.60
    assert metadata["evaluation"]["mode"] == "stratified_k_fold"
    assert metadata["evaluation"]["folds"] == 5
    assert metadata["train_rows"] == 116
    assert metadata["test_rows"] == 116
    fold_metrics = json.loads((result.run_dir / "fold_metrics.json").read_text())
    assert fold_metrics["evaluation_mode"] == "stratified_k_fold"
    assert len(fold_metrics["folds"]) == 5
    assert fold_metrics["folds"][0]["fold"] == 1
    assert fold_metrics["folds"][0]["train_rows"] > fold_metrics["folds"][0]["test_rows"]
    assert fold_metrics["summary"]["f1"]["mean"] >= 0.55
    assert fold_metrics["summary"]["f1"]["std"] >= 0.0
