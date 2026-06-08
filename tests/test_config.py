from pathlib import Path

import pytest

from bc_mlops_showcase.config import DatasetConfig, TrainingConfig, load_training_config
from bc_mlops_showcase.data import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
COIMBRA_DATASET = REPO_ROOT / "data" / "breast-cancer-coimbra.csv"


def test_load_training_config_reads_expected_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment_name: unit-test
random_seed: 17
split:
  test_size: 0.25
  stratify: true
tracking:
  uri: ./mlruns-tests
  experiment_name: config-tests
model:
  kind: sklearn_logreg
  device: cpu
  params:
    c: 0.5
    max_iter: 300
threshold: 0.42
""".strip()
    )

    config = load_training_config(config_path)

    assert isinstance(config, TrainingConfig)
    assert config.experiment_name == "unit-test"
    assert config.random_seed == 17
    assert config.split.test_size == 0.25
    assert config.tracking.uri == "./mlruns-tests"
    assert config.model.kind == "sklearn_logreg"
    assert config.model.device == "cpu"
    assert config.model.params["c"] == 0.5
    assert config.threshold == 0.42


def test_load_training_config_supports_csv_dataset_and_random_forest(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
experiment_name: coimbra-rf
tracking:
  uri: ./mlruns-tests
  experiment_name: config-tests
dataset:
  kind: csv_tabular_binary
  path: {COIMBRA_DATASET}
  target_column: Classification
  positive_label: 2.0
model:
  kind: sklearn_random_forest
  params:
    n_estimators: 32
    max_depth: 4
    min_samples_leaf: 2
""".strip()
    )

    config = load_training_config(config_path)

    assert config.dataset.kind == "csv_tabular_binary"
    assert Path(config.dataset.path) == COIMBRA_DATASET
    assert config.dataset.target_column == "Classification"
    assert float(config.dataset.positive_label) == 2.0
    assert config.model.kind == "sklearn_random_forest"
    assert config.model.params["n_estimators"] == 32
    assert config.model.params["max_depth"] == 4
    assert config.model.params["min_samples_leaf"] == 2


def test_load_training_config_supports_hist_gradient_boosting_on_coimbra(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
experiment_name: coimbra-hgb
tracking:
  uri: ./mlruns-tests
  experiment_name: config-tests
dataset:
  kind: csv_tabular_binary
  path: {COIMBRA_DATASET}
  target_column: Classification
  positive_label: 2.0
model:
  kind: sklearn_hist_gradient_boosting
  params:
    learning_rate: 0.05
    max_iter: 120
    max_depth: 3
    min_samples_leaf: 3
""".strip()
    )

    config = load_training_config(config_path)

    assert config.dataset.kind == "csv_tabular_binary"
    assert config.model.kind == "sklearn_hist_gradient_boosting"
    assert config.model.params["learning_rate"] == 0.05
    assert config.model.params["max_iter"] == 120
    assert config.model.params["max_depth"] == 3
    assert config.model.params["min_samples_leaf"] == 3


def test_load_training_config_supports_stratified_k_fold_evaluation(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
experiment_name: coimbra-kfold
tracking:
  uri: ./mlruns-tests
  experiment_name: config-tests
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
""".strip()
    )

    config = load_training_config(config_path)

    assert config.evaluation.mode == "stratified_k_fold"
    assert config.evaluation.folds == 5
    assert config.model.kind == "sklearn_hist_gradient_boosting"


def test_load_training_config_supports_digits_dataset_and_cnn_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment_name: digits-cnn
tracking:
  uri: ./mlruns-tests
  experiment_name: config-tests
dataset:
  kind: sklearn_digits_binary
model:
  kind: pytorch_cnn
  params:
    conv_channels: [8, 16]
    kernel_size: 3
    epochs: 3
    batch_size: 16
    learning_rate: 0.005
    hidden_dim: 32
""".strip()
    )

    config = load_training_config(config_path)

    assert config.dataset.kind == "sklearn_digits_binary"
    assert config.model.kind == "pytorch_cnn"
    assert config.model.params["conv_channels"] == [8, 16]
    assert config.model.params["kernel_size"] == 3
    assert config.model.params["hidden_dim"] == 32


def test_load_training_config_rejects_cnn_for_non_vision_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-cnn-dataset.yaml"
    config_path.write_text(
        """
experiment_name: invalid-cnn-dataset
tracking:
  uri: ./mlruns-tests
  experiment_name: config-tests
dataset:
  kind: sklearn_breast_cancer
model:
  kind: pytorch_cnn
""".strip()
    )

    with pytest.raises(ValueError, match="pytorch_cnn currently requires"):
        load_training_config(config_path)


def test_load_training_config_rejects_stratified_k_fold_for_pytorch_backend(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid-kfold-pytorch.yaml"
    config_path.write_text(
        """
experiment_name: invalid-kfold-pytorch
tracking:
  uri: ./mlruns-tests
  experiment_name: config-tests
evaluation:
  mode: stratified_k_fold
  folds: 5
model:
  kind: pytorch_mlp
""".strip()
    )

    with pytest.raises(ValueError, match="stratified_k_fold"):
        load_training_config(config_path)


def test_load_dataset_supports_builtin_digits_binary_dataset() -> None:
    dataset = load_dataset(DatasetConfig(kind="sklearn_digits_binary"))

    assert dataset.dataset_name == "sklearn_digits_binary"
    assert dataset.features.shape[1] == 64
    assert set(dataset.target.unique()) == {0, 1}
