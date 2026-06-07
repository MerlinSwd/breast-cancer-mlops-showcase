from pathlib import Path

from bc_mlops_showcase.config import TrainingConfig, load_training_config

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
