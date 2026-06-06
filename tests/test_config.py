from pathlib import Path

from bc_mlops_showcase.config import TrainingConfig, load_training_config


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
