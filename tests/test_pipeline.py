import json
from pathlib import Path

from bc_mlops_showcase.config import TrainingConfig
from bc_mlops_showcase.pipeline import train_and_evaluate


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
