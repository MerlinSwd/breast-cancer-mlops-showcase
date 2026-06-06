"""Model card generation for training run artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def build_model_card(run_dir: str | Path, output_path: str | Path) -> Path:
    """Generate a markdown model card for a completed training run."""

    run_path = Path(run_dir)
    metrics = json.loads((run_path / "metrics.json").read_text())
    metadata = json.loads((run_path / "metadata.json").read_text())

    model_kind = metadata["model"]["kind"]
    runtime = metadata["model"]["runtime"]
    lines = [
        "# Model Card",
        "",
        "## Overview",
        f"- Experiment: {metadata['experiment_name']}",
        f"- Timestamp: {metadata['timestamp']}",
        f"- Train rows: {metadata['train_rows']}",
        f"- Test rows: {metadata['test_rows']}",
        f"- Model kind: {model_kind}",
        f"- Runtime: {runtime['framework']} on {runtime['device']}",
        "",
        "## Metrics",
        f"- Accuracy: {metrics['accuracy']}",
        f"- Precision: {metrics['precision']}",
        f"- Recall: {metrics['recall']}",
        f"- F1: {metrics['f1']}",
        f"- ROC AUC: {metrics['roc_auc']}",
        "",
        "## Tracking",
        f"- MLflow experiment: {metadata['config']['tracking']['experiment_name']}",
        f"- MLflow run id: {metadata['mlflow']['run_id']}",
        f"- Tracking URI: {metadata['mlflow']['tracking_uri']}",
        "",
        "## Notes",
        "- Problem type: binary classification",
        f"- Positive class probability threshold: {metadata['config']['threshold']}",
        "- Architecture is backend-driven: CLI and pipeline stay stable while model kinds",
        "  swap through config.",
        "",
    ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    return destination
