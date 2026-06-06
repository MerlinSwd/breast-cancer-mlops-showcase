from __future__ import annotations

import json
from pathlib import Path


def build_model_card(run_dir: str | Path, output_path: str | Path) -> Path:
    run_path = Path(run_dir)
    metrics = json.loads((run_path / "metrics.json").read_text())
    metadata = json.loads((run_path / "metadata.json").read_text())

    lines = [
        "# Model Card",
        "",
        "## Overview",
        f"- Experiment: {metadata['experiment_name']}",
        f"- Timestamp: {metadata['timestamp']}",
        f"- Train rows: {metadata['train_rows']}",
        f"- Test rows: {metadata['test_rows']}",
        "",
        "## Metrics",
        f"- Accuracy: {metrics['accuracy']}",
        f"- Precision: {metrics['precision']}",
        f"- Recall: {metrics['recall']}",
        f"- F1: {metrics['f1']}",
        f"- ROC AUC: {metrics['roc_auc']}",
        "",
        "## Notes",
        "- Model family: StandardScaler + LogisticRegression",
        "- Problem type: binary classification",
        f"- Positive class probability threshold: {metadata['config']['threshold']}",
        "",
    ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    return destination
