"""Model card generation for training run artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def build_model_card(run_dir: str | Path, output_path: str | Path) -> Path:
    """Generate a markdown model card for a completed training run."""

    run_path = Path(run_dir)
    metrics = json.loads((run_path / "metrics.json").read_text())
    metadata = json.loads((run_path / "metadata.json").read_text())

    dataset = metadata.get("dataset", {})
    evaluation = metadata.get("evaluation", {})
    model_kind = metadata["model"]["kind"]
    runtime = metadata["model"]["runtime"]
    evaluation_mode = evaluation.get("mode", "holdout")
    evaluation_folds = evaluation.get("folds", 1)
    lines = [
        "# Model Card",
        "",
        "## Overview",
        f"- Experiment: {metadata['experiment_name']}",
        f"- Timestamp: {metadata['timestamp']}",
        f"- Train rows: {metadata['train_rows']}",
        f"- Test rows: {metadata['test_rows']}",
        f"- Dataset kind: {dataset.get('kind', metadata.get('dataset_name', 'unknown'))}",
        f"- Target column: {dataset.get('target_column', metadata.get('target_name', 'target'))}",
        f"- Model kind: {model_kind}",
        f"- Runtime: {runtime['framework']} on {runtime['device']}",
        f"- Evaluation mode: {evaluation_mode} ({evaluation_folds} folds)",
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

    fold_metrics_path = run_path / "fold_metrics.json"
    if fold_metrics_path.exists():
        fold_metrics = json.loads(fold_metrics_path.read_text())
        summary = fold_metrics.get("summary", {})
        if "f1" in summary or "roc_auc" in summary:
            lines.append("## Cross-validation Summary")
            if "f1" in summary:
                f1_mean = summary["f1"]["mean"]
                f1_std = summary["f1"]["std"]
                lines.append(f"- Cross-validation F1: {f1_mean:.4f} ± {f1_std:.4f}")
            if "roc_auc" in summary:
                roc_auc_mean = summary["roc_auc"]["mean"]
                roc_auc_std = summary["roc_auc"]["std"]
                lines.append(f"- Cross-validation ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}")
            lines.append("")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    return destination
