"""Training orchestration for the end-to-end MLOps workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .config import TrainingConfig, config_to_dict
from .data import load_dataset
from .modeling import train_backend
from .tracking import fail_training_run, finish_training_run, start_training_run


@dataclass(slots=True)
class TrainingResult:
    """Paths to the primary artifacts produced by a training run."""

    run_dir: Path
    model_path: Path
    metrics_path: Path
    metadata_path: Path

    def summary(self) -> dict[str, str]:
        """Return a JSON-serializable view of the result paths."""

        return {
            "run_dir": str(self.run_dir),
            "model_path": str(self.model_path),
            "metrics_path": str(self.metrics_path),
            "metadata_path": str(self.metadata_path),
        }


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _positive_predictions(probabilities: pd.Series, threshold: float) -> pd.Series:
    return (probabilities >= threshold).astype(int)


def _evaluate_holdout(
    config: TrainingConfig,
    *,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[object, pd.Series, pd.Series, int, int]:
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.split.test_size,
        random_state=config.random_seed,
        stratify=target if config.split.stratify else None,
    )
    backend = train_backend(config=config, X_train=X_train, y_train=y_train)
    probabilities = pd.Series(backend.predict_probabilities(X_test))
    return backend, probabilities, y_test.reset_index(drop=True), len(X_train), len(X_test)


def _evaluate_stratified_k_fold(
    config: TrainingConfig,
    *,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[object, pd.Series, pd.Series, int, int]:
    splitter = StratifiedKFold(
        n_splits=config.evaluation.folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    trained = train_backend(config=config, X_train=features, y_train=target)
    if not config.model.kind.startswith("sklearn_"):
        raise ValueError("stratified_k_fold evaluation currently supports sklearn backends only")

    probabilities = cross_val_predict(
        trained._predictor,
        features,
        target,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    backend = trained
    return (
        backend,
        pd.Series(probabilities),
        target.reset_index(drop=True),
        len(features),
        len(features),
    )


def _write_registry(
    output_root: Path,
    run_name: str,
    metrics: dict[str, float],
    metadata: dict[str, object],
) -> None:
    registry_path = output_root.parent / "registry.json"
    payload = {"runs": []}
    if registry_path.exists():
        payload = json.loads(registry_path.read_text())

    payload.setdefault("runs", []).append(
        {
            "run_name": run_name,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "experiment_name": metadata["experiment_name"],
            "timestamp": metadata["timestamp"],
            "model_kind": metadata["model"]["kind"],
            "mlflow_run_id": metadata["mlflow"]["run_id"],
        }
    )
    payload["best_run"] = max(payload["runs"], key=lambda run: (run["f1"], run["roc_auc"]))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2))


def train_and_evaluate(config: TrainingConfig, output_root: Path) -> TrainingResult:
    """Train the configured backend, evaluate it, and persist run artifacts."""

    dataset = load_dataset(config.dataset)

    run_name = f"{config.experiment_name}-{_timestamp()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path: Path | None = None
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    config_path = run_dir / "config.resolved.yaml"
    feature_importance_path = run_dir / "feature_importance.csv"

    tracking = start_training_run(config)
    try:
        if config.evaluation.mode == "stratified_k_fold":
            backend, probabilities, y_test, train_rows, test_rows = _evaluate_stratified_k_fold(
                config,
                features=dataset.features,
                target=dataset.target,
            )
        else:
            backend, probabilities, y_test, train_rows, test_rows = _evaluate_holdout(
                config,
                features=dataset.features,
                target=dataset.target,
            )
        predictions = _positive_predictions(probabilities, config.threshold)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions)), 4),
            "recall": round(float(recall_score(y_test, predictions)), 4),
            "f1": round(float(f1_score(y_test, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "positive_rate": round(float(predictions.mean()), 4),
        }

        model_path = run_dir / backend.artifact_filename
        backend.save(model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        if backend.feature_importance is not None:
            backend.feature_importance.to_csv(feature_importance_path, index=False)

        metadata = {
            "experiment_name": config.experiment_name,
            "timestamp": _timestamp(),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "target_name": dataset.target_name,
            "dataset_name": dataset.dataset_name,
            "config": config_to_dict(config),
            "evaluation": {
                "mode": config.evaluation.mode,
                "folds": config.evaluation.folds,
            },
            "dataset": {
                "kind": config.dataset.kind,
                "path": config.dataset.path,
                "target_column": config.dataset.target_column,
                "positive_label": config.dataset.positive_label,
                "drop_columns": list(config.dataset.drop_columns),
            },
            "model": {
                "kind": backend.kind,
                "artifact": model_path.name,
                "runtime": backend.runtime,
            },
            "mlflow": {
                "experiment_id": tracking.experiment_id,
                "tracking_uri": tracking.tracking_uri,
                "artifact_root": tracking.artifact_root,
                "run_name": tracking.run.info.run_name,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        config_path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))
        mlflow_info = finish_training_run(metrics=metrics, run_dir=run_dir)
        metadata["mlflow"].update(mlflow_info)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        _write_registry(output_root, run_name, metrics, metadata)
        return TrainingResult(
            run_dir=run_dir,
            model_path=model_path,
            metrics_path=metrics_path,
            metadata_path=metadata_path,
        )
    except Exception:
        fail_training_run()
        raise
