"""MLflow tracking bootstrap and run lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from .config import TrainingConfig, config_to_dict


@dataclass(slots=True)
class TrackingContext:
    """Resolved MLflow state for an active training run."""

    run: Any
    tracking_uri: str
    artifact_root: str
    experiment_id: str


def _flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened


def resolve_tracking_backend(uri: str) -> tuple[str, str]:
    """Resolve a user-facing tracking URI into MLflow DB and artifact locations."""

    if "://" in uri:
        return uri, uri

    root = Path(uri).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
    return tracking_uri, artifact_root.as_uri()


def _resolve_experiment_id(client: MlflowClient, experiment_name: str, artifact_root: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(experiment_name, artifact_location=artifact_root)


def start_training_run(config: TrainingConfig) -> TrackingContext:
    """Create and tag an MLflow run for the given training configuration."""

    tracking_uri, artifact_root = resolve_tracking_backend(config.tracking.uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _resolve_experiment_id(
        client=client,
        experiment_name=config.tracking.experiment_name,
        artifact_root=artifact_root,
    )
    run = mlflow.start_run(experiment_id=experiment_id, run_name=config.experiment_name)
    mlflow.set_tags(
        {
            "model.kind": config.model.kind,
            "project": "bc-mlops-showcase",
        }
    )
    flattened_config = _flatten_dict(config_to_dict(config))
    mlflow.log_params({key: str(value) for key, value in flattened_config.items()})
    return TrackingContext(
        run=run,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        experiment_id=experiment_id,
    )


def finish_training_run(
    metrics: dict[str, float],
    run_dir: Path,
) -> dict[str, str]:
    """Log final metrics and artifacts, then close the active MLflow run."""

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("mlflow run is not active")

    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(str(run_dir), artifact_path="run_artifacts")
    payload = {
        "run_id": active_run.info.run_id,
        "experiment_id": active_run.info.experiment_id,
    }
    mlflow.end_run(status="FINISHED")
    return payload


def fail_training_run() -> None:
    """Mark the active MLflow run as failed when an exception aborts training."""

    if mlflow.active_run() is not None:
        mlflow.end_run(status="FAILED")
