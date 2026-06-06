"""Configuration models and loaders for training runs.

The project keeps backend and dataset selection in configuration so that the CLI
and pipeline stay stable while the model family or benchmark changes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_KIND = "sklearn_logreg"
DEFAULT_DATASET_KIND = "sklearn_breast_cancer"
DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "sklearn_logreg": {
        "c": 1.0,
        "max_iter": 500,
    },
    "sklearn_random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
    },
    "pytorch_mlp": {
        "hidden_dims": [32, 16],
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.01,
        "dropout": 0.1,
    },
}


@dataclass(slots=True)
class SplitConfig:
    """Dataset split settings for train/test evaluation."""

    test_size: float = 0.2
    stratify: bool = True


@dataclass(slots=True)
class TrackingConfig:
    """MLflow tracking configuration."""

    uri: str = "./mlruns"
    experiment_name: str = "bc-mlops-showcase"


@dataclass(slots=True)
class DatasetConfig:
    """Dataset selection and loading parameters."""

    kind: str = DEFAULT_DATASET_KIND
    path: str | None = None
    target_column: str = "target"
    positive_label: float | int | str = 1
    drop_columns: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelConfig:
    """Backend model selection and hyperparameters."""

    kind: str = DEFAULT_MODEL_KIND
    device: str = "auto"
    params: dict[str, Any] = field(
        default_factory=lambda: deepcopy(DEFAULT_MODEL_PARAMS[DEFAULT_MODEL_KIND])
    )


@dataclass(slots=True)
class TrainingConfig:
    """Top-level configuration for a training run."""

    experiment_name: str = "baseline-logreg"
    random_seed: int = 42
    threshold: float = 0.5
    split: SplitConfig = field(default_factory=SplitConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _merge_dataclass(default: Any, values: dict[str, Any] | None) -> Any:
    data = values or {}
    return type(default)(**{**asdict(default), **data})


def _resolve_dataset_config(values: dict[str, Any] | None) -> DatasetConfig:
    raw = values or {}
    kind = raw.get("kind", DEFAULT_DATASET_KIND)
    if kind not in {"sklearn_breast_cancer", "csv_tabular_binary"}:
        raise ValueError(f"unsupported dataset kind: {kind}")
    return DatasetConfig(
        kind=kind,
        path=raw.get("path"),
        target_column=raw.get("target_column", "target"),
        positive_label=raw.get("positive_label", 1),
        drop_columns=list(raw.get("drop_columns", [])),
    )


def _resolve_model_config(values: dict[str, Any] | None) -> ModelConfig:
    raw = values or {}
    kind = raw.get("kind", DEFAULT_MODEL_KIND)
    if kind not in DEFAULT_MODEL_PARAMS:
        raise ValueError(f"unsupported model kind: {kind}")

    base_params = deepcopy(DEFAULT_MODEL_PARAMS[kind])
    base_params.update(raw.get("params", {}))
    return ModelConfig(
        kind=kind,
        device=raw.get("device", "auto"),
        params=base_params,
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a YAML training configuration from disk."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}

    default = TrainingConfig()
    dataset = _resolve_dataset_config(raw.get("dataset"))
    model = _resolve_model_config(raw.get("model"))
    experiment_name = raw.get("experiment_name") or (
        "baseline-logreg"
        if model.kind == "sklearn_logreg"
        else "baseline-pytorch-mlp"
        if model.kind == "pytorch_mlp"
        else "baseline-random-forest"
    )
    return TrainingConfig(
        experiment_name=experiment_name,
        random_seed=raw.get("random_seed", default.random_seed),
        threshold=raw.get("threshold", default.threshold),
        split=_merge_dataclass(default.split, raw.get("split")),
        tracking=_merge_dataclass(default.tracking, raw.get("tracking")),
        dataset=dataset,
        model=model,
    )


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Convert a training configuration into a serializable dictionary."""

    return {
        "experiment_name": config.experiment_name,
        "random_seed": config.random_seed,
        "threshold": config.threshold,
        "split": {
            "test_size": config.split.test_size,
            "stratify": config.split.stratify,
        },
        "tracking": {
            "uri": config.tracking.uri,
            "experiment_name": config.tracking.experiment_name,
        },
        "dataset": {
            "kind": config.dataset.kind,
            "path": config.dataset.path,
            "target_column": config.dataset.target_column,
            "positive_label": config.dataset.positive_label,
            "drop_columns": list(config.dataset.drop_columns),
        },
        "model": {
            "kind": config.model.kind,
            "device": config.model.device,
            "params": deepcopy(config.model.params),
        },
    }
