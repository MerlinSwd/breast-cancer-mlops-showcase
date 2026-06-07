"""Dataset loading helpers for the showcase pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from .config import DatasetConfig


@dataclass(slots=True)
class DatasetBundle:
    """Container for features, labels, and target metadata."""

    features: pd.DataFrame
    target: pd.Series
    target_name: str
    dataset_name: str


def _load_builtin_breast_cancer_dataset() -> DatasetBundle:
    raw = load_breast_cancer(as_frame=True)
    features = raw.data.copy()
    target = raw.target.rename("target")
    return DatasetBundle(
        features=features,
        target=target,
        target_name=raw.target_names[1],
        dataset_name="sklearn_breast_cancer",
    )


def _load_csv_tabular_binary_dataset(config: DatasetConfig) -> DatasetBundle:
    if not config.path:
        raise ValueError("csv_tabular_binary dataset requires a path")

    dataset_path = Path(config.path)
    frame = pd.read_csv(dataset_path)
    if config.target_column not in frame.columns:
        raise ValueError(f"target column not found: {config.target_column}")

    drop_columns = [column for column in config.drop_columns if column in frame.columns]
    target = (frame[config.target_column] == config.positive_label).astype(int).rename("target")
    feature_columns = [
        column
        for column in frame.columns
        if column != config.target_column and column not in set(drop_columns)
    ]
    features = frame[feature_columns].copy()
    return DatasetBundle(
        features=features,
        target=target,
        target_name=str(config.positive_label),
        dataset_name=dataset_path.stem,
    )


def load_dataset(config: DatasetConfig | None = None) -> DatasetBundle:
    """Load a supported binary-classification dataset as pandas objects."""

    dataset_config = config or DatasetConfig()
    if dataset_config.kind == "sklearn_breast_cancer":
        return _load_builtin_breast_cancer_dataset()
    if dataset_config.kind == "csv_tabular_binary":
        return _load_csv_tabular_binary_dataset(dataset_config)
    raise ValueError(f"unsupported dataset kind: {dataset_config.kind}")
