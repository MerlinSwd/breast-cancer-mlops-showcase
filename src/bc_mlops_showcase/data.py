"""Dataset loading helpers for the showcase pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits

from .config import DatasetConfig


@dataclass(slots=True)
class DatasetBundle:
    """Container for features, labels, and target metadata."""

    features: pd.DataFrame
    target: pd.Series
    target_name: str
    dataset_name: str
    labels: dict[str, str]


def _load_builtin_breast_cancer_dataset() -> DatasetBundle:
    raw = load_breast_cancer(as_frame=True)
    features = raw.data.copy()
    target = raw.target.rename("target")
    return DatasetBundle(
        features=features,
        target=target,
        target_name=str(raw.target_names[1]),
        dataset_name="sklearn_breast_cancer",
        labels={"negative": str(raw.target_names[0]), "positive": str(raw.target_names[1])},
    )


def _load_builtin_digits_binary_dataset() -> DatasetBundle:
    raw = load_digits()
    mask = raw.target <= 1
    flattened = raw.data[mask]
    columns = [f"pixel_{index}" for index in range(flattened.shape[1])]
    features = pd.DataFrame(flattened, columns=columns)
    target = pd.Series(raw.target[mask], name="target")
    return DatasetBundle(
        features=features,
        target=target,
        target_name="digit_1",
        dataset_name="sklearn_digits_binary",
        labels={"negative": "digit_0", "positive": "digit_1"},
    )


def _infer_binary_class_labels(frame: pd.DataFrame, config: DatasetConfig) -> dict[str, str]:
    raw_target = frame[config.target_column]
    unique_values = raw_target.dropna().unique().tolist()
    positive = config.positive_label
    if positive not in unique_values:
        raise ValueError(
            f"positive label {config.positive_label!r} not found in target column {config.target_column}"
        )

    negative_values = [value for value in unique_values if value != positive]
    if len(unique_values) != 2 or len(negative_values) != 1:
        raise ValueError(
            "csv_tabular_binary requires exactly two target classes so prediction labels stay well-defined"
        )

    return {"negative": str(negative_values[0]), "positive": str(positive)}


def _load_csv_tabular_binary_dataset(config: DatasetConfig) -> DatasetBundle:
    if not config.path:
        raise ValueError("csv_tabular_binary dataset requires a path")

    dataset_path = Path(config.path)
    frame = pd.read_csv(dataset_path)
    if config.target_column not in frame.columns:
        raise ValueError(f"target column not found: {config.target_column}")

    labels = _infer_binary_class_labels(frame, config)
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
        target_name=labels["positive"],
        dataset_name=dataset_path.stem,
        labels=labels,
    )


DATASET_LOADERS = {
    "sklearn_breast_cancer": lambda _config: _load_builtin_breast_cancer_dataset(),
    "sklearn_digits_binary": lambda _config: _load_builtin_digits_binary_dataset(),
    "csv_tabular_binary": _load_csv_tabular_binary_dataset,
}


def load_dataset(config: DatasetConfig | None = None) -> DatasetBundle:
    """Load a supported binary-classification dataset as pandas objects."""

    dataset_config = config or DatasetConfig()
    try:
        loader = DATASET_LOADERS[dataset_config.kind]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset kind: {dataset_config.kind}") from exc
    return loader(dataset_config)
