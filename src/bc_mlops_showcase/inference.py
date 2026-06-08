"""Inference helpers for loading payloads and scoring trained artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .modeling import load_run_metadata, predict_probabilities_from_path

PredictionRecord = dict[str, float | int | str]
PredictionResult = dict[str, list[PredictionRecord]]


def load_records(path: str | Path) -> pd.DataFrame:
    """Load input records from JSON or CSV into a dataframe."""

    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(input_path.read_text())
        if isinstance(payload, dict):
            payload = [payload]
        return pd.DataFrame(payload)

    if suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError(f"unsupported input format: {input_path.suffix}")


def _load_prediction_labels(model_path: str | Path) -> tuple[str, str]:
    metadata = load_run_metadata(model_path)
    if metadata is None:
        return ("negative", "positive")

    dataset = metadata.get("dataset", {})
    labels = dataset.get("labels", {}) if isinstance(dataset, dict) else {}
    negative_label = labels.get("negative")
    positive_label = labels.get("positive")
    if isinstance(negative_label, str) and isinstance(positive_label, str):
        return negative_label, positive_label
    return ("negative", "positive")


def _build_prediction(index: int, label: str, probability: float) -> PredictionRecord:
    return {
        "index": index,
        "label": label,
        "probability": round(float(probability), 4),
    }


def predict_records(model_path: str | Path, input_path: str | Path) -> PredictionResult:
    """Run offline inference and return labeled predictions."""

    records = load_records(input_path)
    metadata = load_run_metadata(model_path)
    probabilities = predict_probabilities_from_path(
        model_path=model_path,
        records=records,
        metadata=metadata,
    )
    negative_label, positive_label = _load_prediction_labels(model_path)
    labels = [
        positive_label if probability >= 0.5 else negative_label
        for probability in probabilities
    ]

    predictions = [
        _build_prediction(index=index, label=label, probability=probability)
        for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True))
    ]
    return {"predictions": predictions}
