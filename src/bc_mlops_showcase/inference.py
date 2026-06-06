from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PredictionRecord = dict[str, float | int | str]
PredictionResult = dict[str, list[PredictionRecord]]


def load_records(path: str | Path) -> pd.DataFrame:
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


def _build_prediction(index: int, label: str, probability: float) -> PredictionRecord:
    return {
        "index": index,
        "label": label,
        "probability": round(float(probability), 4),
    }


def predict_records(model_path: str | Path, input_path: str | Path) -> PredictionResult:
    model = joblib.load(model_path)
    records = load_records(input_path)
    probabilities = model.predict_proba(records)[:, 1]
    labels = ["malignant" if probability >= 0.5 else "benign" for probability in probabilities]

    predictions = [
        _build_prediction(index=index, label=label, probability=probability)
        for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True))
    ]
    return {"predictions": predictions}
