from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig, config_to_dict
from .data import load_dataset


@dataclass(slots=True)
class TrainingResult:
    run_dir: Path
    model_path: Path
    metrics_path: Path
    metadata_path: Path

    def summary(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "model_path": str(self.model_path),
            "metrics_path": str(self.metrics_path),
            "metadata_path": str(self.metadata_path),
        }


def _build_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.model.c,
                    max_iter=config.model.max_iter,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _positive_predictions(probabilities: pd.Series, threshold: float) -> pd.Series:
    return (probabilities >= threshold).astype(int)


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
        }
    )
    payload["best_run"] = max(payload["runs"], key=lambda run: (run["f1"], run["roc_auc"]))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2))


def train_and_evaluate(config: TrainingConfig, output_root: Path) -> TrainingResult:
    dataset = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=config.split.test_size,
        random_state=config.random_seed,
        stratify=dataset.target if config.split.stratify else None,
    )

    model = _build_pipeline(config)
    model.fit(X_train, y_train)

    probabilities = pd.Series(model.predict_proba(X_test)[:, 1])
    predictions = _positive_predictions(probabilities, config.threshold)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "positive_rate": round(float(predictions.mean()), 4),
    }

    run_name = f"{config.experiment_name}-{_timestamp()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    coefficients_path = run_dir / "feature_importance.csv"
    config_path = run_dir / "config.resolved.yaml"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    coefficients = pd.DataFrame(
        {
            "feature": dataset.features.columns,
            "coefficient": model.named_steps["classifier"].coef_[0],
        }
    ).sort_values("coefficient", ascending=False)
    coefficients.to_csv(coefficients_path, index=False)

    metadata = {
        "experiment_name": config.experiment_name,
        "timestamp": _timestamp(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "config": config_to_dict(config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    config_path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))

    _write_registry(output_root, run_name, metrics, metadata)
    return TrainingResult(
        run_dir=run_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
    )
