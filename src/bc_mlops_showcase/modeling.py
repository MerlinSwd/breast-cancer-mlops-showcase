"""Model backend implementations and artifact loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainingConfig


@dataclass(slots=True)
class BackendTrainingBundle:
    """In-memory training result plus serialization metadata for a backend."""

    kind: str
    artifact_filename: str
    feature_names: list[str]
    runtime: dict[str, Any]
    feature_importance: pd.DataFrame | None
    _predictor: Any
    _serializer_payload: Any

    def save(self, destination: Path) -> None:
        """Persist the trained backend artifact to disk."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix == ".joblib":
            joblib.dump(self._serializer_payload, destination)
            return
        torch.save(self._serializer_payload, destination)

    def predict_probabilities(self, records: pd.DataFrame) -> np.ndarray:
        """Predict positive-class probabilities for input records."""

        if self.kind.startswith("sklearn_"):
            return self._predictor.predict_proba(records[self.feature_names])[:, 1]
        return _predict_pytorch_checkpoint(
            checkpoint=self._serializer_payload,
            records=records[self.feature_names],
        )


class BinaryMLP(nn.Module):
    """Simple feed-forward neural network for binary tabular classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def resolve_device(device_preference: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise fall back to CPU."""

    if device_preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_preference


def _train_sklearn_model(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BackendTrainingBundle:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(config.model.params["c"]),
                    max_iter=int(config.model.params["max_iter"]),
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    classifier = pipeline.named_steps["classifier"]
    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient": classifier.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)
    return BackendTrainingBundle(
        kind="sklearn_logreg",
        artifact_filename="model.joblib",
        feature_names=list(X_train.columns),
        runtime={"framework": "sklearn", "device": "cpu"},
        feature_importance=feature_importance,
        _predictor=pipeline,
        _serializer_payload=pipeline,
    )


def _train_pytorch_model(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BackendTrainingBundle:
    resolved_device = resolve_device(config.model.device)
    device = torch.device(resolved_device)
    hidden_dims = [int(value) for value in config.model.params["hidden_dims"]]
    dropout = float(config.model.params.get("dropout", 0.0))
    batch_size = int(config.model.params["batch_size"])
    epochs = int(config.model.params["epochs"])
    learning_rate = float(config.model.params["learning_rate"])

    train_features = X_train.to_numpy(dtype=np.float32)
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_std = np.where(train_std == 0, 1.0, train_std)
    scaled_features = (train_features - train_mean) / train_std
    train_targets = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

    dataset = TensorDataset(
        torch.tensor(scaled_features, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(config.random_seed)
    model = BinaryMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

    first_layer = next(module for module in model.network if isinstance(module, nn.Linear))
    importance = first_layer.weight.detach().abs().mean(dim=0).cpu().numpy()
    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    checkpoint = {
        "kind": "pytorch_mlp",
        "feature_names": list(X_train.columns),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
    }
    return BackendTrainingBundle(
        kind="pytorch_mlp",
        artifact_filename="model.pt",
        feature_names=list(X_train.columns),
        runtime={"framework": "pytorch", "device": resolved_device},
        feature_importance=feature_importance,
        _predictor=model.cpu(),
        _serializer_payload=checkpoint,
    )


def _train_random_forest_model(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BackendTrainingBundle:
    classifier = RandomForestClassifier(
        n_estimators=int(config.model.params["n_estimators"]),
        max_depth=(
            None
            if config.model.params.get("max_depth") is None
            else int(config.model.params["max_depth"])
        ),
        min_samples_leaf=int(config.model.params.get("min_samples_leaf", 1)),
        random_state=config.random_seed,
    )
    classifier.fit(X_train, y_train)
    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return BackendTrainingBundle(
        kind="sklearn_random_forest",
        artifact_filename="model.joblib",
        feature_names=list(X_train.columns),
        runtime={"framework": "sklearn", "device": "cpu"},
        feature_importance=feature_importance,
        _predictor=classifier,
        _serializer_payload=classifier,
    )


def train_backend(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BackendTrainingBundle:
    """Train the backend selected by ``config.model.kind``."""

    if config.model.kind == "sklearn_logreg":
        return _train_sklearn_model(config=config, X_train=X_train, y_train=y_train)
    if config.model.kind == "sklearn_random_forest":
        return _train_random_forest_model(config=config, X_train=X_train, y_train=y_train)
    if config.model.kind == "pytorch_mlp":
        return _train_pytorch_model(config=config, X_train=X_train, y_train=y_train)
    raise ValueError(f"unsupported model kind: {config.model.kind}")


def _predict_pytorch_checkpoint(checkpoint: dict[str, Any], records: pd.DataFrame) -> np.ndarray:
    feature_names = checkpoint["feature_names"]
    features = records[feature_names].to_numpy(dtype=np.float32)
    mean = np.array(checkpoint["train_mean"], dtype=np.float32)
    std = np.array(checkpoint["train_std"], dtype=np.float32)
    scaled = (features - mean) / std

    model = BinaryMLP(
        input_dim=len(feature_names),
        hidden_dims=[int(value) for value in checkpoint["hidden_dims"]],
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(scaled, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).squeeze(-1).numpy()
    return probabilities


def predict_probabilities_from_path(model_path: str | Path, records: pd.DataFrame) -> np.ndarray:
    """Load a saved backend artifact and score the provided records."""

    path = Path(model_path)
    if path.suffix == ".joblib":
        model = joblib.load(path)
        return model.predict_proba(records)[:, 1]
    if path.suffix == ".pt":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return _predict_pytorch_checkpoint(checkpoint=checkpoint, records=records)
    raise ValueError(f"unsupported model artifact: {path.suffix}")
