"""Model backend implementations and artifact loading helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from math import isqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainingConfig

ARTIFACT_METADATA_VERSION = "1.0"
TASK_KIND_BINARY_CLASSIFICATION = "binary_classification"


@dataclass(frozen=True, slots=True)
class ArtifactLoaderSpec:
    """Serializer/loader contract for a persisted model artifact format."""

    name: str
    version: str
    suffixes: tuple[str, ...]
    save_payload: Callable[[Any, Path], None]
    load_payload: Callable[[Path], Any]
    predict_probabilities: Callable[[Any, pd.DataFrame], np.ndarray]


def _save_joblib_payload(payload: Any, destination: Path) -> None:
    joblib.dump(payload, destination)


def _load_joblib_payload(path: Path) -> Any:
    return joblib.load(path)


def _predict_joblib_payload(payload: Any, records: pd.DataFrame) -> np.ndarray:
    return payload.predict_proba(records)[:, 1]


def _save_torch_checkpoint(payload: Any, destination: Path) -> None:
    torch.save(payload, destination)


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _predict_torch_checkpoint_payload(
    checkpoint: dict[str, Any], records: pd.DataFrame
) -> np.ndarray:
    predictors = {
        "pytorch_cnn": _predict_pytorch_cnn_checkpoint,
        "pytorch_mlp": _predict_pytorch_checkpoint,
    }
    predictor = predictors.get(checkpoint.get("kind"), _predict_pytorch_checkpoint)
    return predictor(checkpoint=checkpoint, records=records)


ARTIFACT_LOADER_SPECS: dict[str, ArtifactLoaderSpec] = {
    "joblib_binary_classifier": ArtifactLoaderSpec(
        name="joblib_binary_classifier",
        version=ARTIFACT_METADATA_VERSION,
        suffixes=(".joblib",),
        save_payload=_save_joblib_payload,
        load_payload=_load_joblib_payload,
        predict_probabilities=_predict_joblib_payload,
    ),
    "torch_checkpoint_binary_classifier": ArtifactLoaderSpec(
        name="torch_checkpoint_binary_classifier",
        version=ARTIFACT_METADATA_VERSION,
        suffixes=(".pt",),
        save_payload=_save_torch_checkpoint,
        load_payload=_load_torch_checkpoint,
        predict_probabilities=_predict_torch_checkpoint_payload,
    ),
}

ARTIFACT_SUFFIX_TO_LOADER = {
    suffix: spec.name
    for spec in ARTIFACT_LOADER_SPECS.values()
    for suffix in spec.suffixes
}


def get_artifact_loader_spec(loader_name: str) -> ArtifactLoaderSpec:
    """Return the registered artifact-loader spec for ``loader_name``."""

    try:
        return ARTIFACT_LOADER_SPECS[loader_name]
    except KeyError as exc:
        raise ValueError(f"unsupported artifact loader: {loader_name}") from exc


def load_run_metadata(model_path: str | Path) -> dict[str, Any] | None:
    """Load adjacent run metadata when present and valid."""

    metadata_path = Path(model_path).with_name("metadata.json")
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_loader_name_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    model = metadata.get("model", {})
    if not isinstance(model, dict):
        return None
    artifact = model.get("artifact", {})
    if isinstance(artifact, dict):
        loader = artifact.get("loader")
        if isinstance(loader, str):
            return loader
    return None


def resolve_artifact_loader(
    model_path: str | Path, metadata: dict[str, Any] | None = None
) -> ArtifactLoaderSpec:
    """Resolve the artifact loader from metadata first, then suffix fallback."""

    loader_name = _resolve_loader_name_from_metadata(metadata)
    if loader_name is not None:
        return get_artifact_loader_spec(loader_name)

    path = Path(model_path)
    suffix_loader = ARTIFACT_SUFFIX_TO_LOADER.get(path.suffix.lower())
    if suffix_loader is None:
        raise ValueError(f"unsupported model artifact: {path.suffix}")
    return get_artifact_loader_spec(suffix_loader)


@dataclass(slots=True)
class BackendTrainingBundle:
    """In-memory training result plus serialization metadata for a backend."""

    kind: str
    artifact_filename: str
    artifact_loader: str
    feature_names: list[str]
    runtime: dict[str, Any]
    feature_importance: pd.DataFrame | None
    _predictor: Any
    _serializer_payload: Any

    def save(self, destination: Path) -> None:
        """Persist the trained backend artifact to disk."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        loader = get_artifact_loader_spec(self.artifact_loader)
        loader.save_payload(self._serializer_payload, destination)

    def artifact_metadata(self) -> dict[str, str]:
        """Return the metadata contract for this backend artifact."""

        loader = get_artifact_loader_spec(self.artifact_loader)
        return {
            "filename": self.artifact_filename,
            "format": loader.name,
            "loader": loader.name,
            "version": loader.version,
        }

    def predict_probabilities(self, records: pd.DataFrame) -> np.ndarray:
        """Predict positive-class probabilities for input records."""

        if self.kind.startswith("sklearn_"):
            return self._predictor.predict_proba(records[self.feature_names])[:, 1]
        if self.kind == "pytorch_cnn":
            return _predict_pytorch_cnn_checkpoint(
                checkpoint=self._serializer_payload,
                records=records[self.feature_names],
            )
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


class BinaryCNN(nn.Module):
    """Compact convolutional network for small grayscale image classification."""

    def __init__(self, conv_channels: list[int], kernel_size: int, hidden_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        padding = kernel_size // 2
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.features(x)
        pooled = self.pool(encoded)
        return self.classifier(pooled)


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
        artifact_loader="joblib_binary_classifier",
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
        artifact_loader="torch_checkpoint_binary_classifier",
        feature_names=list(X_train.columns),
        runtime={"framework": "pytorch", "device": resolved_device},
        feature_importance=feature_importance,
        _predictor=model.cpu(),
        _serializer_payload=checkpoint,
    )


def _reshape_flattened_images(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    side_length = isqrt(frame.shape[1])
    if side_length * side_length != frame.shape[1]:
        raise ValueError("pytorch_cnn expects flattened square image features")
    images = frame.to_numpy(dtype=np.float32).reshape(-1, 1, side_length, side_length)
    return images, side_length


def _train_pytorch_cnn_model(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BackendTrainingBundle:
    resolved_device = resolve_device(config.model.device)
    device = torch.device(resolved_device)
    conv_channels = [int(value) for value in config.model.params.get("conv_channels", [8, 16])]
    kernel_size = int(config.model.params.get("kernel_size", 3))
    hidden_dim = int(config.model.params.get("hidden_dim", 32))
    batch_size = int(config.model.params.get("batch_size", 32))
    epochs = int(config.model.params.get("epochs", 8))
    learning_rate = float(config.model.params.get("learning_rate", 0.005))

    train_images, side_length = _reshape_flattened_images(X_train)
    train_targets = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

    dataset = TensorDataset(
        torch.tensor(train_images / 16.0, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(config.random_seed)
    model = BinaryCNN(
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for batch_images, batch_targets in loader:
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

    checkpoint = {
        "kind": "pytorch_cnn",
        "feature_names": list(X_train.columns),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "conv_channels": conv_channels,
        "kernel_size": kernel_size,
        "hidden_dim": hidden_dim,
        "image_side_length": side_length,
    }
    return BackendTrainingBundle(
        kind="pytorch_cnn",
        artifact_filename="model.pt",
        artifact_loader="torch_checkpoint_binary_classifier",
        feature_names=list(X_train.columns),
        runtime={"framework": "pytorch", "device": resolved_device},
        feature_importance=None,
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
        artifact_loader="joblib_binary_classifier",
        feature_names=list(X_train.columns),
        runtime={"framework": "sklearn", "device": "cpu"},
        feature_importance=feature_importance,
        _predictor=classifier,
        _serializer_payload=classifier,
    )


def _train_hist_gradient_boosting_model(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BackendTrainingBundle:
    classifier = HistGradientBoostingClassifier(
        learning_rate=float(config.model.params.get("learning_rate", 0.1)),
        max_iter=int(config.model.params.get("max_iter", 200)),
        max_depth=(
            None
            if config.model.params.get("max_depth") is None
            else int(config.model.params["max_depth"])
        ),
        min_samples_leaf=int(config.model.params.get("min_samples_leaf", 20)),
        random_state=config.random_seed,
    )
    classifier.fit(X_train, y_train)
    return BackendTrainingBundle(
        kind="sklearn_hist_gradient_boosting",
        artifact_filename="model.joblib",
        artifact_loader="joblib_binary_classifier",
        feature_names=list(X_train.columns),
        runtime={"framework": "sklearn", "device": "cpu"},
        feature_importance=None,
        _predictor=classifier,
        _serializer_payload=classifier,
    )


def train_backend(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BackendTrainingBundle:
    """Train the backend selected by ``config.model.kind``."""

    trainers = {
        "sklearn_logreg": _train_sklearn_model,
        "sklearn_random_forest": _train_random_forest_model,
        "sklearn_hist_gradient_boosting": _train_hist_gradient_boosting_model,
        "pytorch_mlp": _train_pytorch_model,
        "pytorch_cnn": _train_pytorch_cnn_model,
    }
    try:
        trainer = trainers[config.model.kind]
    except KeyError as exc:
        raise ValueError(f"unsupported model kind: {config.model.kind}") from exc
    return trainer(config=config, X_train=X_train, y_train=y_train)


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


def _predict_pytorch_cnn_checkpoint(
    checkpoint: dict[str, Any], records: pd.DataFrame
) -> np.ndarray:
    feature_names = checkpoint["feature_names"]
    images, _ = _reshape_flattened_images(records[feature_names])

    model = BinaryCNN(
        conv_channels=[int(value) for value in checkpoint["conv_channels"]],
        kernel_size=int(checkpoint["kernel_size"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(images / 16.0, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).squeeze(-1).numpy()
    return probabilities


def predict_probabilities_from_path(
    model_path: str | Path,
    records: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a saved backend artifact and score the provided records."""

    path = Path(model_path)
    resolved_metadata = metadata if metadata is not None else load_run_metadata(path)
    loader = resolve_artifact_loader(model_path=path, metadata=resolved_metadata)
    payload = loader.load_payload(path)
    return loader.predict_probabilities(payload, records)
