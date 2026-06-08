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


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Static metadata about a supported dataset kind."""

    kind: str
    label: str
    modality: str


@dataclass(frozen=True, slots=True)
class ModelParamSpec:
    """Canonical definition for one editable model parameter."""

    name: str
    label: str
    kind: str
    default: Any
    placeholder: str = ""
    help_text: str = ""
    order: int = 0
    min_value: float | int | None = None
    min_exclusive: float | int | None = None
    max_value: float | int | None = None
    max_exclusive: float | int | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static metadata about a supported model backend."""

    kind: str
    label: str
    input_modality: str
    default_experiment_name: str
    supported_evaluation_modes: tuple[str, ...]
    parameter_specs: tuple[ModelParamSpec, ...]


DEFAULT_MODEL_KIND = "sklearn_logreg"
DEFAULT_DATASET_KIND = "sklearn_breast_cancer"
DATASET_SPECS: dict[str, DatasetSpec] = {
    "sklearn_breast_cancer": DatasetSpec(
        kind="sklearn_breast_cancer",
        label="Wisconsin breast cancer",
        modality="tabular",
    ),
    "csv_tabular_binary": DatasetSpec(
        kind="csv_tabular_binary",
        label="CSV tabular binary dataset",
        modality="tabular",
    ),
    "sklearn_digits_binary": DatasetSpec(
        kind="sklearn_digits_binary",
        label="Digits 0 vs 1",
        modality="image",
    ),
}
MODEL_SPECS: dict[str, ModelSpec] = {
    "sklearn_logreg": ModelSpec(
        kind="sklearn_logreg",
        label="Logistic regression",
        input_modality="tabular",
        default_experiment_name="baseline-logreg",
        supported_evaluation_modes=("holdout", "stratified_k_fold"),
        parameter_specs=(
            ModelParamSpec(
                name="c",
                label="C",
                kind="float",
                default=1.0,
                placeholder="Logreg C",
                min_exclusive=0.0,
            ),
            ModelParamSpec(
                name="max_iter",
                label="Max iterations",
                kind="int",
                default=500,
                placeholder="Logreg max_iter",
                min_value=1,
            ),
        ),
    ),
    "sklearn_random_forest": ModelSpec(
        kind="sklearn_random_forest",
        label="Random forest",
        input_modality="tabular",
        default_experiment_name="baseline-random-forest",
        supported_evaluation_modes=("holdout", "stratified_k_fold"),
        parameter_specs=(
            ModelParamSpec(
                name="n_estimators",
                label="Trees",
                kind="int",
                default=200,
                placeholder="RF n_estimators",
                min_value=1,
            ),
            ModelParamSpec(
                name="max_depth",
                label="Max depth",
                kind="optional_int",
                default=None,
                placeholder="RF max_depth",
                min_value=1,
            ),
            ModelParamSpec(
                name="min_samples_leaf",
                label="Min samples leaf",
                kind="int",
                default=1,
                placeholder="RF min_samples_leaf",
                min_value=1,
            ),
        ),
    ),
    "sklearn_hist_gradient_boosting": ModelSpec(
        kind="sklearn_hist_gradient_boosting",
        label="Hist gradient boosting",
        input_modality="tabular",
        default_experiment_name="baseline-hist-gradient-boosting",
        supported_evaluation_modes=("holdout", "stratified_k_fold"),
        parameter_specs=(
            ModelParamSpec(
                name="learning_rate",
                label="Learning rate",
                kind="float",
                default=0.1,
                placeholder="HGB learning_rate",
                min_exclusive=0.0,
            ),
            ModelParamSpec(
                name="max_iter",
                label="Max iterations",
                kind="int",
                default=200,
                placeholder="HGB max_iter",
                min_value=1,
            ),
            ModelParamSpec(
                name="max_depth",
                label="Max depth",
                kind="optional_int",
                default=None,
                placeholder="HGB max_depth",
                min_value=1,
            ),
            ModelParamSpec(
                name="min_samples_leaf",
                label="Min samples leaf",
                kind="int",
                default=20,
                placeholder="HGB min_samples_leaf",
                min_value=1,
            ),
        ),
    ),
    "pytorch_mlp": ModelSpec(
        kind="pytorch_mlp",
        label="PyTorch MLP",
        input_modality="tabular",
        default_experiment_name="baseline-pytorch-mlp",
        supported_evaluation_modes=("holdout",),
        parameter_specs=(
            ModelParamSpec(
                name="hidden_dims",
                label="Hidden dims",
                kind="int_list",
                default=[32, 16],
                placeholder="MLP hidden_dims",
                min_value=1,
            ),
            ModelParamSpec(
                name="epochs",
                label="Epochs",
                kind="int",
                default=20,
                placeholder="MLP epochs",
                min_value=1,
            ),
            ModelParamSpec(
                name="batch_size",
                label="Batch size",
                kind="int",
                default=32,
                placeholder="MLP batch_size",
                min_value=1,
            ),
            ModelParamSpec(
                name="learning_rate",
                label="Learning rate",
                kind="float",
                default=0.01,
                placeholder="MLP learning_rate",
                min_exclusive=0.0,
            ),
            ModelParamSpec(
                name="dropout",
                label="Dropout",
                kind="float",
                default=0.1,
                placeholder="MLP dropout",
                min_value=0.0,
                max_exclusive=1.0,
            ),
        ),
    ),
    "pytorch_cnn": ModelSpec(
        kind="pytorch_cnn",
        label="PyTorch CNN",
        input_modality="image",
        default_experiment_name="baseline-pytorch-cnn",
        supported_evaluation_modes=("holdout",),
        parameter_specs=(
            ModelParamSpec(
                name="conv_channels",
                label="Conv channels",
                kind="int_list",
                default=[8, 16],
                placeholder="CNN conv_channels",
                min_value=1,
            ),
            ModelParamSpec(
                name="kernel_size",
                label="Kernel size",
                kind="int",
                default=3,
                placeholder="CNN kernel_size",
                min_value=1,
            ),
            ModelParamSpec(
                name="epochs",
                label="Epochs",
                kind="int",
                default=8,
                placeholder="CNN epochs",
                min_value=1,
            ),
            ModelParamSpec(
                name="batch_size",
                label="Batch size",
                kind="int",
                default=32,
                placeholder="CNN batch_size",
                min_value=1,
            ),
            ModelParamSpec(
                name="learning_rate",
                label="Learning rate",
                kind="float",
                default=0.005,
                placeholder="CNN learning_rate",
                min_exclusive=0.0,
            ),
            ModelParamSpec(
                name="hidden_dim",
                label="Hidden dim",
                kind="int",
                default=32,
                placeholder="CNN hidden_dim",
                min_value=1,
            ),
        ),
    ),
}
SUPPORTED_DATASET_KINDS: tuple[str, ...] = tuple(DATASET_SPECS)
MODEL_KIND_LABELS: dict[str, str] = {kind: spec.label for kind, spec in MODEL_SPECS.items()}
SUPPORTED_MODEL_KINDS: tuple[str, ...] = tuple(MODEL_SPECS)
MODEL_KIND_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (spec.label, kind) for kind, spec in MODEL_SPECS.items()
)
MODEL_DEVICE_OPTIONS: tuple[str, ...] = ("auto", "cpu", "cuda")


def validate_model_device(device: str) -> str:
    """Validate and normalize the requested runtime device."""

    if device not in MODEL_DEVICE_OPTIONS:
        raise ValueError(
            f"unsupported model device: {device}; expected one of {', '.join(MODEL_DEVICE_OPTIONS)}"
        )
    return device


def get_dataset_spec(kind: str) -> DatasetSpec:
    """Return the registered metadata for a dataset kind."""

    try:
        return DATASET_SPECS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset kind: {kind}") from exc


def get_model_spec(kind: str) -> ModelSpec:
    """Return the registered metadata for a model backend kind."""

    try:
        return MODEL_SPECS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported model kind: {kind}") from exc


def get_model_param_specs(kind: str) -> tuple[ModelParamSpec, ...]:
    """Return the registered parameter schema for a model backend."""

    return get_model_spec(kind).parameter_specs


def _parse_int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        try:
            parsed = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated list of integers"
            ) from exc
    elif isinstance(value, list):
        try:
            parsed = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a list of integers") from exc
    else:
        raise ValueError(f"{name} must be a list of integers")

    if not parsed:
        raise ValueError(f"{name} must contain positive integers")
    return parsed


def _coerce_model_param_value(spec: ModelParamSpec, value: Any) -> Any:
    if spec.kind == "int":
        parsed = int(value)
    elif spec.kind == "float":
        parsed = float(value)
    elif spec.kind == "optional_int":
        if value is None:
            parsed = None
        elif isinstance(value, str) and not value.strip():
            parsed = None
        else:
            parsed = int(value)
    elif spec.kind == "int_list":
        parsed = _parse_int_list(value, name=spec.name)
    else:  # pragma: no cover - registry guardrail
        raise ValueError(f"unsupported model param kind: {spec.kind}")

    if parsed is None:
        return None

    values_to_check = parsed if isinstance(parsed, list) else [parsed]
    for item in values_to_check:
        if spec.min_value is not None and item < spec.min_value:
            raise ValueError(f"{spec.name} must be at least {spec.min_value}")
        if spec.min_exclusive is not None and item <= spec.min_exclusive:
            raise ValueError(f"{spec.name} must be greater than {spec.min_exclusive}")
        if spec.max_value is not None and item > spec.max_value:
            raise ValueError(f"{spec.name} must be at most {spec.max_value}")
        if spec.max_exclusive is not None and item >= spec.max_exclusive:
            if spec.min_value == 0 and spec.max_exclusive == 1:
                raise ValueError(f"{spec.name} must be in [0, {spec.max_exclusive})")
            raise ValueError(f"{spec.name} must be less than {spec.max_exclusive}")

    return parsed


def get_default_model_params(kind: str) -> dict[str, Any]:
    """Build default parameters for a model backend from its registered schema."""

    return {spec.name: deepcopy(spec.default) for spec in get_model_param_specs(kind)}


DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    kind: get_default_model_params(kind) for kind in MODEL_SPECS
}


def normalize_model_params(
    kind: str,
    params: dict[str, Any] | None,
    *,
    allow_unknown: bool = False,
    merge_defaults: bool = True,
) -> dict[str, Any]:
    """Normalize model params against the registered backend schema."""

    raw_params = params or {}
    if not isinstance(raw_params, dict):
        raise ValueError("model params must be a mapping")

    specs = get_model_param_specs(kind)
    specs_by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(raw_params) - set(specs_by_name))
    if unknown and not allow_unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown model params for {kind}: {joined}")

    normalized = get_default_model_params(kind) if merge_defaults else {}
    for name, value in raw_params.items():
        spec = specs_by_name.get(name)
        if spec is None:
            continue
        normalized[name] = _coerce_model_param_value(spec, value)
    return normalized


@dataclass(slots=True)
class SplitConfig:
    """Dataset split settings for train/test evaluation."""

    test_size: float = 0.2
    stratify: bool = True


@dataclass(slots=True)
class EvaluationConfig:
    """Evaluation strategy configuration."""

    mode: str = "holdout"
    folds: int = 5


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
    """Model family and hyperparameters."""

    kind: str = DEFAULT_MODEL_KIND
    device: str = "auto"
    params: dict[str, Any] = field(
        default_factory=lambda: get_default_model_params(DEFAULT_MODEL_KIND)
    )


@dataclass(slots=True)
class TrainingConfig:
    """Top-level configuration for a training run."""

    experiment_name: str = "baseline-logreg"
    random_seed: int = 42
    threshold: float = 0.5
    split: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _merge_dataclass(default: Any, values: dict[str, Any] | None) -> Any:
    data = values or {}
    return type(default)(**{**asdict(default), **data})


def _resolve_dataset_config(values: dict[str, Any] | None) -> DatasetConfig:
    raw = values or {}
    kind = raw.get("kind", DEFAULT_DATASET_KIND)
    get_dataset_spec(kind)
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
    get_model_spec(kind)

    normalized_params = normalize_model_params(kind, raw.get("params"), allow_unknown=False)
    return ModelConfig(
        kind=kind,
        device=validate_model_device(str(raw.get("device", "auto"))),
        params=normalized_params,
    )


def _resolve_evaluation_config(values: dict[str, Any] | None) -> EvaluationConfig:
    raw = values or {}
    mode = raw.get("mode", "holdout")
    if mode not in {"holdout", "stratified_k_fold"}:
        raise ValueError(f"unsupported evaluation mode: {mode}")

    folds = int(raw.get("folds", 5))
    if folds < 2:
        raise ValueError("evaluation.folds must be at least 2")

    return EvaluationConfig(mode=mode, folds=folds)


def _validate_dataset_model_compatibility(
    dataset: DatasetConfig,
    model: ModelConfig,
    evaluation: EvaluationConfig,
) -> None:
    dataset_spec = get_dataset_spec(dataset.kind)
    model_spec = get_model_spec(model.kind)

    if dataset_spec.modality != model_spec.input_modality:
        if model.kind == "pytorch_cnn":
            raise ValueError(
                "pytorch_cnn currently requires an image dataset with "
                "flattened square pixel features"
            )
        raise ValueError(
            f"model kind {model.kind} expects {model_spec.input_modality} "
            f"data, got {dataset.kind} ({dataset_spec.modality})"
        )

    if evaluation.mode not in model_spec.supported_evaluation_modes:
        supported = ", ".join(model_spec.supported_evaluation_modes)
        raise ValueError(
            f"evaluation mode {evaluation.mode} is not supported for "
            f"{model.kind}; expected one of {supported}"
        )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a YAML training configuration from disk."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}

    default = TrainingConfig()
    dataset = _resolve_dataset_config(raw.get("dataset"))
    model = _resolve_model_config(raw.get("model"))
    evaluation = _resolve_evaluation_config(raw.get("evaluation"))
    _validate_dataset_model_compatibility(dataset=dataset, model=model, evaluation=evaluation)
    experiment_name = (
        raw.get("experiment_name")
        or get_model_spec(model.kind).default_experiment_name
    )
    return TrainingConfig(
        experiment_name=experiment_name,
        random_seed=raw.get("random_seed", default.random_seed),
        threshold=raw.get("threshold", default.threshold),
        split=_merge_dataclass(default.split, raw.get("split")),
        evaluation=evaluation,
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
        "evaluation": {
            "mode": config.evaluation.mode,
            "folds": config.evaluation.folds,
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
