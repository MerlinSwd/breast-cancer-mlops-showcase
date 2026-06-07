"""Interactive model-designer helpers for building and validating model configs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import yaml

from .config import (
    DEFAULT_MODEL_KIND,
    DEFAULT_MODEL_PARAMS,
    MODEL_DEVICE_OPTIONS,
    SUPPORTED_MODEL_KINDS,
    ModelConfig,
    TrainingConfig,
)
from .designer import DesignerDraft

SUPPORTED_MODEL_KINDS = tuple(SUPPORTED_MODEL_KINDS)
MODEL_DEVICE_OPTIONS: tuple[str, ...] = tuple(MODEL_DEVICE_OPTIONS)


@dataclass(slots=True)
class ModelDesignerDraft:
    model_kind: str
    device: str
    logreg_c: str
    logreg_max_iter: str
    rf_n_estimators: str
    rf_max_depth: str
    rf_min_samples_leaf: str
    hgb_learning_rate: str
    hgb_max_iter: str
    hgb_max_depth: str
    hgb_min_samples_leaf: str
    mlp_hidden_dims: str
    mlp_epochs: str
    mlp_batch_size: str
    mlp_learning_rate: str
    mlp_dropout: str
    source_name: str | None = None


@dataclass(slots=True)
class ModelDesignerValidationResult:
    ok: bool
    errors: list[str]
    resolved_model: ModelConfig | None


def _stringify_max_depth(value: object) -> str:
    return "" if value is None else str(value)


def _hidden_dims_to_string(value: object) -> str:
    dims = value if isinstance(value, list) else []
    return ",".join(str(item) for item in dims)


def build_default_model_designer_draft(
    model_kind: str = DEFAULT_MODEL_KIND,
) -> ModelDesignerDraft:
    if model_kind not in DEFAULT_MODEL_PARAMS:
        model_kind = DEFAULT_MODEL_KIND
    defaults = deepcopy(DEFAULT_MODEL_PARAMS[model_kind])
    return build_model_designer_draft_from_model_config(
        ModelConfig(kind=model_kind, device="auto", params=defaults)
    )


def build_model_designer_draft_from_model_config(
    model: ModelConfig, source_name: str | None = None
) -> ModelDesignerDraft:
    params = deepcopy(
        DEFAULT_MODEL_PARAMS.get(model.kind, DEFAULT_MODEL_PARAMS[DEFAULT_MODEL_KIND])
    )
    params.update(model.params)
    return ModelDesignerDraft(
        model_kind=model.kind,
        device=model.device,
        logreg_c=str(params.get("c", 1.0)),
        logreg_max_iter=str(params.get("max_iter", 500)),
        rf_n_estimators=str(params.get("n_estimators", 200)),
        rf_max_depth=_stringify_max_depth(params.get("max_depth")),
        rf_min_samples_leaf=str(params.get("min_samples_leaf", 1)),
        hgb_learning_rate=str(params.get("learning_rate", 0.1)),
        hgb_max_iter=str(params.get("max_iter", 200)),
        hgb_max_depth=_stringify_max_depth(params.get("max_depth")),
        hgb_min_samples_leaf=str(params.get("min_samples_leaf", 20)),
        mlp_hidden_dims=_hidden_dims_to_string(params.get("hidden_dims", [32, 16])),
        mlp_epochs=str(params.get("epochs", 20)),
        mlp_batch_size=str(params.get("batch_size", 32)),
        mlp_learning_rate=str(params.get("learning_rate", 0.01)),
        mlp_dropout=str(params.get("dropout", 0.1)),
        source_name=source_name,
    )


def build_model_designer_draft_from_training_config(
    config: TrainingConfig, source_name: str | None = None
) -> ModelDesignerDraft:
    return build_model_designer_draft_from_model_config(config.model, source_name=source_name)


def _parse_hidden_dims(raw: str) -> list[int]:
    try:
        dims = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("hidden_dims must be a comma-separated list of integers") from exc
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError("hidden_dims must contain positive integers")
    return dims


def _parse_optional_int(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("max_depth must be at least 1")
    return parsed


def model_designer_draft_to_model_config(draft: ModelDesignerDraft) -> ModelConfig:
    if draft.model_kind not in SUPPORTED_MODEL_KINDS:
        raise ValueError(f"unsupported model kind: {draft.model_kind}")
    if draft.device not in MODEL_DEVICE_OPTIONS:
        raise ValueError(f"unsupported device: {draft.device}")

    if draft.model_kind == "sklearn_logreg":
        c = float(draft.logreg_c)
        max_iter = int(draft.logreg_max_iter)
        if c <= 0:
            raise ValueError("c must be greater than 0")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        params = {"c": c, "max_iter": max_iter}
    elif draft.model_kind == "sklearn_random_forest":
        n_estimators = int(draft.rf_n_estimators)
        min_samples_leaf = int(draft.rf_min_samples_leaf)
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        params = {
            "n_estimators": n_estimators,
            "max_depth": _parse_optional_int(draft.rf_max_depth),
            "min_samples_leaf": min_samples_leaf,
        }
    elif draft.model_kind == "sklearn_hist_gradient_boosting":
        learning_rate = float(draft.hgb_learning_rate)
        max_iter = int(draft.hgb_max_iter)
        min_samples_leaf = int(draft.hgb_min_samples_leaf)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        params = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_depth": _parse_optional_int(draft.hgb_max_depth),
            "min_samples_leaf": min_samples_leaf,
        }
    else:
        epochs = int(draft.mlp_epochs)
        batch_size = int(draft.mlp_batch_size)
        learning_rate = float(draft.mlp_learning_rate)
        dropout = float(draft.mlp_dropout)
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        params = {
            "hidden_dims": _parse_hidden_dims(draft.mlp_hidden_dims),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
        }

    return ModelConfig(kind=draft.model_kind, device=draft.device, params=params)


def validate_model_designer_draft(draft: ModelDesignerDraft) -> ModelDesignerValidationResult:
    try:
        resolved = model_designer_draft_to_model_config(draft)
    except (TypeError, ValueError) as exc:
        return ModelDesignerValidationResult(ok=False, errors=[str(exc)], resolved_model=None)
    return ModelDesignerValidationResult(ok=True, errors=[], resolved_model=resolved)


def render_model_designer_preview_text(draft: ModelDesignerDraft) -> str:
    result = validate_model_designer_draft(draft)
    if not result.ok or result.resolved_model is None:
        return "\n".join(["Normalized model preview", "", *result.errors])
    preview = yaml.safe_dump(
        {
            "model": {
                "kind": result.resolved_model.kind,
                "device": result.resolved_model.device,
                "params": deepcopy(result.resolved_model.params),
            }
        },
        sort_keys=False,
    ).strip()
    return f"Normalized model preview\n\n{preview}"


def apply_model_designer_draft_to_run_draft(
    model_draft: ModelDesignerDraft, run_draft: DesignerDraft
) -> DesignerDraft:
    result = validate_model_designer_draft(model_draft)
    if not result.ok or result.resolved_model is None:
        raise ValueError("cannot apply invalid model designer draft")
    updated = deepcopy(run_draft)
    updated.model_kind = result.resolved_model.kind
    updated.device = result.resolved_model.device
    updated.model_params_json = __import__("json").dumps(
        result.resolved_model.params, indent=2, sort_keys=True
    )
    return updated
