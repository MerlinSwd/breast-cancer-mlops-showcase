"""Interactive model-designer helpers for building and validating model configs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import yaml

from .config import (
    DEFAULT_MODEL_KIND,
    MODEL_DEVICE_OPTIONS,
    ModelConfig,
    ModelParamSpec,
    TrainingConfig,
    get_default_model_params,
    get_model_param_specs,
    get_model_spec,
    normalize_model_params,
    validate_model_device,
)
from .designer import DesignerDraft

MODEL_DEVICE_OPTIONS: tuple[str, ...] = tuple(MODEL_DEVICE_OPTIONS)


@dataclass(slots=True)
class ModelDesignerDraft:
    model_kind: str
    device: str
    param_values: dict[str, str]
    source_name: str | None = None


@dataclass(frozen=True, slots=True)
class ModelDesignerFieldSpec:
    name: str
    input_id: str
    placeholder: str


@dataclass(slots=True)
class ModelDesignerValidationResult:
    ok: bool
    errors: list[str]
    resolved_model: ModelConfig | None


def _format_model_param_value(spec: ModelParamSpec, value: object) -> str:
    if value is None:
        return ""
    if spec.kind == "int_list":
        items = value if isinstance(value, list) else []
        return ",".join(str(item) for item in items)
    return str(value)



def iter_model_designer_fields(model_kind: str) -> tuple[ModelDesignerFieldSpec, ...]:
    """Return the UI field specs for one registered model family."""

    return tuple(
        ModelDesignerFieldSpec(
            name=spec.name,
            input_id=f"model-designer-param-{spec.name.replace('_', '-')}",
            placeholder=spec.placeholder or spec.label,
        )
        for spec in get_model_param_specs(model_kind)
    )



def iter_all_model_designer_fields() -> tuple[ModelDesignerFieldSpec, ...]:
    """Return the union of all model-designer fields across registered backends."""

    fields: dict[str, ModelDesignerFieldSpec] = {}
    for model_kind in get_registered_model_kinds():
        for field in iter_model_designer_fields(model_kind):
            fields.setdefault(field.name, field)
    return tuple(fields[name] for name in sorted(fields))



def get_registered_model_kinds() -> tuple[str, ...]:
    """Return all registered model kinds in declaration order."""

    from .config import SUPPORTED_MODEL_KINDS

    return tuple(SUPPORTED_MODEL_KINDS)



def build_default_model_designer_draft(
    model_kind: str = DEFAULT_MODEL_KIND,
) -> ModelDesignerDraft:
    try:
        get_model_spec(model_kind)
    except ValueError:
        model_kind = DEFAULT_MODEL_KIND
    params = get_default_model_params(model_kind)
    return build_model_designer_draft_from_model_config(
        ModelConfig(kind=model_kind, device="auto", params=params)
    )



def build_model_designer_draft_from_model_config(
    model: ModelConfig, source_name: str | None = None
) -> ModelDesignerDraft:
    normalized_params = normalize_model_params(
        model.kind,
        model.params,
        allow_unknown=True,
        merge_defaults=True,
    )
    param_values = {
        spec.name: _format_model_param_value(spec, normalized_params.get(spec.name))
        for spec in get_model_param_specs(model.kind)
    }
    return ModelDesignerDraft(
        model_kind=model.kind,
        device=model.device,
        param_values=param_values,
        source_name=source_name,
    )



def build_model_designer_draft_from_training_config(
    config: TrainingConfig, source_name: str | None = None
) -> ModelDesignerDraft:
    return build_model_designer_draft_from_model_config(config.model, source_name=source_name)



def model_designer_draft_to_model_config(draft: ModelDesignerDraft) -> ModelConfig:
    get_model_spec(draft.model_kind)
    device = validate_model_device(draft.device)
    params = normalize_model_params(
        draft.model_kind,
        draft.param_values,
        allow_unknown=True,
        merge_defaults=True,
    )
    return ModelConfig(kind=draft.model_kind, device=device, params=params)



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
