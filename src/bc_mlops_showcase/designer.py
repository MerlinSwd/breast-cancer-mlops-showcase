"""Run-designer draft state, validation, persistence, and launch helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import (
    DEFAULT_DATASET_KIND,
    DEFAULT_MODEL_KIND,
    DEFAULT_MODEL_PARAMS,
    TrainingConfig,
    config_to_dict,
    load_training_config,
    validate_model_device,
)


@dataclass(slots=True)
class DesignerDraft:
    """UI-friendly draft values for a pending training config."""

    experiment_name: str
    config_slug: str
    random_seed: str
    threshold: str
    dataset_kind: str
    dataset_path: str
    target_column: str
    model_kind: str
    device: str
    evaluation_mode: str
    folds: str
    test_size: str
    stratify: bool
    tracking_experiment_name: str
    model_params_json: str
    source_name: str | None = None


@dataclass(slots=True)
class DesignerValidationResult:
    """Normalized validation outcome for a designer draft."""

    ok: bool
    errors: list[str]
    resolved_config: TrainingConfig | None


@dataclass(slots=True)
class DesignerActionResult:
    """Side-effect result from saving or launching a designer draft."""

    ok: bool
    title: str
    message: str
    output: str
    config_path: Path | None = None
    run_dir: Path | None = None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "designer-run"


def build_default_designer_draft() -> DesignerDraft:
    """Return a fresh designer draft seeded from project defaults."""

    return DesignerDraft(
        experiment_name="baseline-logreg",
        config_slug="baseline-logreg",
        random_seed="42",
        threshold="0.5",
        dataset_kind=DEFAULT_DATASET_KIND,
        dataset_path="",
        target_column="target",
        model_kind=DEFAULT_MODEL_KIND,
        device="auto",
        evaluation_mode="holdout",
        folds="5",
        test_size="0.2",
        stratify=True,
        tracking_experiment_name="bc-mlops-showcase",
        model_params_json=json.dumps(
            DEFAULT_MODEL_PARAMS[DEFAULT_MODEL_KIND], indent=2, sort_keys=True
        ),
    )


def build_designer_draft_from_config(
    config: TrainingConfig, source_name: str | None = None
) -> DesignerDraft:
    """Create a UI draft from an existing training config."""

    return DesignerDraft(
        experiment_name=config.experiment_name,
        config_slug=_slugify(config.experiment_name),
        random_seed=str(config.random_seed),
        threshold=str(config.threshold),
        dataset_kind=config.dataset.kind,
        dataset_path=config.dataset.path or "",
        target_column=config.dataset.target_column,
        model_kind=config.model.kind,
        device=config.model.device,
        evaluation_mode=config.evaluation.mode,
        folds=str(config.evaluation.folds),
        test_size=str(config.split.test_size),
        stratify=bool(config.split.stratify),
        tracking_experiment_name=config.tracking.experiment_name,
        model_params_json=json.dumps(config.model.params, indent=2, sort_keys=True),
        source_name=source_name,
    )


def designer_draft_to_config(draft: DesignerDraft) -> TrainingConfig:
    """Normalize a draft into a concrete TrainingConfig."""

    params = json.loads(draft.model_params_json)
    if not isinstance(params, dict):
        raise ValueError("model params must be a JSON object")

    config = TrainingConfig()
    config.experiment_name = draft.experiment_name.strip() or "designer-run"
    config.random_seed = int(draft.random_seed)
    config.threshold = float(draft.threshold)

    config.split.test_size = float(draft.test_size)
    config.split.stratify = bool(draft.stratify)

    config.evaluation.mode = draft.evaluation_mode
    config.evaluation.folds = int(draft.folds)
    if config.evaluation.mode == "stratified_k_fold" and config.evaluation.folds < 2:
        raise ValueError("evaluation.folds must be at least 2")

    config.tracking.experiment_name = draft.tracking_experiment_name.strip() or "bc-mlops-showcase"

    config.dataset.kind = draft.dataset_kind
    config.dataset.path = draft.dataset_path.strip() or None
    config.dataset.target_column = draft.target_column.strip() or "target"

    config.model.kind = draft.model_kind
    config.model.device = validate_model_device(draft.device.strip() or "auto")
    config.model.params = params

    if config.dataset.kind == "csv_tabular_binary" and not config.dataset.path:
        raise ValueError("dataset.path is required for csv_tabular_binary")

    return config


def validate_designer_draft(draft: DesignerDraft) -> DesignerValidationResult:
    """Validate a draft and return normalized feedback instead of throwing."""

    errors: list[str] = []
    try:
        params = json.loads(draft.model_params_json)
        if not isinstance(params, dict):
            errors.append("model params must be a JSON object")
    except json.JSONDecodeError as exc:
        errors.append(f"model params JSON is invalid: {exc.msg}")
        params = None

    if draft.dataset_kind == "csv_tabular_binary" and not draft.dataset_path.strip():
        errors.append("dataset.path is required for csv_tabular_binary")

    try:
        if int(draft.folds) < 2 and draft.evaluation_mode == "stratified_k_fold":
            errors.append("evaluation.folds must be at least 2")
    except ValueError:
        errors.append("evaluation.folds must be an integer")

    if errors:
        return DesignerValidationResult(ok=False, errors=errors, resolved_config=None)

    try:
        config = designer_draft_to_config(draft)
    except (TypeError, ValueError) as exc:
        return DesignerValidationResult(ok=False, errors=[str(exc)], resolved_config=None)

    del params
    return DesignerValidationResult(ok=True, errors=[], resolved_config=config)


def render_designer_preview_text(draft: DesignerDraft) -> str:
    """Render the normalized YAML preview shown in the designer pane."""

    result = validate_designer_draft(draft)
    if not result.ok or result.resolved_config is None:
        return "\n".join(["Normalized config preview", "", *result.errors])

    payload = config_to_dict(result.resolved_config)
    preview = yaml.safe_dump(payload, sort_keys=False).strip()
    return f"Normalized config preview\n\n{preview}"


def save_designer_draft(draft: DesignerDraft, config_root: Path) -> Path:
    """Persist a normalized draft into the config directory."""

    result = validate_designer_draft(draft)
    if not result.ok or result.resolved_config is None:
        raise ValueError("cannot save invalid designer draft")

    config_root.mkdir(parents=True, exist_ok=True)
    slug = _slugify(draft.config_slug or draft.experiment_name)
    path = config_root / f"{slug}.yaml"
    if path.exists():
        raise FileExistsError(path)

    payload = config_to_dict(result.resolved_config)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def launch_designer_run(
    draft: DesignerDraft, *, config_root: Path, output_root: Path
) -> DesignerActionResult:
    """Save a draft, train it, and return a structured outcome."""

    from .pipeline import train_and_evaluate

    try:
        config_path = save_designer_draft(draft, config_root)
        config = load_training_config(config_path)
        result = train_and_evaluate(config=config, output_root=output_root)
    except Exception as exc:  # pragma: no cover - exercised via failure tests later if needed
        return DesignerActionResult(
            ok=False,
            title="Run designer launch failed",
            message=f"Could not launch {draft.experiment_name}: {exc}",
            output=str(exc),
        )

    return DesignerActionResult(
        ok=True,
        title="Run launched",
        message=f"Launched {draft.experiment_name} from {config_path.name}.",
        output=json.dumps(result.summary(), indent=2),
        config_path=config_path,
        run_dir=result.run_dir,
    )
