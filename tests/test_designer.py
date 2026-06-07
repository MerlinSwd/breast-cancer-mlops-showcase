import json
from pathlib import Path

import pytest
import yaml

from bc_mlops_showcase.config import TrainingConfig
from bc_mlops_showcase.designer import (
    DesignerActionResult,
    DesignerDraft,
    build_default_designer_draft,
    build_designer_draft_from_config,
    designer_draft_to_config,
    launch_designer_run,
    render_designer_preview_text,
    save_designer_draft,
    validate_designer_draft,
)


def test_build_default_designer_draft_uses_project_defaults() -> None:
    draft = build_default_designer_draft()

    assert draft.experiment_name == "baseline-logreg"
    assert draft.config_slug == "baseline-logreg"
    assert draft.dataset_kind == "sklearn_breast_cancer"
    assert draft.model_kind == "sklearn_logreg"
    assert json.loads(draft.model_params_json)["max_iter"] == 500


def test_build_designer_draft_from_config_clones_existing_config() -> None:
    config = TrainingConfig()
    config.experiment_name = "coimbra-rf"
    config.dataset.kind = "csv_tabular_binary"
    config.dataset.path = "data/breast-cancer-coimbra.csv"
    config.dataset.target_column = "Classification"
    config.model.kind = "sklearn_random_forest"
    config.model.params = {"n_estimators": 50, "max_depth": 4, "min_samples_leaf": 2}

    draft = build_designer_draft_from_config(config, source_name="train-coimbra-random-forest")

    assert draft.experiment_name == "coimbra-rf"
    assert draft.source_name == "train-coimbra-random-forest"
    assert draft.dataset_kind == "csv_tabular_binary"
    assert draft.dataset_path == "data/breast-cancer-coimbra.csv"
    assert draft.model_kind == "sklearn_random_forest"
    assert json.loads(draft.model_params_json)["n_estimators"] == 50


def test_designer_draft_to_config_builds_training_config() -> None:
    draft = DesignerDraft(
        experiment_name="designer-demo",
        config_slug="designer-demo",
        random_seed="7",
        threshold="0.61",
        dataset_kind="csv_tabular_binary",
        dataset_path="data/demo.csv",
        target_column="Classification",
        model_kind="sklearn_hist_gradient_boosting",
        device="cpu",
        evaluation_mode="stratified_k_fold",
        folds="4",
        test_size="0.2",
        stratify=True,
        tracking_experiment_name="bc-mlops-showcase",
        model_params_json='{"learning_rate": 0.05, "max_iter": 120, "max_depth": 3}',
    )

    config = designer_draft_to_config(draft)

    assert isinstance(config, TrainingConfig)
    assert config.experiment_name == "designer-demo"
    assert config.random_seed == 7
    assert config.threshold == 0.61
    assert config.dataset.kind == "csv_tabular_binary"
    assert config.dataset.path == "data/demo.csv"
    assert config.model.kind == "sklearn_hist_gradient_boosting"
    assert config.evaluation.mode == "stratified_k_fold"
    assert config.evaluation.folds == 4
    assert config.model.params["max_iter"] == 120


def test_validate_designer_draft_reports_invalid_json() -> None:
    draft = build_default_designer_draft()
    draft.model_params_json = "{not valid json}"

    result = validate_designer_draft(draft)

    assert result.ok is False
    assert any("model params" in error.lower() for error in result.errors)
    assert result.resolved_config is None


def test_validate_designer_draft_requires_csv_path() -> None:
    draft = build_default_designer_draft()
    draft.dataset_kind = "csv_tabular_binary"
    draft.dataset_path = ""

    result = validate_designer_draft(draft)

    assert result.ok is False
    assert any("dataset.path" in error for error in result.errors)


def test_validate_designer_draft_requires_at_least_two_folds() -> None:
    draft = build_default_designer_draft()
    draft.evaluation_mode = "stratified_k_fold"
    draft.folds = "1"

    result = validate_designer_draft(draft)

    assert result.ok is False
    assert any("at least 2" in error for error in result.errors)


def test_validate_designer_draft_rejects_invalid_device() -> None:
    draft = build_default_designer_draft()
    draft.device = "tpu"

    result = validate_designer_draft(draft)

    assert result.ok is False
    assert any("unsupported model device" in error for error in result.errors)


def test_render_designer_preview_text_renders_normalized_yaml() -> None:
    draft = build_default_designer_draft()
    draft.experiment_name = "preview-me"

    preview = render_designer_preview_text(draft)

    assert "normalized config preview" in preview.lower()
    assert "experiment_name: preview-me" in preview
    assert "model:" in preview
    assert "dataset:" in preview


def test_save_designer_draft_writes_yaml_without_overwriting(tmp_path: Path) -> None:
    draft = build_default_designer_draft()
    draft.experiment_name = "designer-save"
    draft.config_slug = "designer-save"

    saved_path = save_designer_draft(draft, tmp_path)

    assert saved_path == tmp_path / "designer-save.yaml"
    payload = yaml.safe_load(saved_path.read_text())
    assert payload["experiment_name"] == "designer-save"

    with pytest.raises(FileExistsError):
        save_designer_draft(draft, tmp_path)


def test_launch_designer_run_saves_config_and_trains(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    output_root = tmp_path / "artifacts" / "runs"
    draft = build_default_designer_draft()
    draft.experiment_name = "designer-train"
    draft.config_slug = "designer-train"

    result = launch_designer_run(draft, config_root=config_root, output_root=output_root)

    assert isinstance(result, DesignerActionResult)
    assert result.ok is True
    assert result.config_path == config_root / "designer-train.yaml"
    assert result.run_dir is not None
    assert result.run_dir.exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "config.resolved.yaml").exists()
    assert "designer-train" in result.message
