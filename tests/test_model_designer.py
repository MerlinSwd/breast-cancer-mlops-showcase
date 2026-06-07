import json
from pathlib import Path

import pytest

from bc_mlops_showcase.config import ModelConfig, TrainingConfig, load_training_config
from bc_mlops_showcase.designer import build_default_designer_draft
from bc_mlops_showcase.model_designer import (
    apply_model_designer_draft_to_run_draft,
    build_default_model_designer_draft,
    build_model_designer_draft_from_model_config,
    build_model_designer_draft_from_training_config,
    model_designer_draft_to_model_config,
    render_model_designer_preview_text,
    validate_model_designer_draft,
)


def test_build_default_model_designer_draft_uses_project_defaults() -> None:
    draft = build_default_model_designer_draft()

    assert draft.model_kind == "sklearn_logreg"
    assert draft.device == "auto"
    assert draft.logreg_c == "1.0"
    assert draft.logreg_max_iter == "500"


def test_build_model_designer_draft_from_model_config_clones_rf_values() -> None:
    model = ModelConfig(
        kind="sklearn_random_forest",
        device="cpu",
        params={"n_estimators": 75, "max_depth": 6, "min_samples_leaf": 2},
    )

    draft = build_model_designer_draft_from_model_config(model, source_name="rf-template")

    assert draft.model_kind == "sklearn_random_forest"
    assert draft.device == "cpu"
    assert draft.rf_n_estimators == "75"
    assert draft.rf_max_depth == "6"
    assert draft.rf_min_samples_leaf == "2"
    assert draft.source_name == "rf-template"


def test_build_model_designer_draft_from_training_config_uses_model_slice() -> None:
    config = TrainingConfig()
    config.model.kind = "pytorch_mlp"
    config.model.device = "cuda"
    config.model.params = {
        "hidden_dims": [64, 32],
        "epochs": 30,
        "batch_size": 16,
        "learning_rate": 0.005,
        "dropout": 0.2,
    }

    draft = build_model_designer_draft_from_training_config(config, source_name="run-draft")

    assert draft.model_kind == "pytorch_mlp"
    assert draft.device == "cuda"
    assert draft.mlp_hidden_dims == "64,32"
    assert draft.mlp_epochs == "30"
    assert draft.source_name == "run-draft"


def test_model_designer_draft_to_model_config_builds_hist_gradient_boosting() -> None:
    draft = build_default_model_designer_draft("sklearn_hist_gradient_boosting")
    draft.device = "cpu"
    draft.hgb_learning_rate = "0.05"
    draft.hgb_max_iter = "120"
    draft.hgb_max_depth = "4"
    draft.hgb_min_samples_leaf = "7"

    config = model_designer_draft_to_model_config(draft)

    assert config.kind == "sklearn_hist_gradient_boosting"
    assert config.device == "cpu"
    assert config.params["learning_rate"] == 0.05
    assert config.params["max_iter"] == 120
    assert config.params["max_depth"] == 4
    assert config.params["min_samples_leaf"] == 7


def test_validate_model_designer_draft_rejects_invalid_hidden_dims() -> None:
    draft = build_default_model_designer_draft("pytorch_mlp")
    draft.mlp_hidden_dims = "64, nope"

    result = validate_model_designer_draft(draft)

    assert result.ok is False
    assert any("hidden_dims" in error for error in result.errors)


def test_validate_model_designer_draft_rejects_invalid_device() -> None:
    draft = build_default_model_designer_draft()
    draft.device = "tpu"

    result = validate_model_designer_draft(draft)

    assert result.ok is False
    assert any("device" in error.lower() for error in result.errors)


def test_load_training_config_rejects_invalid_model_device(tmp_path: Path) -> None:
    path = tmp_path / "invalid-device.yaml"
    path.write_text(
        "experiment_name: invalid-device\n"
        "model:\n"
        "  kind: sklearn_logreg\n"
        "  device: tpu\n"
    )

    with pytest.raises(ValueError, match="unsupported model device"):
        load_training_config(path)


def test_blank_max_depth_normalizes_to_none() -> None:
    draft = build_default_model_designer_draft("sklearn_random_forest")
    draft.rf_max_depth = ""

    config = model_designer_draft_to_model_config(draft)

    assert config.params["max_depth"] is None


def test_apply_model_designer_draft_to_run_draft_updates_only_model_slice() -> None:
    run_draft = build_default_designer_draft()
    run_draft.dataset_kind = "csv_tabular_binary"
    run_draft.dataset_path = "data/demo.csv"
    run_draft.target_column = "Classification"

    model_draft = build_default_model_designer_draft("sklearn_random_forest")
    model_draft.device = "cpu"
    model_draft.rf_n_estimators = "50"
    model_draft.rf_max_depth = "8"
    model_draft.rf_min_samples_leaf = "2"

    updated = apply_model_designer_draft_to_run_draft(model_draft, run_draft)

    assert updated.dataset_kind == "csv_tabular_binary"
    assert updated.dataset_path == "data/demo.csv"
    assert updated.target_column == "Classification"
    assert updated.model_kind == "sklearn_random_forest"
    assert updated.device == "cpu"
    assert json.loads(updated.model_params_json) == {
        "n_estimators": 50,
        "max_depth": 8,
        "min_samples_leaf": 2,
    }


def test_render_model_designer_preview_text_renders_normalized_yaml() -> None:
    draft = build_default_model_designer_draft("sklearn_random_forest")
    draft.rf_n_estimators = "80"

    preview = render_model_designer_preview_text(draft)

    assert "normalized model preview" in preview.lower()
    assert "kind: sklearn_random_forest" in preview
    assert "n_estimators: 80" in preview


def test_apply_model_designer_draft_preserves_existing_run_metadata() -> None:
    run_draft = build_default_designer_draft()
    run_draft.experiment_name = "custom-run"
    run_draft.config_slug = "custom-run"
    run_draft.dataset_kind = "csv_tabular_binary"
    run_draft.dataset_path = "data/custom.csv"
    run_draft.target_column = "Diagnosis"

    model_draft = build_default_model_designer_draft("pytorch_mlp")
    model_draft.device = "cuda"
    model_draft.mlp_hidden_dims = "128,64"

    updated = apply_model_designer_draft_to_run_draft(model_draft, run_draft)

    assert updated.experiment_name == "custom-run"
    assert updated.config_slug == "custom-run"
    assert updated.dataset_path == "data/custom.csv"
    assert updated.target_column == "Diagnosis"
    assert updated.model_kind == "pytorch_mlp"
    assert updated.device == "cuda"
