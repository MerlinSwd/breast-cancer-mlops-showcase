import json
from pathlib import Path

import pytest

from bc_mlops_showcase.config import (
    ModelConfig,
    TrainingConfig,
    get_default_model_params,
    get_model_spec,
    load_training_config,
)
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
    assert draft.param_values == {"c": "1.0", "max_iter": "500"}


def test_model_spec_defaults_are_derived_from_parameter_registry() -> None:
    defaults = get_default_model_params("pytorch_cnn")
    spec = get_model_spec("pytorch_cnn")

    assert [param.name for param in spec.parameter_specs] == [
        "conv_channels",
        "kernel_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_dim",
    ]
    assert defaults == {
        "conv_channels": [8, 16],
        "kernel_size": 3,
        "epochs": 8,
        "batch_size": 32,
        "learning_rate": 0.005,
        "hidden_dim": 32,
    }


def test_build_model_designer_draft_from_model_config_clones_rf_values() -> None:
    model = ModelConfig(
        kind="sklearn_random_forest",
        device="cpu",
        params={"n_estimators": 75, "max_depth": 6, "min_samples_leaf": 2},
    )

    draft = build_model_designer_draft_from_model_config(model, source_name="rf-template")

    assert draft.model_kind == "sklearn_random_forest"
    assert draft.device == "cpu"
    assert draft.param_values == {
        "n_estimators": "75",
        "max_depth": "6",
        "min_samples_leaf": "2",
    }
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
    assert draft.param_values["hidden_dims"] == "64,32"
    assert draft.param_values["epochs"] == "30"
    assert draft.source_name == "run-draft"


def test_model_designer_draft_to_model_config_builds_hist_gradient_boosting() -> None:
    draft = build_default_model_designer_draft("sklearn_hist_gradient_boosting")
    draft.device = "cpu"
    draft.param_values.update(
        {
            "learning_rate": "0.05",
            "max_iter": "120",
            "max_depth": "4",
            "min_samples_leaf": "7",
        }
    )

    config = model_designer_draft_to_model_config(draft)

    assert config.kind == "sklearn_hist_gradient_boosting"
    assert config.device == "cpu"
    assert config.params["learning_rate"] == 0.05
    assert config.params["max_iter"] == 120
    assert config.params["max_depth"] == 4
    assert config.params["min_samples_leaf"] == 7


def test_validate_model_designer_draft_rejects_invalid_hidden_dims() -> None:
    draft = build_default_model_designer_draft("pytorch_mlp")
    draft.param_values["hidden_dims"] = "64, nope"

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
        "experiment_name: invalid-device\nmodel:\n  kind: sklearn_logreg\n  device: tpu\n"
    )

    with pytest.raises(ValueError, match="unsupported model device"):
        load_training_config(path)


def test_blank_max_depth_normalizes_to_none() -> None:
    draft = build_default_model_designer_draft("sklearn_random_forest")
    draft.param_values["max_depth"] = ""

    config = model_designer_draft_to_model_config(draft)

    assert config.params["max_depth"] is None


def test_model_designer_supports_registered_cnn_params_without_custom_draft_fields() -> None:
    draft = build_default_model_designer_draft("pytorch_cnn")
    draft.device = "cpu"
    draft.param_values.update(
        {
            "conv_channels": "16,32",
            "kernel_size": "5",
            "epochs": "4",
            "batch_size": "8",
            "learning_rate": "0.01",
            "hidden_dim": "64",
        }
    )

    config = model_designer_draft_to_model_config(draft)

    assert config.kind == "pytorch_cnn"
    assert config.device == "cpu"
    assert config.params == {
        "conv_channels": [16, 32],
        "kernel_size": 5,
        "epochs": 4,
        "batch_size": 8,
        "learning_rate": 0.01,
        "hidden_dim": 64,
    }


def test_load_training_config_rejects_unknown_model_params_for_backend(tmp_path: Path) -> None:
    path = tmp_path / "invalid-param.yaml"
    path.write_text(
        "experiment_name: invalid-param\n"
        "model:\n"
        "  kind: sklearn_logreg\n"
        "  params:\n"
        "    c: 1.0\n"
        "    nope: 123\n"
    )

    with pytest.raises(ValueError, match="unknown model params"):
        load_training_config(path)


def test_apply_model_designer_draft_to_run_draft_updates_only_model_slice() -> None:
    run_draft = build_default_designer_draft()
    run_draft.dataset_kind = "csv_tabular_binary"
    run_draft.dataset_path = "data/demo.csv"
    run_draft.target_column = "Classification"

    model_draft = build_default_model_designer_draft("sklearn_random_forest")
    model_draft.device = "cpu"
    model_draft.param_values.update(
        {
            "n_estimators": "50",
            "max_depth": "8",
            "min_samples_leaf": "2",
        }
    )

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
    draft.param_values["n_estimators"] = "80"

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
    model_draft.param_values["hidden_dims"] = "128,64"

    updated = apply_model_designer_draft_to_run_draft(model_draft, run_draft)

    assert updated.experiment_name == "custom-run"
    assert updated.config_slug == "custom-run"
    assert updated.dataset_path == "data/custom.csv"
    assert updated.target_column == "Diagnosis"
    assert updated.model_kind == "pytorch_mlp"
    assert updated.device == "cuda"
