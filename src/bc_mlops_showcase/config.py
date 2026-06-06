from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SplitConfig:
    test_size: float = 0.2
    stratify: bool = True


@dataclass(slots=True)
class ModelConfig:
    c: float = 1.0
    max_iter: int = 500


@dataclass(slots=True)
class TrainingConfig:
    experiment_name: str = "baseline-logreg"
    random_seed: int = 42
    threshold: float = 0.5
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _merge_dataclass(default: Any, values: dict[str, Any] | None) -> Any:
    data = values or {}
    return type(default)(**{**asdict(default), **data})


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}

    default = TrainingConfig()
    return TrainingConfig(
        experiment_name=raw.get("experiment_name", default.experiment_name),
        random_seed=raw.get("random_seed", default.random_seed),
        threshold=raw.get("threshold", default.threshold),
        split=_merge_dataclass(default.split, raw.get("split")),
        model=_merge_dataclass(default.model, raw.get("model")),
    )


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "random_seed": config.random_seed,
        "threshold": config.threshold,
        "split": {
            "test_size": config.split.test_size,
            "stratify": config.split.stratify,
        },
        "model": {
            "c": config.model.c,
            "max_iter": config.model.max_iter,
        },
    }
