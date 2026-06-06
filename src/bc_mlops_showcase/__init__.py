"""Breast cancer MLOps showcase package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "load_training_config",
    "train_and_evaluate",
]


def __getattr__(name: str) -> Any:
    """Lazily expose top-level package symbols without importing heavy ML deps eagerly."""

    if name in {"TrainingConfig", "load_training_config"}:
        module = import_module(".config", __name__)
        return getattr(module, name)

    if name in {"TrainingResult", "train_and_evaluate"}:
        module = import_module(".pipeline", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
