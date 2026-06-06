"""Dataset loading helpers for the showcase pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass(slots=True)
class DatasetBundle:
    """Container for features, labels, and target metadata."""

    features: pd.DataFrame
    target: pd.Series
    target_name: str


def load_dataset() -> DatasetBundle:
    """Load the scikit-learn breast cancer dataset as pandas objects."""

    raw = load_breast_cancer(as_frame=True)
    features = raw.data.copy()
    target = raw.target.rename("target")
    return DatasetBundle(features=features, target=target, target_name=raw.target_names[1])
