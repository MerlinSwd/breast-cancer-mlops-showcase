from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass(slots=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    target_name: str


def load_dataset() -> DatasetBundle:
    raw = load_breast_cancer(as_frame=True)
    features = raw.data.copy()
    target = raw.target.rename("target")
    return DatasetBundle(features=features, target=target, target_name=raw.target_names[1])
