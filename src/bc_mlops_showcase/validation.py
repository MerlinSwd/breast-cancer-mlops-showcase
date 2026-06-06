"""Quality gate validation for persisted model metrics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

METRIC_FIELDS = {
    "min_accuracy": "accuracy",
    "min_f1": "f1",
    "min_roc_auc": "roc_auc",
}


def validate_metrics(metrics_path: str | Path, gates_path: str | Path) -> dict[str, object]:
    """Check a metrics file against configured minimum thresholds."""

    metrics = json.loads(Path(metrics_path).read_text())
    gates = yaml.safe_load(Path(gates_path).read_text()) or {}

    checks = []
    for gate_key, metric_name in METRIC_FIELDS.items():
        threshold = float(gates[gate_key])
        actual = float(metrics[metric_name])
        checks.append(
            {
                "metric": metric_name,
                "actual": round(actual, 4),
                "threshold": round(threshold, 4),
                "passed": actual >= threshold,
            }
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
