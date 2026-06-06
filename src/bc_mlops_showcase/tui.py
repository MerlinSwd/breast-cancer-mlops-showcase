"""Branded terminal dashboard for recent training runs and artifact health."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass(slots=True)
class RunArtifactStatus:
    """File health for a single training run directory."""

    run_name: str
    model_kind: str
    model_status: str
    metrics_status: str
    card_status: str


@dataclass(slots=True)
class DashboardSummary:
    """Derived dashboard state built from the registry and artifact tree."""

    runs: list[dict[str, object]]
    best_run: dict[str, object] | None
    artifact_statuses: list[RunArtifactStatus]


BRAND_TITLE = "MERLIN // ONCO-OPS COMMAND DECK"
BRAND_TAGLINE = "A slightly dramatic bridge view for breast-cancer MLOps runs."


def load_dashboard_summary(registry_path: Path, run_root: Path) -> DashboardSummary:
    """Load registry data and enrich it with artifact presence checks."""

    payload: dict[str, object] = {"runs": []}
    if registry_path.exists():
        payload = _safe_json_file(registry_path, default=payload)

    runs = list(payload.get("runs", []))
    best_run = payload.get("best_run")
    if best_run is None and runs:
        best_run = max(
            runs,
            key=lambda run: (_metric_value(run, "f1"), _metric_value(run, "roc_auc")),
        )

    artifact_statuses: list[RunArtifactStatus] = []
    enriched_runs: list[dict[str, object]] = []
    for run in runs:
        run_dir = run_root / str(run["run_name"])
        metadata = _load_run_metadata(run_dir)
        model_artifact = metadata.get("artifact", "model artifact")
        enriched_run = dict(run)
        enriched_run["model_kind"] = enriched_run.get("model_kind") or metadata.get(
            "kind", "unknown"
        )
        enriched_runs.append(enriched_run)
        artifact_statuses.append(
            RunArtifactStatus(
                run_name=str(enriched_run["run_name"]),
                model_kind=str(enriched_run["model_kind"]),
                model_status=_ok_or_missing(run_dir / model_artifact, model_artifact),
                metrics_status=_ok_or_missing(run_dir / "metrics.json", "metrics.json"),
                card_status=_ok_or_missing(run_dir / "MODEL_CARD.md", "MODEL_CARD"),
            )
        )

    runs = enriched_runs
    if best_run is not None:
        best_run = next((run for run in runs if run["run_name"] == best_run["run_name"]), best_run)

    return DashboardSummary(runs=runs, best_run=best_run, artifact_statuses=artifact_statuses)


def _load_run_metadata(run_dir: Path) -> dict[str, str]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}

    payload = _safe_json_file(metadata_path, default={})
    model = payload.get("model", {})
    return {
        "artifact": str(model.get("artifact", "model artifact")),
        "kind": str(model.get("kind", "unknown")),
    }


def _safe_json_file(path: Path, default: dict[str, object]) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default

    return payload if isinstance(payload, dict) else default


def _metric_value(run: dict[str, object], key: str, default: float = -1.0) -> float:
    value = run.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_metric(run: dict[str, object], key: str) -> str:
    value = run.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _ok_or_missing(path: Path, label: str) -> str:
    return f"OK: {label}" if path.exists() else f"{label} missing"


def render_dashboard_text(
    registry_path: Path, run_root: Path, width: int = 110, *, color: bool = False
) -> str:
    """Render the dashboard to plain text for CLI output and tests."""

    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
    stream = StringIO()
    console = Console(
        file=stream,
        record=True,
        width=width,
        force_terminal=color,
        color_system="truecolor" if color else None,
    )
    console.print(_build_dashboard(summary, registry_path=registry_path, run_root=run_root))
    return stream.getvalue()


def _build_dashboard(summary: DashboardSummary, registry_path: Path, run_root: Path) -> Group:
    return Group(
        _brand_panel(),
        _summary_panel(summary),
        _leaderboard_panel(summary),
        _artifact_health_panel(summary),
        _operator_hints_panel(summary, registry_path=registry_path, run_root=run_root),
    )


def _brand_panel() -> Panel:
    title = Text(BRAND_TITLE, style="bold magenta")
    subtitle = Text(BRAND_TAGLINE, style="italic cyan")
    banner = Group(title, Text(""), subtitle)
    return Panel(banner, border_style="bright_magenta", title="Command Bridge")


def _summary_panel(summary: DashboardSummary) -> Panel:
    if not summary.runs or summary.best_run is None:
        return Panel(
            Group(
                Text("No tracked runs yet.", style="bold yellow"),
                Text("Train with: bc-mlops train --config configs/train.yaml"),
            ),
            border_style="yellow",
            title="Champion Run",
        )

    best_run = summary.best_run
    lines = Group(
        Text(f"Champion: {best_run['run_name']}", style="bold green"),
        Text(f"Model kind: {best_run.get('model_kind', 'unknown')}"),
        Text(f"Accuracy: {_format_metric(best_run, 'accuracy')}"),
        Text(f"F1: {_format_metric(best_run, 'f1')}"),
        Text(f"ROC AUC: {_format_metric(best_run, 'roc_auc')}"),
        Text(f"Tracked runs: {len(summary.runs)}"),
    )
    return Panel(lines, border_style="green", title="Champion Run")


def _leaderboard_panel(summary: DashboardSummary) -> Panel:
    table = Table(expand=True)
    table.add_column("Run")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ROC AUC", justify="right")

    for run in sorted(
        summary.runs,
        key=lambda item: (_metric_value(item, "f1"), _metric_value(item, "roc_auc")),
        reverse=True,
    ):
        table.add_row(
            str(run["run_name"]),
            str(run.get("model_kind", "unknown")),
            _format_metric(run, "accuracy"),
            _format_metric(run, "f1"),
            _format_metric(run, "roc_auc"),
        )

    if not summary.runs:
        table.add_row("—", "—", "—", "—", "—")

    return Panel(table, border_style="blue", title="Leaderboard")


def _artifact_health_panel(summary: DashboardSummary) -> Panel:
    table = Table(expand=True)
    table.add_column("Run")
    table.add_column("Model kind", no_wrap=True)
    table.add_column("Model artifact", no_wrap=True)
    table.add_column("Metrics", no_wrap=True)
    table.add_column("Model card", no_wrap=True)

    for status in summary.artifact_statuses:
        table.add_row(
            status.run_name,
            status.model_kind,
            status.model_status,
            status.metrics_status,
            status.card_status,
        )

    if not summary.artifact_statuses:
        table.add_row("—", "—", "—", "—", "—")

    return Panel(table, border_style="red", title="Artifact Health")


def _operator_hints_panel(summary: DashboardSummary, registry_path: Path, run_root: Path) -> Panel:
    if not summary.runs:
        hints = Group(
            Text("Next move:"),
            Text("- Train baseline: bc-mlops train --config configs/train.yaml"),
            Text("- Compare runs: bc-mlops compare --registry artifacts/registry.json"),
        )
        return Panel(hints, border_style="cyan", title="Operator Hints")

    missing_cards = sum("missing" in status.card_status for status in summary.artifact_statuses)
    hints = Group(
        Text(f"Registry: {registry_path}"),
        Text(f"Run root: {run_root}"),
        Text(f"Runs missing model cards: {missing_cards}"),
        Text("Next move: bc-mlops report --run-dir <run_dir> --output <run_dir>/MODEL_CARD.md"),
    )
    return Panel(hints, border_style="cyan", title="Operator Hints")
