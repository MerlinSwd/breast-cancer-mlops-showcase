"""Command-line interface for training, validation, inference, and reporting."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the project CLI."""

    parser = argparse.ArgumentParser(description="Breast cancer MLOps showcase CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model and emit artifacts")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))

    compare_parser = subparsers.add_parser("compare", help="Print experiment registry summary")
    compare_parser.add_argument("--registry", type=Path, default=Path("artifacts/registry.json"))
    compare_parser.add_argument(
        "--summary",
        action="store_true",
        help="Render a human-readable comparison view instead of raw registry JSON",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Render a branded terminal dashboard for run performance and artifact health",
    )
    dashboard_parser.add_argument("--registry", type=Path, default=Path("artifacts/registry.json"))
    dashboard_parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    dashboard_parser.add_argument("--width", type=int, default=110)
    dashboard_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive Textual command deck instead of static text output",
    )

    predict_parser = subparsers.add_parser("predict", help="Score records with a trained model")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="Check metrics against quality gates")
    validate_parser.add_argument("--metrics", type=Path, required=True)
    validate_parser.add_argument("--gates", type=Path, required=True)

    report_parser = subparsers.add_parser("report", help="Generate a markdown model card")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)

    kaggle_parser = subparsers.add_parser(
        "kaggle", help="Pull Kaggle datasets and competition files"
    )
    kaggle_subparsers = kaggle_parser.add_subparsers(dest="kaggle_command", required=True)
    kaggle_pull_parser = kaggle_subparsers.add_parser(
        "pull", help="Download a Kaggle dataset or competition bundle"
    )
    kaggle_pull_parser.add_argument(
        "--resource-type",
        choices=("dataset", "competition"),
        required=True,
    )
    kaggle_pull_parser.add_argument("--handle", required=True)
    kaggle_pull_parser.add_argument("--output-dir", type=Path, required=True)
    kaggle_pull_parser.add_argument("--file-name")
    kaggle_pull_parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep dataset archives zipped instead of auto-unzipping them",
    )
    kaggle_pull_parser.add_argument(
        "--force",
        action="store_true",
        help="Force a re-download even if Kaggle thinks the files are already present",
    )
    return parser


def _print_registry(registry_path: Path) -> int:
    if not registry_path.exists():
        print(json.dumps({"runs": []}, indent=2))
        return 0

    print(json.dumps(json.loads(registry_path.read_text()), indent=2))
    return 0


def _load_registry_payload(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {"runs": []}
    try:
        payload = json.loads(registry_path.read_text())
    except json.JSONDecodeError:
        return {"runs": []}
    return payload if isinstance(payload, dict) else {"runs": []}


def _normalize_runs(candidates: object) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    return [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]


def _metric_value(run: dict[str, Any], key: str, default: float = -1.0) -> float:
    value = run.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_metric(run: dict[str, Any], key: str) -> str:
    value = run.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_delta(best_run: dict[str, Any] | None, current_run: dict[str, Any]) -> str:
    if best_run is None:
        return "n/a"
    delta = _metric_value(current_run, "f1") - _metric_value(best_run, "f1")
    return f"{delta:+.4f}"


def _format_evaluation_mode(run: dict[str, Any]) -> str:
    mode = run.get("evaluation_mode")
    if not mode:
        return "n/a"
    if str(mode) != "stratified_k_fold":
        return str(mode)
    try:
        folds = int(run.get("evaluation_folds"))
    except (TypeError, ValueError):
        return str(mode)
    return f"{mode} ({folds} folds)"


def _render_compare_summary(registry_path: Path) -> str:
    payload = _load_registry_payload(registry_path)
    runs = _normalize_runs(payload.get("runs", []))
    best_run_raw = payload.get("best_run")
    best_run = dict(best_run_raw) if isinstance(best_run_raw, dict) else None
    if best_run is None and runs:
        best_run = max(
            runs, key=lambda run: (_metric_value(run, "f1"), _metric_value(run, "roc_auc"))
        )

    lines = ["Compare Summary", f"Registry: {registry_path}"]
    champion_name = str(best_run.get("run_name", "n/a")) if best_run is not None else "n/a"
    lines.append(f"Champion: {champion_name}")
    lines.append("")
    lines.append("Rank | Run | Model | Evaluation | F1 | ΔF1 vs champ | F1 σ")

    ordered_runs = sorted(
        runs,
        key=lambda run: (
            _metric_value(run, "f1"),
            _metric_value(run, "roc_auc"),
            str(run.get("run_name", "")),
        ),
        reverse=True,
    )
    for index, run in enumerate(ordered_runs, start=1):
        lines.append(
            " | ".join(
                [
                    str(index),
                    str(run.get("run_name", "unknown")),
                    str(run.get("model_kind", "unknown")),
                    _format_evaluation_mode(run),
                    _format_metric(run, "f1"),
                    _format_delta(best_run, run),
                    _format_metric(run, "cv_f1_std"),
                ]
            )
        )

    if not ordered_runs:
        lines.append("— | — | — | — | — | — | —")

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entrypoint and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from .config import load_training_config
        from .pipeline import train_and_evaluate

        config = load_training_config(args.config)
        result = train_and_evaluate(config=config, output_root=args.output_dir)
        print(json.dumps(result.summary(), indent=2))
        return 0

    if args.command == "compare":
        if args.summary:
            print(_render_compare_summary(args.registry), end="")
            return 0
        return _print_registry(args.registry)

    if args.command == "dashboard":
        from .tui import launch_dashboard_app, render_dashboard_text

        if args.interactive:
            launch_dashboard_app(registry_path=args.registry, run_root=args.run_root)
            return 0
        print(render_dashboard_text(args.registry, args.run_root, width=args.width), end="")
        return 0

    if args.command == "predict":
        from .inference import predict_records

        print(json.dumps(predict_records(model_path=args.model, input_path=args.input), indent=2))
        return 0

    if args.command == "validate":
        from .validation import validate_metrics

        validation_result = validate_metrics(metrics_path=args.metrics, gates_path=args.gates)
        print(json.dumps(validation_result, indent=2))
        return 0 if validation_result["passed"] else 1

    if args.command == "kaggle":
        from .kaggle import pull_kaggle_resource

        result = pull_kaggle_resource(
            resource_type=args.resource_type,
            handle=args.handle,
            output_dir=args.output_dir,
            file_name=args.file_name,
            unzip=not args.keep_zip,
            force=args.force,
        )
        print(json.dumps(result.summary(), indent=2))
        return 0

    from .reporting import build_model_card

    build_model_card(run_dir=args.run_dir, output_path=args.output)
    print(json.dumps({"model_card": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
