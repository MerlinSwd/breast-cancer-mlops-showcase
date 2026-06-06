from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import load_training_config
from .inference import predict_records
from .pipeline import train_and_evaluate
from .reporting import build_model_card
from .validation import validate_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Breast cancer MLOps showcase CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model and emit artifacts")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))

    compare_parser = subparsers.add_parser("compare", help="Print experiment registry summary")
    compare_parser.add_argument("--registry", type=Path, default=Path("artifacts/registry.json"))

    predict_parser = subparsers.add_parser("predict", help="Score records with a trained model")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="Check metrics against quality gates")
    validate_parser.add_argument("--metrics", type=Path, required=True)
    validate_parser.add_argument("--gates", type=Path, required=True)

    report_parser = subparsers.add_parser("report", help="Generate a markdown model card")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    return parser


def _print_registry(registry_path: Path) -> int:
    if not registry_path.exists():
        print(json.dumps({"runs": []}, indent=2))
        return 0

    print(json.dumps(json.loads(registry_path.read_text()), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = load_training_config(args.config)
        result = train_and_evaluate(config=config, output_root=args.output_dir)
        print(json.dumps(result.summary(), indent=2))
        return 0

    if args.command == "compare":
        return _print_registry(args.registry)

    if args.command == "predict":
        print(json.dumps(predict_records(model_path=args.model, input_path=args.input), indent=2))
        return 0

    if args.command == "validate":
        validation_result = validate_metrics(metrics_path=args.metrics, gates_path=args.gates)
        print(json.dumps(validation_result, indent=2))
        return 0 if validation_result["passed"] else 1

    build_model_card(run_dir=args.run_dir, output_path=args.output)
    print(json.dumps({"model_card": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
