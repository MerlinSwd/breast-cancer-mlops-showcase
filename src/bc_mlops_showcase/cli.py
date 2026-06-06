from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import load_training_config
from .pipeline import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Breast cancer MLOps showcase CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model and emit artifacts")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))

    compare_parser = subparsers.add_parser("compare", help="Print experiment registry summary")
    compare_parser.add_argument("--registry", type=Path, default=Path("artifacts/registry.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = load_training_config(args.config)
        result = train_and_evaluate(config=config, output_root=args.output_dir)
        print(json.dumps(result.summary(), indent=2))
        return 0

    registry_path = args.registry
    if not registry_path.exists():
        print(json.dumps({"runs": []}, indent=2))
        return 0

    print(json.dumps(json.loads(registry_path.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
