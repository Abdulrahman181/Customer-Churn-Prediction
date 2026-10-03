"""Command-line interface for local churn model training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import DEFAULT_FILENAME, default_data_path, train_from_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an educational churn classifier from a local CSV."
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("CHURN_DATA_PATH", str(default_data_path())),
        help=f"Local CSV path (default: CHURN_DATA_PATH or {default_data_path()}; expected filename {DEFAULT_FILENAME}).",
    )
    parser.add_argument("--output-dir", default="artifacts", help="Local output directory (default: artifacts/).")
    parser.add_argument("--target", default="Churn", help="Binary target column (default: Churn).")
    parser.add_argument("--random-state", type=int, default=42, help="Deterministic split/model seed (default: 42).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = train_from_csv(
            Path(args.data),
            output_dir=Path(args.output_dir),
            target_column=args.target,
            random_state=args.random_state,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
