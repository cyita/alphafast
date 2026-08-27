#!/usr/bin/env python3
"""Compare five-sample confidence and ranking outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from validate_e2e import validate_artifacts, write_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frozen-features", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument(
        "--tolerances",
        type=Path,
        default=Path(__file__).with_name("ranking_tolerances.json"),
    )
    parser.add_argument(
        "--result-file", type=Path, default=Path("ae-results/latest_ranking.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = args.reference
    candidate = args.candidate
    if args.frozen_features is not None or args.input_json is not None:
        if args.frozen_features is None or args.input_json is None:
            reason = "--frozen-features and --input-json must be used together"
            write_result(
                args.result_file,
                {"stage": "ranking", "status": "error", "reason": reason},
            )
            print(f"Ranking validation error: {reason}", file=sys.stderr)
            return 2
        from postprocess_ranking import postprocess_artifact

        stem = args.result_file.stem
        reference = args.result_file.parent / f"{stem}_reference.npz"
        candidate = args.result_file.parent / f"{stem}_candidate.npz"
        try:
            postprocess_artifact(
                args.reference, args.frozen_features, args.input_json, reference
            )
            postprocess_artifact(
                args.candidate, args.frozen_features, args.input_json, candidate
            )
        except (OSError, ValueError, KeyError) as error:
            result = {
                "stage": "ranking",
                "status": "error",
                "reason": str(error),
            }
            write_result(args.result_file, result)
            print(f"Ranking validation error: {error}", file=sys.stderr)
            return 2

    exit_code, result = validate_artifacts(
        reference, candidate, args.tolerances, stage="ranking"
    )
    result["raw_reference"] = str(args.reference)
    result["raw_candidate"] = str(args.candidate)
    write_result(args.result_file, result)
    if exit_code == 2:
        print(f"Ranking validation error: {result['reason']}", file=sys.stderr)
    elif exit_code == 1:
        print(f"Ranking validation failed: {result['reason']}", file=sys.stderr)
    else:
        print(f"Ranking validation passed: {len(result['tensors'])} checkpoints")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
