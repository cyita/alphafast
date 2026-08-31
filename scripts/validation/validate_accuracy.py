#!/usr/bin/env python3
"""Run the benchmark ground-truth accuracy gate through one interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        type=Path,
        default=Path(__file__).with_name("score_ground_truth.py"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument(
        "--result-file", type=Path, default=Path("ae-results/latest_accuracy.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        str(args.tool),
        str(args.output_root),
        "--profile",
        args.profile,
        "--json-out",
        str(args.result_file),
        "--gate",
    ]
    command.extend(("--ground-truth-root", str(args.ground_truth_root)))
    if args.thresholds is not None:
        command.extend(("--thresholds", str(args.thresholds)))
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode not in (0, 1) or not args.result_file.is_file():
        reason = f"accuracy tool exited with code {completed.returncode}"
        args.result_file.write_text(
            json.dumps(
                {"stage": "accuracy", "status": "error", "reason": reason},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Accuracy validation could not run: {reason}", file=sys.stderr)
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
