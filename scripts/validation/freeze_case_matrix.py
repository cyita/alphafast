#!/usr/bin/env python3
"""Freeze every case/seed pair from an AF3 benchmark manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from freeze_features import freeze_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--input-kind", choices=("smoke", "no_msa", "pipeline"), default="no_msa"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.benchmark_root / "manifest.json").read_text())
    input_field = f"{args.input_kind}_input"
    written = 0
    skipped = 0
    for case in manifest["cases"]:
        bucket_size = math.ceil(case["estimated_token_count"] / 128) * 128
        input_json = args.benchmark_root / case[input_field]
        for seed in args.seeds:
            output_file = (
                args.output_root
                / case["id"]
                / f"seed_{seed}"
                / "featurised_input.npz"
            )
            if output_file.is_file():
                skipped += 1
                continue
            freeze_features(input_json, output_file, bucket_size, seed)
            written += 1
    print(f"Frozen case matrix complete: {written} written, {skipped} existing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
