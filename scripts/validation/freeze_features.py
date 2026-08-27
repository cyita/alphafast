#!/usr/bin/env python3
"""Featurise one processed AF3 input once and save its model input arrays."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model.components import utils
import numpy as np

from parity_io import sha256_file, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--bucket-size", type=int, default=128)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def freeze_features(
    input_json: Path,
    output_file: Path,
    bucket_size: int,
    seed: int | None,
) -> None:
    fold_inputs = list(folding_input.load_fold_inputs_from_path(input_json))
    if len(fold_inputs) != 1:
        raise ValueError("parity input must contain one job")

    fold_input = fold_inputs[0]
    if seed is None:
        if len(fold_input.rng_seeds) != 1:
            raise ValueError("--seed is required when the input has multiple seeds")
    else:
        if seed not in fold_input.rng_seeds:
            raise ValueError(f"seed {seed} is not present in the input")
        fold_input = dataclasses.replace(fold_input, rng_seeds=(seed,))
    examples = list(
        featurisation.featurise_input(
            fold_input=fold_input,
            buckets=(bucket_size,),
            ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd),
            verbose=True,
        )
    )
    arrays = {
        key: np.asarray(value)
        for key, value in utils.remove_invalidly_typed_feats(examples[0]).items()
    }
    metadata = {
        "artifact_type": "alphafold3_frozen_features",
        "format_version": 1,
        "input_json_sha256": sha256_file(input_json),
        "job_name": fold_input.name,
        "model_seed": int(fold_input.rng_seeds[0]),
        "bucket_size": bucket_size,
    }
    write_npz(output_file, arrays, metadata)
    print(f"Wrote {len(arrays)} frozen feature arrays to {output_file}")


def main() -> int:
    args = parse_args()
    freeze_features(
        args.input_json, args.output_file, args.bucket_size, args.seed
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
