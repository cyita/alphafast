#!/usr/bin/env python3
"""Append AF3 confidence and ranking outputs to a raw parity artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphafold3.model import model
import numpy as np

from confidence_checkpoints import confidence_arrays
from featurise_ranking_batch import featurise_ranking_batch
from parity_io import METADATA_KEY, load_metadata, sha256_file, write_npz


MODEL_OUTPUT_KEYS = (
    "atom_positions",
    "distogram_contact_probs",
    "predicted_lddt",
    "predicted_experimentally_resolved",
    "full_pde",
    "average_pde",
    "full_pae",
    "tmscore_adjusted_pae_global",
    "tmscore_adjusted_pae_interface",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def postprocess_artifact(
    checkpoints: Path,
    frozen_features: Path,
    input_json: Path,
    output_file: Path,
) -> None:
    with np.load(checkpoints, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in MODEL_OUTPUT_KEYS}
        metadata = json.loads(str(archive[METADATA_KEY]))
    feature_metadata = load_metadata(frozen_features)
    if sha256_file(frozen_features) != metadata["frozen_features_sha256"]:
        raise ValueError("--frozen-features does not match the checkpoints")
    if sha256_file(input_json) != feature_metadata["input_json_sha256"]:
        raise ValueError("--input-json does not match the frozen features")
    batch = featurise_ranking_batch(input_json, int(feature_metadata["bucket_size"]))
    result = {
        "diffusion_samples": {"atom_positions": arrays["atom_positions"]},
        "distogram": {"contact_probs": arrays["distogram_contact_probs"]},
        "__identifier__": b"parity-postprocessing",
    }
    for key in MODEL_OUTPUT_KEYS[2:]:
        result[key] = arrays[key]
    inference_results = list(
        model.Model.get_inference_result(
            batch=batch,
            result=result,
            target_name=metadata["job_name"],
        )
    )
    write_npz(output_file, confidence_arrays(inference_results), metadata)


def main() -> int:
    args = parse_args()
    postprocess_artifact(
        args.checkpoints,
        args.frozen_features,
        args.input_json,
        args.output_file,
    )
    print(f"Wrote ranked parity checkpoints to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
