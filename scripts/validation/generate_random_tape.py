#!/usr/bin/env python3
"""Generate framework-independent diffusion random tensors for parity tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from parity_io import load_npz, sha256_file, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features, _ = load_npz(args.frozen_features)
    atom_shape = features["pred_dense_atom_mask"].shape
    rng = np.random.default_rng(args.seed)
    sample_shape = (args.num_samples,) + atom_shape + (3,)
    msa_mask = features["msa_mask"]
    valid_msa_rows = np.flatnonzero(np.any(msa_mask, axis=-1))
    padded_msa_rows = np.flatnonzero(~np.any(msa_mask, axis=-1))
    msa_row_order = np.stack(
        [
            np.concatenate((rng.permutation(valid_msa_rows), padded_msa_rows))
            for _ in range(11)
        ]
    ).astype(np.int32)
    arrays = {
        "msa_row_order": msa_row_order,
        "initial_positions_noise": rng.standard_normal(sample_shape, dtype=np.float32),
        "rotation_noise": rng.standard_normal(
            (args.diffusion_steps, args.num_samples, 2, 3), dtype=np.float32
        ),
        "translation_noise": rng.standard_normal(
            (args.diffusion_steps, args.num_samples, 3), dtype=np.float32
        ),
        "diffusion_noise": rng.standard_normal(
            (args.diffusion_steps,) + sample_shape, dtype=np.float32
        ),
    }
    metadata = {
        "artifact_type": "alphafold3_diffusion_random_tape",
        "format_version": 1,
        "frozen_features_sha256": sha256_file(args.frozen_features),
        "generator": "numpy.default_rng",
        "seed": args.seed,
        "num_samples": args.num_samples,
        "diffusion_steps": args.diffusion_steps,
        "num_trunk_passes": 11,
        "atom_shape": list(atom_shape),
    }
    write_npz(args.output_file, arrays, metadata)
    print(f"Wrote diffusion random tape to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
