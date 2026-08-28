#!/usr/bin/env python3
"""Run AlphaFast/JAX from frozen features and save parity checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from absl import flags
from alphafold3.model.inference import make_model_config, ModelRunner
import jax
import numpy as np

from parity_io import load_npz, sha256_file, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--random-tape", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument(
        "--precision", choices=("default", "fp32"), default="default"
    )
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def checkpoint_arrays(
    result: dict[str, object], steps: int, num_trunk_passes: int
) -> dict[str, np.ndarray]:
    diffusion = result["diffusion_samples"]
    distogram = result["distogram"]
    arrays = {
        "single_embeddings": np.asarray(result["single_embeddings"]),
        "pair_embeddings": np.asarray(result["pair_embeddings"]),
        "atom_positions": np.asarray(diffusion["atom_positions"]),
        "distogram_contact_probs": np.asarray(distogram["contact_probs"]),
        "distogram_logits": np.asarray(distogram["distogram"]),
        "pairformer_block_1_single": np.asarray(
            result["pairformer_block_1_single"]
        ),
        "pairformer_block_1_pair": np.asarray(result["pairformer_block_1_pair"]),
        "denoised_step_1": np.asarray(diffusion["denoised_step_1"]),
    }
    checkpoint_passes = range(1, num_trunk_passes + 1)
    for index, trunk_pass in enumerate(checkpoint_passes):
        arrays[f"trunk_pass_{trunk_pass}_single"] = np.asarray(
            result["trunk_single_checkpoints"][index]
        )
        arrays[f"trunk_pass_{trunk_pass}_pair"] = np.asarray(
            result["trunk_pair_checkpoints"][index]
        )
    for name in (
        "trunk_pass_6_pre_pairformer_single",
        "trunk_pass_6_pre_pairformer_pair",
        "trunk_pass_6_pairformer_block_1_single",
        "trunk_pass_6_pairformer_block_1_pair",
    ):
        arrays[name] = np.asarray(result[name])
    trajectory = np.asarray(diffusion["trajectory"])
    for step in (0, steps // 2 - 1, steps - 1):
        arrays[f"diffusion_step_{step + 1}"] = trajectory[step]
    for key in (
        "predicted_lddt",
        "predicted_experimentally_resolved",
        "full_pde",
        "average_pde",
        "full_pae",
        "tmscore_adjusted_pae_global",
        "tmscore_adjusted_pae_interface",
    ):
        arrays[key] = np.asarray(result[key])
    return arrays


def main() -> int:
    args = parse_args()
    if not flags.FLAGS.is_parsed():
        flags.FLAGS([sys.argv[0]])
    frozen_features, feature_metadata = load_npz(args.frozen_features)
    random_tape, tape_metadata = load_npz(args.random_tape)
    num_samples = int(tape_metadata["num_samples"])
    diffusion_steps = int(tape_metadata["diffusion_steps"])
    devices = jax.local_devices(backend="gpu")
    config = make_model_config(
        flash_attention_implementation="xla",
        num_diffusion_samples=num_samples,
        num_recycles=args.num_recycles,
        return_embeddings=True,
        return_distogram=True,
    )
    if args.precision == "fp32":
        config.global_config.bfloat16 = "none"
    runner = ModelRunner(
        config=config,
        device=devices[args.device],
        model_dir=args.model_dir,
    )
    result = runner.run_inference(
        frozen_features, jax.random.PRNGKey(args.seed), random_tape=random_tape
    )
    metadata = {
        "artifact_type": "alphafold3_parity_checkpoints",
        "format_version": 1,
        "backend": "jax",
        "model_seed": args.seed,
        "num_recycles": args.num_recycles,
        "num_trunk_passes": args.num_recycles + 1,
        "num_samples": num_samples,
        "diffusion_steps": diffusion_steps,
        "precision": args.precision,
        "job_name": feature_metadata["job_name"],
        "frozen_features_sha256": sha256_file(args.frozen_features),
        "random_tape_sha256": sha256_file(args.random_tape),
        "weights_sha256": sha256_file(args.weights_file),
    }
    arrays = checkpoint_arrays(result, diffusion_steps, args.num_recycles + 1)
    write_npz(args.output_file, arrays, metadata)
    print(f"Wrote JAX parity checkpoints to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
