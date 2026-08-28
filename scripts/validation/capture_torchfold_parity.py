#!/usr/bin/env python3
"""Run TorchFold from frozen features and save parity checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphafold3.model.components import utils
import numpy as np
import torch
import torch.utils._pytree as pytree
from torchfold.alphafold3 import AlphaFold3
from torchfold.fastnn import config as fastnn_config
from torchfold.params import import_jax_weights_

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
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def checkpoint_arrays(
    result: dict[str, object], steps: int
) -> dict[str, np.ndarray]:
    diffusion = result["diffusion_samples"]
    distogram = result["distogram"]
    arrays = {
        "single_embeddings": to_numpy(result["single_embeddings"]),
        "pair_embeddings": to_numpy(result["pair_embeddings"]),
        "atom_positions": to_numpy(diffusion["atom_positions"]),
        "distogram_contact_probs": to_numpy(distogram["contact_probs"]),
        "distogram_logits": to_numpy(distogram["distogram"]),
        "denoised_step_1": to_numpy(diffusion["denoised_step_1"]),
    }
    for name in ("pairformer_block_1_single", "pairformer_block_1_pair"):
        if name in result:
            arrays[name] = to_numpy(result[name])
    for name in (
        "trunk_pass_6_pre_pairformer_single",
        "trunk_pass_6_pre_pairformer_pair",
        "trunk_pass_6_pairformer_block_1_single",
        "trunk_pass_6_pairformer_block_1_pair",
        "trunk_pass_6_pairformer_block_24_single",
        "trunk_pass_6_pairformer_block_24_pair",
    ):
        if name in result:
            arrays[name] = to_numpy(result[name])

    checkpoint_passes = result["trunk_checkpoint_passes"]
    for index, trunk_pass in enumerate(checkpoint_passes):
        arrays[f"trunk_pass_{trunk_pass}_single"] = to_numpy(
            result["trunk_single_checkpoints"][index]
        )
        arrays[f"trunk_pass_{trunk_pass}_pair"] = to_numpy(
            result["trunk_pair_checkpoints"][index]
        )
    trajectory = diffusion["trajectory"]
    for step in (0, steps // 2 - 1, steps - 1):
        arrays[f"diffusion_step_{step + 1}"] = to_numpy(trajectory[step])
    for key in (
        "predicted_lddt",
        "predicted_experimentally_resolved",
        "full_pde",
        "average_pde",
        "full_pae",
        "tmscore_adjusted_pae_global",
        "tmscore_adjusted_pae_interface",
    ):
        arrays[key] = to_numpy(result[key])
    return arrays


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    frozen_features, feature_metadata = load_npz(args.frozen_features)
    random_tape, tape_metadata = load_npz(args.random_tape)
    num_samples = int(tape_metadata["num_samples"])
    diffusion_steps = int(tape_metadata["diffusion_steps"])

    fastnn_config.layer_norm_implementation = "torch"
    fastnn_config.dot_product_attention_implementation = "torch"
    fastnn_config.gated_linear_unit_implementation = "torch"
    model = AlphaFold3(
        num_recycles=args.num_recycles,
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
    )
    import_jax_weights_(model, args.model_dir)
    if args.precision == "fp32":
        model.float()
    model.eval().to(device)

    batch = pytree.tree_map(
        torch.from_numpy, utils.remove_invalidly_typed_feats(frozen_features)
    )
    batch = pytree.tree_map_only(torch.Tensor, lambda value: value.to(device), batch)
    batch["deletion_mean"] = batch["deletion_mean"].float()
    tape = {key: torch.from_numpy(value).to(device) for key, value in random_tape.items()}
    with torch.inference_mode():
        result = model(batch, random_tape=tape)
    actual_num_trunk_passes = int(result["num_trunk_passes"])

    metadata = {
        "artifact_type": "alphafold3_parity_checkpoints",
        "format_version": 1,
        "backend": "torchfold",
        "model_seed": args.seed,
        "num_recycles": args.num_recycles,
        "num_trunk_passes": actual_num_trunk_passes,
        "expected_num_trunk_passes": args.num_recycles + 1,
        "num_samples": num_samples,
        "diffusion_steps": diffusion_steps,
        "precision": args.precision,
        "job_name": feature_metadata["job_name"],
        "frozen_features_sha256": sha256_file(args.frozen_features),
        "random_tape_sha256": sha256_file(args.random_tape),
        "weights_sha256": sha256_file(args.weights_file),
    }
    arrays = checkpoint_arrays(result, diffusion_steps)
    write_npz(args.output_file, arrays, metadata)
    print(
        f"Wrote TorchFold parity checkpoints to {args.output_file} "
        f"({actual_num_trunk_passes} trunk passes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
