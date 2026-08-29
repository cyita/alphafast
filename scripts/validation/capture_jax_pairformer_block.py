#!/usr/bin/env python3
"""Run one isolated JAX Pairformer block from a trusted checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from parity_io import configure_deterministic_gpu

configure_deterministic_gpu()

from absl import flags
from alphafold3.model import params as model_params
from alphafold3.model.inference import make_model_config
from alphafold3.model.network import modules
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

from parity_io import gpu_determinism_metadata, load_npz, sha256_file, write_npz


PAIRFORMER_PREFIX = (
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-file", type=Path, required=True)
    parser.add_argument("--block", type=int, choices=range(1, 49), required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def block_params(params: hk.Params, block_index: int) -> hk.Params:
    return hk.data_structures.to_immutable_dict(
        {
            "trunk_pairformer/" + module_name.removeprefix(PAIRFORMER_PREFIX): {
                name: value[block_index] for name, value in values.items()
            }
            for module_name, values in params.items()
            if module_name.startswith(PAIRFORMER_PREFIX)
        }
    )


def main() -> int:
    args = parse_args()
    if not flags.FLAGS.is_parsed():
        flags.FLAGS([sys.argv[0]])
    reference, _ = load_npz(args.reference)
    frozen_features, _ = load_npz(args.frozen_features)
    device = jax.local_devices(backend="gpu")[args.device]
    config = make_model_config(flash_attention_implementation="xla")
    config.global_config.bfloat16 = "none"

    @hk.transform
    def run_block(pair, single, pair_mask, seq_mask):
        return modules.PairFormerIteration(
            config.evoformer.pairformer,
            config.global_config,
            with_single=True,
            name="trunk_pairformer",
        )(pair, pair_mask, single, seq_mask)

    apply_block = jax.jit(run_block.apply, device=device)
    params = model_params.get_model_haiku_params(model_dir=args.model_dir)
    pair = jax.device_put(
        jnp.asarray(reference["trunk_pass_6_pre_pairformer_pair"]), device
    )
    single = jax.device_put(
        jnp.asarray(reference["trunk_pass_6_pre_pairformer_single"]), device
    )
    seq_mask = jax.device_put(jnp.asarray(frozen_features["seq_mask"]), device)
    pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(jnp.float32)

    target_input_pair = None
    target_input_single = None
    for block_index in range(args.block):
        if block_index == args.block - 1:
            target_input_pair = pair
            target_input_single = single
        pair, single = apply_block(
            block_params(params, block_index),
            None,
            pair,
            single,
            pair_mask,
            seq_mask,
        )

    arrays = {
        "input_pair": np.asarray(target_input_pair),
        "input_single": np.asarray(target_input_single),
        "output_pair": np.asarray(pair),
        "output_single": np.asarray(single),
        "pair_mask": np.asarray(pair_mask),
        "seq_mask": np.asarray(seq_mask),
    }
    metadata = {
        "artifact_type": "pairformer_block_parity",
        "format_version": 1,
        "backend": "jax",
        "block": args.block,
        "precision": "fp32",
        "source_reference_sha256": sha256_file(args.reference),
        "weights_sha256": sha256_file(args.weights_file),
        "gpu_determinism": gpu_determinism_metadata(),
    }
    write_npz(args.output_file, arrays, metadata)
    print(f"Wrote JAX Pairformer block {args.block} to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
