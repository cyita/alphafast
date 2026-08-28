#!/usr/bin/env python3
"""Capture the first JAX Evoformer pass without running downstream heads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from absl import flags
from alphafold3.model import feat_batch
from alphafold3.model import model
from alphafold3.model import params as model_params
from alphafold3.model.components import utils
from alphafold3.model.inference import make_model_config
from alphafold3.model.network import evoformer as evoformer_network
import haiku as hk
import jax
from jax import numpy as jnp
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
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def evoformer_params(params: hk.Params) -> hk.Params:
    prefix = "diffuser/"
    return hk.data_structures.to_immutable_dict(
        {
            module_name.removeprefix(prefix): values
            for module_name, values in params.items()
            if module_name.startswith("diffuser/evoformer")
        }
    )


def main() -> int:
    args = parse_args()
    if not flags.FLAGS.is_parsed():
        flags.FLAGS([sys.argv[0]])
    frozen, _ = load_npz(args.frozen_features)
    tape, _ = load_npz(args.random_tape)
    device = jax.local_devices(backend="gpu")[args.device]
    config = make_model_config(flash_attention_implementation="xla")
    config.global_config.bfloat16 = "none"

    @hk.transform
    def run_pass(batch_dict, key, msa_row_order):
        batch = feat_batch.Batch.from_data_dict(batch_dict)
        embedding_module = evoformer_network.Evoformer(
            config.evoformer, config.global_config
        )
        target_feat = model.create_target_feat_embedding(
            batch, embedding_module.config, config.global_config
        )
        prev = {
            "pair": jnp.zeros(
                [batch.num_res, batch.num_res, config.evoformer.pair_channel],
                dtype=jnp.float32,
            ),
            "single": jnp.zeros(
                [batch.num_res, config.evoformer.seq_channel], dtype=jnp.float32
            ),
            "target_feat": target_feat,
        }
        embeddings = embedding_module(
            batch=batch,
            prev=prev,
            target_feat=target_feat,
            key=key,
            msa_row_order=msa_row_order,
            capture_pairformer=True,
        )
        return target_feat, embeddings

    params = model_params.get_model_haiku_params(model_dir=args.model_dir)
    batch = jax.device_put(
        jax.tree.map(jnp.asarray, utils.remove_invalidly_typed_feats(frozen)),
        device,
    )
    _, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    target_feat, result = jax.jit(run_pass.apply, device=device)(
        evoformer_params(params),
        None,
        batch,
        subkey,
        jax.device_put(jnp.asarray(tape["msa_row_order"][0]), device),
    )
    arrays = {
        "target_feat": np.asarray(target_feat),
        "target_feat_atom": np.asarray(target_feat[..., -384:]),
        "pre_pair": np.asarray(result["pairformer_pre_pair"]),
        "pre_single": np.asarray(result["pairformer_pre_single"]),
        "block_1_pair": np.asarray(result["pairformer_block_1_pair"]),
        "block_1_single": np.asarray(result["pairformer_block_1_single"]),
        "output_pair": np.asarray(result["pair"]),
        "output_single": np.asarray(result["single"]),
    }
    metadata = {
        "artifact_type": "evoformer_pass_parity",
        "format_version": 1,
        "backend": "jax",
        "pass": 1,
        "precision": "fp32",
        "frozen_features_sha256": sha256_file(args.frozen_features),
        "random_tape_sha256": sha256_file(args.random_tape),
        "weights_sha256": sha256_file(args.weights_file),
    }
    write_npz(args.output_file, arrays, metadata)
    print(f"Wrote JAX Evoformer pass 1 to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
