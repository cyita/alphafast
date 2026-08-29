#!/usr/bin/env python3
"""Capture the first JAX Evoformer pass without running downstream heads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from parity_io import configure_deterministic_gpu

configure_deterministic_gpu()

from absl import flags
from alphafold3.model import feat_batch
from alphafold3.model import params as model_params
from alphafold3.model.components import utils
from alphafold3.model.inference import make_model_config
from alphafold3.model.network import atom_cross_attention
from alphafold3.model.network import evoformer as evoformer_network
from alphafold3.model.network import featurization
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

from parity_io import gpu_determinism_metadata, load_npz, sha256_file, write_npz


ATOM_SINGLE_EMBEDDINGS = {
    "evoformer_conditioning_embed_ref_pos": "atom_embed_ref_pos",
    "evoformer_conditioning_embed_ref_mask": "atom_embed_ref_mask",
    "evoformer_conditioning_embed_ref_element": "atom_embed_ref_element",
    "evoformer_conditioning_embed_ref_charge": "atom_embed_ref_charge",
    "evoformer_conditioning_embed_ref_atom_name": "atom_embed_ref_atom_name",
}

ATOM_PAIR_PROJECTIONS = {
    "evoformer_conditioning_single_to_pair_cond_row_1": "atom_pair_row",
    "evoformer_conditioning_single_to_pair_cond_col_1": "atom_pair_col",
    "evoformer_conditioning_embed_pair_offsets_1": "atom_pair_offsets",
    "evoformer_conditioning_embed_pair_distances_1": "atom_pair_distances",
    "evoformer_conditioning_embed_pair_offsets_valid": "atom_pair_offsets_valid",
    "evoformer_conditioning_pair_mlp_1": "atom_pair_mlp_1",
    "evoformer_conditioning_pair_mlp_2": "atom_pair_mlp_2",
    "evoformer_conditioning_pair_mlp_3": "atom_pair_mlp_3",
}

ATOM_BOUNDARIES = ATOM_SINGLE_EMBEDDINGS | ATOM_PAIR_PROJECTIONS

ATOM_ATTENTION_INTERMEDIATES = (
    "q_adaln_x_norm",
    "q_adaln_single_cond_norm",
    "q_adaln_single_scale",
    "q_adaln_single_bias",
    "k_adaln_x_norm",
    "k_adaln_single_cond_norm",
    "k_adaln_single_scale",
    "k_adaln_single_bias",
    "q_norm",
    "k_norm",
    "q_projection",
    "k_projection",
    "logits",
    "weights",
    "v_projection",
    "weighted_average",
    "gate_logits",
    "gated_average",
    "attention_update",
)


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
        atom_terms = {}

        def capture_atom_terms(next_f, call_args, call_kwargs, context):
            output = next_f(*call_args, **call_kwargs)
            tensor_name = ATOM_BOUNDARIES.get(context.module.name)
            if tensor_name and context.method_name == "__call__":
                atom_terms[f"{tensor_name}_input"] = call_args[0]
                atom_terms[tensor_name] = output
            return output

        target_feat_base = featurization.create_target_feat(
            batch, append_per_atom_features=False
        )
        with hk.intercept_methods(capture_atom_terms):
            (
                enc,
                atom_transformer_intermediates,
            ) = atom_cross_attention.atom_cross_att_encoder(
                token_atoms_act=None,
                trunk_single_cond=None,
                trunk_pair_cond=None,
                config=embedding_module.config.per_atom_conditioning,
                global_config=config.global_config,
                batch=batch,
                name="evoformer_conditioning",
                capture_transformer=True,
            )
        target_feat = jnp.concatenate([target_feat_base, enc.token_act], axis=-1)
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
        return (
            target_feat_base,
            atom_terms,
            enc,
            atom_transformer_intermediates,
            target_feat,
            embeddings,
        )

    params = model_params.get_model_haiku_params(model_dir=args.model_dir)
    batch = jax.device_put(
        jax.tree.map(jnp.asarray, utils.remove_invalidly_typed_feats(frozen)),
        device,
    )
    _, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    (
        target_feat_base,
        atom_terms,
        enc,
        atom_transformer_intermediates,
        target_feat,
        result,
    ) = jax.jit(run_pass.apply, device=device)(
        evoformer_params(params),
        None,
        batch,
        subkey,
        jax.device_put(jnp.asarray(tape["msa_row_order"][0]), device),
    )
    arrays = {
        **{name: np.asarray(value) for name, value in atom_terms.items()},
        "target_feat_base": np.asarray(target_feat_base),
        "target_feat": np.asarray(target_feat),
        "target_feat_atom": np.asarray(enc.token_act),
        "atom_queries_single_cond": np.asarray(enc.queries_single_cond),
        "atom_queries_mask": np.asarray(enc.queries_mask),
        "atom_keys_single_cond": np.asarray(enc.keys_single_cond),
        "atom_keys_mask": np.asarray(enc.keys_mask),
        "atom_pair_cond": np.asarray(enc.pair_cond),
        "atom_skip_connection": np.asarray(enc.skip_connection),
        "atom_transformer_same_input": np.asarray(enc.skip_connection),
        "pre_pair": np.asarray(result["pairformer_pre_pair"]),
        "pre_single": np.asarray(result["pairformer_pre_single"]),
        "block_1_pair": np.asarray(result["pairformer_block_1_pair"]),
        "block_1_single": np.asarray(result["pairformer_block_1_single"]),
        "output_pair": np.asarray(result["pair"]),
        "output_single": np.asarray(result["single"]),
    }
    query_mask = np.asarray(enc.queries_mask)
    key_mask = np.asarray(enc.keys_mask)
    atom_mask = query_mask[..., None]
    attention_acts = atom_transformer_intermediates["attention_state"]
    transition_acts = atom_transformer_intermediates["transition_state"]
    for block_index in range(attention_acts.shape[0]):
        arrays[f"atom_transformer_block_{block_index + 1}_attention"] = (
            np.asarray(attention_acts[block_index]) * atom_mask
        )
        arrays[f"atom_transformer_block_{block_index + 1}_transition"] = (
            np.asarray(transition_acts[block_index]) * atom_mask
        )
    query_terms = {
        "q_adaln_x_norm",
        "q_adaln_single_cond_norm",
        "q_adaln_single_scale",
        "q_adaln_single_bias",
        "q_norm",
        "weighted_average",
        "gate_logits",
        "gated_average",
        "attention_update",
    }
    key_terms = {
        "k_adaln_x_norm",
        "k_adaln_single_cond_norm",
        "k_adaln_single_scale",
        "k_adaln_single_bias",
        "k_norm",
    }
    query_head_terms = {"q_projection"}
    key_head_terms = {"k_projection", "v_projection"}
    pair_mask = query_mask[:, None, :, None] * key_mask[:, None, None, :]
    for name in ATOM_ATTENTION_INTERMEDIATES:
        value = np.asarray(atom_transformer_intermediates[name][0])
        if name in query_terms:
            value = value * query_mask[..., None]
        elif name in key_terms:
            value = value * key_mask[..., None]
        elif name in query_head_terms:
            value = value * query_mask[..., None, None]
        elif name in key_head_terms:
            value = value * key_mask[..., None, None]
        else:
            value = value * pair_mask
        arrays[f"atom_transformer_block_1_{name}"] = value
    metadata = {
        "artifact_type": "evoformer_pass_parity",
        "format_version": 1,
        "backend": "jax",
        "pass": 1,
        "precision": "fp32",
        "frozen_features_sha256": sha256_file(args.frozen_features),
        "random_tape_sha256": sha256_file(args.random_tape),
        "weights_sha256": sha256_file(args.weights_file),
        "gpu_determinism": gpu_determinism_metadata(),
    }
    write_npz(args.output_file, arrays, metadata)
    print(f"Wrote JAX Evoformer pass 1 to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
