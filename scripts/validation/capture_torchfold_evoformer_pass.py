#!/usr/bin/env python3
"""Capture the first TorchFold Evoformer pass without downstream heads."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from alphafold3.model.components import utils
import torch
import torch.utils._pytree as pytree
from torchfold import feat_batch
from torchfold.alphafold3 import AlphaFold3
from torchfold.fastnn import config as fastnn_config
from torchfold.nn import featurization
from torchfold.params import import_jax_weights_

from parity_io import load_npz, sha256_file, write_npz
from capture_torchfold_parity import to_numpy


ATOM_SINGLE_EMBEDDINGS = {
    "embed_ref_pos": "atom_embed_ref_pos",
    "embed_ref_mask": "atom_embed_ref_mask",
    "embed_ref_element": "atom_embed_ref_element",
    "embed_ref_charge": "atom_embed_ref_charge",
    "embed_ref_atom_name": "atom_embed_ref_atom_name",
}

ATOM_PAIR_PROJECTIONS = {
    "single_to_pair_cond_row_1": "atom_pair_row",
    "single_to_pair_cond_col_1": "atom_pair_col",
    "embed_pair_offsets_1": "atom_pair_offsets",
    "embed_pair_distances_1": "atom_pair_distances",
    "embed_pair_offsets_valid": "atom_pair_offsets_valid",
    "pair_mlp_1": "atom_pair_mlp_1",
    "pair_mlp_2": "atom_pair_mlp_2",
    "pair_mlp_3": "atom_pair_mlp_3",
}

ATOM_BOUNDARIES = ATOM_SINGLE_EMBEDDINGS | ATOM_PAIR_PROJECTIONS

ATOM_ATTENTION_INTERMEDIATES = (
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
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-inputs", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen, _ = load_npz(args.frozen_features)
    tape, _ = load_npz(args.random_tape)
    device = torch.device(args.device)
    fastnn_config.layer_norm_implementation = "torch"
    fastnn_config.dot_product_attention_implementation = "torch"
    fastnn_config.gated_linear_unit_implementation = "torch"
    model = AlphaFold3()
    import_jax_weights_(model, args.model_dir)
    model.float().eval().to(device)

    batch_dict = pytree.tree_map(
        torch.from_numpy, utils.remove_invalidly_typed_feats(frozen)
    )
    batch_dict = pytree.tree_map_only(
        torch.Tensor, lambda value: value.to(device), batch_dict
    )
    batch_dict["deletion_mean"] = batch_dict["deletion_mean"].float()
    batch = feat_batch.Batch.from_data_dict(batch_dict)
    atom_terms = {}

    def capture_atom_term(tensor_name):
        def hook(_module, inputs, output):
            atom_terms[f"{tensor_name}_input"] = inputs[0].clone()
            atom_terms[tensor_name] = output.clone()

        return hook

    hooks = [
        getattr(model.evoformer_conditioning, module_name).register_forward_hook(
            capture_atom_term(tensor_name)
        )
        for module_name, tensor_name in ATOM_BOUNDARIES.items()
    ]
    with torch.inference_mode():
        try:
            target_feat_base = featurization.create_target_feat(
                batch, append_per_atom_features=False
            )
            enc = model.evoformer_conditioning(
                token_atoms_act=None,
                trunk_single_cond=None,
                trunk_pair_cond=None,
                batch=batch,
            )
        finally:
            for hook in hooks:
                hook.remove()
        atom_transformer_same_input = None
        atom_transformer_block_acts = {}
        if args.reference_inputs is not None:
            reference, _ = load_npz(args.reference_inputs)
            queries_single_cond = torch.from_numpy(
                reference["atom_queries_single_cond"]
            ).to(device)
            queries_mask = torch.from_numpy(reference["atom_queries_mask"]).to(device)
            keys_single_cond = torch.from_numpy(
                reference["atom_keys_single_cond"]
            ).to(device)
            keys_mask = torch.from_numpy(reference["atom_keys_mask"]).to(device)
            pair_cond = torch.from_numpy(reference["atom_pair_cond"]).to(device)
            transformer = model.evoformer_conditioning.atom_transformer_encoder
            block_one_attention = transformer.cross_attention[0]
            block_one_attention.capture_intermediates = True
            block_one_attention.last_intermediates = None
            attention_updates = []
            transition_updates = []

            def capture_update(outputs):
                def hook(_module, _inputs, output):
                    outputs.append(output.clone())

                return hook

            transformer_hooks = [
                module.register_forward_hook(capture_update(attention_updates))
                for module in transformer.cross_attention
            ] + [
                module.register_forward_hook(capture_update(transition_updates))
                for module in transformer.transition_block
            ]
            transformer.first_run = True
            transformer.pair_logits = None
            try:
                atom_transformer_same_input = transformer(
                    queries_act=queries_single_cond.clone(),
                    queries_mask=queries_mask,
                    queries_to_keys=batch.atom_cross_att.queries_to_keys,
                    keys_mask=keys_mask,
                    queries_single_cond=queries_single_cond,
                    keys_single_cond=keys_single_cond,
                    pair_cond=pair_cond,
                )
            finally:
                for hook in transformer_hooks:
                    hook.remove()
                block_one_attention.capture_intermediates = False
            atom_transformer_same_input *= queries_mask[..., None]
            block_act = queries_single_cond.clone()
            for block_index, (attention_update, transition_update) in enumerate(
                zip(attention_updates, transition_updates), start=1
            ):
                block_act += attention_update
                atom_transformer_block_acts[
                    f"atom_transformer_block_{block_index}_attention"
                ] = block_act.clone() * queries_mask[..., None]
                block_act += transition_update
                atom_transformer_block_acts[
                    f"atom_transformer_block_{block_index}_transition"
                ] = block_act.clone() * queries_mask[..., None]
            query_terms = {
                "q_norm",
                "weighted_average",
                "gate_logits",
                "gated_average",
                "attention_update",
            }
            key_terms = {"k_norm"}
            query_head_terms = {"q_projection"}
            key_head_terms = {"k_projection", "v_projection"}
            pair_mask = queries_mask[:, None, :, None] * keys_mask[:, None, None, :]
            for name in ATOM_ATTENTION_INTERMEDIATES:
                value = block_one_attention.last_intermediates[name]
                if name in query_terms:
                    value = value * queries_mask[..., None]
                elif name in key_terms:
                    value = value * keys_mask[..., None]
                elif name in query_head_terms:
                    value = value * queries_mask[..., None, None]
                elif name in key_head_terms:
                    value = value * keys_mask[..., None, None]
                else:
                    value = value * pair_mask
                atom_transformer_block_acts[
                    f"atom_transformer_block_1_{name}"
                ] = value
        target_feat = torch.concatenate([target_feat_base, enc.token_act], dim=-1)
        prev = {
            "pair": torch.zeros(
                [batch.num_res, batch.num_res, model.evoformer_pair_channel],
                dtype=torch.float32,
                device=device,
            ),
            "single": torch.zeros(
                [batch.num_res, model.evoformer_seq_channel],
                dtype=torch.float32,
                device=device,
            ),
            "target_feat": target_feat,
        }
        result = model.evoformer(
            batch=batch,
            prev=prev,
            target_feat=target_feat,
            idx=0,
            msa_row_order=torch.from_numpy(tape["msa_row_order"][0]).to(device),
            capture_pairformer=True,
        )

    candidate_commit = subprocess.run(
        ["git", "-C", str(args.candidate_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    arrays = {
        **{name: to_numpy(value) for name, value in atom_terms.items()},
        "target_feat_base": to_numpy(target_feat_base),
        "target_feat": to_numpy(target_feat),
        "target_feat_atom": to_numpy(enc.token_act),
        "atom_queries_single_cond": to_numpy(enc.queries_single_cond),
        "atom_queries_mask": to_numpy(enc.queries_mask),
        "atom_keys_single_cond": to_numpy(enc.keys_single_cond),
        "atom_keys_mask": to_numpy(enc.keys_mask),
        "atom_pair_cond": to_numpy(enc.pair_cond),
        "atom_skip_connection": to_numpy(enc.skip_connection),
        "pre_pair": to_numpy(result["pairformer_pre_pair"]),
        "pre_single": to_numpy(result["pairformer_pre_single"]),
        "block_1_pair": to_numpy(result["pairformer_block_1_pair"]),
        "block_1_single": to_numpy(result["pairformer_block_1_single"]),
        "output_pair": to_numpy(result["pair"]),
        "output_single": to_numpy(result["single"]),
    }
    if atom_transformer_same_input is not None:
        arrays["atom_transformer_same_input"] = to_numpy(
            atom_transformer_same_input
        )
        arrays.update(
            {
                name: to_numpy(value)
                for name, value in atom_transformer_block_acts.items()
            }
        )
    metadata = {
        "artifact_type": "evoformer_pass_parity",
        "format_version": 1,
        "backend": "torchfold",
        "pass": 1,
        "precision": "fp32",
        "frozen_features_sha256": sha256_file(args.frozen_features),
        "random_tape_sha256": sha256_file(args.random_tape),
        "weights_sha256": sha256_file(args.weights_file),
        "candidate_commit": candidate_commit,
    }
    write_npz(args.output_file, arrays, metadata)
    print(f"Wrote TorchFold Evoformer pass 1 to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
