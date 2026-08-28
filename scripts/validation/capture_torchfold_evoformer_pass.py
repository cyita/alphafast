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
from torchfold.params import import_jax_weights_

from parity_io import load_npz, sha256_file, write_npz
from capture_torchfold_parity import to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--random-tape", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
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
    with torch.inference_mode():
        target_feat = model.create_target_feat_embedding(batch)
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
        "pre_pair": to_numpy(result["pairformer_pre_pair"]),
        "pre_single": to_numpy(result["pairformer_pre_single"]),
        "block_1_pair": to_numpy(result["pairformer_block_1_pair"]),
        "block_1_single": to_numpy(result["pairformer_block_1_single"]),
        "output_pair": to_numpy(result["pair"]),
        "output_single": to_numpy(result["single"]),
    }
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
