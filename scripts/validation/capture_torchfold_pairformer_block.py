#!/usr/bin/env python3
"""Run one isolated TorchFold Pairformer block from a JAX checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import torch
from torchfold.alphafold3 import AlphaFold3
from torchfold.fastnn import config as fastnn_config
from torchfold.params import import_jax_weights_

from parity_io import load_npz, write_npz
from capture_torchfold_parity import to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-block", type=Path, required=True)
    parser.add_argument("--chain-reference", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, source_metadata = load_npz(args.jax_block)
    block_number = int(source_metadata["block"])
    device = torch.device(args.device)

    fastnn_config.layer_norm_implementation = "torch"
    fastnn_config.dot_product_attention_implementation = "torch"
    fastnn_config.gated_linear_unit_implementation = "torch"
    model = AlphaFold3()
    import_jax_weights_(model, args.model_dir)
    model.float().eval().to(device)

    if args.chain_reference is None:
        pair_input = arrays["input_pair"]
        single_input = arrays["input_single"]
    else:
        chain_reference, _ = load_npz(args.chain_reference)
        pair_input = chain_reference["trunk_pass_6_pre_pairformer_pair"]
        single_input = chain_reference["trunk_pass_6_pre_pairformer_single"]
    pair = torch.from_numpy(pair_input).to(device)
    single = torch.from_numpy(single_input).to(device)
    pair_mask = torch.from_numpy(arrays["pair_mask"]).to(device)
    seq_mask = torch.from_numpy(arrays["seq_mask"]).to(device)
    with torch.inference_mode():
        if args.chain_reference is None:
            output_pair, output_single = model.evoformer.trunk_pairformer[
                block_number - 1
            ](pair, pair_mask, single, seq_mask)
        else:
            output_pair, output_single = pair, single
            for block in model.evoformer.trunk_pairformer[:block_number]:
                output_pair, output_single = block(
                    output_pair, pair_mask, output_single, seq_mask
                )

    candidate_commit = subprocess.run(
        ["git", "-C", str(args.candidate_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "artifact_type": "pairformer_block_parity",
        "format_version": 1,
        "backend": "torchfold",
        "block": block_number,
        "precision": "fp32",
        "source_reference_sha256": source_metadata["source_reference_sha256"],
        "weights_sha256": source_metadata["weights_sha256"],
        "candidate_commit": candidate_commit,
        "mode": "isolated" if args.chain_reference is None else "chain",
    }
    write_npz(
        args.output_file,
        {
            "output_pair": to_numpy(output_pair),
            "output_single": to_numpy(output_single),
        },
        metadata,
    )
    print(f"Wrote TorchFold Pairformer block {block_number} to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
