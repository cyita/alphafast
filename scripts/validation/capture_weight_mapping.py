#!/usr/bin/env python3
"""Load official AF3 weights into TorchFold and report the actual mapping."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess


META_KEY = "__meta__/__identifier__"
MANUAL_PARAMETERS = {
    "diffusion_head.fourier_embeddings.bias",
    "diffusion_head.fourier_embeddings.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def target_tensors(mapping: object) -> list[object]:
    value = mapping.param
    return list(value) if isinstance(value, list) else [value]


def transformed_tensors(mapping: object, source: object) -> list[object]:
    targets = target_tensors(mapping)
    if mapping.stacked:
        if len(targets) == source.shape[0]:
            values = list(source.unbind(0))
        elif source.ndim > 1 and len(targets) == source.shape[0] * source.shape[1]:
            values = list(source.reshape(-1, *source.shape[2:]).unbind(0))
        else:
            return []
    else:
        values = [source]
    return [mapping.param_type.transformation(value) for value in values]


def git_commit(source_file: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_file.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main() -> int:
    args = parse_args()
    import torch
    from torchfold.alphafold3 import AlphaFold3
    from torchfold import params as torchfold_params

    weights_file = args.weights_file.resolve()
    source = torchfold_params.get_alphafold3_params(weights_file)
    model = AlphaFold3()
    mappings = torchfold_params._process_translations_dict(
        torchfold_params.get_translation_dict(model),
        _key_prefix="diffuser/",
    )

    source_keys = set(source)
    mapped_source_keys = set(mappings) & source_keys
    source_without_target = sorted(source_keys - set(mappings) - {META_KEY})

    mapped_target_ids = {
        id(tensor)
        for mapping in mappings.values()
        for tensor in target_tensors(mapping)
    }
    target_without_source = sorted(
        name
        for name, parameter in model.named_parameters()
        if id(parameter) not in mapped_target_ids and name not in MANUAL_PARAMETERS
    )

    shape_mismatches = []
    dtype_mismatches = []
    comparisons = []
    for key in sorted(mapped_source_keys):
        targets = target_tensors(mappings[key])
        expected = transformed_tensors(mappings[key], source[key])
        if len(targets) != len(expected):
            shape_mismatches.append({
                "source": key,
                "source_shape": list(source[key].shape),
                "target_count": len(targets),
            })
            continue
        for index, (target, value) in enumerate(zip(targets, expected)):
            suffix = f"[{index}]" if len(targets) > 1 else ""
            if target.shape != value.shape:
                shape_mismatches.append({
                    "source": key + suffix,
                    "source_shape": list(value.shape),
                    "target_shape": list(target.shape),
                })
            elif target.dtype != value.dtype:
                dtype_mismatches.append({
                    "source": key + suffix,
                    "source_dtype": str(value.dtype),
                    "target_dtype": str(target.dtype),
                })
            else:
                comparisons.append((key + suffix, target, value))

    value_mismatches = []
    if not shape_mismatches and not dtype_mismatches:
        try:
            torchfold_params.import_jax_weights_(model, weights_file.parent)
        except Exception as error:
            value_mismatches.append({"loader_error": str(error)})
        else:
            for name, target, expected in comparisons:
                if not torch.equal(target.detach(), expected.to(target.device)):
                    value_mismatches.append(name)

    implementation_file = Path(inspect.getfile(AlphaFold3)).resolve()
    report = {
        "source_sha256": sha256_file(weights_file),
        "records_total": len(source),
        "records_mapped": len(mapped_source_keys) + int(META_KEY in source_keys),
        "source_records_without_target": source_without_target,
        "target_parameters_without_source": target_without_source,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
        "implementation": {
            "file": str(implementation_file),
            "commit": git_commit(implementation_file),
            "torch_parameters": len(list(model.named_parameters())),
            "translation_records": len(mappings),
            "manual_parameters": sorted(MANUAL_PARAMETERS),
        },
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote weight mapping report to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
