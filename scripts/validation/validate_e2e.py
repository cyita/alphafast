#!/usr/bin/env python3
"""Compare deterministic JAX and Torch parity checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from parity_io import load_npz


DEFAULT_METADATA_FIELDS = (
    "artifact_type",
    "format_version",
    "model_seed",
    "num_recycles",
    "num_trunk_passes",
    "num_samples",
    "diffusion_steps",
    "precision",
    "job_name",
    "frozen_features_sha256",
    "random_tape_sha256",
    "weights_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument(
        "--result-file", type=Path, default=Path("ae-results/latest_result.json")
    )
    return parser.parse_args()


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def compare_tensor(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance: dict[str, float],
) -> tuple[bool, dict[str, object]]:
    metrics: dict[str, object] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        return False, metrics
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        metrics["finite"] = False
        return False, metrics

    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    difference = np.abs(candidate64 - reference64)
    reference_norm = float(np.linalg.norm(reference64.ravel()))
    candidate_norm = float(np.linalg.norm(candidate64.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    normalized_rmse = (
        difference_norm / reference_norm
        if reference_norm
        else (0.0 if difference_norm == 0.0 else float("inf"))
    )
    cosine = (
        float(
            np.vdot(reference64.ravel(), candidate64.ravel())
            / (reference_norm * candidate_norm)
        )
        if reference_norm and candidate_norm
        else (1.0 if difference_norm == 0.0 else 0.0)
    )
    metrics.update(
        {
            "finite": True,
            "max_abs_error": float(np.max(difference)),
            "mean_abs_error": float(np.mean(difference)),
            "normalized_rmse": normalized_rmse,
            "cosine_similarity": cosine,
            "total_elements": int(reference.size),
        }
    )
    checks = []
    if "atol" in tolerance or "rtol" in tolerance:
        atol = float(tolerance.get("atol", 0.0))
        rtol = float(tolerance.get("rtol", 0.0))
        failed = difference > atol + rtol * np.abs(reference64)
        failed_elements = int(np.count_nonzero(failed))
        failed_fraction = failed_elements / reference.size
        metrics.update(
            {
                "atol": atol,
                "rtol": rtol,
                "failed_elements": failed_elements,
                "failed_fraction": failed_fraction,
            }
        )
        checks.append(failed_fraction <= tolerance.get("max_failed_fraction", 0.0))
    if "max_normalized_rmse" in tolerance:
        checks.append(normalized_rmse <= tolerance["max_normalized_rmse"])
    if "min_cosine_similarity" in tolerance:
        checks.append(cosine >= tolerance["min_cosine_similarity"])
    return all(checks), metrics


def validate_artifacts(
    reference_path: Path,
    candidate_path: Path,
    tolerances_path: Path,
    *,
    stage: str = "e2e",
) -> tuple[int, dict[str, object]]:
    try:
        reference, reference_metadata = load_npz(reference_path)
        candidate, candidate_metadata = load_npz(candidate_path)
        contract = json.loads(tolerances_path.read_text())
        tensor_tolerances = contract["tensors"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return 2, {"stage": stage, "status": "error", "reason": str(error)}

    failures = []
    metadata_mismatches = {}
    for field in contract.get("metadata_fields", DEFAULT_METADATA_FIELDS):
        if reference_metadata.get(field) != candidate_metadata.get(field):
            metadata_mismatches[field] = {
                "reference": reference_metadata.get(field),
                "candidate": candidate_metadata.get(field),
            }
            failures.append(f"metadata mismatch: {field}")

    tensor_metrics = {}
    for name, tolerance in tensor_tolerances.items():
        if name not in reference or name not in candidate:
            failures.append(f"missing checkpoint: {name}")
            continue
        passed, metrics = compare_tensor(reference[name], candidate[name], tolerance)
        tensor_metrics[name] = metrics
        if not passed:
            failures.append(f"tensor mismatch: {name}")

    status = "failed" if failures else "passed"
    result = {
        "stage": stage,
        "status": status,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "metadata_mismatches": metadata_mismatches,
        "tensors": tensor_metrics,
        "reason": "; ".join(failures) if failures else "all checkpoints matched",
    }
    return (1 if failures else 0), result


def main() -> int:
    args = parse_args()
    exit_code, result = validate_artifacts(
        args.reference, args.candidate, args.tolerances
    )
    write_result(args.result_file, result)
    if exit_code == 2:
        print(f"E2E validation error: {result['reason']}", file=sys.stderr)
    elif exit_code == 1:
        print(f"E2E validation failed: {result['reason']}", file=sys.stderr)
    else:
        print(f"E2E validation passed: {len(result['tensors'])} checkpoints")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
