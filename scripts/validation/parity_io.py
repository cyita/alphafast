#!/usr/bin/env python3
"""Small helpers for deterministic JAX/Torch parity artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


METADATA_KEY = "_metadata_json"
DETERMINISTIC_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops"


def configure_deterministic_gpu() -> None:
    """Configure reproducible XLA/cuBLAS kernels before JAX initializes."""
    flags = os.environ.get("XLA_FLAGS", "").split()
    if DETERMINISTIC_XLA_FLAG not in flags:
        flags.append(DETERMINISTIC_XLA_FLAG)
    os.environ["XLA_FLAGS"] = " ".join(flags)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def gpu_determinism_metadata() -> dict[str, str]:
    return {
        "xla_flags": os.environ["XLA_FLAGS"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: archive[key]
            for key in archive.files
            if key != METADATA_KEY
        }
        metadata = json.loads(str(archive[METADATA_KEY])) if METADATA_KEY in archive else {}
    return arrays, metadata


def load_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            json.loads(str(archive[METADATA_KEY]))
            if METADATA_KEY in archive
            else {}
        )


def write_npz(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        **{METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True))},
    )
