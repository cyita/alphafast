import os
from unittest import mock

from scripts.validation.parity_io import (
    DETERMINISTIC_XLA_FLAG,
    configure_deterministic_gpu,
    gpu_determinism_metadata,
)


def test_configure_deterministic_gpu_preserves_existing_flags():
    with mock.patch.dict(
        os.environ,
        {"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"},
        clear=True,
    ):
        configure_deterministic_gpu()
        configure_deterministic_gpu()
        metadata = gpu_determinism_metadata()

    assert metadata["xla_flags"].split().count(DETERMINISTIC_XLA_FLAG) == 1
    assert "--xla_gpu_enable_triton_gemm=false" in metadata["xla_flags"]
    assert metadata["cublas_workspace_config"] == ":4096:8"


def test_configure_deterministic_gpu_keeps_explicit_workspace():
    with mock.patch.dict(
        os.environ,
        {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
        clear=True,
    ):
        configure_deterministic_gpu()
        assert gpu_determinism_metadata()["cublas_workspace_config"] == ":16:8"
