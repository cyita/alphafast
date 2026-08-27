from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_torch.sh"


def write_artifact(
    path: Path, value: np.ndarray, *, num_trunk_passes: int = 11
) -> None:
    metadata = {
        "artifact_type": "alphafold3_parity_checkpoints",
        "format_version": 1,
        "model_seed": 101,
        "num_recycles": 10,
        "num_trunk_passes": num_trunk_passes,
        "num_samples": 1,
        "diffusion_steps": 200,
        "job_name": "1UBQ",
        "frozen_features_sha256": "features",
        "random_tape_sha256": "tape",
        "weights_sha256": "weights",
    }
    np.savez_compressed(
        path,
        single_embeddings=value,
        _metadata_json=np.asarray(json.dumps(metadata)),
    )


class ValidateE2ETest(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.reference = self.root / "jax.npz"
        self.candidate = self.root / "torch.npz"
        self.tolerances = self.root / "tolerances.json"
        self.result = self.root / "result.json"
        self.tolerances.write_text(
            json.dumps(
                {
                    "tensors": {
                        "single_embeddings": {"atol": 1e-3, "rtol": 1e-3}
                    }
                }
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_validator(self) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        environment["ALPHAFAST_VALIDATION_PYTHON"] = sys.executable
        return subprocess.run(
            [
                "bash",
                str(VALIDATE_SCRIPT),
                "e2e",
                "--reference",
                str(self.reference),
                "--candidate",
                str(self.candidate),
                "--tolerances",
                str(self.tolerances),
                "--result-file",
                str(self.result),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_passes_within_tolerance(self) -> None:
        write_artifact(self.reference, np.asarray([1.0, 2.0], dtype=np.float32))
        write_artifact(
            self.candidate, np.asarray([1.0005, 1.9995], dtype=np.float32)
        )

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["tensors"]["single_embeddings"]["failed_elements"], 0
        )

    def test_fails_outside_tolerance(self) -> None:
        write_artifact(self.reference, np.asarray([1.0], dtype=np.float32))
        write_artifact(self.candidate, np.asarray([1.1], dtype=np.float32))

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 1)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIn("tensor mismatch", result["reason"])

    def test_fails_when_run_configuration_differs(self) -> None:
        value = np.asarray([1.0], dtype=np.float32)
        write_artifact(self.reference, value)
        write_artifact(self.candidate, value, num_trunk_passes=10)

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 1)
        result = json.loads(self.result.read_text())
        self.assertEqual(
            result["metadata_mismatches"]["num_trunk_passes"]["candidate"], 10
        )

    def test_supports_normalized_tensor_checks(self) -> None:
        value = np.asarray([1000.0, -1000.0], dtype=np.float32)
        write_artifact(self.reference, value)
        write_artifact(self.candidate, value * 1.01)
        self.tolerances.write_text(
            json.dumps(
                {
                    "tensors": {
                        "single_embeddings": {
                            "max_normalized_rmse": 0.02,
                            "min_cosine_similarity": 0.999,
                        }
                    }
                }
            )
        )

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.result.read_text())
        self.assertAlmostEqual(
            result["tensors"]["single_embeddings"]["normalized_rmse"], 0.01
        )

    def test_handles_matching_zero_tensors(self) -> None:
        value = np.zeros(2, dtype=np.float32)
        write_artifact(self.reference, value)
        write_artifact(self.candidate, value)

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        metrics = json.loads(self.result.read_text())["tensors"][
            "single_embeddings"
        ]
        self.assertEqual(metrics["normalized_rmse"], 0.0)
        self.assertEqual(metrics["cosine_similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
