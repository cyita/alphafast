from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


VALIDATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "validation"
sys.path.insert(0, str(VALIDATION_DIR))
import validate_ranking  # noqa: E402


def write_ranked_artifact(path: Path) -> None:
    metadata = {
        "artifact_type": "alphafold3_parity_checkpoints",
        "format_version": 1,
        "model_seed": 101,
        "num_recycles": 0,
        "num_trunk_passes": 1,
        "num_samples": 1,
        "diffusion_steps": 200,
        "precision": "float32",
        "job_name": "test",
        "frozen_features_sha256": "features",
        "random_tape_sha256": "tape",
        "weights_sha256": "weights",
    }
    np.savez_compressed(
        path,
        confidence_ptm=np.asarray([0.9], dtype=np.float32),
        _metadata_json=np.asarray(json.dumps(metadata)),
    )


class ValidateRawRankingTest(unittest.TestCase):

    def test_postprocesses_both_raw_artifacts_before_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "ranking.json"
            tolerances = root / "tolerances.json"
            tolerances.write_text(
                json.dumps(
                    {"tensors": {"confidence_ptm": {"atol": 0, "rtol": 0}}}
                )
            )
            calls = []

            fake_module = types.ModuleType("postprocess_ranking")

            def postprocess_artifact(
                checkpoints: Path,
                frozen_features: Path,
                input_json: Path,
                output_file: Path,
            ) -> None:
                calls.append((checkpoints, frozen_features, input_json))
                write_ranked_artifact(output_file)

            fake_module.postprocess_artifact = postprocess_artifact
            arguments = [
                "validate_ranking.py",
                "--reference",
                str(root / "jax_raw.npz"),
                "--candidate",
                str(root / "torch_raw.npz"),
                "--frozen-features",
                str(root / "features.npz"),
                "--input-json",
                str(root / "input.json"),
                "--tolerances",
                str(tolerances),
                "--result-file",
                str(result_path),
            ]
            with mock.patch.dict(sys.modules, {"postprocess_ranking": fake_module}):
                with mock.patch.object(sys, "argv", arguments):
                    exit_code = validate_ranking.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue((root / "ranking_reference.npz").is_file())
            self.assertTrue((root / "ranking_candidate.npz").is_file())
            self.assertEqual(json.loads(result_path.read_text())["status"], "passed")


if __name__ == "__main__":
    unittest.main()
