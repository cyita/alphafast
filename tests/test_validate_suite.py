from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_torch.sh"


def write_artifact(path: Path, tensor_name: str, value: np.ndarray) -> None:
    metadata = {
        "artifact_type": "alphafold3_parity_checkpoints",
        "format_version": 1,
        "model_seed": 101,
        "num_recycles": 0,
        "num_trunk_passes": 1,
        "num_samples": 5,
        "diffusion_steps": 200,
        "precision": "float32",
        "job_name": "test",
        "frozen_features_sha256": "features",
        "random_tape_sha256": "tape",
        "weights_sha256": "weights",
    }
    np.savez_compressed(
        path,
        **{tensor_name: value, "_metadata_json": np.asarray(json.dumps(metadata))},
    )


def write_accuracy_tool(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            import argparse
            import json

            parser = argparse.ArgumentParser()
            parser.add_argument("output_root")
            parser.add_argument("--profile")
            parser.add_argument("--json-out")
            parser.add_argument("--ground-truth-root")
            parser.add_argument("--thresholds")
            parser.add_argument("--gate", action="store_true")
            args = parser.parse_args()
            status = "FAIL" if args.output_root.endswith("fail") else "PASS"
            is_torch = args.output_root.endswith("torch_predictions")
            report = {
                "status": status,
                "cases": [{
                    "name": "1UBQ",
                    "top_sample": 3 if is_torch else 2,
                    "best_sample": 2,
                    "validation": {"status": status},
                    "samples": [{
                        "sample": 3 if is_torch else 2,
                        "metrics": {"ca_rmsd": 1.1 if is_torch else 1.0},
                    }],
                }],
            }
            with open(args.json_out, "w") as stream:
                json.dump(report, stream)
            raise SystemExit(1 if args.gate and status == "FAIL" else 0)
            """
        )
    )


class ValidateSuiteTest(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.environment = os.environ.copy()
        self.environment["ALPHAFAST_VALIDATION_PYTHON"] = sys.executable

        value = np.asarray([1.0, 2.0], dtype=np.float32)
        write_artifact(self.root / "jax_e2e.npz", "single_embeddings", value)
        write_artifact(self.root / "torch_e2e.npz", "single_embeddings", value)
        write_artifact(self.root / "jax_rank.npz", "confidence_ptm", value)
        write_artifact(self.root / "torch_rank.npz", "confidence_ptm", value)
        (self.root / "e2e_tolerances.json").write_text(
            json.dumps({"tensors": {"single_embeddings": {"atol": 0, "rtol": 0}}})
        )
        (self.root / "ranking_tolerances.json").write_text(
            json.dumps({"tensors": {"confidence_ptm": {"atol": 0, "rtol": 0}}})
        )
        (self.root / "accuracy_delta_tolerances.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": {
                        "1UBQ": {
                            "ca_rmsd": {
                                "direction": "lower",
                                "max_regression": 0.2,
                            }
                        }
                    },
                }
            )
        )
        write_accuracy_tool(self.root / "score_accuracy.py")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_stage(self, stage: str, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(VALIDATE_SCRIPT), stage, *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_suite_runs_e2e_ranking_and_accuracy(self) -> None:
        manifest = {
            "schema_version": 1,
            "checks": [
                {
                    "name": "module-parity",
                    "stage": "e2e",
                    "reference": "jax_e2e.npz",
                    "candidate": "torch_e2e.npz",
                    "tolerances": "e2e_tolerances.json",
                },
                {
                    "name": "ranking-parity",
                    "stage": "ranking",
                    "reference": "jax_rank.npz",
                    "candidate": "torch_rank.npz",
                    "tolerances": "ranking_tolerances.json",
                },
                {
                    "name": "jax-structure-accuracy",
                    "stage": "accuracy",
                    "tool": "score_accuracy.py",
                    "output_root": "jax_predictions",
                    "profile": "full",
                    "result_file": "jax_accuracy.json",
                },
                {
                    "name": "torch-structure-accuracy",
                    "stage": "accuracy",
                    "tool": "score_accuracy.py",
                    "output_root": "torch_predictions",
                    "profile": "full",
                    "result_file": "torch_accuracy.json",
                },
                {
                    "name": "accuracy-delta",
                    "stage": "accuracy_delta",
                    "reference_report": "jax_accuracy.json",
                    "candidate_report": "torch_accuracy.json",
                    "tolerances": "accuracy_delta_tolerances.json",
                },
            ],
        }
        manifest_path = self.root / "suite.json"
        result_path = self.root / "suite_result.json"
        manifest_path.write_text(json.dumps(manifest))

        completed = self.run_stage(
            "suite",
            "--manifest",
            str(manifest_path),
            "--result-file",
            str(result_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(result_path.read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["counts"], {"total": 5, "passed": 5, "failed": 0, "error": 0}
        )
        self.assertEqual(
            report["checks"][2]["summary"]["cases"][0],
            {
                "name": "1UBQ",
                "status": "PASS",
                "top_sample": 2,
                "best_sample": 2,
            },
        )
        self.assertTrue(
            report["checks"][4]["summary"]["cases"][0]["top_sample_changed"]
        )
        for check in report["checks"]:
            self.assertTrue(Path(check["stdout_file"]).is_file())
            self.assertTrue(Path(check["stderr_file"]).is_file())

    def test_suite_continues_after_a_failed_check(self) -> None:
        write_artifact(
            self.root / "torch_e2e.npz",
            "single_embeddings",
            np.asarray([9.0, 9.0], dtype=np.float32),
        )
        manifest = {
            "schema_version": 1,
            "checks": [
                {
                    "name": "failing-e2e",
                    "stage": "e2e",
                    "reference": "jax_e2e.npz",
                    "candidate": "torch_e2e.npz",
                    "tolerances": "e2e_tolerances.json",
                },
                {
                    "name": "ranking-still-runs",
                    "stage": "ranking",
                    "reference": "jax_rank.npz",
                    "candidate": "torch_rank.npz",
                    "tolerances": "ranking_tolerances.json",
                },
            ],
        }
        manifest_path = self.root / "suite.json"
        result_path = self.root / "suite_result.json"
        manifest_path.write_text(json.dumps(manifest))

        completed = self.run_stage(
            "suite",
            "--manifest",
            str(manifest_path),
            "--result-file",
            str(result_path),
        )

        self.assertEqual(completed.returncode, 1)
        report = json.loads(result_path.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["failed"], 1)
        self.assertEqual(report["counts"]["passed"], 1)
        self.assertEqual(report["checks"][1]["status"], "passed")

    def test_accuracy_stage_propagates_gate_failure(self) -> None:
        result_path = self.root / "accuracy.json"

        completed = self.run_stage(
            "accuracy",
            "--tool",
            str(self.root / "score_accuracy.py"),
            "--output-root",
            str(self.root / "fail"),
            "--result-file",
            str(result_path),
        )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(json.loads(result_path.read_text())["status"], "FAIL")

    def test_accuracy_delta_fails_on_excessive_regression(self) -> None:
        reference = self.root / "reference_accuracy.json"
        candidate = self.root / "candidate_accuracy.json"
        result = self.root / "delta_result.json"
        report = {
            "status": "PASS",
            "cases": [
                {
                    "name": "1UBQ",
                    "top_sample": 0,
                    "best_sample": 0,
                    "samples": [{"sample": 0, "metrics": {"ca_rmsd": 1.0}}],
                }
            ],
        }
        reference.write_text(json.dumps(report))
        report["cases"][0]["samples"][0]["metrics"]["ca_rmsd"] = 1.3
        candidate.write_text(json.dumps(report))

        completed = self.run_stage(
            "accuracy_delta",
            "--reference-report",
            str(reference),
            "--candidate-report",
            str(candidate),
            "--tolerances",
            str(self.root / "accuracy_delta_tolerances.json"),
            "--result-file",
            str(result),
        )

        self.assertEqual(completed.returncode, 1)
        detail = json.loads(result.read_text())
        self.assertEqual(detail["status"], "failed")
        self.assertIn("ca_rmsd", detail["reason"])


if __name__ == "__main__":
    unittest.main()
