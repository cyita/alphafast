from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Optional
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_torch.sh"
RECORD_COUNT = 405


def write_weights(path: Path) -> None:
    with path.open("wb") as stream:
        for index in range(RECORD_COUNT):
            scope = f"scope_{index}".encode()
            name = b"weights"
            dtype = b"float32"
            payload = struct.pack("<f", float(index))
            stream.write(
                struct.pack(
                    "<5i", len(scope), len(name), len(dtype), 1, len(payload)
                )
            )
            stream.write(scope)
            stream.write(name)
            stream.write(dtype)
            stream.write(struct.pack("<i", 1))
            stream.write(payload)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mapping_report(path: Path, weights: Path, **updates: object) -> None:
    report: dict[str, object] = {
        "source_sha256": sha256(weights),
        "records_total": RECORD_COUNT,
        "records_mapped": RECORD_COUNT,
        "source_records_without_target": [],
        "target_parameters_without_source": [],
        "shape_mismatches": [],
        "dtype_mismatches": [],
        "value_mismatches": [],
    }
    report.update(updates)
    path.write_text(json.dumps(report))


class ValidateWeightsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.weights = self.root / "af3.bin"
        self.mapping_report = self.root / "mapping.json"
        self.result = self.root / "result.json"
        write_weights(self.weights)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_validator(
        self, weights: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        environment["ALPHAFAST_VALIDATION_PYTHON"] = sys.executable
        return subprocess.run(
            [
                "bash",
                str(VALIDATE_SCRIPT),
                "weight",
                "--weights-file",
                str(weights or self.weights),
                "--mapping-report",
                str(self.mapping_report),
                "--result-file",
                str(self.result),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_passes_for_complete_mapping(self) -> None:
        write_mapping_report(self.mapping_report, self.weights)

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["metrics"]["source_records"], RECORD_COUNT)

    def test_fails_for_unmapped_parameters(self) -> None:
        write_mapping_report(
            self.mapping_report,
            self.weights,
            target_parameters_without_source=["torch_model.unmapped_weight"],
        )

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 1)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["metrics"]["target_parameters_without_source_count"], 1
        )
        self.assertEqual(
            result["reason"], "1 target parameter(s) have no source weight"
        )

    def test_allows_bfloat16_source_upcast_to_float32(self) -> None:
        write_mapping_report(
            self.mapping_report,
            self.weights,
            dtype_mismatches=[{
                "source": "scope/weights",
                "source_dtype": "torch.bfloat16",
                "target_dtype": "torch.float32",
            }],
        )

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["metrics"]["dtype_mismatches_count"], 1)
        self.assertEqual(result["metrics"]["safe_dtype_upcasts_count"], 1)
        self.assertEqual(result["metrics"]["unsafe_dtype_mismatches_count"], 0)

    def test_fails_for_unsafe_dtype_mismatch(self) -> None:
        write_mapping_report(
            self.mapping_report,
            self.weights,
            dtype_mismatches=[{
                "source": "scope/weights",
                "source_dtype": "torch.float32",
                "target_dtype": "torch.bfloat16",
            }],
        )

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 1)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["safe_dtype_upcasts_count"], 0)
        self.assertEqual(result["metrics"]["unsafe_dtype_mismatches_count"], 1)
        self.assertEqual(result["reason"], "1 unsafe dtype mismatch(es)")

    def test_reports_invalid_weight_stream_as_error(self) -> None:
        self.weights.write_bytes(b"not a weight record")
        write_mapping_report(self.mapping_report, self.weights)

        completed = self.run_validator()

        self.assertEqual(completed.returncode, 2)
        result = json.loads(self.result.read_text())
        self.assertEqual(result["status"], "error")

    @unittest.skipUnless(shutil.which("zstd"), "zstd executable not installed")
    def test_reads_zstd_compressed_weights(self) -> None:
        compressed = self.root / "af3.bin.zst"
        subprocess.run(
            ["zstd", "-q", "-f", str(self.weights), "-o", str(compressed)],
            check=True,
        )
        write_mapping_report(self.mapping_report, compressed)

        completed = self.run_validator(compressed)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(self.result.read_text())["status"], "passed")

    def test_unknown_stage_is_an_execution_error(self) -> None:
        completed = subprocess.run(
            ["bash", str(VALIDATE_SCRIPT), "module"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("not implemented", completed.stderr)


if __name__ == "__main__":
    unittest.main()
