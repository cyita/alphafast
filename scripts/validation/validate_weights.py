#!/usr/bin/env python3
"""Validate an official AF3 weight stream and a Torch mapping report.

The mapping report must contain source_sha256, records_total, records_mapped,
source_records_without_target, target_parameters_without_source,
shape_mismatches, dtype_mismatches, and value_mismatches. BF16 source weights
may be losslessly upcast to FP32 target parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import BinaryIO, Iterator, Optional


EXPECTED_RECORDS = 405
HEADER = struct.Struct("<5i")
ISSUE_FIELDS = {
    "source_records_without_target": "source record(s) have no target parameter",
    "target_parameters_without_source": "target parameter(s) have no source weight",
    "shape_mismatches": "shape mismatch(es)",
    "value_mismatches": "value mismatch(es)",
}
SAFE_DTYPE_UPCASTS = {("torch.bfloat16", "torch.float32")}


class WeightFormatError(Exception):
    """The source weight stream is unreadable or malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-file", type=Path, required=True)
    parser.add_argument("--mapping-report", type=Path, required=True)
    parser.add_argument(
        "--result-file",
        type=Path,
        default=Path("ae-results/latest_result.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def open_weight_stream(path: Path) -> Iterator[BinaryIO]:
    if not path.name.endswith(".zst"):
        with path.open("rb") as stream:
            yield stream
        return

    process = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        yield process.stdout
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code:
            raise WeightFormatError(stderr.strip() or "zstd decompression failed")
    except BaseException:
        process.kill()
        process.wait()
        raise


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise WeightFormatError(f"truncated record: expected {size} bytes")
    return data


def discard_exact(stream: BinaryIO, size: int) -> None:
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise WeightFormatError(
                f"truncated tensor payload: {remaining} bytes missing"
            )
        remaining -= len(chunk)


def inspect_records(path: Path) -> tuple[int, int, list[str]]:
    keys: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    record_count = 0
    tensor_bytes = 0

    with open_weight_stream(path) as stream:
        while True:
            header = stream.read(HEADER.size)
            if not header:
                break
            if len(header) != HEADER.size:
                raise WeightFormatError("truncated record header")

            scope_len, name_len, dtype_len, shape_len, payload_len = (
                HEADER.unpack(header)
            )
            if min(scope_len, name_len, dtype_len, shape_len, payload_len) < 0:
                raise WeightFormatError("negative record length")

            scope = read_exact(stream, scope_len).decode("utf-8")
            name = read_exact(stream, name_len).decode("utf-8")
            read_exact(stream, dtype_len).decode("utf-8")
            shape = struct.unpack(
                f"<{shape_len}i", read_exact(stream, shape_len * 4)
            )
            if any(dimension < 0 for dimension in shape):
                raise WeightFormatError(f"negative tensor dimension: {scope}/{name}")

            discard_exact(stream, payload_len)
            key = (scope, name)
            if key in keys:
                duplicates.append(f"{scope}/{name}")
            keys.add(key)
            record_count += 1
            tensor_bytes += payload_len

    return record_count, tensor_bytes, duplicates


def load_mapping_report(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("mapping report must be a JSON object")
    return report


def validate_mapping(
    report: dict[str, object], source_sha256: str, source_records: int
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []

    report_total = report.get("records_total")
    report_mapped = report.get("records_mapped")
    if type(report_total) is not int:
        failures.append("records_total must be an integer")
    elif report_total != source_records:
        failures.append(
            f"records_total={report_total}, source contains {source_records}"
        )

    if type(report_mapped) is not int:
        failures.append("records_mapped must be an integer")
    elif report_mapped != source_records:
        failures.append(
            f"records_mapped={report_mapped}, expected {source_records}"
        )

    if report.get("source_sha256") != source_sha256:
        failures.append("source_sha256 does not match the weight file")

    issue_counts: dict[str, Optional[int]] = {}
    for field, description in ISSUE_FIELDS.items():
        issues = report.get(field)
        if not isinstance(issues, list):
            failures.append(f"{field} must be a list")
            issue_counts[field] = None
        else:
            issue_counts[field] = len(issues)
            if issues:
                failures.append(f"{len(issues)} {description}")

    dtype_mismatches = report.get("dtype_mismatches")
    if not isinstance(dtype_mismatches, list):
        failures.append("dtype_mismatches must be a list")
        issue_counts["dtype_mismatches"] = None
        safe_dtype_upcasts: Optional[int] = None
        unsafe_dtype_mismatches: Optional[int] = None
    else:
        safe_dtype_upcasts = sum(
            isinstance(issue, dict)
            and (issue.get("source_dtype"), issue.get("target_dtype"))
            in SAFE_DTYPE_UPCASTS
            for issue in dtype_mismatches
        )
        unsafe_dtype_mismatches = len(dtype_mismatches) - safe_dtype_upcasts
        issue_counts["dtype_mismatches"] = len(dtype_mismatches)
        if unsafe_dtype_mismatches:
            failures.append(
                f"{unsafe_dtype_mismatches} unsafe dtype mismatch(es)"
            )

    metrics: dict[str, object] = {
        "source_records": source_records,
        "mapped_records": report_mapped,
        **{f"{field}_count": count for field, count in issue_counts.items()},
        "safe_dtype_upcasts_count": safe_dtype_upcasts,
        "unsafe_dtype_mismatches_count": unsafe_dtype_mismatches,
    }
    return failures, metrics


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    source: dict[str, object] = {"file": str(args.weights_file)}

    try:
        if not args.weights_file.is_file():
            raise WeightFormatError(f"weight file not found: {args.weights_file}")

        source_sha256 = sha256_file(args.weights_file)
        source_records, tensor_bytes, duplicates = inspect_records(
            args.weights_file
        )
        source.update(
            {
                "sha256": source_sha256,
                "records": source_records,
                "tensor_bytes": tensor_bytes,
            }
        )
    except (OSError, UnicodeDecodeError, struct.error, WeightFormatError) as error:
        result = {
            "stage": "weight",
            "status": "error",
            "source": source,
            "reason": str(error),
        }
        write_result(args.result_file, result)
        print(f"Weight validation error: {error}", file=sys.stderr)
        return 2

    failures: list[str] = []
    if source_records != EXPECTED_RECORDS:
        failures.append(
            f"source contains {source_records} records, expected {EXPECTED_RECORDS}"
        )
    if duplicates:
        failures.append(f"source contains {len(duplicates)} duplicate record(s)")

    try:
        mapping_report = load_mapping_report(args.mapping_report)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"invalid mapping report: {error}")
        metrics: dict[str, object] = {"source_records": source_records}
    else:
        mapping_failures, metrics = validate_mapping(
            mapping_report, source_sha256, source_records
        )
        failures.extend(mapping_failures)
    metrics["duplicate_source_records"] = len(duplicates)
    metrics["tensor_bytes"] = tensor_bytes

    status = "failed" if failures else "passed"
    safe_upcasts = metrics.get("safe_dtype_upcasts_count")
    passed_reason = "all weights mapped"
    if safe_upcasts:
        passed_reason += f"; {safe_upcasts} safe BF16-to-FP32 upcast(s)"
    result = {
        "stage": "weight",
        "status": status,
        "source": source,
        "mapping_report": str(args.mapping_report),
        "metrics": metrics,
        "reason": "; ".join(failures) if failures else passed_reason,
    }
    write_result(args.result_file, result)

    if failures:
        print(f"Weight validation failed: {result['reason']}", file=sys.stderr)
        return 1

    print(f"Weight validation passed: {source_records}/{source_records} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
