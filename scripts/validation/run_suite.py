#!/usr/bin/env python3
"""Run a validation manifest and write one aggregate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


VALIDATOR = Path(__file__).resolve().parents[1] / "validate_torch.sh"
REQUIRED_FIELDS = {
    "weight": ("weights_file", "mapping_report"),
    "e2e": ("reference", "candidate", "tolerances"),
    "ranking": ("reference", "candidate"),
    "accuracy": ("tool", "output_root"),
    "accuracy_delta": ("reference_report", "candidate_report", "tolerances"),
}
PATH_OPTIONS = {
    "weights_file": "--weights-file",
    "mapping_report": "--mapping-report",
    "reference": "--reference",
    "candidate": "--candidate",
    "tolerances": "--tolerances",
    "frozen_features": "--frozen-features",
    "input_json": "--input-json",
    "tool": "--tool",
    "output_root": "--output-root",
    "ground_truth_root": "--ground-truth-root",
    "thresholds": "--thresholds",
    "reference_report": "--reference-report",
    "candidate_report": "--candidate-report",
}
STAGE_FIELDS = {
    "weight": ("weights_file", "mapping_report"),
    "e2e": ("reference", "candidate", "tolerances"),
    "ranking": (
        "reference",
        "candidate",
        "frozen_features",
        "input_json",
        "tolerances",
    ),
    "accuracy": ("tool", "output_root", "ground_truth_root", "thresholds"),
    "accuracy_delta": ("reference_report", "candidate_report", "tolerances"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--result-file", type=Path, default=Path("ae-results/suite_result.json")
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve_path(value: object, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else base / path


def build_command(
    check: dict[str, object], manifest_dir: Path, result_file: Path
) -> tuple[str, str, list[str]]:
    name = check.get("name")
    stage = check.get("stage")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if stage not in REQUIRED_FIELDS:
        raise ValueError(f"unsupported stage: {stage}")
    assert isinstance(stage, str)
    missing = [field for field in REQUIRED_FIELDS[stage] if field not in check]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")

    command = ["bash", str(VALIDATOR), stage]
    for field in STAGE_FIELDS[stage]:
        if field in check:
            command.extend(
                (
                    PATH_OPTIONS[field],
                    str(resolve_path(check[field], manifest_dir, field)),
                )
            )
    if stage == "accuracy" and "profile" in check:
        profile = check["profile"]
        if not isinstance(profile, str) or not profile:
            raise ValueError("profile must be a non-empty string")
        command.extend(("--profile", profile))
    command.extend(("--result-file", str(result_file)))
    return name, stage, command


def compact_result(stage: str, result: dict[str, object]) -> dict[str, object]:
    if stage == "accuracy_delta":
        cases = result.get("cases", [])
        return {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "cases": [
                {
                    "name": case.get("name"),
                    "status": case.get("status"),
                    "top_sample_changed": case.get("top_sample_changed"),
                    "best_sample_changed": case.get("best_sample_changed"),
                }
                for case in cases
                if isinstance(case, dict)
            ],
        }
    if stage != "accuracy":
        return {
            key: result[key]
            for key in ("status", "reason")
            if key in result
        }
    cases = result.get("cases", [])
    return {
        "status": result.get("status"),
        "cases": [
            {
                "name": case.get("name"),
                "status": case.get("validation", {}).get("status"),
                "top_sample": case.get("top_sample"),
                "best_sample": case.get("best_sample"),
            }
            for case in cases
            if isinstance(case, dict) and isinstance(case.get("validation"), dict)
        ],
    }


def read_check_result(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("check result must be a JSON object")
    return value


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    try:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        if manifest.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        checks = manifest.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError("checks must be a non-empty list")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {
            "schema_version": 1,
            "status": "error",
            "manifest": str(manifest_path),
            "reason": str(error),
        }
        write_json(args.result_file, result)
        print(f"Suite validation error: {error}", file=sys.stderr)
        return 2

    check_dir = args.result_file.parent / f"{args.result_file.stem}_checks"
    records: list[dict[str, object]] = []
    for index, check in enumerate(checks):
        result_file = check_dir / f"{index:03d}.json"
        try:
            if not isinstance(check, dict):
                raise ValueError("check must be a JSON object")
            if "result_file" in check:
                result_file = resolve_path(
                    check["result_file"], manifest_path.parent, "result_file"
                )
            name, stage, command = build_command(
                check, manifest_path.parent, result_file
            )
        except ValueError as error:
            record = {
                "name": check.get("name", f"check-{index}")
                if isinstance(check, dict)
                else f"check-{index}",
                "stage": check.get("stage") if isinstance(check, dict) else None,
                "status": "error",
                "exit_code": 2,
                "summary": {"reason": str(error)},
            }
            records.append(record)
            print(f"[ERROR] {record['name']}: {error}")
            continue

        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        status = (
            "passed"
            if completed.returncode == 0
            else "failed"
            if completed.returncode == 1
            else "error"
        )
        try:
            detail = read_check_result(result_file)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            detail = {"reason": f"invalid check result: {error}"}
            status = "error"
        record = {
            "name": name,
            "stage": stage,
            "status": status,
            "exit_code": completed.returncode if status != "error" else 2,
            "result_file": str(result_file),
            "summary": compact_result(stage, detail),
        }
        if status != "passed":
            output = completed.stderr.strip() or completed.stdout.strip()
            if output:
                record["message"] = output.splitlines()[-1]
        records.append(record)
        print(f"[{status.upper()}] {name} ({stage})")

    counts = {
        state: sum(r["status"] == state for r in records)
        for state in ("passed", "failed", "error")
    }
    status = (
        "error" if counts["error"] else "failed" if counts["failed"] else "passed"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "manifest": str(manifest_path),
        "counts": {"total": len(records), **counts},
        "checks": records,
    }
    write_json(args.result_file, report)
    print(
        f"Suite {status}: {counts['passed']} passed, "
        f"{counts['failed']} failed, {counts['error']} error"
    )
    return 2 if counts["error"] else (1 if counts["failed"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
