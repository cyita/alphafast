#!/usr/bin/env python3
"""Compare Torch ground-truth metrics against a JAX accuracy report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument(
        "--tolerances",
        type=Path,
        default=Path(__file__).with_name("accuracy_delta_tolerances.json"),
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=Path("ae-results/latest_accuracy_delta.json"),
    )
    return parser.parse_args()


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def case_map(report: dict[str, object]) -> dict[str, dict[str, object]]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("accuracy report cases must be a list")
    result = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise ValueError("accuracy report contains an invalid case")
        result[case["name"]] = case
    return result


def top_metrics(case: dict[str, object]) -> dict[str, object]:
    samples = case.get("samples")
    top_sample = case.get("top_sample")
    if not isinstance(samples, list):
        raise ValueError(f"{case.get('name')}: samples must be a list")
    for sample in samples:
        if isinstance(sample, dict) and sample.get("sample") == top_sample:
            metrics = sample.get("metrics")
            if isinstance(metrics, dict):
                return metrics
    raise ValueError(f"{case.get('name')}: top sample metrics not found")


def numeric_metric(metrics: dict[str, object], name: str, case_name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{case_name}: metric {name} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{case_name}: metric {name} is not finite")
    return value


def main() -> int:
    args = parse_args()
    try:
        reference_report = load_json(args.reference_report)
        candidate_report = load_json(args.candidate_report)
        contract = load_json(args.tolerances)
        if contract.get("schema_version") != 1:
            raise ValueError("accuracy delta tolerances schema_version must be 1")
        case_rules = contract.get("cases")
        if not isinstance(case_rules, dict) or not case_rules:
            raise ValueError("accuracy delta tolerances cases must be an object")
        reference_cases = case_map(reference_report)
        candidate_cases = case_map(candidate_report)

        failures = []
        case_results = []
        for case_name, metric_rules in case_rules.items():
            if case_name not in reference_cases or case_name not in candidate_cases:
                raise ValueError(f"missing accuracy case: {case_name}")
            if not isinstance(metric_rules, dict) or not metric_rules:
                raise ValueError(f"{case_name}: metric rules must be an object")
            reference_case = reference_cases[case_name]
            candidate_case = candidate_cases[case_name]
            reference_metrics = top_metrics(reference_case)
            candidate_metrics = top_metrics(candidate_case)
            metric_results = {}
            case_failed = False
            for metric_name, rule in metric_rules.items():
                if not isinstance(rule, dict):
                    raise ValueError(f"{case_name}/{metric_name}: invalid rule")
                direction = rule.get("direction")
                max_regression = rule.get("max_regression")
                if direction not in ("lower", "higher") or not isinstance(
                    max_regression, (int, float)
                ):
                    raise ValueError(f"{case_name}/{metric_name}: invalid rule")
                reference_value = numeric_metric(
                    reference_metrics, metric_name, case_name
                )
                candidate_value = numeric_metric(
                    candidate_metrics, metric_name, case_name
                )
                delta = candidate_value - reference_value
                regression = delta if direction == "lower" else -delta
                passed = regression <= float(max_regression)
                metric_results[metric_name] = {
                    "reference": reference_value,
                    "candidate": candidate_value,
                    "delta": delta,
                    "regression": regression,
                    "max_regression": float(max_regression),
                    "direction": direction,
                    "passed": passed,
                }
                if not passed:
                    case_failed = True
                    failures.append(
                        f"{case_name}/{metric_name} regression "
                        f"{regression:.6g} > {float(max_regression):.6g}"
                    )
            case_results.append(
                {
                    "name": case_name,
                    "status": "failed" if case_failed else "passed",
                    "reference_top_sample": reference_case.get("top_sample"),
                    "candidate_top_sample": candidate_case.get("top_sample"),
                    "top_sample_changed": reference_case.get("top_sample")
                    != candidate_case.get("top_sample"),
                    "reference_best_sample": reference_case.get("best_sample"),
                    "candidate_best_sample": candidate_case.get("best_sample"),
                    "best_sample_changed": reference_case.get("best_sample")
                    != candidate_case.get("best_sample"),
                    "metrics": metric_results,
                }
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        result = {
            "stage": "accuracy_delta",
            "status": "error",
            "reason": str(error),
        }
        write_result(args.result_file, result)
        print(f"Accuracy delta validation error: {error}", file=sys.stderr)
        return 2

    result = {
        "stage": "accuracy_delta",
        "status": "failed" if failures else "passed",
        "reference_report": str(args.reference_report),
        "candidate_report": str(args.candidate_report),
        "cases": case_results,
        "reason": "; ".join(failures) if failures else "all accuracy deltas passed",
    }
    write_result(args.result_file, result)
    if failures:
        print(f"Accuracy delta validation failed: {result['reason']}", file=sys.stderr)
        return 1
    print(f"Accuracy delta validation passed: {len(case_results)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
