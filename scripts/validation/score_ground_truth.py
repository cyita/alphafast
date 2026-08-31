#!/usr/bin/env python3
"""Score fixed AlphaFold 3 predictions against external ground-truth structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shlex
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
PROFILES = {
    "smoke": {"nvidia_single_protein_1ubq"},
    "suite": {
        "nvidia_single_protein_1ubq",
        "nvidia_protein_complex_1brs",
        "nvidia_protein_ligand_3hs4",
    },
    "full": {
        "nvidia_single_protein_1ubq",
        "nvidia_protein_complex_1brs",
        "nvidia_protein_ligand_3hs4",
    },
}
BACKBONE_ATOMS = ("N", "CA", "C", "O")


class AccuracyError(ValueError):
    pass


@dataclass(frozen=True)
class Atom:
    group: str
    element: str
    name: str
    comp_id: str
    chain: str
    seq_id: str
    xyz: np.ndarray


@dataclass
class Residue:
    seq_id: str
    comp_id: str
    atoms: dict[str, Atom]


def atom_site_rows(path: pathlib.Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        header_index = index + 1
        headers: list[str] = []
        while header_index < len(lines) and lines[header_index].lstrip().startswith("_"):
            headers.append(lines[header_index].split()[0])
            header_index += 1
        if not headers or not headers[0].startswith("_atom_site."):
            continue

        names = [header.removeprefix("_atom_site.") for header in headers]
        rows: list[dict[str, str]] = []
        pending: list[str] = []
        for line in lines[header_index:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "#" or stripped == "loop_" or stripped.startswith(("_", "data_")):
                break
            pending.extend(shlex.split(line, comments=False, posix=True))
            while len(pending) >= len(names):
                values, pending = pending[: len(names)], pending[len(names) :]
                rows.append(dict(zip(names, values)))
        if pending:
            raise AccuracyError(f"{path}: incomplete atom_site row")
        return rows
    raise AccuracyError(f"{path}: no atom_site loop")


def parse_atoms(path: pathlib.Path) -> list[Atom]:
    atoms: list[Atom] = []
    for row in atom_site_rows(path):
        if row.get("pdbx_PDB_model_num", "1") != "1":
            continue
        if row.get("label_alt_id", ".") not in {".", "?", "A"}:
            continue
        seq_id = row.get("label_seq_id", ".")
        if seq_id in {".", "?"}:
            seq_id = row.get("auth_seq_id", ".")
        try:
            xyz = np.array(
                [float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])],
                dtype=np.float64,
            )
        except (KeyError, ValueError) as exc:
            raise AccuracyError(f"{path}: invalid atom coordinate") from exc
        atoms.append(
            Atom(
                group=row.get("group_PDB", ""),
                element=row.get("type_symbol", "").upper(),
                name=row.get("label_atom_id", ""),
                comp_id=row.get("label_comp_id", ""),
                chain=row.get("label_asym_id", ""),
                seq_id=seq_id,
                xyz=xyz,
            )
        )
    if not atoms:
        raise AccuracyError(f"{path}: no model-1 atoms")
    return atoms


def protein_residues(atoms: list[Atom], chain: str) -> list[Residue]:
    residues: dict[str, Residue] = {}
    for atom in atoms:
        if atom.group != "ATOM" or atom.chain != chain:
            continue
        residue = residues.setdefault(atom.seq_id, Residue(atom.seq_id, atom.comp_id, {}))
        residue.atoms.setdefault(atom.name, atom)
    return list(residues.values())


def matched_residues(
    predicted: list[Atom], reference: list[Atom], predicted_chain: str, reference_chain: str
) -> list[tuple[Residue, Residue]]:
    pred = {residue.seq_id: residue for residue in protein_residues(predicted, predicted_chain)}
    ref = {residue.seq_id: residue for residue in protein_residues(reference, reference_chain)}
    return [
        (pred[seq_id], residue)
        for seq_id, residue in ref.items()
        if seq_id in pred and pred[seq_id].comp_id == residue.comp_id
    ]


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(mobile) < 3 or mobile.shape != target.shape:
        raise AccuracyError("at least three matched coordinates are required")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mobile_center).T @ (target - target_center))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    return rotation, target_center - mobile_center @ rotation


def transformed(xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return xyz @ rotation + translation


def rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((mobile - target) ** 2, axis=1))))


def ca_lddt(predicted: np.ndarray, reference: np.ndarray) -> float:
    reference_distances = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
    predicted_distances = np.linalg.norm(predicted[:, None] - predicted[None, :], axis=-1)
    mask = (reference_distances > 0) & (reference_distances < 15.0)
    errors = np.abs(predicted_distances - reference_distances)[mask]
    return float(np.mean([(errors < cutoff).mean() for cutoff in (0.5, 1.0, 2.0, 4.0)]))


def protein_accuracy(
    predicted: list[Atom], reference: list[Atom], chain_mapping: dict[str, str]
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, dict[str, list[tuple[Residue, Residue]]]]:
    matches = {
        pred_chain: matched_residues(predicted, reference, pred_chain, ref_chain)
        for pred_chain, ref_chain in chain_mapping.items()
    }
    ca_pairs = [
        (pred_res.atoms["CA"].xyz, ref_res.atoms["CA"].xyz)
        for chain_matches in matches.values()
        for pred_res, ref_res in chain_matches
        if "CA" in pred_res.atoms and "CA" in ref_res.atoms
    ]
    if len(ca_pairs) < 3:
        raise AccuracyError("fewer than three matched C-alpha atoms")
    pred_ca = np.array([pair[0] for pair in ca_pairs])
    ref_ca = np.array([pair[1] for pair in ca_pairs])
    rotation, translation = kabsch(pred_ca, ref_ca)
    aligned_ca = transformed(pred_ca, rotation, translation)

    backbone_pairs = [
        (pred_res.atoms[name].xyz, ref_res.atoms[name].xyz)
        for chain_matches in matches.values()
        for pred_res, ref_res in chain_matches
        for name in BACKBONE_ATOMS
        if name in pred_res.atoms and name in ref_res.atoms
    ]
    pred_backbone = np.array([pair[0] for pair in backbone_pairs])
    ref_backbone = np.array([pair[1] for pair in backbone_pairs])
    target_residues = sum(
        len(protein_residues(predicted, predicted_chain))
        for predicted_chain in chain_mapping
    )
    reference_residues = sum(
        len(protein_residues(reference, reference_chain))
        for reference_chain in chain_mapping.values()
    )
    d0 = max(0.5, 1.24 * max(target_residues - 15, 1) ** (1.0 / 3.0) - 1.8)
    ca_distances = np.linalg.norm(aligned_ca - ref_ca, axis=1)
    metrics: dict[str, float | int] = {
        "matched_residues": len(ca_pairs),
        "target_residues": target_residues,
        "reference_residues": reference_residues,
        "target_coverage": len(ca_pairs) / target_residues,
        "resolved_reference_coverage": len(ca_pairs) / reference_residues,
        "ca_rmsd": rmsd(aligned_ca, ref_ca),
        "backbone_rmsd": rmsd(transformed(pred_backbone, rotation, translation), ref_backbone),
        "ca_tm_score": float(np.sum(1.0 / (1.0 + (ca_distances / d0) ** 2)) / target_residues),
        "ca_lddt": ca_lddt(pred_ca, ref_ca),
    }
    return metrics, rotation, translation, matches


def heavy_xyz(residue: Residue) -> np.ndarray:
    return np.array([atom.xyz for atom in residue.atoms.values() if atom.element != "H"])


def residue_contacts(
    left: list[Residue], right: list[Residue], cutoff: float = 5.0
) -> set[tuple[str, str]]:
    contacts: set[tuple[str, str]] = set()
    cutoff_squared = cutoff**2
    for left_residue in left:
        left_xyz = heavy_xyz(left_residue)
        for right_residue in right:
            right_xyz = heavy_xyz(right_residue)
            if np.any(np.sum((left_xyz[:, None] - right_xyz[None, :]) ** 2, axis=-1) < cutoff_squared):
                contacts.add((left_residue.seq_id, right_residue.seq_id))
    return contacts


def precision_recall_f1(predicted: set[Any], reference: set[Any]) -> tuple[float, float, float]:
    overlap = len(predicted & reference)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def complex_accuracy(
    predicted: list[Atom], reference: list[Atom], config: dict[str, Any]
) -> dict[str, float | int]:
    mapping = config["protein_chains"]
    metrics, _, _, matches = protein_accuracy(predicted, reference, mapping)
    receptor = config["receptor_chain"]
    partner = config["partner_chain"]
    _, rotation, translation, _ = protein_accuracy(
        predicted, reference, {receptor: mapping[receptor]}
    )
    partner_pairs = matches[partner]
    pred_partner = np.array([pred.atoms["CA"].xyz for pred, ref in partner_pairs if "CA" in pred.atoms and "CA" in ref.atoms])
    ref_partner = np.array([ref.atoms["CA"].xyz for pred, ref in partner_pairs if "CA" in pred.atoms and "CA" in ref.atoms])

    common_left = [pair[0] for pair in matches[receptor]]
    common_right = [pair[0] for pair in matches[partner]]
    reference_left = [pair[1] for pair in matches[receptor]]
    reference_right = [pair[1] for pair in matches[partner]]
    predicted_contacts = residue_contacts(common_left, common_right)
    reference_contacts = residue_contacts(reference_left, reference_right)
    precision, recall, f1 = precision_recall_f1(predicted_contacts, reference_contacts)

    metrics.update(
        {
            "partner_ca_rmsd_after_receptor_fit": rmsd(
                transformed(pred_partner, rotation, translation), ref_partner
            ),
            "reference_interface_contacts": len(reference_contacts),
            "predicted_interface_contacts": len(predicted_contacts),
            "interface_contact_precision": precision,
            "interface_contact_recall": recall,
            "interface_contact_f1": f1,
        }
    )
    return metrics


def component_atoms(atoms: list[Atom], chain: str, comp_id: str) -> dict[str, Atom]:
    return {
        atom.name: atom
        for atom in atoms
        if atom.chain == chain and atom.comp_id == comp_id and atom.element != "H"
    }


def pocket_residues(residues: list[Residue], ligand_xyz: np.ndarray, cutoff: float = 5.0) -> set[str]:
    pocket: set[str] = set()
    for residue in residues:
        distances = np.linalg.norm(heavy_xyz(residue)[:, None] - ligand_xyz[None, :], axis=-1)
        if np.any(distances < cutoff):
            pocket.add(residue.seq_id)
    return pocket


def ligand_accuracy(
    predicted: list[Atom], reference: list[Atom], config: dict[str, Any]
) -> dict[str, float | int]:
    mapping = config["protein_chains"]
    metrics, rotation, translation, matches = protein_accuracy(predicted, reference, mapping)
    ligand = config["ligand"]
    pred_ligand = component_atoms(predicted, ligand["predicted_chain"], ligand["comp_id"])
    ref_ligand = component_atoms(reference, ligand["reference_chain"], ligand["comp_id"])
    common_names = sorted(pred_ligand.keys() & ref_ligand.keys())
    if len(common_names) < 3:
        raise AccuracyError("fewer than three matched ligand atoms")
    pred_ligand_raw = np.array([pred_ligand[name].xyz for name in common_names])
    pred_ligand_xyz = transformed(pred_ligand_raw, rotation, translation)
    ref_ligand_xyz = np.array([ref_ligand[name].xyz for name in common_names])

    ion = config["ion"]
    pred_ion = list(component_atoms(predicted, ion["predicted_chain"], ion["comp_id"]).values())
    ref_ion = list(component_atoms(reference, ion["reference_chain"], ion["comp_id"]).values())
    if len(pred_ion) != 1 or len(ref_ion) != 1:
        raise AccuracyError("expected one matched ion atom")
    pred_ion_xyz = transformed(pred_ion[0].xyz[None, :], rotation, translation)[0]
    ref_ion_xyz = ref_ion[0].xyz

    protein_chain = next(iter(mapping))
    pred_residues = [pair[0] for pair in matches[protein_chain]]
    ref_residues = [pair[1] for pair in matches[protein_chain]]
    predicted_pocket = pocket_residues(pred_residues, pred_ligand_raw)
    reference_pocket = pocket_residues(ref_residues, ref_ligand_xyz)
    precision, recall, f1 = precision_recall_f1(predicted_pocket, reference_pocket)
    pred_ion_ligand_distance = float(np.min(np.linalg.norm(pred_ligand_xyz - pred_ion_xyz, axis=1)))
    ref_ion_ligand_distance = float(np.min(np.linalg.norm(ref_ligand_xyz - ref_ion_xyz, axis=1)))
    metrics.update(
        {
            "matched_ligand_atoms": len(common_names),
            "ligand_rmsd": rmsd(pred_ligand_xyz, ref_ligand_xyz),
            "ion_distance": float(np.linalg.norm(pred_ion_xyz - ref_ion_xyz)),
            "ion_ligand_distance_error": abs(pred_ion_ligand_distance - ref_ion_ligand_distance),
            "reference_pocket_residues": len(reference_pocket),
            "predicted_pocket_residues": len(predicted_pocket),
            "pocket_precision": precision,
            "pocket_recall": recall,
            "pocket_f1": f1,
        }
    )
    return metrics


def locate_job(output_root: pathlib.Path, name: str) -> pathlib.Path:
    matches = list(output_root.rglob(f"{name}_ranking_scores.csv"))
    if len(matches) != 1:
        raise AccuracyError(f"{output_root}: expected one output for {name}, found {len(matches)}")
    return matches[0].parent


def score_case(
    output_root: pathlib.Path, case: dict[str, Any], ground_truth_root: pathlib.Path
) -> dict[str, Any]:
    name = case["name"]
    job_dir = locate_job(output_root, name)
    truth_path = ground_truth_root / case["ground_truth"]
    reference = parse_atoms(truth_path)
    with (job_dir / f"{name}_ranking_scores.csv").open(newline="", encoding="utf-8") as handle:
        ranking_rows = list(csv.DictReader(handle))

    samples: list[dict[str, Any]] = []
    for row in ranking_rows:
        seed, sample = int(row["seed"]), int(row["sample"])
        sample_dir = job_dir / f"seed-{seed}_sample-{sample}"
        prefix = f"{name}_seed-{seed}_sample-{sample}"
        predicted = parse_atoms(sample_dir / f"{prefix}_model.cif")
        config = case["accuracy"]
        if case["kind"] == "single_protein":
            metrics, _, _, _ = protein_accuracy(predicted, reference, config["protein_chains"])
        elif case["kind"] == "protein_complex":
            metrics = complex_accuracy(predicted, reference, config)
        else:
            metrics = ligand_accuracy(predicted, reference, config)
        summary = json.loads(
            (sample_dir / f"{prefix}_summary_confidences.json").read_text(encoding="utf-8")
        )
        samples.append(
            {
                "seed": seed,
                "sample": sample,
                "ranking_score": float(row["ranking_score"]),
                "ptm": summary.get("ptm"),
                "iptm": summary.get("iptm"),
                "metrics": metrics,
            }
        )

    top = max(samples, key=lambda item: item["ranking_score"])
    if case["kind"] == "single_protein":
        best = max(samples, key=lambda item: item["metrics"]["ca_tm_score"])
    elif case["kind"] == "protein_complex":
        best = max(samples, key=lambda item: item["metrics"]["interface_contact_f1"])
    else:
        best = min(samples, key=lambda item: item["metrics"]["ligand_rmsd"])
    return {
        "name": name,
        "kind": case["kind"],
        "ground_truth": str(truth_path),
        "top_sample": top["sample"],
        "best_sample": best["sample"],
        "samples": samples,
    }


def evaluate_thresholds(
    metrics: dict[str, float | int], rules: dict[str, dict[str, float]]
) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []
    for metric, limits in rules.items():
        value = float(metrics[metric])
        if "fail_min" in limits and value < limits["fail_min"]:
            failures.append(f"{metric}={value:.3f} < {limits['fail_min']:.3f}")
        elif "fail_max" in limits and value > limits["fail_max"]:
            failures.append(f"{metric}={value:.3f} > {limits['fail_max']:.3f}")
        elif "warn_min" in limits and value < limits["warn_min"]:
            warnings.append(f"{metric}={value:.3f} < {limits['warn_min']:.3f}")
        elif "warn_max" in limits and value > limits["warn_max"]:
            warnings.append(f"{metric}={value:.3f} > {limits['warn_max']:.3f}")
    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    return {"status": status, "warnings": warnings, "failures": failures}


def apply_thresholds(report: dict[str, Any], thresholds: dict[str, Any]) -> None:
    statuses = []
    for case in report["cases"]:
        top = next(sample for sample in case["samples"] if sample["sample"] == case["top_sample"])
        case["validation"] = evaluate_thresholds(
            top["metrics"], thresholds["cases"][case["name"]]
        )
        statuses.append(case["validation"]["status"])
    report["status"] = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"


def print_report(report: dict[str, Any]) -> None:
    for case in report["cases"]:
        top = next(sample for sample in case["samples"] if sample["sample"] == case["top_sample"])
        metrics = top["metrics"]
        if case["kind"] == "single_protein":
            detail = (
                f"CA-RMSD={metrics['ca_rmsd']:.3f}A "
                f"TM={metrics['ca_tm_score']:.3f} lDDT={metrics['ca_lddt']:.3f}"
            )
        elif case["kind"] == "protein_complex":
            detail = (
                f"partner-RMSD={metrics['partner_ca_rmsd_after_receptor_fit']:.3f}A "
                f"contact-F1={metrics['interface_contact_f1']:.3f}"
            )
        else:
            detail = (
                f"ligand-RMSD={metrics['ligand_rmsd']:.3f}A "
                f"Zn-error={metrics['ion_distance']:.3f}A pocket-F1={metrics['pocket_f1']:.3f}"
            )
        print(
            f"{case['name']}: {case['validation']['status']} "
            f"top={case['top_sample']} best={case['best_sample']} "
            f"target-coverage={metrics['target_coverage']:.3f} {detail}"
        )
        for issue in case["validation"]["warnings"] + case["validation"]["failures"]:
            print(f"  {issue}")
    print(f"overall: {report['status']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--ground-truth-root", type=pathlib.Path, required=True)
    parser.add_argument("--json-out", type=pathlib.Path)
    parser.add_argument(
        "--thresholds",
        type=pathlib.Path,
        default=ROOT / "ground_truth_accuracy_thresholds.json",
    )
    parser.add_argument("--gate", action="store_true", help="exit 1 when a top-1 case fails")
    args = parser.parse_args(argv)

    manifest = json.loads(
        (ROOT / "ground_truth_cases.json").read_text(encoding="utf-8")
    )
    wanted = PROFILES[args.profile]
    report = {
        "schema_version": 1,
        "profile": args.profile,
        "output_root": str(args.output_root.resolve()),
        "cases": [
            score_case(args.output_root, case, args.ground_truth_root)
            for case in manifest["cases"]
            if case["name"] in wanted
        ],
    }
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    apply_thresholds(report, thresholds)
    report["thresholds"] = str(args.thresholds.resolve())
    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if args.gate and report["status"] == "FAIL" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AccuracyError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
