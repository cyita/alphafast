from __future__ import annotations

import csv
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np


VALIDATION_ROOT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validation"
SPEC = importlib.util.spec_from_file_location(
    "score_ground_truth", VALIDATION_ROOT / "score_ground_truth.py"
)
assert SPEC is not None and SPEC.loader is not None
accuracy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = accuracy
SPEC.loader.exec_module(accuracy)


def protein(chain: str, origins: list[tuple[float, float, float]]) -> list[accuracy.Atom]:
    names = ("ALA", "GLY", "SER", "VAL", "LEU")
    offsets = {
        "N": np.array([-0.4, 0.0, 0.0]),
        "CA": np.array([0.0, 0.0, 0.0]),
        "C": np.array([0.4, 0.1, 0.0]),
        "O": np.array([0.6, 0.2, 0.0]),
        "CB": np.array([0.0, 0.5, 0.2]),
    }
    atoms = []
    for index, origin in enumerate(origins, 1):
        for atom_name, offset in offsets.items():
            atoms.append(
                accuracy.Atom(
                    "ATOM",
                    atom_name[0],
                    atom_name,
                    names[index - 1],
                    chain,
                    str(index),
                    np.array(origin) + offset,
                )
            )
    return atoms


def rigid_transform(atoms: list[accuracy.Atom]) -> list[accuracy.Atom]:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([8.0, -3.0, 2.0])
    return [
        accuracy.Atom(
            atom.group,
            atom.element,
            atom.name,
            atom.comp_id,
            atom.chain,
            atom.seq_id,
            atom.xyz @ rotation + translation,
        )
        for atom in atoms
    ]


def write_cif(path: pathlib.Path, atoms: list[accuracy.Atom]) -> None:
    headers = (
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "auth_seq_id",
        "auth_asym_id",
        "pdbx_PDB_model_num",
    )
    lines = ["data_test", "#", "loop_", *[f"_atom_site.{header}" for header in headers]]
    for index, atom in enumerate(atoms, 1):
        label_seq_id = atom.seq_id if atom.group == "ATOM" else "."
        lines.append(
            f"{atom.group} {index} {atom.element} {atom.name} . {atom.comp_id} "
            f"{atom.chain} 1 {label_seq_id} ? {atom.xyz[0]:.6f} {atom.xyz[1]:.6f} "
            f"{atom.xyz[2]:.6f} 1.0 0.0 {atom.seq_id} {atom.chain} 1"
        )
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


class AccuracyTest(unittest.TestCase):
    def test_protein_metrics_are_rigid_transform_invariant(self) -> None:
        reference = protein("A", [(0, 0, 0), (3, 0, 0), (3, 3, 0), (3, 3, 3)])
        predicted = rigid_transform(reference)
        metrics, _, _, _ = accuracy.protein_accuracy(predicted, reference, {"A": "A"})
        self.assertAlmostEqual(metrics["ca_rmsd"], 0.0, places=10)
        self.assertAlmostEqual(metrics["backbone_rmsd"], 0.0, places=10)
        self.assertAlmostEqual(metrics["ca_tm_score"], 1.0, places=10)
        self.assertAlmostEqual(metrics["ca_lddt"], 1.0, places=10)

    def test_missing_reference_residue_reduces_target_coverage(self) -> None:
        predicted = protein("A", [(0, 0, 0), (3, 0, 0), (3, 3, 0), (3, 3, 3)])
        reference = [atom for atom in predicted if atom.seq_id != "1"]
        metrics, _, _, _ = accuracy.protein_accuracy(predicted, reference, {"A": "A"})
        self.assertAlmostEqual(metrics["target_coverage"], 0.75)
        self.assertAlmostEqual(metrics["resolved_reference_coverage"], 1.0)

    def test_complex_contacts_use_explicit_chain_mapping(self) -> None:
        reference = protein("A", [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
        reference += protein("D", [(0, 4, 0), (3, 4, 0), (6, 4, 0)])
        predicted = rigid_transform(
            protein("A", [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
            + protein("B", [(0, 4, 0), (3, 4, 0), (6, 4, 0)])
        )
        metrics = accuracy.complex_accuracy(
            predicted,
            reference,
            {
                "protein_chains": {"A": "A", "B": "D"},
                "receptor_chain": "A",
                "partner_chain": "B",
            },
        )
        self.assertAlmostEqual(metrics["partner_ca_rmsd_after_receptor_fit"], 0.0, places=10)
        self.assertAlmostEqual(metrics["interface_contact_f1"], 1.0)

    def test_ligand_and_ion_follow_the_protein_fit(self) -> None:
        reference = protein("A", [(0, 0, 0), (3, 0, 0), (3, 3, 0), (3, 3, 3)])
        reference += [
            accuracy.Atom("HETATM", "C", "C1", "AZM", "D", "701", np.array([1.0, 2.0, 1.0])),
            accuracy.Atom("HETATM", "N", "N1", "AZM", "D", "701", np.array([2.0, 2.0, 1.0])),
            accuracy.Atom("HETATM", "S", "S1", "AZM", "D", "701", np.array([1.0, 3.0, 1.0])),
            accuracy.Atom("HETATM", "ZN", "ZN", "ZN", "B", "301", np.array([1.5, 2.5, 2.0])),
        ]
        predicted_source = [
            accuracy.Atom(
                atom.group,
                atom.element,
                atom.name,
                atom.comp_id,
                "B" if atom.chain == "D" else "C" if atom.chain == "B" else atom.chain,
                atom.seq_id,
                atom.xyz,
            )
            for atom in reference
        ]
        predicted = rigid_transform(predicted_source)
        metrics = accuracy.ligand_accuracy(
            predicted,
            reference,
            {
                "protein_chains": {"A": "A"},
                "ligand": {"predicted_chain": "B", "reference_chain": "D", "comp_id": "AZM"},
                "ion": {"predicted_chain": "C", "reference_chain": "B", "comp_id": "ZN"},
            },
        )
        self.assertAlmostEqual(metrics["ligand_rmsd"], 0.0, places=10)
        self.assertAlmostEqual(metrics["ion_distance"], 0.0, places=10)
        self.assertAlmostEqual(metrics["pocket_f1"], 1.0)

    def test_case_report_separates_top_ranked_and_best_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            truth_root = root / "truth"
            output_root = root / "output"
            job = output_root / "nvidia_single_protein_1ubq"
            truth_root.mkdir()
            job.mkdir(parents=True)
            reference = protein("A", [(0, 0, 0), (3, 0, 0), (3, 3, 0), (3, 3, 3)])
            write_cif(truth_root / "1ubq.cif", reference)
            with (job / "nvidia_single_protein_1ubq_ranking_scores.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["seed", "sample", "ranking_score"])
                writer.writerow([101, 0, 0.9])
                writer.writerow([101, 1, 0.8])
            for sample in (0, 1):
                sample_dir = job / f"seed-101_sample-{sample}"
                sample_dir.mkdir()
                prefix = f"nvidia_single_protein_1ubq_seed-101_sample-{sample}"
                atoms = rigid_transform(reference)
                if sample == 0:
                    atoms[-4] = accuracy.Atom(
                        atoms[-4].group,
                        atoms[-4].element,
                        atoms[-4].name,
                        atoms[-4].comp_id,
                        atoms[-4].chain,
                        atoms[-4].seq_id,
                        atoms[-4].xyz + np.array([5.0, 0.0, 0.0]),
                    )
                write_cif(sample_dir / f"{prefix}_model.cif", atoms)
                (sample_dir / f"{prefix}_summary_confidences.json").write_text(
                    json.dumps({"ptm": 0.8, "iptm": None})
                )
            case = json.loads((VALIDATION_ROOT / "ground_truth_cases.json").read_text())["cases"][0]
            report = accuracy.score_case(output_root, case, truth_root)
            self.assertEqual(report["top_sample"], 0)
            self.assertEqual(report["best_sample"], 1)

            thresholds = root / "thresholds.json"
            thresholds.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cases": {
                            case["name"]: {
                                "ca_rmsd": {"warn_max": 0.01, "fail_max": 0.02}
                            }
                        },
                    }
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = accuracy.main(
                    [
                        str(output_root),
                        "--profile",
                        "smoke",
                        "--ground-truth-root",
                        str(truth_root),
                        "--thresholds",
                        str(thresholds),
                        "--gate",
                    ]
                )
            self.assertEqual(exit_code, 1)

    def test_thresholds_distinguish_pass_warn_and_fail(self) -> None:
        rules = {
            "score": {"warn_min": 0.9, "fail_min": 0.8},
            "error": {"warn_max": 1.0, "fail_max": 2.0},
        }
        self.assertEqual(
            accuracy.evaluate_thresholds({"score": 0.95, "error": 0.5}, rules)["status"],
            "PASS",
        )
        self.assertEqual(
            accuracy.evaluate_thresholds({"score": 0.85, "error": 1.5}, rules)["status"],
            "WARN",
        )
        self.assertEqual(
            accuracy.evaluate_thresholds({"score": 0.7, "error": 3.0}, rules)["status"],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
