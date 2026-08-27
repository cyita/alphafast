"""Recreate the non-numeric batch fields needed by AF3 postprocessing."""

from __future__ import annotations

from pathlib import Path

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation


def featurise_ranking_batch(input_json: Path, bucket_size: int):
    fold_input = next(iter(folding_input.load_fold_inputs_from_path(input_json)))
    return next(
        iter(
            featurisation.featurise_input(
                fold_input=fold_input,
                buckets=(bucket_size,),
                ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd),
                verbose=False,
            )
        )
    )
