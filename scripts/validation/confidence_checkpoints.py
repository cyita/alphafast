"""Convert AF3 inference-result metadata into ranking checkpoints."""

from __future__ import annotations

import numpy as np


CONFIDENCE_FIELDS = (
    "ptm",
    "iptm",
    "ptm_iptm_average",
    "ranking_confidence",
    "predicted_distance_error",
    "fraction_disordered",
    "ranking_score",
)


def confidence_arrays(inference_results) -> dict[str, np.ndarray]:
    arrays = {
        f"confidence_{field}": np.asarray(
            [result.metadata[field] for result in inference_results],
            dtype=np.float32,
        )
        for field in CONFIDENCE_FIELDS
    }
    arrays["confidence_has_clash"] = np.asarray(
        [result.metadata["has_clash"] for result in inference_results],
        dtype=np.int8,
    )
    arrays["ranking_order"] = np.argsort(
        -arrays["confidence_ranking_score"], kind="stable"
    ).astype(np.int32)
    return arrays
