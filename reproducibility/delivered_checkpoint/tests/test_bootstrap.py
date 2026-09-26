from __future__ import annotations

import pandas as pd
import pytest

from sfibai_b.bootstrap import paired_patient_cluster_bootstrap
from sfibai_b.evaluation import evaluate_predictions


def _predictions(*, perfect: bool) -> pd.DataFrame:
    rows = []
    for center, patients in {"c1": ("p1", "p2"), "c2": ("p3", "p4")}.items():
        for patient_index, patient in enumerate(patients):
            for image_index, truth in enumerate((0.4 + patient_index, 0.8 + patient_index)):
                rows.append(
                    {
                        "image_uid": f"{patient}_{image_index}",
                        "patient_uid": patient,
                        "center_id": center,
                        "true_score": truth,
                        "pred_score": truth if perfect else min(3.5, truth + 0.8),
                    }
                )
    return pd.DataFrame(rows)


def test_center_stratified_patient_cluster_bootstrap_is_paired_and_exact() -> None:
    candidate = _predictions(perfect=True)
    reference = _predictions(perfect=False)

    result = paired_patient_cluster_bootstrap(
        candidate,
        reference,
        candidate_name="C",
        reference_name="A",
        n_resamples=250,
        seed=2026,
    )

    expected_delta = (
        evaluate_predictions(candidate)["r_final"]
        - evaluate_predictions(reference)["r_final"]
    )
    assert result["observed_candidate_minus_reference"] == pytest.approx(expected_delta)
    assert result["resamples"] == 250
    assert result["centers"] == 2
    assert result["patients"] == 4
    assert result["probability_candidate_better"] == 1.0
    assert result["ci_95"][1] < 0.0
