from __future__ import annotations

import math

import pandas as pd
import pytest

from sfibai_b.evaluation import (
    checkpoint_key,
    evaluate_predictions,
    is_checkpoint_eligible,
    score_cor,
)


def test_cor_uses_strict_thresholds_and_accumulating_penalties() -> None:
    scored = score_cor(
        true_scores=[0.0, 1.0, 3.0],
        pred_scores=[0.3, 1.6, 1.4],
    )

    assert scored["cor_continuous"] == pytest.approx(1.0 / 12.0)
    assert scored["cor_error_gt_0_3"] == pytest.approx(0.1)
    assert scored["cor_error_gt_0_5"] == pytest.approx(0.1)
    assert scored["cor_stage_distance"] == pytest.approx(1.0 / 12.0)
    assert scored["cor_severe_stage"] == pytest.approx(1.0 / 30.0)
    assert scored["cor"] == pytest.approx(0.4)


def test_evaluator_combines_image_patient_max_and_sqrt_center_views() -> None:
    predictions = pd.DataFrame(
        {
            "image_uid": ["i1", "i2", "i3", "i4"],
            "patient_uid": ["p1", "p1", "p2", "p3"],
            "center_id": ["a", "a", "a", "b"],
            "true_score": [0.0, 2.0, 3.0, 1.0],
            "pred_score": [0.4, 1.8, 2.0, 1.6],
        }
    )

    result = evaluate_predictions(predictions)

    expected_center_a = (0.02 + 29.0 / 60.0) / 2.0
    expected_center_b = 133.0 / 300.0
    expected_center_balanced = (
        math.sqrt(2.0) * expected_center_a + expected_center_b
    ) / (math.sqrt(2.0) + 1.0)
    expected_image = (0.19 + 0.02 + 29.0 / 60.0 + 133.0 / 300.0) / 4.0
    expected_patient_max = (0.02 + 29.0 / 60.0 + 133.0 / 300.0) / 3.0

    assert result["image"]["cor"] == pytest.approx(expected_image)
    assert result["patient_max"]["cor"] == pytest.approx(expected_patient_max)
    assert result["center_balanced_patient_max"]["cor"] == pytest.approx(
        expected_center_balanced
    )
    assert result["center_balanced_patient_max"]["center_patient_counts"] == {
        "a": 2,
        "b": 1,
    }
    assert result["r_final"] == pytest.approx(
        0.4 * expected_image
        + 0.4 * expected_patient_max
        + 0.2 * expected_center_balanced
    )


def test_checkpoint_key_prefers_cor_then_image_cor_then_earlier_epoch() -> None:
    assert checkpoint_key({"r_final": 0.2, "image": {"cor": 0.4}}, 9) < checkpoint_key(
        {"r_final": 0.3, "image": {"cor": 0.1}}, 1
    )
    assert checkpoint_key({"r_final": 0.2, "image": {"cor": 0.3}}, 9) < checkpoint_key(
        {"r_final": 0.2, "image": {"cor": 0.4}}, 1
    )
    assert checkpoint_key({"r_final": 0.2, "image": {"cor": 0.3}}, 3) < checkpoint_key(
        {"r_final": 0.2, "image": {"cor": 0.3}}, 4
    )


def test_formal_checkpoint_selection_excludes_epochs_one_through_twenty() -> None:
    assert not is_checkpoint_eligible(1, formal=True)
    assert not is_checkpoint_eligible(20, formal=True)
    assert is_checkpoint_eligible(21, formal=True)
    assert is_checkpoint_eligible(120, formal=True)
    assert is_checkpoint_eligible(1, formal=False)


def test_evaluator_rejects_patient_crossing_centers() -> None:
    predictions = pd.DataFrame(
        {
            "image_uid": ["i1", "i2"],
            "patient_uid": ["p1", "p1"],
            "center_id": ["a", "b"],
            "true_score": [1.0, 2.0],
            "pred_score": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="multiple centers"):
        evaluate_predictions(predictions)


def test_image_metrics_include_multiclass_auc_precision_recall_and_calibration() -> None:
    true_scores = [0.1, 0.2, 1.1, 1.2, 2.1, 2.2, 3.1, 3.2]
    rows = []
    for index, score in enumerate(true_scores):
        grade = index // 2
        probabilities = [0.0, 0.0, 0.0, 0.0]
        probabilities[grade] = 1.0
        rows.append(
            {
                "image_uid": f"i{index}",
                "patient_uid": f"p{index}",
                "center_id": f"c{index % 2}",
                "true_score": score,
                "pred_score": score,
                **{f"prob_f{level}": probabilities[level] for level in range(4)},
            }
        )

    metrics = evaluate_predictions(pd.DataFrame(rows))["image"]

    assert metrics["grade_macro_auroc"] == pytest.approx(1.0)
    assert metrics["grade_macro_auprc"] == pytest.approx(1.0)
    assert metrics["grade_brier"] == pytest.approx(0.0)
    assert metrics["grade_ece"] == pytest.approx(0.0)
    for grade in range(4):
        assert metrics[f"grade_f{grade}_auroc"] == pytest.approx(1.0)
        assert metrics[f"grade_f{grade}_auprc"] == pytest.approx(1.0)


def test_image_probability_metrics_do_not_hide_real_out_of_range_values() -> None:
    frame = pd.DataFrame(
        {
            "image_uid": ["i1"],
            "patient_uid": ["p1"],
            "center_id": ["c1"],
            "true_score": [0.1],
            "pred_score": [0.1],
            "prob_f0": [-0.1],
            "prob_f1": [1.1],
            "prob_f2": [0.0],
            "prob_f3": [0.0],
        }
    )

    with pytest.raises(ValueError, match="lie in"):
        evaluate_predictions(frame)
