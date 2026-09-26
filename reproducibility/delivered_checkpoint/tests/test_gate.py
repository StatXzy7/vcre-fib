from __future__ import annotations

import pandas as pd
import pytest

from sfibai_b.gate import audit_history, validate_full_prediction_schema
from sfibai_b.training import learning_rate_for_epoch


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch": range(1, 121),
            "learning_rate": [
                learning_rate_for_epoch(epoch) for epoch in range(1, 121)
            ],
            "val__r_final": [0.5 - epoch / 1000 for epoch in range(1, 121)],
            "val__image__cor": [0.4 - epoch / 2000 for epoch in range(1, 121)],
        }
    )


def test_history_audit_confirms_120_epochs_schedule_and_best_tie_breaker() -> None:
    result = audit_history(_history(), best_epoch=120)
    assert result["ok"] is True
    assert result["best_epoch"] == 120


def test_history_audit_excludes_epochs_1_through_20_from_best_selection() -> None:
    history = _history()
    history.loc[4, "val__r_final"] = 0.0
    result = audit_history(history, best_epoch=120)
    assert result["best_epoch"] == 120


def test_history_audit_rejects_wrong_learning_rate() -> None:
    history = _history()
    history.loc[15, "learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="learning rate"):
        audit_history(history, best_epoch=120)


def test_full_prediction_schema_requires_branch_outputs_and_all_logits() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test"],
            "image_uid": ["i"],
            "patient_uid": ["p"],
            "center_id": ["c"],
            "true_score": [1.0],
            "pred_score": [1.1],
            "pred_bin": [11],
            **{f"prob_f{g}": [0.25] for g in range(4)},
            **{f"logit_{index:02d}": [0.0] for index in range(36)},
            "position_true": [1],
            "position_pred": [1],
            **{f"position_prob_{position}": [1 / 6] for position in range(1, 7)},
            **{f"position_logit_{position}": [0.0] for position in range(1, 7)},
            "position_gate": [0.1],
            "lesion_valid": [True],
            "lesion_gate": [0.2],
            "lesion_box_present": [True],
            "inside_attention": [0.5],
            "outside_ratio": [0.4],
            "inside_outside_ratio": [1.2],
            "dice_at_0_5": [0.3],
            "iou_at_0_5": [0.2],
        }
    )
    validate_full_prediction_schema(frame, arm="E", split="test")
    with pytest.raises(ValueError, match="position"):
        validate_full_prediction_schema(frame.drop(columns=["position_gate"]), arm="E", split="test")
