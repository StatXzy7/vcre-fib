from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from sfibai_b.evaluation import evaluate_predictions
from sfibai_b.prediction import PredictionCollector
from sfibai_b.storage import (
    load_metrics,
    save_prediction_bundle,
)


def grading_outputs(batch_size: int) -> dict[str, torch.Tensor]:
    logits = torch.full((batch_size, 36), -4.0)
    logits[0, 4] = 4.0
    logits[1, 24] = 4.0
    return {"logits": logits}


def batch() -> dict[str, object]:
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, 1:3, 1:3] = 1
    return {
        "image_uid": ["i1", "i2"],
        "patient_uid": ["p1", "p2"],
        "center_id": ["c1", "c2"],
        "split": ["val", "val"],
        "true_score": torch.tensor([0.4, 2.4]),
        "label_bin": torch.tensor([4, 24]),
        "position_norm": torch.tensor([1, 2]),
        "lesion_mask": mask,
        "lesion_box_valid": torch.tensor([True, False]),
    }


def test_compact_baseline_predictions_have_plotting_probabilities_without_aux_placeholders() -> None:
    collector = PredictionCollector(arm="A", full=False, collect_attention=False)
    collector.add_batch(batch(), grading_outputs(2))

    frame = collector.to_frame()

    assert set(f"prob_f{grade}" for grade in range(4)).issubset(frame.columns)
    assert not any(column.startswith("position_") for column in frame.columns)
    assert not any(column.startswith("lesion_") for column in frame.columns)
    assert not any(column.startswith("logit_") for column in frame.columns)
    assert np.allclose(
        frame[[f"prob_f{grade}" for grade in range(4)]].sum(axis=1), 1.0
    )


def test_full_e_predictions_include_logits_auxiliary_scalars_and_attention() -> None:
    outputs = grading_outputs(2)
    outputs.update(
        {
            "position_logits": torch.zeros(2, 6),
            "position_probs": torch.full((2, 6), 1 / 6),
            "position_gate": torch.tensor([[0.1], [0.2]]),
            "lesion_logits": torch.zeros(2, 1, 4, 4),
            "lesion_attention": torch.full((2, 1, 4, 4), 0.5),
            "lesion_gate": torch.tensor([[0.3], [0.4]]),
        }
    )
    collector = PredictionCollector(arm="E", full=True, collect_attention=True)
    collector.add_batch(batch(), outputs)

    frame = collector.to_frame()
    attention = collector.attention_payload()

    assert set(f"logit_{index:02d}" for index in range(36)).issubset(frame.columns)
    assert set(f"position_logit_{index}" for index in range(1, 7)).issubset(
        frame.columns
    )
    assert frame["lesion_valid"].tolist() == [True, False]
    assert attention["attention"].dtype == np.float16
    assert attention["attention"].shape == (2, 4, 4)
    assert attention["image_uid"].tolist() == ["i1", "i2"]


def test_saved_bundle_loads_single_canonical_metrics_and_preserves_attention(tmp_path) -> None:
    outputs = grading_outputs(2)
    outputs.update(
        {
            "position_logits": torch.zeros(2, 6),
            "position_probs": torch.full((2, 6), 1 / 6),
            "position_gate": torch.tensor([[0.1], [0.2]]),
            "lesion_logits": torch.zeros(2, 1, 4, 4),
            "lesion_attention": torch.full((2, 1, 4, 4), 0.5),
            "lesion_gate": torch.tensor([[0.3], [0.4]]),
        }
    )
    collector = PredictionCollector(arm="E", full=True, collect_attention=True)
    collector.add_batch(batch(), outputs)
    frame = collector.to_frame()
    metrics = evaluate_predictions(frame)

    save_prediction_bundle(
        directory=tmp_path,
        frame=frame,
        metrics=metrics,
        attention=collector.attention_payload(),
        full=True,
    )
    verified = load_metrics(tmp_path)

    assert verified["r_final"] == pytest.approx(metrics["r_final"])
    assert (tmp_path / "predictions_full.csv.gz").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "lesion_attention_float16.npz").is_file()
    with (tmp_path / "metrics.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["r_final"] == pytest.approx(metrics["r_final"])
