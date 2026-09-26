from __future__ import annotations

import pandas as pd
import pytest

from sfibai_b.figures import (
    build_plot_tables,
    generate_plot_pack,
    generate_selection_trajectory,
    validate_plot_ready_predictions,
    verify_plot_pack,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for index, true_score in enumerate([0.1, 0.8, 1.8, 2.9, 0.2, 1.2, 2.2, 3.2]):
        grade = min(int(true_score + 0.5), 3)
        probabilities = [0.05, 0.05, 0.05, 0.05]
        probabilities[grade] = 0.85
        rows.append(
            {
                "image_uid": f"i{index}",
                "patient_uid": f"p{index // 2}",
                "center_id": f"c{index % 2}",
                "true_score": true_score,
                "pred_score": true_score + 0.1,
                **{f"prob_f{g}": probabilities[g] for g in range(4)},
            }
        )
    return pd.DataFrame(rows)


def test_plot_tables_cover_roc_pr_calibration_confusion_and_aggregations() -> None:
    frame = _predictions()
    validate_plot_ready_predictions(frame)
    tables = build_plot_tables(frame)

    assert {"roc", "pr", "calibration", "confusion", "patient", "center", "errors"} <= set(tables)
    assert set(tables["roc"]["grade"]) == {0, 1, 2, 3}
    assert set(tables["pr"]["grade"]) == {0, 1, 2, 3}
    assert int(tables["confusion"]["count"].sum()) == len(frame)
    assert set(tables["patient"]["aggregation"]) == {"max", "median"}


def test_plot_readiness_requires_all_four_grade_probabilities() -> None:
    frame = _predictions().drop(columns=["prob_f3"])
    with pytest.raises(ValueError, match="prob_f3"):
        validate_plot_ready_predictions(frame)


def test_plotting_accepts_float32_probability_roundoff_but_rejects_real_overflow() -> None:
    frame = _predictions()
    frame.loc[0, ["prob_f0", "prob_f1", "prob_f2", "prob_f3"]] = [
        1.000000238418579,
        1.0e-9,
        1.0e-9,
        1.0e-9,
    ]

    validate_plot_ready_predictions(frame)
    build_plot_tables(frame)
    assert frame.loc[0, "prob_f0"] == 1.000000238418579

    invalid = _predictions()
    invalid.loc[0, ["prob_f0", "prob_f1", "prob_f2", "prob_f3"]] = [
        1.0001,
        0.0,
        0.0,
        0.0,
    ]
    with pytest.raises(ValueError, match="probabilities"):
        validate_plot_ready_predictions(invalid)

    negative = _predictions()
    negative.loc[0, ["prob_f0", "prob_f1", "prob_f2", "prob_f3"]] = [
        -1.0e-8,
        0.2,
        0.3,
        0.50000001,
    ]
    with pytest.raises(ValueError, match="probabilities"):
        validate_plot_ready_predictions(negative)

    not_normalized = _predictions()
    not_normalized.loc[
        0, ["prob_f0", "prob_f1", "prob_f2", "prob_f3"]
    ] = [0.25004, 0.25004, 0.25004, 0.25004]
    with pytest.raises(ValueError, match="sum to one"):
        validate_plot_ready_predictions(not_normalized)


def test_plot_pack_is_complete_and_refuses_overwrite(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.csv.gz"
    _predictions().to_csv(prediction_path, index=False, compression="gzip")
    output = tmp_path / "plots"

    generate_plot_pack(prediction_path, output)
    verified = verify_plot_pack(output)

    assert verified["rows"] == 8
    assert (output / "evaluation_overview.png").is_file()
    assert not (tmp_path / "plots.partial").exists()
    with pytest.raises(FileExistsError):
        generate_plot_pack(prediction_path, output)


def test_selection_trajectory_marks_burn_in_and_best(tmp_path) -> None:
    history = pd.DataFrame(
        {
            "epoch": range(1, 121),
            "val__r_final": [0.5 - epoch / 1000 for epoch in range(1, 121)],
            "val__image__cor": [0.4 - epoch / 2000 for epoch in range(1, 121)],
        }
    )
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "selection.png"
    history.to_csv(history_path, index=False)

    result = generate_selection_trajectory(
        history_path, output_path, best_epoch=120
    )

    assert output_path.is_file()
    assert result["best_epoch"] == 120
    assert result["eligible_start_epoch"] == 21
