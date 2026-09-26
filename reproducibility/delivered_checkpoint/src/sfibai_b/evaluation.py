from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sfibai_b.protocol import BEST_SELECTION_START_EPOCH


GRADE_BOUNDARIES = np.asarray([0.5, 1.5, 2.5], dtype=np.float64)
COR_WEIGHTS = {
    "continuous": 0.35,
    "error_gt_0_3": 0.15,
    "error_gt_0_5": 0.15,
    "stage_distance": 0.25,
    "severe_stage": 0.10,
}


def score_to_grade(scores: Iterable[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    return np.searchsorted(GRADE_BOUNDARIES, values, side="right").astype(np.int64)


def _score_arrays(
    true_scores: Iterable[float] | np.ndarray,
    pred_scores: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(list(true_scores), dtype=np.float64)
    pred = np.asarray(list(pred_scores), dtype=np.float64)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError("true_scores and pred_scores must be equal-length vectors")
    if true.size == 0:
        raise ValueError("At least one prediction is required")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Scores must be finite")
    return true, pred


def score_cor(
    true_scores: Iterable[float] | np.ndarray,
    pred_scores: Iterable[float] | np.ndarray,
) -> dict[str, float]:
    true, pred = _score_arrays(true_scores, pred_scores)
    error = np.abs(pred - true)
    stage_distance = np.abs(score_to_grade(pred) - score_to_grade(true))

    components = {
        "cor_continuous": COR_WEIGHTS["continuous"] * float(np.mean(error / 3.5)),
        "cor_error_gt_0_3": COR_WEIGHTS["error_gt_0_3"]
        * float(np.mean(error > 0.3)),
        "cor_error_gt_0_5": COR_WEIGHTS["error_gt_0_5"]
        * float(np.mean(error > 0.5)),
        "cor_stage_distance": COR_WEIGHTS["stage_distance"]
        * float(np.mean(stage_distance / 3.0)),
        "cor_severe_stage": COR_WEIGHTS["severe_stage"]
        * float(np.mean(stage_distance >= 2)),
    }
    components["cor"] = float(sum(components.values()))
    return components


def _macro_f1(true_grade: np.ndarray, pred_grade: np.ndarray, classes: int) -> float:
    values: list[float] = []
    for grade in range(classes):
        true_positive = np.sum((true_grade == grade) & (pred_grade == grade))
        false_positive = np.sum((true_grade != grade) & (pred_grade == grade))
        false_negative = np.sum((true_grade == grade) & (pred_grade != grade))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(values))


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(confidence.size)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (confidence >= lower) & (confidence <= upper)
            if index == bins - 1
            else (confidence >= lower) & (confidence < upper)
        )
        if mask.any():
            error += float(mask.sum() / total) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(error)


def _probabilistic_grade_metrics(
    true_grade: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    if probabilities.shape != (true_grade.size, 4):
        raise ValueError("Grade probabilities must have shape N x 4")
    if not np.isfinite(probabilities).all():
        raise ValueError("Grade probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0 + 1e-6):
        raise ValueError("Grade probabilities must lie in [0, 1]")
    probabilities = np.clip(probabilities.astype(np.float64, copy=False), 0.0, 1.0)
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("Grade probabilities must sum to one")
    targets = np.eye(4, dtype=np.float64)[true_grade]
    output: dict[str, float | None] = {
        "grade_brier": float(np.mean(np.sum(np.square(probabilities - targets), axis=1)))
    }
    aurocs: list[float] = []
    auprcs: list[float] = []
    for grade in range(4):
        binary = targets[:, grade]
        probability = probabilities[:, grade]
        if np.unique(binary).size == 2:
            auroc = float(roc_auc_score(binary, probability))
            auprc = float(average_precision_score(binary, probability))
            aurocs.append(auroc)
            auprcs.append(auprc)
        else:
            auroc = None
            auprc = None
        output[f"grade_f{grade}_auroc"] = auroc
        output[f"grade_f{grade}_auprc"] = auprc
        output[f"grade_f{grade}_brier"] = float(
            np.mean(np.square(probability - binary))
        )
        output[f"grade_f{grade}_ece"] = _expected_calibration_error(
            probability, binary
        )
    output["grade_macro_auroc"] = float(np.mean(aurocs)) if aurocs else None
    output["grade_macro_auprc"] = float(np.mean(auprcs)) if auprcs else None
    confidence = probabilities.max(axis=1)
    predicted_grade = probabilities.argmax(axis=1)
    output["grade_ece"] = _expected_calibration_error(
        confidence, (predicted_grade == true_grade).astype(np.float64)
    )
    return output


def _classification_metrics(
    true: np.ndarray, pred: np.ndarray, probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    error = np.abs(pred - true)
    true_grade = score_to_grade(true)
    pred_grade = score_to_grade(pred)
    metrics: dict[str, Any] = {
        "n": int(true.size),
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "accuracy_within_0_3": float(np.mean(error <= 0.3)),
        "accuracy_within_0_5": float(np.mean(error <= 0.5)),
        "grade_accuracy": float(np.mean(true_grade == pred_grade)),
        "grade_macro_f1": _macro_f1(true_grade, pred_grade, 4),
        "tmae": float(np.mean(np.maximum(error - 0.5, 0.0))),
        "severe_error_rate": float(np.mean(error > 1.0)),
    }
    for grade in range(4):
        mask = true_grade == grade
        metrics[f"grade_f{grade}_recall"] = (
            float(np.mean(pred_grade[mask] == grade)) if mask.any() else None
        )
        metrics[f"grade_f{grade}_support"] = int(mask.sum())
    if probabilities is not None:
        metrics.update(_probabilistic_grade_metrics(true_grade, probabilities))
    metrics.update(score_cor(true, pred))
    return metrics


def _patient_predictions(predictions: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    if aggregation not in {"max", "median"}:
        raise ValueError("aggregation must be 'max' or 'median'")
    center_counts = predictions.groupby("patient_uid", sort=False)["center_id"].nunique()
    crossing = center_counts[center_counts != 1]
    if not crossing.empty:
        raise ValueError(
            f"Patients assigned to multiple centers: {', '.join(map(str, crossing.index[:3]))}"
        )
    reducer = "max" if aggregation == "max" else "median"
    grouped = predictions.groupby("patient_uid", sort=False, as_index=False).agg(
        center_id=("center_id", "first"),
        true_score=("true_score", reducer),
        pred_score=("pred_score", reducer),
        image_count=("image_uid", "size"),
    )
    return grouped


def _center_balanced_patient_max(patient_max: pd.DataFrame) -> dict[str, Any]:
    center_patient_counts: dict[str, int] = {}
    per_center_cor: dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for center, group in patient_max.groupby("center_id", sort=True):
        center_name = str(center)
        patient_count = int(len(group))
        center_cor = score_cor(group["true_score"], group["pred_score"])["cor"]
        weight = math.sqrt(patient_count)
        center_patient_counts[center_name] = patient_count
        per_center_cor[center_name] = center_cor
        weighted_sum += weight * center_cor
        weight_sum += weight
    if not weight_sum:
        raise ValueError("No centers available for center-balanced COR")
    return {
        "center_count": len(center_patient_counts),
        "patient_count": int(len(patient_max)),
        "center_patient_counts": center_patient_counts,
        "per_center_cor": per_center_cor,
        "weighting": "sqrt_center_patient_count",
        "cor": float(weighted_sum / weight_sum),
    }


def _position_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"position_true", "position_pred"}
    if not required.issubset(predictions.columns):
        return {"available": False}
    true = predictions["position_true"].to_numpy(dtype=np.int64)
    pred = predictions["position_pred"].to_numpy(dtype=np.int64)
    valid = (true >= 1) & (true <= 6)
    if not valid.any():
        return {"available": True, "n": 0}
    true = true[valid]
    pred = pred[valid]
    output: dict[str, Any] = {
        "available": True,
        "n": int(true.size),
        "accuracy": float(np.mean(true == pred)),
        "macro_f1": _macro_f1(true - 1, pred - 1, 6),
    }
    probability_columns = [f"position_prob_{index}" for index in range(1, 7)]
    if set(probability_columns).issubset(predictions.columns):
        probabilities = predictions.loc[valid, probability_columns].to_numpy(
            dtype=np.float64
        )
        true_probability = probabilities[np.arange(true.size), true - 1]
        output["ce"] = float(np.mean(-np.log(np.clip(true_probability, 1e-12, 1.0))))
    else:
        output["ce"] = None
    for position in range(1, 7):
        mask = true == position
        output[f"recall_{position}"] = (
            float(np.mean(pred[mask] == position)) if mask.any() else None
        )
        output[f"predicted_frequency_{position}"] = float(np.mean(pred == position))
    return output


def _lesion_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if "lesion_valid" not in predictions.columns:
        return {"available": False}
    valid = predictions["lesion_valid"].astype(bool).to_numpy()
    output: dict[str, Any] = {"available": True, "valid_box_count": int(valid.sum())}
    columns = {
        "inside_attention": "inside_attention",
        "outside_ratio": "outside_ratio",
        "inside_outside_ratio": "inside_outside_ratio",
        "dice_at_0_5": "dice_at_0_5",
        "iou_at_0_5": "iou_at_0_5",
    }
    for metric_name, column in columns.items():
        output[metric_name] = (
            float(predictions.loc[valid, column].astype(float).mean())
            if valid.any() and column in predictions.columns
            else None
        )
    return output


def _gate_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for branch in ("position", "lesion"):
        column = f"{branch}_gate"
        if column not in predictions.columns:
            output[branch] = {"available": False}
            continue
        values = predictions[column].dropna().to_numpy(dtype=np.float64)
        if not values.size:
            output[branch] = {"available": False}
            continue
        output[branch] = {
            "available": True,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=0)),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
    return output


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"image_uid", "patient_uid", "center_id", "true_score", "pred_score"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction columns are missing: {', '.join(missing)}")
    if predictions.empty:
        raise ValueError("Prediction frame is empty")
    if predictions["image_uid"].duplicated().any():
        raise ValueError("image_uid must be unique in one evaluation")
    true, pred = _score_arrays(predictions["true_score"], predictions["pred_score"])
    probability_columns = [f"prob_f{grade}" for grade in range(4)]
    image_probabilities = (
        predictions[probability_columns].to_numpy(dtype=np.float64)
        if set(probability_columns).issubset(predictions.columns)
        else None
    )
    image = _classification_metrics(true, pred, image_probabilities)
    patient_max_frame = _patient_predictions(predictions, "max")
    patient_median_frame = _patient_predictions(predictions, "median")
    patient_max = _classification_metrics(
        patient_max_frame["true_score"].to_numpy(dtype=np.float64),
        patient_max_frame["pred_score"].to_numpy(dtype=np.float64),
    )
    patient_median = _classification_metrics(
        patient_median_frame["true_score"].to_numpy(dtype=np.float64),
        patient_median_frame["pred_score"].to_numpy(dtype=np.float64),
    )
    center_balanced = _center_balanced_patient_max(patient_max_frame)
    r_final = float(
        0.4 * image["cor"]
        + 0.4 * patient_max["cor"]
        + 0.2 * center_balanced["cor"]
    )
    return {
        "image": image,
        "patient_max": patient_max,
        "patient_median": patient_median,
        "center_balanced_patient_max": center_balanced,
        "position": _position_metrics(predictions),
        "lesion": _lesion_metrics(predictions),
        "gates": _gate_metrics(predictions),
        "r_final": r_final,
        "r_final_weights": {
            "image": 0.4,
            "patient_max": 0.4,
            "center_balanced_patient_max": 0.2,
        },
    }


def checkpoint_key(metrics: Mapping[str, Any], epoch: int) -> tuple[float, float, int]:
    return (
        float(metrics["r_final"]),
        float(metrics["image"]["cor"]),
        int(epoch),
    )


def is_checkpoint_eligible(epoch: int, *, formal: bool) -> bool:
    if int(epoch) <= 0:
        raise ValueError("Epochs are one-based")
    return not formal or int(epoch) >= BEST_SELECTION_START_EPOCH


def flatten_metrics(
    metrics: Mapping[str, Any],
    *,
    separator: str = "/",
    prefix: str = "",
) -> dict[str, float | int | str | bool | None]:
    flattened: dict[str, float | int | str | bool | None] = {}
    for key, value in metrics.items():
        name = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(
                flatten_metrics(value, separator=separator, prefix=name)
            )
        elif isinstance(value, (str, bool, int, float)) or value is None:
            flattened[name] = value
    return flattened


def evaluate_prediction_file(path: str | Path) -> dict[str, Any]:
    prediction_path = Path(path)
    return evaluate_predictions(pd.read_csv(prediction_path))
