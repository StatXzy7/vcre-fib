from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sfibai_b.ranking import CONTRASTS


REQUIRED_COLUMNS = {
    "image_uid",
    "patient_uid",
    "center_id",
    "true_score",
    "pred_score",
}


def _cor_risk(true_score: np.ndarray, pred_score: np.ndarray) -> np.ndarray:
    true = np.asarray(true_score, dtype=np.float64)
    pred = np.asarray(pred_score, dtype=np.float64)
    error = np.abs(pred - true)
    boundaries = np.asarray([0.5, 1.5, 2.5], dtype=np.float64)
    true_grade = np.searchsorted(boundaries, true, side="right")
    pred_grade = np.searchsorted(boundaries, pred, side="right")
    stage_distance = np.abs(pred_grade - true_grade)
    return (
        0.35 * error / 3.5
        + 0.15 * (error > 0.3)
        + 0.15 * (error > 0.5)
        + 0.25 * stage_distance / 3.0
        + 0.10 * (stage_distance >= 2)
    ).astype(np.float64)


def _validate_pair(candidate: pd.DataFrame, reference: pd.DataFrame) -> None:
    for name, frame in (("candidate", candidate), ("reference", reference)):
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{name} predictions missing columns: {missing}")
        if frame.empty or frame["image_uid"].duplicated().any():
            raise ValueError(f"{name} predictions require unique non-empty image_uid")
    metadata = ["image_uid", "patient_uid", "center_id", "true_score"]
    left = candidate[metadata].sort_values("image_uid").reset_index(drop=True)
    right = reference[metadata].sort_values("image_uid").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Paired bootstrap requires identical test images and labels")


def _cluster_table(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    working["image_risk"] = _cor_risk(
        working["true_score"].to_numpy(), working["pred_score"].to_numpy()
    )
    centers_per_patient = working.groupby("patient_uid")["center_id"].nunique()
    if not (centers_per_patient == 1).all():
        raise ValueError("One patient cannot occur in multiple centers")
    clusters = working.groupby("patient_uid", sort=True, as_index=False).agg(
        center_id=("center_id", "first"),
        image_risk_sum=("image_risk", "sum"),
        image_count=("image_uid", "size"),
        patient_true=("true_score", "max"),
        patient_pred=("pred_score", "max"),
    )
    clusters["patient_risk"] = _cor_risk(
        clusters["patient_true"].to_numpy(),
        clusters["patient_pred"].to_numpy(),
    )
    return clusters


def _observed_r_final(clusters: pd.DataFrame) -> float:
    image_cor = float(clusters["image_risk_sum"].sum() / clusters["image_count"].sum())
    patient_cor = float(clusters["patient_risk"].mean())
    weighted = 0.0
    weight_sum = 0.0
    for _center, group in clusters.groupby("center_id", sort=True):
        weight = math.sqrt(len(group))
        weighted += weight * float(group["patient_risk"].mean())
        weight_sum += weight
    center_balanced = weighted / weight_sum
    return float(0.4 * image_cor + 0.4 * patient_cor + 0.2 * center_balanced)


def paired_patient_cluster_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    candidate_name: str,
    reference_name: str,
    n_resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    if n_resamples <= 1:
        raise ValueError("n_resamples must be greater than one")
    _validate_pair(candidate, reference)
    candidate_clusters = _cluster_table(candidate).set_index("patient_uid")
    reference_clusters = _cluster_table(reference).set_index("patient_uid")
    if not candidate_clusters[["center_id", "image_count", "patient_true"]].equals(
        reference_clusters[["center_id", "image_count", "patient_true"]]
    ):
        raise ValueError("Paired bootstrap patient clusters are inconsistent")

    rng = np.random.default_rng(seed)
    centers = sorted(candidate_clusters["center_id"].astype(str).unique())
    total_patient_count = len(candidate_clusters)

    candidate_image_sum = np.zeros(n_resamples, dtype=np.float64)
    reference_image_sum = np.zeros(n_resamples, dtype=np.float64)
    image_count_sum = np.zeros(n_resamples, dtype=np.float64)
    candidate_patient_sum = np.zeros(n_resamples, dtype=np.float64)
    reference_patient_sum = np.zeros(n_resamples, dtype=np.float64)
    candidate_center_weighted = np.zeros(n_resamples, dtype=np.float64)
    reference_center_weighted = np.zeros(n_resamples, dtype=np.float64)
    center_weight_sum = 0.0

    for center in centers:
        center_mask = candidate_clusters["center_id"].astype(str).to_numpy() == center
        indices = np.flatnonzero(center_mask)
        count = len(indices)
        draws = indices[rng.integers(0, count, size=(n_resamples, count))]

        candidate_image = candidate_clusters["image_risk_sum"].to_numpy()[draws]
        reference_image = reference_clusters["image_risk_sum"].to_numpy()[draws]
        image_count = candidate_clusters["image_count"].to_numpy()[draws]
        candidate_patient = candidate_clusters["patient_risk"].to_numpy()[draws]
        reference_patient = reference_clusters["patient_risk"].to_numpy()[draws]

        candidate_image_sum += candidate_image.sum(axis=1)
        reference_image_sum += reference_image.sum(axis=1)
        image_count_sum += image_count.sum(axis=1)
        candidate_patient_sum += candidate_patient.sum(axis=1)
        reference_patient_sum += reference_patient.sum(axis=1)
        weight = math.sqrt(count)
        candidate_center_weighted += weight * candidate_patient.mean(axis=1)
        reference_center_weighted += weight * reference_patient.mean(axis=1)
        center_weight_sum += weight

    candidate_r = (
        0.4 * candidate_image_sum / image_count_sum
        + 0.4 * candidate_patient_sum / total_patient_count
        + 0.2 * candidate_center_weighted / center_weight_sum
    )
    reference_r = (
        0.4 * reference_image_sum / image_count_sum
        + 0.4 * reference_patient_sum / total_patient_count
        + 0.2 * reference_center_weighted / center_weight_sum
    )
    deltas = candidate_r - reference_r
    observed = _observed_r_final(candidate_clusters.reset_index()) - _observed_r_final(
        reference_clusters.reset_index()
    )
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "contrast": f"{candidate_name}-{reference_name}",
        "observed_candidate_minus_reference": float(observed),
        "bootstrap_mean": float(deltas.mean()),
        "bootstrap_sample_sd": float(deltas.std(ddof=1)),
        "ci_95": [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ],
        "probability_candidate_better": float(np.mean(deltas < 0.0)),
        "probability_equal": float(np.mean(deltas == 0.0)),
        "resamples": int(n_resamples),
        "seed": int(seed),
        "stratification": "center",
        "resampling_unit": "patient_cluster",
        "shared_draws": True,
        "centers": int(len(centers)),
        "patients": int(total_patient_count),
        "images": int(len(candidate)),
    }


def write_paired_bootstrap_artifacts(
    experiment_root: str | Path,
    *,
    n_resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    root = Path(experiment_root)
    frames = {
        arm: pd.read_csv(
            root
            / "seed_2026"
            / arm
            / "best"
            / "test"
            / "predictions_full.csv.gz"
        )
        for arm in "ABCDE"
    }
    results = [
        paired_patient_cluster_bootstrap(
            frames[candidate],
            frames[reference],
            candidate_name=candidate,
            reference_name=reference,
            n_resamples=n_resamples,
            seed=seed,
        )
        for candidate, reference in CONTRASTS
    ]
    payload = {
        "round": "AE_COR_v2",
        "endpoint": "best-test R_final",
        "results": results,
    }
    output = root / "final_ranking"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "paired_patient_cluster_bootstrap.json"
    temporary = json_path.with_name(json_path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, json_path)
    pd.DataFrame(
        [
            {
                **{key: value for key, value in item.items() if key != "ci_95"},
                "ci_95_low": item["ci_95"][0],
                "ci_95_high": item["ci_95"][1],
            }
            for item in results
        ]
    ).to_csv(output / "paired_patient_cluster_bootstrap.csv", index=False, encoding="utf-8")
    return payload
