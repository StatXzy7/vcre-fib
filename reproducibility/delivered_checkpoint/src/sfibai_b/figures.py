from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from sfibai_b.evaluation import score_cor, score_to_grade


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PLOT_REQUIRED_COLUMNS = {
    "image_uid",
    "patient_uid",
    "center_id",
    "true_score",
    "pred_score",
    "prob_f0",
    "prob_f1",
    "prob_f2",
    "prob_f3",
}
GRADE_PROBABILITY_ATOL = 1e-6


def validate_plot_ready_predictions(frame: pd.DataFrame) -> None:
    missing = sorted(PLOT_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("Plot-ready predictions are missing: " + ", ".join(missing))
    if frame.empty or frame["image_uid"].duplicated().any():
        raise ValueError("Plot-ready predictions require unique non-empty image rows")
    probabilities = frame[[f"prob_f{grade}" for grade in range(4)]].to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Grade probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0 + GRADE_PROBABILITY_ATOL
    ):
        raise ValueError("Grade probabilities must lie in [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=GRADE_PROBABILITY_ATOL
    ):
        raise ValueError("Four grade probabilities must sum to one")


def _curve_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    true_grade = score_to_grade(frame["true_score"].to_numpy(float))
    roc_rows: list[dict[str, float | int]] = []
    pr_rows: list[dict[str, float | int]] = []
    calibration_rows: list[dict[str, float | int]] = []
    for grade in range(4):
        target = (true_grade == grade).astype(int)
        probability = np.clip(
            frame[f"prob_f{grade}"].to_numpy(float), 0.0, 1.0
        )
        if np.unique(target).size != 2:
            raise ValueError(f"ROC/PR require positive and negative examples for F{grade}")
        false_positive, true_positive, thresholds = roc_curve(target, probability)
        for x, y, threshold in zip(false_positive, true_positive, thresholds):
            roc_rows.append({"grade": grade, "fpr": x, "tpr": y, "threshold": threshold})
        precision, recall, thresholds = precision_recall_curve(target, probability)
        padded = np.append(thresholds, np.nan)
        for x, y, threshold in zip(recall, precision, padded):
            pr_rows.append({"grade": grade, "recall": x, "precision": y, "threshold": threshold})
        observed, predicted = calibration_curve(target, probability, n_bins=10, strategy="uniform")
        for bin_index, (x, y) in enumerate(zip(predicted, observed)):
            calibration_rows.append(
                {"grade": grade, "bin": bin_index, "mean_probability": x, "observed_frequency": y}
            )
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows), pd.DataFrame(calibration_rows)


def _patient_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for aggregation in ("max", "median"):
        grouped = frame.groupby("patient_uid", as_index=False, sort=False).agg(
            center_id=("center_id", "first"),
            true_score=("true_score", aggregation),
            pred_score=("pred_score", aggregation),
            image_count=("image_uid", "size"),
        )
        grouped.insert(0, "aggregation", aggregation)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_plot_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_plot_ready_predictions(frame)
    true_grade = score_to_grade(frame["true_score"].to_numpy(float))
    pred_grade = score_to_grade(frame["pred_score"].to_numpy(float))
    roc, pr, calibration = _curve_tables(frame)
    matrix = confusion_matrix(true_grade, pred_grade, labels=[0, 1, 2, 3])
    confusion = pd.DataFrame(
        [
            {"true_grade": true, "pred_grade": pred, "count": int(matrix[true, pred])}
            for true in range(4)
            for pred in range(4)
        ]
    )
    patient = _patient_table(frame)
    patient_max = patient[patient["aggregation"] == "max"]
    center_rows = []
    for center, group in patient_max.groupby("center_id", sort=True):
        center_rows.append(
            {
                "center_id": str(center),
                "patient_count": len(group),
                **score_cor(group["true_score"], group["pred_score"]),
            }
        )
    errors = frame[["image_uid", "patient_uid", "center_id", "true_score", "pred_score"]].copy()
    errors["true_grade"] = true_grade
    errors["pred_grade"] = pred_grade
    errors["absolute_error"] = np.abs(errors["pred_score"] - errors["true_score"])
    return {
        "roc": roc,
        "pr": pr,
        "calibration": calibration,
        "confusion": confusion,
        "patient": patient,
        "center": pd.DataFrame(center_rows),
        "errors": errors,
    }


def generate_plot_pack(prediction_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(prediction_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Plot pack already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Incomplete plot pack already exists: {temporary}")
    temporary.mkdir()
    frame = pd.read_csv(source)
    tables = build_plot_tables(frame)
    try:
        for name, table in tables.items():
            table.to_csv(temporary / f"{name}.csv", index=False, encoding="utf-8")

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for grade, group in tables["roc"].groupby("grade"):
            axes[0, 0].plot(group["fpr"], group["tpr"], label=f"F{grade}")
        axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[0, 0].set(title="One-vs-rest ROC", xlabel="FPR", ylabel="TPR")
        axes[0, 0].legend()
        for grade, group in tables["pr"].groupby("grade"):
            axes[0, 1].plot(group["recall"], group["precision"], label=f"F{grade}")
        axes[0, 1].set(title="One-vs-rest PR", xlabel="Recall", ylabel="Precision")
        axes[0, 1].legend()
        for grade, group in tables["calibration"].groupby("grade"):
            axes[0, 2].plot(group["mean_probability"], group["observed_frequency"], marker="o", label=f"F{grade}")
        axes[0, 2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[0, 2].set(title="Calibration", xlabel="Predicted", ylabel="Observed")
        axes[0, 2].legend()
        matrix = tables["confusion"].pivot(index="true_grade", columns="pred_grade", values="count").to_numpy()
        axes[1, 0].imshow(matrix, cmap="Blues")
        axes[1, 0].set(title="Grade confusion", xlabel="Predicted", ylabel="True")
        for y in range(4):
            for x in range(4):
                axes[1, 0].text(x, y, str(matrix[y, x]), ha="center", va="center")
        axes[1, 1].hist(
            tables["errors"]["absolute_error"], bins=np.linspace(0.0, 3.5, 36)
        )
        axes[1, 1].set(title="Image absolute error", xlabel="Absolute error", ylabel="Images")
        center = tables["center"].sort_values("cor")
        axes[1, 2].bar(center["center_id"].astype(str), center["cor"])
        axes[1, 2].set(title="Patient-max COR by center", xlabel="Center", ylabel="COR")
        axes[1, 2].tick_params(axis="x", rotation=90)
        fig.tight_layout()
        figure_path = temporary / "evaluation_overview.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
        manifest = {
            "source_predictions": str(source.resolve()),
            "rows": len(frame),
            "tables": sorted(f"{name}.csv" for name in tables),
            "figure": figure_path.name,
        }
        (temporary / "plot_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output)
        verify_plot_pack(output)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_plot_pack(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "plot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Plot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = [*manifest["tables"], manifest["figure"]]
    missing = [name for name in expected if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Plot pack is incomplete: " + ", ".join(missing))
    return manifest


def generate_selection_trajectory(
    history_path: str | Path,
    output_path: str | Path,
    *,
    best_epoch: int,
) -> dict[str, Any]:
    history = pd.read_csv(history_path)
    required = {"epoch", "val__r_final", "val__image__cor"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Selection trajectory history columns are missing: {missing}")
    epochs = history["epoch"].astype(int).tolist()
    if epochs != list(range(1, 121)):
        raise ValueError("Selection trajectory requires exactly epochs 1 through 120")
    if not 21 <= int(best_epoch) <= 120:
        raise ValueError("Best epoch must be in the locked eligible interval 21..120")
    eligible = history.loc[history["epoch"].astype(int) >= 21]
    selected = min(
        eligible.to_dict("records"),
        key=lambda row: (
            float(row["val__r_final"]),
            float(row["val__image__cor"]),
            int(row["epoch"]),
        ),
    )
    if int(selected["epoch"]) != int(best_epoch):
        raise ValueError("Trajectory best epoch differs from the locked selector")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    axis.plot(history["epoch"], history["val__r_final"], label="val R_final")
    axis.plot(
        history["epoch"],
        history["val__image__cor"],
        label="val image COR",
        alpha=0.75,
    )
    axis.axvspan(1, 20, color="0.85", alpha=0.65, label="burn-in (ineligible)")
    axis.axvline(best_epoch, color="tab:red", linestyle="--", label=f"best={best_epoch}")
    axis.scatter(
        [best_epoch],
        [float(selected["val__r_final"])],
        color="tab:red",
        zorder=3,
    )
    axis.set(
        title="Validation checkpoint-selection trajectory",
        xlabel="Epoch",
        ylabel="Risk (lower is better)",
        xlim=(1, 120),
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return {
        "history": str(Path(history_path).resolve()),
        "output": str(output.resolve()),
        "best_epoch": int(best_epoch),
        "eligible_start_epoch": 21,
        "best_r_final": float(selected["val__r_final"]),
    }
