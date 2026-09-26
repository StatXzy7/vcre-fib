"""Package Legacy SynAP-Fib E; reuse canonical metrics, export missing tables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from package_common import (
    PACK,
    WORKSPACE,
    ROUND,
    RUN,
    DATA,
    VIEW_NAMES,
    sha256,
    read_json,
    write_json,
    write_table,
    grades,
    markdown_table,
)

COPIES: list[dict] = []


def copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(source)
    shutil.copy2(source, destination)
    after = sha256(destination)
    if before != after:
        raise ValueError(f"Copy integrity failure: {source}")
    COPIES.append(
        {
            "source": str(source),
            "destination": str(destination.relative_to(PACK)),
            "sha256": after,
            "bytes": destination.stat().st_size,
        }
    )


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            copy_verified(path, destination / path.relative_to(source))


def save_confusion(
    true: np.ndarray, pred: np.ndarray, labels: list[int], output: Path, prefix: str
) -> None:
    matrix = confusion_matrix(true, pred, labels=labels)
    rows = pd.DataFrame(matrix, columns=[f"pred_{v}" for v in labels])
    rows.insert(0, "true_label", labels)
    write_table(rows, output / f"{prefix}.csv")
    write_json(
        output / f"{prefix}.json",
        {
            "labels": labels,
            "rows": "true",
            "columns": "predicted",
            "matrix": matrix.tolist(),
            "true_support": matrix.sum(axis=1).tolist(),
            "n": int(matrix.sum()),
        },
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues")
    for y in range(len(labels)):
        for x in range(len(labels)):
            ax.text(
                x,
                y,
                str(matrix[y, x]),
                ha="center",
                va="center",
                color="white" if matrix[y, x] > matrix.max() * 0.55 else "black",
            )
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=prefix.replace("_", " "),
    )
    fig.savefig(output / f"{prefix}.png", dpi=180)
    fig.savefig(output / f"{prefix}.pdf")
    plt.close(fig)


def export_split(split: str, complete: dict, manifest: pd.DataFrame) -> dict:
    output = PACK / "results" / split
    source = RUN / "best" / split
    for name, target in [
        ("metrics.json", "metrics.json"),
        ("predictions_full.csv.gz", "predictions.csv.gz"),
        ("lesion_attention_float16.npz", "lesion_attention_float16.npz"),
    ]:
        copy_verified(source / name, output / target)
    metrics = read_json(output / "metrics.json")
    frame = pd.read_csv(output / "predictions.csv.gz", float_precision="round_trip")
    expected = manifest[manifest.split == split]
    if (
        frame.image_uid.duplicated().any()
        or set(frame.image_uid) != set(expected.image_uid)
        or set(frame.split) != {split}
        or set(frame.arm) != {"E"}
    ):
        raise ValueError(f"Split identity mismatch: {split}")
    if (frame.groupby("patient_uid").center_id.nunique() != 1).any():
        raise ValueError("Patient spans multiple centers")
    probabilities = frame[[f"prob_f{i}" for i in range(4)]].to_numpy()
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(frame.pred_score).all()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid predictions")
    with np.load(output / "lesion_attention_float16.npz", allow_pickle=False) as maps:
        if (
            list(maps["image_uid"]) != frame.image_uid.tolist()
            or maps["attention"].shape != (len(frame), 32, 32)
            or not np.isfinite(maps["attention"]).all()
        ):
            raise ValueError("Attention map alignment failure")
    for reducer in ["max", "median"]:
        patients = frame.groupby("patient_uid", as_index=False, sort=False).agg(
            center_id=("center_id", "first"),
            true_score=("true_score", reducer),
            pred_score=("pred_score", reducer),
            image_count=("image_uid", "size"),
        )
        patients["split"] = split
        patients["true_grade"] = grades(patients.true_score)
        patients["pred_grade"] = grades(patients.pred_score)
        write_table(patients, output / f"patient_{reducer}_predictions.csv.gz")
    save_confusion(
        grades(frame.true_score),
        grades(frame.pred_score),
        list(range(4)),
        output,
        "grade_confusion",
    )
    valid = frame.position_true.between(1, 6) & frame.position_pred.between(1, 6)
    save_confusion(
        frame.loc[valid, "position_true"].to_numpy(),
        frame.loc[valid, "position_pred"].to_numpy(),
        list(range(1, 7)),
        output,
        "view_confusion",
    )
    view = metrics["position"]
    counts = frame.loc[valid].position_true.value_counts()
    views = pd.DataFrame(
        [
            {
                "view_id": i,
                "name": VIEW_NAMES[i],
                "support": int(counts.get(i, 0)),
                "recall": view[f"recall_{i}"],
                "predicted_frequency": view[f"predicted_frequency_{i}"],
            }
            for i in range(1, 7)
        ]
    )
    write_table(views, output / "view_classes.csv")
    center = metrics["center_balanced_patient_max"]
    weight_sum = sum(np.sqrt(n) for n in center["center_patient_counts"].values())
    centers = pd.DataFrame(
        [
            {
                "center_id": c,
                "images": int((frame.center_id == c).sum()),
                "patients": n,
                "patient_max_cor": center["per_center_cor"][c],
                "sqrt_patient_weight": np.sqrt(n),
                "normalized_weight": np.sqrt(n) / weight_sum,
                "weighted_cor_contribution": np.sqrt(n)
                / weight_sum
                * center["per_center_cor"][c],
            }
            for c, n in center["center_patient_counts"].items()
        ]
    )
    write_table(centers, output / "centers.csv")
    write_json(
        output / "provenance.json",
        {
            "model": "SynAP-Fib",
            "architecture": "Legacy E gated residual",
            "arm": "E",
            "round": "AE_COR_v2",
            "seed": 2026,
            "split": split,
            "frozen": True,
            "checkpoint": "../../model/checkpoints/best.pt",
            "epoch": complete["best_epoch"],
            "checkpoint_sha256": complete["best_checkpoint_sha256"],
            "metrics_origin": "byte-identical frozen canonical evaluation; no metric reevaluation",
            "derived_exports": [
                "patient predictions",
                "grade/view confusion",
                "center weights",
                "view support",
            ],
            "images": len(frame),
            "patients": int(frame.patient_uid.nunique()),
            "centers": int(frame.center_id.nunique()),
            "inference_rerun": False,
            "metrics_sha256": sha256(output / "metrics.json"),
            "prediction_sha256": sha256(output / "predictions.csv.gz"),
        },
    )
    write_table(expected, output / "image_manifest.csv.gz")
    return metrics


def summarize(metrics: dict[str, dict]) -> None:
    for split, m in metrics.items():
        directory = PACK / "tables" / split
        main = {
            "model": "SynAP-Fib E",
            "round": "AE_COR_v2",
            "seed": 2026,
            "epoch": 103,
            "frozen": True,
            "images": m["image"]["n"],
            "patients": m["patient_max"]["n"],
            "centers": m["center_balanced_patient_max"]["center_count"],
            "R_final": m["r_final"],
            "image_COR": m["image"]["cor"],
            "patient_max_COR": m["patient_max"]["cor"],
            "patient_median_COR": m["patient_median"]["cor"],
            "center_balanced_COR": m["center_balanced_patient_max"]["cor"],
        }
        grade_keys = [k for k in m["patient_max"]]
        grading = pd.DataFrame(
            [
                {"level": level, **{k: m[level][k] for k in grade_keys}}
                for level in ["image", "patient_max", "patient_median"]
            ]
        )
        probabilities = pd.DataFrame(
            [
                {"metric": k, "value": v}
                for k, v in m["image"].items()
                if any(term in k for term in ["auroc", "auprc", "brier", "ece"])
            ]
        )
        spatial = pd.DataFrame(
            [
                {"group": group, "metric": k, "value": v}
                for group in ["position", "lesion"]
                for k, v in m[group].items()
            ]
        )
        for name, table in [
            ("main", pd.DataFrame([main])),
            ("grading_all_levels", grading),
            ("image", grading[grading.level == "image"]),
            ("patients", grading[grading.level != "image"]),
            ("probabilities", probabilities),
            ("spatial", spatial),
        ]:
            write_table(table, directory / f"{name}.csv")
            (directory / f"{name}.md").write_text(
                markdown_table(table) + "\n", encoding="utf-8"
            )


def training_plots(complete: dict) -> None:
    history = pd.read_csv(PACK / "training/history.csv", float_precision="round_trip")
    if history.epoch.tolist() != list(range(1, 121)):
        raise ValueError("Incomplete training trajectory")
    eligible = history[history.epoch >= 21].sort_values(
        ["val__r_final", "val__image__cor", "epoch"]
    )
    if int(eligible.iloc[0].epoch) != complete["best_epoch"]:
        raise ValueError("Frozen selection mismatch")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = [
        [("train__total", "Total loss")],
        [
            ("val__r_final", "Validation R_final"),
            ("val__image__cor", "Validation image COR"),
        ],
        [
            ("train__grading", "Grading"),
            ("train__position", "View"),
            ("train__box_raw", "Weak-box raw"),
        ],
        [
            ("train__grading", "Grading"),
            ("train__weighted_position", "0.1 view"),
            ("train__weighted_box", "0.1 weak-box"),
        ],
    ]
    for ax, columns in zip(axes.flat, series):
        for column, name in columns:
            ax.plot(history.epoch, history[column], label=name, linewidth=1.3)
        ax.axvline(103, color="#c0392b", linestyle="--", label="Selected epoch 103")
        ax.set(xlabel="Epoch", xlim=(1, 120))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0, 0].set_title("Training total loss")
    axes[0, 1].set_title("Validation checkpoint selection")
    axes[1, 0].set_title("Unweighted loss components")
    axes[1, 1].set_title("Weighted loss contributions")
    fig.savefig(PACK / "training/training_curves.png", dpi=200)
    fig.savefig(PACK / "training/training_curves.pdf")
    plt.close(fig)
    write_json(
        PACK / "training/selection.json",
        {
            "training_epochs": 120,
            "selection_window": [21, 120],
            "best_epoch": 103,
            "best_val_R_final": complete["best_val"]["r_final"],
            "best_val_image_COR": complete["best_val"]["image"]["cor"],
            "tie_break": ["val_R_final", "val_image_COR", "earlier_epoch"],
            "amp_skipped_steps_sum": int(
                history.train__amp_overflow_skipped_steps.sum()
            ),
            "history_rows": len(history),
            "history_columns": len(history.columns),
        },
    )


def main() -> None:
    if (PACK / "provenance/copied_sources.json").exists():
        raise FileExistsError("Evidence package already built; do not overwrite")
    complete = read_json(RUN / "RUN_COMPLETE.json")
    if complete["status"] != "COMPLETE" or complete["best_epoch"] != 103:
        raise ValueError("Unexpected frozen run")
    copy_tree(ROUND / "round_snapshot", PACK / "source/round_snapshot")
    copy_tree(RUN / "checkpoints", PACK / "model/checkpoints")
    for name in ["task.json", "initialization.json", "RUN_COMPLETE.json"]:
        copy_verified(RUN / name, PACK / "provenance" / name)
    for name in ["SNAPSHOT_AMENDMENT_20260903_FP32_LESION.md"]:
        copy_verified(ROUND / name, PACK / "provenance" / name)
    for name in ["history.csv", "tensorboard", "val_epochs"]:
        src, dst = RUN / name, PACK / "training" / name
        copy_tree(src, dst) if src.is_dir() else copy_verified(src, dst)
    copy_tree(RUN / "last", PACK / "training/last_checkpoint_diagnostics")
    copy_verified(
        ROUND / "selection_trajectories/seed_2026_E.png",
        PACK / "training/original_selection_trajectory.png",
    )
    copy_verified(
        WORKSPACE / "data/数据清洗要点-人工标注.md",
        PACK / "provenance/view_names_source.md",
    )
    write_json(
        PACK / "provenance/view_names.json",
        {"source": "view_names_source.md, lines 17-31", "mapping": VIEW_NAMES},
    )
    manifest = pd.read_csv(DATA / "manifests/images.csv")
    task = read_json(RUN / "task.json")
    for name, key in [
        ("images.csv", "manifest_sha256"),
        ("annotations.jsonl", "annotations_sha256"),
    ]:
        if sha256(DATA / "manifests" / name) != task["data_audit"][key]:
            raise ValueError(f"Dataset hash mismatch: {name}")
    metrics = {
        split: export_split(split, complete, manifest) for split in ["test", "val"]
    }
    dest = PACK / "provenance/annotations_val_test.jsonl"
    with (DATA / "manifests/annotations.jsonl").open(
        encoding="utf-8"
    ) as source_stream, dest.open("w", encoding="utf-8") as destination_stream:
        for line in source_stream:
            row = json.loads(line)
            if row["split"] in {"val", "test"}:
                destination_stream.write(line)
    summarize(metrics)
    training_plots(complete)
    write_json(PACK / "provenance/copied_sources.json", COPIES)
    print(
        json.dumps(
            {
                "status": "EVIDENCE_EXPORTED",
                "copied_files": len(COPIES),
                "package": str(PACK),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
