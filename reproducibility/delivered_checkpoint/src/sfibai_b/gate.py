from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sfibai_b.evaluation import (
    checkpoint_key,
    flatten_metrics,
    is_checkpoint_eligible,
)
from sfibai_b.figures import validate_plot_ready_predictions
from sfibai_b.model import ARM_BRANCHES
from sfibai_b.protocol import ARMS, EPOCHS, EXPECTED_SPLITS
from sfibai_b.provenance import audit_seed_initialization
from sfibai_b.runner import (
    validate_checkpoint_payload,
    validate_disk_history_matches_checkpoint,
)
from sfibai_b.snapshot import verify_source_snapshot
from sfibai_b.storage import compare_metrics, load_metrics
from sfibai_b.tensorboard_schema import evaluation_tag, training_tag
from sfibai_b.training import TaskSpec, learning_rate_for_epoch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_history(history: pd.DataFrame, *, best_epoch: int) -> dict[str, Any]:
    required = {"epoch", "learning_rate", "val__r_final", "val__image__cor"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError("History columns are missing: " + ", ".join(missing))
    epochs = history["epoch"].astype(int).tolist()
    if epochs != list(range(1, EPOCHS + 1)):
        raise ValueError(f"History must contain exactly epochs 1 through {EPOCHS}")
    for row in history.itertuples(index=False):
        actual = float(getattr(row, "learning_rate"))
        expected = learning_rate_for_epoch(int(getattr(row, "epoch")))
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"History learning rate mismatch at epoch {getattr(row, 'epoch')}"
            )
    selected = min(
        [
            row
            for row in history.to_dict("records")
            if is_checkpoint_eligible(int(row["epoch"]), formal=True)
        ],
        key=lambda row: (
            float(row["val__r_final"]),
            float(row["val__image__cor"]),
            int(row["epoch"]),
        ),
    )
    if int(selected["epoch"]) != int(best_epoch):
        raise ValueError("Saved best epoch differs from the locked COR tie-breaker")
    return {"ok": True, "epochs": EPOCHS, "best_epoch": int(best_epoch)}


def validate_full_prediction_schema(
    frame: pd.DataFrame, *, arm: str, split: str
) -> None:
    validate_plot_ready_predictions(frame)
    required = {"split", "pred_bin", *(f"logit_{index:02d}" for index in range(36))}
    use_position, use_lesion = ARM_BRANCHES[arm]
    position_columns = {
        "position_true",
        "position_pred",
        "position_gate",
        *(f"position_prob_{position}" for position in range(1, 7)),
        *(f"position_logit_{position}" for position in range(1, 7)),
    }
    lesion_columns = {
        "lesion_valid",
        "lesion_gate",
        "lesion_box_present",
        "inside_attention",
        "outside_ratio",
        "inside_outside_ratio",
        "dice_at_0_5",
        "iou_at_0_5",
    }
    if use_position:
        required.update(position_columns)
    elif position_columns & set(frame.columns):
        raise ValueError(f"Arm {arm} unexpectedly contains position outputs")
    if use_lesion:
        required.update(lesion_columns)
    elif lesion_columns & set(frame.columns):
        raise ValueError(f"Arm {arm} unexpectedly contains lesion outputs")
    missing = sorted(required - set(frame.columns))
    if missing:
        branch = "position" if any(name.startswith("position") for name in missing) else "lesion" if any(name.startswith("lesion") for name in missing) else "grading"
        raise ValueError(f"Full {branch} prediction columns are missing: {', '.join(missing)}")
    if set(frame["split"].astype(str)) != {split}:
        raise ValueError(f"Prediction split column is not exactly {split}")


def _validate_compact_prediction_schema(frame: pd.DataFrame, *, arm: str) -> None:
    validate_plot_ready_predictions(frame)
    required = {"split", "pred_bin"}
    use_position, use_lesion = ARM_BRANCHES[arm]
    if use_position:
        required.update(
            {
                "position_true",
                "position_pred",
                "position_gate",
                *(f"position_prob_{position}" for position in range(1, 7)),
            }
        )
    if use_lesion:
        required.update(
            {
                "lesion_valid",
                "lesion_gate",
                "lesion_box_present",
                "inside_attention",
                "outside_ratio",
                "inside_outside_ratio",
                "dice_at_0_5",
                "iou_at_0_5",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Compact prediction columns are missing: " + ", ".join(missing))
    if set(frame["split"].astype(str)) != {"val"}:
        raise ValueError("Compact epoch prediction split must be val")


def _values_match(actual: Any, expected: Any) -> bool:
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        return pd.isna(actual)
    if isinstance(expected, bool):
        return bool(actual) == expected
    if isinstance(expected, (int, float, np.integer, np.floating)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    return str(actual) == str(expected)


def _audit_history_row(history_row: pd.Series, metrics: dict[str, Any], *, epoch: int) -> None:
    flattened = flatten_metrics(metrics, separator="__", prefix="val")
    for name, expected in flattened.items():
        if name not in history_row.index or not _values_match(history_row[name], expected):
            raise ValueError(f"Epoch {epoch} history differs from compact metrics at {name}")


def _tensorboard_scalars(tensorboard_root: Path) -> dict[str, list[Any]]:
    event_files = sorted(tensorboard_root.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"TensorBoard event file is missing: {tensorboard_root}")
    output: dict[str, list[Any]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            output.setdefault(tag, []).extend(accumulator.Scalars(tag))
    return output


def _require_tensorboard_value(
    scalars: dict[str, list[Any]], *, tag: str, step: int, expected: float
) -> None:
    matches = [event.value for event in scalars.get(tag, []) if int(event.step) == step]
    if not matches or not any(
        math.isclose(float(value), float(expected), rel_tol=1e-5, abs_tol=1e-6)
        for value in matches
    ):
        raise ValueError(f"TensorBoard scalar missing or inconsistent: {tag} step={step}")


def _audit_tensorboard(
    root: Path, history: pd.DataFrame, epoch_metrics: dict[int, dict[str, Any]]
) -> None:
    scalars = _tensorboard_scalars(root / "tensorboard")
    for epoch, metrics in epoch_metrics.items():
        for name, value in flatten_metrics(metrics, separator="/").items():
            if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
                continue
            if math.isfinite(float(value)):
                _require_tensorboard_value(
                    scalars,
                    tag=evaluation_tag(name),
                    step=epoch,
                    expected=float(value),
                )
        row = history.loc[history["epoch"].astype(int) == epoch].iloc[0]
        for column, value in row.items():
            if column.startswith("train__") and pd.notna(value):
                _require_tensorboard_value(
                    scalars,
                    tag=training_tag(column.removeprefix("train__")),
                    step=epoch,
                    expected=float(value),
                )
        _require_tensorboard_value(
            scalars,
            tag=training_tag("learning_rate"),
            step=epoch,
            expected=float(row["learning_rate"]),
        )


def _read_full_bundle(directory: Path, *, arm: str, split: str) -> dict[str, Any]:
    metrics = load_metrics(directory)
    prediction_path = directory / "predictions_full.csv.gz"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Full prediction file is missing: {prediction_path}")
    frame = pd.read_csv(prediction_path)
    validate_full_prediction_schema(frame, arm=arm, split=split)
    expected = EXPECTED_SPLITS[split]
    if len(frame) != expected["images"]:
        raise ValueError(f"{arm} {split} image count mismatch")
    if int(metrics["patient_max"]["n"]) != expected["patients"]:
        raise ValueError(f"{arm} {split} patient count mismatch")
    if int(metrics["center_balanced_patient_max"]["center_count"]) != expected["centers"]:
        raise ValueError(f"{arm} {split} center count mismatch")
    attention = directory / "lesion_attention_float16.npz"
    if ARM_BRANCHES[arm][1] != attention.is_file():
        raise ValueError(f"{arm} {split} attention archive presence mismatch")
    if attention.is_file():
        with np.load(attention) as payload:
            if payload["attention"].shape != (len(frame), 32, 32):
                raise ValueError(f"{arm} {split} attention shape must be N x 32 x 32")
    return metrics


def audit_completed_task(task_root: str | Path, *, arm: str, seed: int) -> dict[str, Any]:
    root = Path(task_root)
    completion_path = root / "RUN_COMPLETE.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"Completed marker is missing: {completion_path}")
    if list(root.rglob("*.partial")):
        raise ValueError(f"Partial files remain under completed task {root}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "COMPLETE"
        or completion.get("test_checkpoint") != "best"
        or completion.get("last_test_was_run") is not True
    ):
        raise ValueError(f"Completion state violates the best-ranked plus last-test protocol: {root}")
    task_record = json.loads((root / "task.json").read_text(encoding="utf-8"))
    task_payload = task_record["task"]
    reconstructed_task = TaskSpec(**task_payload)
    if (
        reconstructed_task.arm != arm
        or reconstructed_task.seed != seed
        or not reconstructed_task.formal
        or reconstructed_task.identity_sha256() != task_record["task_identity_sha256"]
    ):
        raise ValueError(f"Formal task identity mismatch under {root}")
    history = pd.read_csv(root / "history.csv")
    audit_history(history, best_epoch=int(completion["best_epoch"]))
    epoch_directories = sorted((root / "val_epochs").glob("epoch_*"))
    if [path.name for path in epoch_directories] != [f"epoch_{epoch:03d}" for epoch in range(1, EPOCHS + 1)]:
        raise ValueError(f"{arm} seed {seed} does not contain exactly {EPOCHS} val epochs")
    epoch_metrics: dict[int, dict[str, Any]] = {}
    for directory in epoch_directories:
        if not (directory / "predictions_compact.csv.gz").is_file() or not (directory / "metrics.json").is_file():
            raise FileNotFoundError(f"Compact validation bundle is incomplete: {directory}")
        epoch = int(directory.name.split("_")[-1])
        metrics = load_metrics(directory)
        frame = pd.read_csv(directory / "predictions_compact.csv.gz")
        _validate_compact_prediction_schema(frame, arm=arm)
        if len(frame) != EXPECTED_SPLITS["val"]["images"]:
            raise ValueError(f"{arm} epoch {epoch} compact val image count mismatch")
        if int(metrics["patient_max"]["n"]) != EXPECTED_SPLITS["val"]["patients"]:
            raise ValueError(f"{arm} epoch {epoch} compact val patient count mismatch")
        if (
            int(metrics["center_balanced_patient_max"]["center_count"])
            != EXPECTED_SPLITS["val"]["centers"]
        ):
            raise ValueError(f"{arm} epoch {epoch} compact val center count mismatch")
        _audit_history_row(history.loc[history["epoch"].astype(int) == epoch].iloc[0], metrics, epoch=epoch)
        epoch_metrics[epoch] = metrics
    _audit_tensorboard(root, history, epoch_metrics)
    checkpoint_files = sorted(root.rglob("*.pt"))
    expected_checkpoint_files = sorted(
        [root / "checkpoints" / "best.pt", root / "checkpoints" / "last.pt"]
    )
    if checkpoint_files != expected_checkpoint_files:
        raise ValueError("Completed task must contain exactly best.pt and last.pt")
    checkpoints: dict[str, dict[str, Any]] = {}
    for label in ("best", "last"):
        checkpoint_path = root / "checkpoints" / f"{label}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        validate_checkpoint_payload(checkpoint)
        if (
            checkpoint["checkpoint_type"] != label
            or checkpoint["task_identity_sha256"] != task_record["task_identity_sha256"]
        ):
            raise ValueError(f"{label} checkpoint type mismatch")
        checkpoints[label] = checkpoint
    best_checkpoint = checkpoints["best"]
    last_checkpoint = checkpoints["last"]
    best_epoch = int(completion["best_epoch"])
    if (
        int(last_checkpoint["epoch"]) != EPOCHS
        or int(last_checkpoint["best_epoch"]) != best_epoch
        or int(best_checkpoint["epoch"]) != best_epoch
        or tuple(last_checkpoint["best_key"]) != tuple(best_checkpoint["best_key"])
        or last_checkpoint["best_metrics"] != best_checkpoint["val_metrics"]
        or checkpoint_key(epoch_metrics[best_epoch], best_epoch)
        != tuple(last_checkpoint["best_key"])
    ):
        raise ValueError("Best/last checkpoint, history, and compact metrics are inconsistent")
    compare_metrics(
        best_checkpoint["val_metrics"],
        epoch_metrics[best_epoch],
        "checkpoint.best.val_metrics",
    )
    compare_metrics(
        last_checkpoint["val_metrics"],
        epoch_metrics[EPOCHS],
        "checkpoint.last.val_metrics",
    )
    validate_disk_history_matches_checkpoint(history, last_checkpoint["history_rows"])
    if _sha256(root / "checkpoints" / "best.pt") != completion["best_checkpoint_sha256"]:
        raise ValueError("Best checkpoint SHA-256 mismatch")
    if _sha256(root / "checkpoints" / "last.pt") != completion["last_checkpoint_sha256"]:
        raise ValueError("Last checkpoint SHA-256 mismatch")
    best_val = _read_full_bundle(root / "best" / "val", arm=arm, split="val")
    last_val = _read_full_bundle(root / "last" / "val", arm=arm, split="val")
    best_test = _read_full_bundle(root / "best" / "test", arm=arm, split="test")
    last_test = _read_full_bundle(root / "last" / "test", arm=arm, split="test")
    compare_metrics(completion["best_val"], best_val, "completion.best_val")
    compare_metrics(completion["last_val"], last_val, "completion.last_val")
    compare_metrics(completion["best_test"], best_test, "completion.best_test")
    compare_metrics(completion["last_test"], last_test, "completion.last_test")
    return {
        "ok": True,
        "arm": arm,
        "seed": seed,
        "best_epoch": int(completion["best_epoch"]),
        "test_r_final": float(best_test["r_final"]),
    }


def audit_seed_gate(experiment_root: str | Path, *, seed: int = 2026) -> dict[str, Any]:
    root = Path(experiment_root)
    snapshot = verify_source_snapshot(root / "round_snapshot")
    tasks = {
        arm: audit_completed_task(root / f"seed_{seed}" / arm, arm=arm, seed=seed)
        for arm in ARMS
    }
    pairing = audit_seed_initialization(root / f"seed_{seed}", seed=seed)
    return {
        "status": "PASS",
        "seed": seed,
        "snapshot_file_count": int(snapshot["file_count"]),
        "tasks": tasks,
        "paired_initialization": pairing,
        "performance_outcome_gate_applied": False,
        "runtime_benchmark_selection_verified": True,
    }
