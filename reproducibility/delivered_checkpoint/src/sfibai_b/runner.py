from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sfibai_b.data import (
    EpochShuffleSampler,
    FormalImageDataset,
    validate_data_v4,
)
from sfibai_b.evaluation import (
    checkpoint_key,
    evaluate_predictions,
    flatten_metrics,
    is_checkpoint_eligible,
)
from sfibai_b.loss import build_objective
from sfibai_b.model import ARM_BRANCHES, build_model
from sfibai_b.prediction import PredictionCollector
from sfibai_b.provenance import save_initialization_record
from sfibai_b.protocol import (
    ANNOTATIONS_JSONL,
    DATASET_ROOT,
    EXPECTED_SPLITS,
    EXPERIMENT_ROOT,
    IMAGES_CSV,
)
from sfibai_b.storage import (
    compare_metrics,
    load_metrics,
    save_prediction_bundle,
)
from sfibai_b.tensorboard_schema import (
    custom_scalars_layout,
    evaluation_tag,
    training_tag,
)
from sfibai_b.training import RuntimeSelection, TaskSpec, learning_rate_for_epoch


BEST_CHECKPOINT_FIELDS = {
    "format_version",
    "checkpoint_type",
    "task",
    "task_identity_sha256",
    "epoch",
    "model",
    "val_metrics",
    "best_key",
    "best_epoch",
}

LAST_CHECKPOINT_FIELDS = BEST_CHECKPOINT_FIELDS | {
    "optimizer",
    "scheduler",
    "scaler",
    "rng_state",
    "best_metrics",
    "history_rows",
}

TRAINING_COUNTER_METRICS = {
    "valid_box_count",
    "grade0_ignored_box_count",
    "missing_box_count",
}


@dataclass(frozen=True)
class RunLimits:
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None
    image_size: int = 512

    def __post_init__(self) -> None:
        values = (
            self.max_train_samples,
            self.max_val_samples,
            self.max_test_samples,
            self.limit_train_batches,
            self.limit_eval_batches,
        )
        if any(value is not None and value <= 0 for value in values):
            raise ValueError("Run limits must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    checkpoint_type = str(payload.get("checkpoint_type", ""))
    if checkpoint_type not in {"best", "last"}:
        raise ValueError("Checkpoint type must be best or last")
    required = (
        BEST_CHECKPOINT_FIELDS if checkpoint_type == "best" else LAST_CHECKPOINT_FIELDS
    )
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint fields are missing: {', '.join(missing)}")
    extra = sorted(set(payload) - required)
    if extra:
        raise ValueError(f"Checkpoint fields are unexpected: {', '.join(extra)}")
    if int(payload["epoch"]) <= 0:
        raise ValueError("Checkpoint epoch must be positive")


def atomic_save_checkpoint(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_checkpoint_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def guard_run_directory(directory: str | Path, *, resume: bool) -> None:
    root = Path(directory)
    if (root / "RUN_COMPLETE.json").exists():
        raise FileExistsError(f"Run is completed and immutable: {root}")
    if not root.exists():
        root.mkdir(parents=True)
        return
    entries = list(root.iterdir())
    if not entries:
        return
    if not resume:
        raise FileExistsError(f"Run directory is non-empty; refusing overwrite: {root}")
    if not (root / "checkpoints" / "last.pt").is_file():
        raise FileNotFoundError(f"Resume requires checkpoints/last.pt under {root}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_history(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def _worker_init(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(0)


def _make_loader(
    *,
    dataset: FormalImageDataset,
    task: TaskSpec,
    training: bool,
) -> tuple[DataLoader, EpochShuffleSampler | None]:
    sampler = (
        EpochShuffleSampler(size=len(dataset), seed=task.seed) if training else None
    )
    generator = torch.Generator()
    generator.manual_seed(task.seed + (0 if training else 50_000))
    loader = DataLoader(
        dataset,
        batch_size=task.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=task.runtime.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=_worker_init,
        generator=generator,
    )
    return loader, sampler


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _train_one_epoch(
    *,
    forward_model: torch.nn.Module,
    loader: DataLoader,
    objective: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    limit_batches: int | None,
) -> dict[str, float]:
    forward_model.train()
    totals: dict[str, torch.Tensor] = {}
    batches = 0
    amp_overflow_skipped_steps = 0
    for batch_index, cpu_batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        batch = _move_tensor_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = forward_model(batch["image"])
            components = objective.loss_components(outputs, batch)
            loss = components["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scale_before_step = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < scale_before_step:
            amp_overflow_skipped_steps += 1
        for name, value in components.items():
            if value.ndim == 0:
                detached = value.detach()
                if name in TRAINING_COUNTER_METRICS:
                    detached = detached.to(dtype=torch.float64)
                totals[name] = totals.get(
                    name,
                    torch.zeros_like(detached),
                ) + detached
        batches += 1
    if batches == 0:
        raise RuntimeError("Training epoch processed zero batches")
    metrics = {
        name: float(
            (value if name in TRAINING_COUNTER_METRICS else value / batches).cpu()
        )
        for name, value in totals.items()
    }
    metrics["amp_overflow_skipped_steps"] = float(amp_overflow_skipped_steps)
    return metrics


@torch.no_grad()
def _infer(
    *,
    forward_model: torch.nn.Module,
    loader: DataLoader,
    arm: str,
    device: torch.device,
    amp: bool,
    full: bool,
    limit_batches: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray] | None]:
    forward_model.eval()
    use_lesion = ARM_BRANCHES[arm][1]
    collector = PredictionCollector(
        arm=arm,
        full=full,
        collect_attention=full and use_lesion,
    )
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = forward_model(images)
        collector.add_batch(batch, outputs)
    frame = collector.to_frame()
    if frame.empty:
        raise RuntimeError("Evaluation produced zero predictions")
    if limit_batches is None and len(frame) != len(loader.dataset):
        raise RuntimeError(
            f"Evaluation prediction count mismatch: {len(frame)} != {len(loader.dataset)}"
        )
    return frame, evaluate_predictions(frame), collector.attention_payload()


def _tensorboard_metrics(
    writer: SummaryWriter,
    prefix: str,
    metrics: Mapping[str, Any],
    step: int,
) -> None:
    for name, value in flatten_metrics(metrics, separator="/").items():
        if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            writer.add_scalar(f"{prefix}/{name}", float(value), step)


def _tensorboard_evaluation_metrics(
    writer: SummaryWriter,
    metrics: Mapping[str, Any],
    step: int,
    *,
    split: str = "val",
) -> None:
    for name, value in flatten_metrics(metrics, separator="/").items():
        if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            writer.add_scalar(evaluation_tag(name, split=split), float(value), step)


def _best_checkpoint_payload(
    *,
    task: TaskSpec,
    epoch: int,
    model: torch.nn.Module,
    val_metrics: Mapping[str, Any],
    best_key: tuple[float, float, int],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "checkpoint_type": "best",
        "task": task.to_dict(),
        "task_identity_sha256": task.identity_sha256(),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "val_metrics": dict(val_metrics),
        "best_key": tuple(best_key),
        "best_epoch": int(epoch),
    }


def _last_checkpoint_payload(
    *,
    task: TaskSpec,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    val_metrics: Mapping[str, Any],
    best_key: tuple[float, float, int] | None,
    best_epoch: int,
    best_metrics: Mapping[str, Any],
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "checkpoint_type": "last",
        "task": task.to_dict(),
        "task_identity_sha256": task.identity_sha256(),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "val_metrics": dict(val_metrics),
        "best_key": tuple(best_key) if best_key is not None else None,
        "best_epoch": int(best_epoch),
        "best_metrics": dict(best_metrics),
        "history_rows": list(history_rows),
    }


def _validate_formal_snapshot(task: TaskSpec) -> None:
    if not task.formal:
        return
    snapshot_root = EXPERIMENT_ROOT / "round_snapshot"
    snapshot_manifest = snapshot_root / "snapshot_manifest.json"
    if not snapshot_manifest.is_file():
        raise FileNotFoundError(
            "Formal training requires the immutable round snapshot before task launch"
        )
    from sfibai_b.snapshot import validate_runtime_environment, verify_source_snapshot

    snapshot = verify_source_snapshot(snapshot_root)
    validate_runtime_environment(snapshot["environment"])
    runtime = RuntimeSelection.from_json(snapshot_root / "runtime_selection.json")
    if runtime != task.runtime:
        raise ValueError("Task runtime selection differs from the round snapshot")
    expected_runner = (snapshot_root / "src" / "sfibai_b" / "runner.py").resolve()
    if Path(__file__).resolve() != expected_runner:
        raise RuntimeError("Formal tasks must execute the source inside round_snapshot")


def _validate_history_for_resume(root: Path, checkpoint: Mapping[str, Any]) -> None:
    partials = list(root.rglob("*.partial"))
    if partials:
        raise ValueError(f"Strict resume refuses leftover partial files: {partials[0]}")
    history_path = root / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError("Resume checkpoint exists but history.csv is missing")
    history = pd.read_csv(history_path)
    epoch = int(checkpoint["epoch"])
    if len(history) != epoch or int(history.iloc[-1]["epoch"]) != epoch:
        raise ValueError("Resume history and last.pt epoch are inconsistent")
    validate_disk_history_matches_checkpoint(history, checkpoint["history_rows"])
    epoch_root = root / "val_epochs"
    directories = sorted(epoch_root.glob("epoch_*")) if epoch_root.exists() else []
    expected_names = [f"epoch_{value:03d}" for value in range(1, epoch + 1)]
    if [path.name for path in directories] != expected_names:
        raise ValueError("Resume requires exactly one compact validation bundle per saved epoch")
    epoch_metrics = {
        int(path.name.split("_")[-1]): load_metrics(path)
        for path in directories
    }
    for saved_epoch, metrics in epoch_metrics.items():
        history_row = history.loc[history["epoch"].astype(int) == saved_epoch].iloc[0]
        validate_epoch_metrics_match_history(
            metrics, history_row, epoch=saved_epoch
        )
    if checkpoint.get("checkpoint_type") != "last":
        raise ValueError("Strict resume requires checkpoint_type=last")
    compare_metrics(
        checkpoint["val_metrics"], epoch_metrics[epoch], "resume.last_val_metrics"
    )
    formal = bool(checkpoint["task"].get("formal", False))
    eligible_epochs = [
        saved_epoch
        for saved_epoch in epoch_metrics
        if is_checkpoint_eligible(saved_epoch, formal=formal)
    ]
    best_path = root / "checkpoints" / "best.pt"
    if not eligible_epochs:
        if best_path.exists():
            raise ValueError("best.pt must not exist before the first eligible epoch")
        if checkpoint["best_key"] is not None or int(checkpoint["best_epoch"]) != 0:
            raise ValueError("Last checkpoint contains an ineligible best selection")
        return
    if not best_path.is_file():
        raise FileNotFoundError("Strict resume requires checkpoints/best.pt")
    best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
    validate_checkpoint_payload(best)
    best_epoch = int(checkpoint["best_epoch"])
    if (
        best.get("checkpoint_type") != "best"
        or int(best["epoch"]) != best_epoch
        or best["task_identity_sha256"] != checkpoint["task_identity_sha256"]
        or tuple(best["best_key"]) != tuple(checkpoint["best_key"])
        or best["val_metrics"] != checkpoint["best_metrics"]
    ):
        raise ValueError("best.pt is inconsistent with last.pt resume state")
    if checkpoint_key(epoch_metrics[best_epoch], best_epoch) != tuple(
        checkpoint["best_key"]
    ):
        raise ValueError("Best compact prediction metrics differ from checkpoint state")
    compare_metrics(
        checkpoint["best_metrics"],
        epoch_metrics[best_epoch],
        "resume.best_val_metrics",
    )
    recomputed_best_epoch = min(
        eligible_epochs,
        key=lambda saved_epoch: checkpoint_key(
            epoch_metrics[saved_epoch], saved_epoch
        ),
    )
    if recomputed_best_epoch != best_epoch:
        raise ValueError("Best epoch differs after recomputing all compact bundles")


def validate_disk_history_matches_checkpoint(
    history: pd.DataFrame, checkpoint_rows: list[dict[str, Any]]
) -> None:
    expected = pd.DataFrame(checkpoint_rows)
    if list(history.columns) != list(expected.columns) or len(history) != len(expected):
        raise ValueError("On-disk history differs from checkpoint history rows")
    for column in history.columns:
        for actual, wanted in zip(history[column].tolist(), expected[column].tolist()):
            actual_missing = pd.isna(actual)
            wanted_missing = pd.isna(wanted)
            if actual_missing or wanted_missing:
                if actual_missing and wanted_missing:
                    continue
                raise ValueError("On-disk history differs from checkpoint history rows")
            if isinstance(wanted, (int, float, np.integer, np.floating)):
                if not math.isclose(
                    float(actual), float(wanted), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError("On-disk history differs from checkpoint history rows")
            elif str(actual) != str(wanted):
                raise ValueError("On-disk history differs from checkpoint history rows")


def validate_epoch_metrics_match_history(
    metrics: Mapping[str, Any], history_row: pd.Series, *, epoch: int
) -> None:
    flattened = flatten_metrics(metrics, separator="__", prefix="val")
    for column, expected in flattened.items():
        if column not in history_row.index:
            raise ValueError(f"Resume epoch {epoch} metrics are missing from history")
        actual = history_row[column]
        actual_missing = pd.isna(actual)
        expected_missing = expected is None or (
            isinstance(expected, float) and math.isnan(expected)
        )
        if actual_missing or expected_missing:
            if actual_missing and expected_missing:
                continue
            raise ValueError(f"Resume epoch {epoch} metrics differ from history")
        if isinstance(expected, bool):
            if bool(actual) != expected:
                raise ValueError(f"Resume epoch {epoch} metrics differ from history")
        elif isinstance(expected, (int, float)):
            if not math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"Resume epoch {epoch} metrics differ from history")
        elif str(actual) != str(expected):
            raise ValueError(f"Resume epoch {epoch} metrics differ from history")


def _load_checkpoint_into_training(
    *,
    checkpoint: Mapping[str, Any],
    task: TaskSpec,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[
    int,
    tuple[float, float, int] | None,
    int,
    dict[str, Any],
    list[dict[str, Any]],
]:
    validate_checkpoint_payload(checkpoint)
    if checkpoint["task_identity_sha256"] != task.identity_sha256():
        raise ValueError("Resume task identity differs from last.pt")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["epoch"]),
        tuple(checkpoint["best_key"]) if checkpoint["best_key"] is not None else None,
        int(checkpoint["best_epoch"]),
        dict(checkpoint["best_metrics"] or {}),
        list(checkpoint["history_rows"]),
    )


def _save_task_identity(root: Path, task: TaskSpec, data_audit: Mapping[str, Any]) -> None:
    task_path = root / "task.json"
    snapshot_manifest = EXPERIMENT_ROOT / "round_snapshot" / "snapshot_manifest.json"
    payload = {
        "task": task.to_dict(),
        "task_identity_sha256": task.identity_sha256(),
        "data_audit": data_audit,
        "execution_provenance": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__)),
            "round_snapshot_manifest_sha256": (
                _sha256(snapshot_manifest) if task.formal else None
            ),
            "runtime_environment_sha256": (
                json.loads(snapshot_manifest.read_text(encoding="utf-8"))["environment"]
                .get("runtime_environment_sha256")
                if task.formal
                else None
            ),
        },
    }
    if task_path.exists():
        existing = json.loads(task_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Existing task.json differs from the requested task")
    else:
        _atomic_json(payload, task_path)


def _final_evaluation(
    *,
    label: str,
    checkpoint_path: Path,
    task: TaskSpec,
    raw_model: torch.nn.Module,
    forward_model: torch.nn.Module,
    loader: DataLoader,
    root: Path,
    split: str,
    device: torch.device,
    amp: bool,
    limit_batches: int | None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(checkpoint)
    if checkpoint["task_identity_sha256"] != task.identity_sha256():
        raise ValueError(f"{label} checkpoint task identity differs from current task")
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    frame, metrics, attention = _infer(
        forward_model=forward_model,
        loader=loader,
        arm=task.arm,
        device=device,
        amp=amp,
        full=True,
        limit_batches=limit_batches,
    )
    bundle = root / label / split
    save_prediction_bundle(
        directory=bundle,
        frame=frame,
        metrics=metrics,
        attention=attention,
        full=True,
    )
    return metrics


def run_task(
    task: TaskSpec,
    *,
    resume: bool = False,
    limits: RunLimits | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    if task.formal and limits is not None:
        raise ValueError("Formal tasks cannot use smoke limits")
    _validate_formal_snapshot(task)
    root = Path(task.output_dir)
    guard_run_directory(root, resume=resume)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "val_epochs").mkdir(exist_ok=True)

    data_audit = validate_data_v4(DATASET_ROOT, IMAGES_CSV, ANNOTATIONS_JSONL)
    _save_task_identity(root, task, data_audit)
    set_global_seed(task.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if task.formal and device.type != "cuda":
        raise RuntimeError("Formal tasks require the local CUDA GPU")
    amp = device.type == "cuda"
    run_limits = limits or RunLimits()

    train_dataset = FormalImageDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=IMAGES_CSV,
        annotations_path=ANNOTATIONS_JSONL if ARM_BRANCHES[task.arm][1] else None,
        split="train",
        arm=task.arm,
        seed=task.seed,
        training=True,
        image_size=run_limits.image_size,
        max_samples=run_limits.max_train_samples,
    )
    val_dataset = FormalImageDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=IMAGES_CSV,
        annotations_path=ANNOTATIONS_JSONL if ARM_BRANCHES[task.arm][1] else None,
        split="val",
        arm=task.arm,
        seed=task.seed,
        training=False,
        image_size=run_limits.image_size,
        max_samples=run_limits.max_val_samples,
    )
    if task.formal:
        if len(train_dataset) != EXPECTED_SPLITS["train"]["images"]:
            raise RuntimeError("Formal train image count differs from Data V4 protocol")
        if len(val_dataset) != EXPECTED_SPLITS["val"]["images"]:
            raise RuntimeError("Formal val image count differs from Data V4 protocol")
    train_loader, train_sampler = _make_loader(
        dataset=train_dataset, task=task, training=True
    )
    val_loader, _ = _make_loader(dataset=val_dataset, task=task, training=False)
    assert train_sampler is not None

    raw_model = build_model(
        arm=task.arm, seed=task.seed, pretrained=task.pretrained
    ).to(device)
    save_initialization_record(
        raw_model,
        seed=task.seed,
        arm=task.arm,
        path=root / "initialization.json",
    )
    objective = build_objective(task.arm).to(device)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=task.learning_rate,
        weight_decay=task.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=task.scheduler_step_size,
        gamma=task.scheduler_gamma,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)

    completed_epoch = 0
    best_key: tuple[float, float, int] | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    if resume:
        checkpoint = torch.load(
            root / "checkpoints" / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        _validate_history_for_resume(root, checkpoint)
        (
            completed_epoch,
            restored_key,
            best_epoch,
            best_metrics,
            history_rows,
        ) = _load_checkpoint_into_training(
            checkpoint=checkpoint,
            task=task,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        best_key = restored_key

    forward_model: torch.nn.Module = raw_model
    if task.runtime.backend == "compile":
        forward_model = torch.compile(raw_model)

    writer = SummaryWriter(
        log_dir=str(root / "tensorboard"),
        purge_step=completed_epoch + 1 if completed_epoch else None,
    )
    writer.add_custom_scalars(custom_scalars_layout())
    writer.add_text("protocol/task", json.dumps(task.to_dict(), sort_keys=True), 0)
    try:
        for epoch in range(completed_epoch + 1, task.epochs + 1):
            current_lr = float(optimizer.param_groups[0]["lr"])
            expected_lr = learning_rate_for_epoch(epoch) if task.formal else current_lr
            if not math.isclose(current_lr, expected_lr, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"Learning rate mismatch at epoch {epoch}: {current_lr} != {expected_lr}"
                )
            train_sampler.set_epoch(epoch)
            train_metrics = _train_one_epoch(
                forward_model=forward_model,
                loader=train_loader,
                objective=objective,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp=amp,
                limit_batches=run_limits.limit_train_batches,
            )
            val_frame, val_metrics, _attention = _infer(
                forward_model=forward_model,
                loader=val_loader,
                arm=task.arm,
                device=device,
                amp=amp,
                full=False,
                limit_batches=run_limits.limit_eval_batches,
            )
            epoch_bundle = root / "val_epochs" / f"epoch_{epoch:03d}"
            save_prediction_bundle(
                directory=epoch_bundle,
                frame=val_frame,
                metrics=val_metrics,
                attention=None,
                full=False,
            )

            row: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": current_lr,
                **{f"train__{key}": value for key, value in train_metrics.items()},
                **flatten_metrics(val_metrics, separator="__", prefix="val"),
            }
            history_rows.append(row)
            _atomic_history(history_rows, root / "history.csv")
            for name, value in train_metrics.items():
                if math.isfinite(value):
                    writer.add_scalar(training_tag(name), value, epoch)
            writer.add_scalar(training_tag("learning_rate"), current_lr, epoch)
            _tensorboard_evaluation_metrics(writer, val_metrics, epoch)

            eligible = is_checkpoint_eligible(epoch, formal=task.formal)
            candidate_key = checkpoint_key(val_metrics, epoch)
            is_best = eligible and (best_key is None or candidate_key < best_key)
            if is_best:
                best_key = candidate_key
                best_epoch = epoch
                best_metrics = dict(val_metrics)
            scheduler.step()
            last_payload = _last_checkpoint_payload(
                task=task,
                epoch=epoch,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                val_metrics=val_metrics,
                best_key=best_key,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                history_rows=history_rows,
            )
            if is_best:
                assert best_key is not None
                best_payload = _best_checkpoint_payload(
                    task=task,
                    epoch=epoch,
                    model=raw_model,
                    val_metrics=val_metrics,
                    best_key=best_key,
                )
                atomic_save_checkpoint(
                    best_payload, root / "checkpoints" / "best.pt"
                )
            atomic_save_checkpoint(last_payload, root / "checkpoints" / "last.pt")
            writer.flush()

        best_path = root / "checkpoints" / "best.pt"
        last_path = root / "checkpoints" / "last.pt"
        if not best_path.is_file():
            raise RuntimeError("Training completed without an eligible best checkpoint")
        best_val = _final_evaluation(
            label="best",
            checkpoint_path=best_path,
            task=task,
            raw_model=raw_model,
            forward_model=forward_model,
            loader=val_loader,
            root=root,
            split="val",
            device=device,
            amp=amp,
            limit_batches=run_limits.limit_eval_batches,
        )
        last_val = _final_evaluation(
            label="last",
            checkpoint_path=last_path,
            task=task,
            raw_model=raw_model,
            forward_model=forward_model,
            loader=val_loader,
            root=root,
            split="val",
            device=device,
            amp=amp,
            limit_batches=run_limits.limit_eval_batches,
        )
        test_dataset = FormalImageDataset(
            dataset_root=DATASET_ROOT,
            manifest_path=IMAGES_CSV,
            annotations_path=ANNOTATIONS_JSONL if ARM_BRANCHES[task.arm][1] else None,
            split="test",
            arm=task.arm,
            seed=task.seed,
            training=False,
            image_size=run_limits.image_size,
            max_samples=run_limits.max_test_samples,
        )
        if task.formal and len(test_dataset) != EXPECTED_SPLITS["test"]["images"]:
            raise RuntimeError("Formal test image count differs from Data V4 protocol")
        test_loader, _ = _make_loader(dataset=test_dataset, task=task, training=False)
        best_test = _final_evaluation(
            label="best",
            checkpoint_path=best_path,
            task=task,
            raw_model=raw_model,
            forward_model=forward_model,
            loader=test_loader,
            root=root,
            split="test",
            device=device,
            amp=amp,
            limit_batches=run_limits.limit_eval_batches,
        )
        last_test = _final_evaluation(
            label="last",
            checkpoint_path=last_path,
            task=task,
            raw_model=raw_model,
            forward_model=forward_model,
            loader=test_loader,
            root=root,
            split="test",
            device=device,
            amp=amp,
            limit_batches=run_limits.limit_eval_batches,
        )
        _tensorboard_metrics(writer, "final/best_val", best_val, task.epochs)
        _tensorboard_metrics(writer, "final/last_val", last_val, task.epochs)
        _tensorboard_metrics(writer, "final/best_test", best_test, task.epochs)
        _tensorboard_metrics(writer, "final/last_test", last_test, task.epochs)
        writer.flush()

        completion = {
            "status": "COMPLETE",
            "task_identity_sha256": task.identity_sha256(),
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": _sha256(best_path),
            "last_checkpoint_sha256": _sha256(last_path),
            "best_val": best_val,
            "last_val": last_val,
            "best_test": best_test,
            "last_test": last_test,
            "test_checkpoint": "best",
            "last_test_was_run": True,
        }
        _atomic_json(completion, root / "RUN_COMPLETE.json")
        return completion
    finally:
        writer.close()
