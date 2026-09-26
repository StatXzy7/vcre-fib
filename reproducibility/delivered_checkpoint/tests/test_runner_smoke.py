from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sfibai_b.runner import RunLimits, run_task
from sfibai_b.training import RuntimeSelection, TaskSpec


@pytest.mark.smoke
def test_one_epoch_smoke_closes_train_val_and_best_last_test(tmp_path: Path) -> None:
    runtime = RuntimeSelection(
        num_workers=4,
        backend="eager",
        benchmark_sha256="f" * 64,
    )
    task = TaskSpec(
        arm="A",
        seed=2026,
        output_dir=tmp_path / "run",
        runtime=runtime,
        epochs=1,
        batch_size=8,
        pretrained=False,
        formal=False,
    )

    completion = run_task(
        task,
        limits=RunLimits(
            max_train_samples=16,
            max_val_samples=8,
            max_test_samples=8,
            limit_train_batches=1,
            limit_eval_batches=1,
            image_size=64,
        ),
        device_name="cuda",
    )

    root = Path(task.output_dir)
    assert completion["status"] == "COMPLETE"
    assert (root / "RUN_COMPLETE.json").is_file()
    assert (root / "checkpoints" / "best.pt").is_file()
    assert (root / "checkpoints" / "last.pt").is_file()
    assert (root / "val_epochs" / "epoch_001" / "predictions_compact.csv.gz").is_file()
    assert (root / "best" / "val" / "predictions_full.csv.gz").is_file()
    assert (root / "last" / "val" / "predictions_full.csv.gz").is_file()
    assert (root / "best" / "test" / "predictions_full.csv.gz").is_file()
    assert (root / "last" / "test" / "predictions_full.csv.gz").is_file()
    assert completion["last_test_was_run"] is True
    history = pd.read_csv(root / "history.csv")
    assert len(history) == 1
    assert history.loc[0, "train__amp_overflow_skipped_steps"] == 0.0
