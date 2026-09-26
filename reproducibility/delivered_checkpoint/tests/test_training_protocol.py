from __future__ import annotations

import json

import pytest

from sfibai_b.protocol import EXPERIMENT_ROOT
from sfibai_b.training import (
    RuntimeSelection,
    TaskSpec,
    build_task_matrix,
    learning_rate_for_epoch,
)


def test_formal_matrix_is_single_seed_a_to_e() -> None:
    runtime = RuntimeSelection(
        num_workers=6,
        backend="eager",
        benchmark_sha256="a" * 64,
    )

    tasks = build_task_matrix(runtime)

    assert len(tasks) == 5
    assert [(task.seed, task.arm) for task in tasks] == [
        (2026, "A"),
        (2026, "B"),
        (2026, "C"),
        (2026, "D"),
        (2026, "E"),
    ]
    assert tasks[0].output_dir == EXPERIMENT_ROOT / "seed_2026" / "A"


def test_formal_task_rejects_unlocked_training_values() -> None:
    runtime = RuntimeSelection(6, "eager", "b" * 64)
    task = TaskSpec.for_formal_run(arm="A", seed=2026, runtime=runtime)

    assert task.epochs == 120
    assert task.batch_size == 24
    assert task.pretrained
    assert not task.early_stopping
    with pytest.raises(ValueError, match="120 epochs"):
        TaskSpec(
            **{**task.to_dict(), "epochs": 30, "output_dir": str(task.output_dir)}
        )


def test_step_lr_keeps_fifteen_epoch_decay_across_120_epochs() -> None:
    assert learning_rate_for_epoch(1) == pytest.approx(1e-4)
    assert learning_rate_for_epoch(15) == pytest.approx(1e-4)
    assert learning_rate_for_epoch(16) == pytest.approx(6e-5)
    assert learning_rate_for_epoch(31) == pytest.approx(3.6e-5)
    assert learning_rate_for_epoch(46) == pytest.approx(2.16e-5)
    assert learning_rate_for_epoch(60) == pytest.approx(2.16e-5)
    assert learning_rate_for_epoch(61) == pytest.approx(1.296e-5)
    assert learning_rate_for_epoch(106) == pytest.approx(2.79936e-6)
    assert learning_rate_for_epoch(120) == pytest.approx(2.79936e-6)


def test_runtime_selection_requires_measured_backend_and_workers(tmp_path) -> None:
    path = tmp_path / "runtime_selection.json"
    path.write_text(
        json.dumps(
            {
                "num_workers": 8,
                "backend": "compile",
                "benchmark_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )

    selected = RuntimeSelection.from_json(path)

    assert selected.num_workers == 8
    assert selected.backend == "compile"
    with pytest.raises(ValueError):
        RuntimeSelection(5, "eager", "d" * 64)
    with pytest.raises(ValueError):
        RuntimeSelection(6, "automatic", "e" * 64)


def test_formal_task_factory_keeps_formal_boolean_and_identity_json_safe() -> None:
    runtime = RuntimeSelection(8, "eager", "f" * 64)
    task = TaskSpec.for_formal_run(arm="A", seed=2026, runtime=runtime)

    assert task.formal is True
    assert isinstance(task.to_dict()["formal"], bool)
    assert len(task.identity_sha256()) == 64
