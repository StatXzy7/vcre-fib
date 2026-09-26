from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from sfibai_b.runner import (
    BEST_CHECKPOINT_FIELDS,
    LAST_CHECKPOINT_FIELDS,
    _train_one_epoch,
    atomic_save_checkpoint,
    capture_rng_state,
    guard_run_directory,
    restore_rng_state,
    validate_disk_history_matches_checkpoint,
    validate_epoch_metrics_match_history,
    validate_checkpoint_payload,
)


class _FiniteLossOverflowModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"finite_loss": self.weight * image.float().sum()}


class _TotalOnlyObjective(torch.nn.Module):
    def loss_components(
        self,
        outputs: dict[str, torch.Tensor],
        _batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {"total": outputs["finite_loss"]}


class _FiniteCounterObjective(torch.nn.Module):
    def loss_components(
        self,
        outputs: dict[str, torch.Tensor],
        _batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "total": outputs["finite_loss"] * 0.0 + 1.0,
            "valid_box_count": torch.tensor(32_768.0, dtype=torch.float16),
        }


def test_training_counter_metrics_use_epoch_totals_without_float16_overflow() -> None:
    device = torch.device("cpu")
    model = _FiniteLossOverflowModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    metrics = _train_one_epoch(
        forward_model=model,
        loader=[{"image": torch.ones(1)}] * 3,
        objective=_FiniteCounterObjective(),
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        amp=False,
        limit_batches=None,
    )

    assert metrics["total"] == 1.0
    assert metrics["valid_box_count"] == 98_304.0
    assert np.isfinite(metrics["valid_box_count"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is required")
def test_training_continues_and_records_an_amp_overflow_skip() -> None:
    device = torch.device("cuda")
    model = _FiniteLossOverflowModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=65536.0, growth_interval=2000
    )
    initial_weight = model.weight.detach().clone()

    metrics = _train_one_epoch(
        forward_model=model,
        loader=[{"image": torch.tensor([[1.0e35]], dtype=torch.float32)}],
        objective=_TotalOnlyObjective(),
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        amp=True,
        limit_batches=None,
    )

    assert metrics["amp_overflow_skipped_steps"] == 1.0
    assert scaler.get_scale() == 32768.0
    assert torch.equal(model.weight.detach(), initial_weight)


def test_rng_state_restores_python_numpy_torch_and_cuda_when_available() -> None:
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    state = capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, device="cuda").cpu() if torch.cuda.is_available() else None,
    )

    restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, device="cuda").cpu() if torch.cuda.is_available() else None,
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    if expected[3] is not None:
        assert torch.equal(actual[3], expected[3])


def test_run_directory_is_immutable_and_resume_requires_last_checkpoint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    guard_run_directory(run_dir, resume=False)
    (run_dir / "unexpected.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        guard_run_directory(run_dir, resume=False)
    with pytest.raises(FileNotFoundError, match="last.pt"):
        guard_run_directory(run_dir, resume=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    torch.save({}, checkpoints / "last.pt")
    guard_run_directory(run_dir, resume=True)
    (run_dir / "RUN_COMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="completed"):
        guard_run_directory(run_dir, resume=True)


def test_best_is_model_only_while_last_requires_full_training_state(tmp_path) -> None:
    payload = {field: {} for field in BEST_CHECKPOINT_FIELDS}
    payload["checkpoint_type"] = "best"
    payload["epoch"] = 1
    path = tmp_path / "best.pt"

    validate_checkpoint_payload(payload)
    atomic_save_checkpoint(payload, path)

    assert path.is_file()
    assert not (tmp_path / "best.pt.partial").exists()
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert set(restored) == BEST_CHECKPOINT_FIELDS
    assert "optimizer" not in restored

    full = {field: {} for field in LAST_CHECKPOINT_FIELDS}
    full["checkpoint_type"] = "last"
    full["epoch"] = 1
    validate_checkpoint_payload(full)
    incomplete = dict(full)
    incomplete.pop("optimizer")
    with pytest.raises(ValueError, match="optimizer"):
        validate_checkpoint_payload(incomplete)


def test_resume_history_must_match_checkpoint_rows_value_by_value() -> None:
    rows = [
        {"epoch": 1, "learning_rate": 1e-4, "val__r_final": 0.3, "note": None},
        {"epoch": 2, "learning_rate": 1e-4, "val__r_final": 0.2, "note": "ok"},
    ]
    frame = pd.DataFrame(rows)
    validate_disk_history_matches_checkpoint(frame, rows)
    changed = frame.copy()
    changed.loc[0, "val__r_final"] = 0.31
    with pytest.raises(ValueError, match="history differs"):
        validate_disk_history_matches_checkpoint(changed, rows)


def test_resume_epoch_metrics_must_match_the_corresponding_history_row() -> None:
    metrics = {"r_final": 0.2, "image": {"cor": 0.1, "n": 10}}
    row = pd.Series(
        {
            "epoch": 3,
            "val__r_final": 0.2,
            "val__image__cor": 0.1,
            "val__image__n": 10,
        }
    )
    validate_epoch_metrics_match_history(metrics, row, epoch=3)
    row["val__image__n"] = 9
    with pytest.raises(ValueError, match="epoch 3"):
        validate_epoch_metrics_match_history(metrics, row, epoch=3)
