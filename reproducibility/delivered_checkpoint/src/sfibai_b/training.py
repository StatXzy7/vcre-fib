from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from sfibai_b.protocol import (
    ARMS,
    BATCH_SIZE,
    EPOCHS,
    EXPERIMENT_ROOT,
    LEARNING_RATE,
    SCHEDULER_GAMMA,
    SCHEDULER_STEP_SIZE,
    SEEDS,
    WEIGHT_DECAY,
)


@dataclass(frozen=True)
class RuntimeSelection:
    num_workers: int
    backend: str
    benchmark_sha256: str

    def __post_init__(self) -> None:
        if self.num_workers not in {4, 6, 8}:
            raise ValueError("num_workers must be selected from 4, 6, or 8")
        if self.backend not in {"eager", "compile"}:
            raise ValueError("backend must be eager or compile")
        digest = self.benchmark_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("benchmark_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "benchmark_sha256", digest)

    @classmethod
    def from_json(cls, path: str | Path) -> "RuntimeSelection":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            num_workers=int(payload["num_workers"]),
            backend=str(payload["backend"]),
            benchmark_sha256=str(payload["benchmark_sha256"]),
        )


@dataclass(frozen=True)
class TaskSpec:
    arm: str
    seed: int
    output_dir: str | Path
    runtime: RuntimeSelection | Mapping[str, Any]
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    pretrained: bool = True
    early_stopping: bool = False
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    scheduler_step_size: int = SCHEDULER_STEP_SIZE
    scheduler_gamma: float = SCHEDULER_GAMMA
    formal: bool = True

    def __post_init__(self) -> None:
        normalized_arm = self.arm.upper()
        object.__setattr__(self, "arm", normalized_arm)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.runtime, RuntimeSelection):
            object.__setattr__(self, "runtime", RuntimeSelection(**dict(self.runtime)))
        if normalized_arm not in ARMS:
            raise ValueError(f"Unknown formal arm: {self.arm!r}")
        if self.formal:
            if self.seed not in SEEDS:
                raise ValueError(f"Formal seed must be one of {SEEDS}")
            if self.epochs != EPOCHS:
                raise ValueError("Formal tasks must train exactly 120 epochs")
            if self.batch_size != BATCH_SIZE:
                raise ValueError("Formal tasks must use batch size 24")
            if not self.pretrained:
                raise ValueError("Formal tasks require IMAGENET1K_V2 pretraining")
            if self.early_stopping:
                raise ValueError("Formal tasks cannot use early stopping")
            if self.learning_rate != LEARNING_RATE:
                raise ValueError("Formal tasks require AdamW learning rate 1e-4")
            if self.weight_decay != WEIGHT_DECAY:
                raise ValueError("Formal tasks require AdamW weight decay 1e-4")
            if self.scheduler_step_size != SCHEDULER_STEP_SIZE:
                raise ValueError("Formal tasks require StepLR step_size 15")
            if self.scheduler_gamma != SCHEDULER_GAMMA:
                raise ValueError("Formal tasks require StepLR gamma 0.6")
            expected_output = EXPERIMENT_ROOT / f"seed_{self.seed}" / self.arm
            if self.output_dir.resolve() != expected_output.resolve():
                raise ValueError(f"Formal output_dir must be {expected_output}")

    @classmethod
    def for_formal_run(
        cls, *, arm: str, seed: int, runtime: RuntimeSelection
    ) -> "TaskSpec":
        normalized_arm = arm.upper()
        return cls(
            arm=normalized_arm,
            seed=seed,
            output_dir=EXPERIMENT_ROOT / f"seed_{seed}" / normalized_arm,
            runtime=runtime,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_task_matrix(runtime: RuntimeSelection) -> list[TaskSpec]:
    return [
        TaskSpec.for_formal_run(arm=arm, seed=seed, runtime=runtime)
        for seed in SEEDS
        for arm in ARMS
    ]


def learning_rate_for_epoch(epoch: int) -> float:
    if not 1 <= epoch <= EPOCHS:
        raise ValueError(f"epoch must be in 1..{EPOCHS}")
    decay_count = (epoch - 1) // SCHEDULER_STEP_SIZE
    return LEARNING_RATE * (SCHEDULER_GAMMA**decay_count)
