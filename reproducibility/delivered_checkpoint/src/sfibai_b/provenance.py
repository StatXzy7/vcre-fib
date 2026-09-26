from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from sfibai_b.protocol import ARMS


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_initialization_record(
    model: torch.nn.Module, *, seed: int, arm: str
) -> dict[str, Any]:
    tensors = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for name, value in model.state_dict().items()
    }
    return {"format_version": 1, "seed": int(seed), "arm": arm.upper(), "tensors": tensors}


def save_initialization_record(
    model: torch.nn.Module, *, seed: int, arm: str, path: str | Path
) -> dict[str, Any]:
    target = Path(path)
    record = model_initialization_record(model, seed=seed, arm=arm)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError("Existing initialization record differs from rebuilt model")
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    return record


def _subset(record: Mapping[str, Any], prefixes: tuple[str, ...]) -> dict[str, str]:
    return {
        name: str(item["sha256"])
        for name, item in record["tensors"].items()
        if name.startswith(prefixes)
    }


def _same_named_tensors(
    records: Mapping[str, Mapping[str, Any]],
    arms: tuple[str, ...],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    subsets = {arm: _subset(records[arm], prefixes) for arm in arms}
    reference = subsets[arms[0]]
    mismatches: list[str] = []
    if not reference:
        mismatches.append("no matching tensors")
    for arm in arms[1:]:
        if set(subsets[arm]) != set(reference):
            mismatches.append(f"{arm}: tensor names differ")
            continue
        for name, digest in reference.items():
            if subsets[arm][name] != digest:
                mismatches.append(f"{arm}:{name}")
    return {
        "arms": list(arms),
        "prefixes": list(prefixes),
        "tensor_count": len(reference),
        "ok": not mismatches,
        "mismatches": mismatches,
    }


def audit_initialization_records(
    records: Mapping[str, Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    if set(records) != set(ARMS):
        raise ValueError(f"Initialization audit requires exactly arms {ARMS}")
    for arm in ARMS:
        if int(records[arm]["seed"]) != int(seed):
            raise ValueError(f"Initialization seed mismatch for arm {arm}")
    comparisons = {
        "shared_all_arms": _same_named_tensors(
            records, ARMS, ("backbone.", "grading_head.")
        ),
        "position_c_e": _same_named_tensors(
            records,
            ("C", "E"),
            ("position_head.", "position_residual.", "position_gate."),
        ),
        "lesion_d_e": _same_named_tensors(
            records,
            ("D", "E"),
            ("lesion_head.", "lesion_residual.", "lesion_gate."),
        ),
    }
    failures = [name for name, result in comparisons.items() if not result["ok"]]
    if failures:
        raise ValueError(
            "Strict paired initialization audit failed: " + ", ".join(failures)
        )
    return {"ok": True, "seed": int(seed), "comparisons": comparisons}


def audit_seed_initialization(seed_root: str | Path, *, seed: int) -> dict[str, Any]:
    root = Path(seed_root)
    records = {
        arm: json.loads((root / arm / "initialization.json").read_text(encoding="utf-8"))
        for arm in ARMS
    }
    return audit_initialization_records(records, seed=seed)
