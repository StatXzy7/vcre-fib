from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _atomic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_npz(payload: Mapping[str, np.ndarray], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def compare_metrics(expected: Any, actual: Any, path: str = "metrics") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"Metric key mismatch at {path}")
        for key in expected:
            compare_metrics(expected[key], actual[key], f"{path}.{key}")
        return
    if expected is None or actual is None:
        if expected is not actual:
            raise ValueError(f"Metric null mismatch at {path}")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            raise ValueError(f"Metric boolean mismatch at {path}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Metric numeric mismatch at {path}: {expected} != {actual}")
        return
    if expected != actual:
        raise ValueError(f"Metric value mismatch at {path}: {expected!r} != {actual!r}")


def save_prediction_bundle(
    *,
    directory: str | Path,
    frame: pd.DataFrame,
    metrics: Mapping[str, Any],
    attention: Mapping[str, np.ndarray] | None,
    full: bool,
) -> None:
    if frame.empty or frame["image_uid"].astype(str).duplicated().any():
        raise ValueError("Prediction bundles require unique non-empty image rows")
    if "r_final" not in metrics or not math.isfinite(float(metrics["r_final"])):
        raise ValueError("Canonical metrics require a finite r_final")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    filename = "predictions_full.csv.gz" if full else "predictions_compact.csv.gz"
    _atomic_csv_gzip(frame, output / filename)
    _atomic_json(metrics, output / "metrics.json")
    if attention is not None:
        if not full:
            raise ValueError("Attention maps cannot be stored in a compact bundle")
        if len(attention["image_uid"]) != len(frame):
            raise ValueError("Attention payload does not align with prediction rows")
        if list(map(str, attention["image_uid"])) != frame["image_uid"].astype(str).tolist():
            raise ValueError("Attention image_uid order differs from prediction order")
        _atomic_npz(attention, output / "lesion_attention_float16.npz")


def load_metrics(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    full_path = root / "predictions_full.csv.gz"
    compact_path = root / "predictions_compact.csv.gz"
    prediction_path = full_path if full_path.is_file() else compact_path
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction bundle is missing under {root}")
    metrics_path = root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json is missing under {root}")
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "r_final" not in saved or not math.isfinite(float(saved["r_final"])):
        raise ValueError("Saved canonical metrics require a finite r_final")
    return saved
