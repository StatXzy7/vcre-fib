"""Shared read-only evidence helpers; writes are restricted to the delivery root."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PACK = Path(os.environ.get("VCRE_DELIVERY_ROOT", "outputs/delivery")).resolve()
WORKSPACE = Path(os.environ.get("VCRE_WORKSPACE", Path.cwd())).resolve()
ROUND = WORKSPACE / "research-private/experiments/SFibAI-B_AE_COR_v2"
RUN = ROUND / "seed_2026/E"
DATA = WORKSPACE / "data/processed/schisto_2024_clean_v4"
VIEW_NAMES = {
    1: "剑突下矢状切面：肝左叶",
    2: "左肋下横切面：左叶和门脉矢状部",
    3: "右肋缘下切面：第二肝门及三支肝静脉",
    4: "右肋缘下切面：右肝、右肝静脉、膈肌",
    5: "右肋间斜切面：肝右叶、门静脉和胆囊",
    6: "右肋间斜切面：肝右叶和右肾",
}


def sha256(path: Path) -> str:
    """Hash a file without loading large checkpoints into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write strict JSON, so unhandled NaN cannot silently reach deliverables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, encoding="utf-8", compression=compression)


def grades(values: Any) -> np.ndarray:
    return np.asarray(
        np.searchsorted(
            [0.5, 1.5, 2.5], np.asarray(values, dtype=np.float64), side="right"
        ),
        dtype=np.int64,
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render without depending on optional tabulate."""

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "/").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)
