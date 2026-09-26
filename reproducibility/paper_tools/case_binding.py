"""Verify private case ordering and labels against their exact source panel PDF."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_case_binding(source: Path, metadata: Path, binding: Path, case_ids: tuple[str, ...]) -> None:
    record = json.loads(binding.read_text(encoding="utf-8"))
    if record["source_pdf_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Source case-panel PDF differs from its private binding")
    if record["metadata_sha256"] != hashlib.sha256(metadata.read_bytes()).hexdigest():
        raise ValueError("Case metadata differs from the source-panel binding")
    if tuple(record["case_ids"]) != case_ids:
        raise ValueError("Case ordering differs from the source-panel binding")
