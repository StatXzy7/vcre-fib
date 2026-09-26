"""A metadata reorder must never silently relabel fixed publication panels."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("case_binding", Path(__file__).resolve().parents[1] / "reproducibility/paper_tools/case_binding.py")
binding_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding_module)


@pytest.mark.parametrize("change", ["source", "metadata", "order"])
def test_case_source_binding_rejects_drift(tmp_path, change):
    source, metadata, binding = [tmp_path / name for name in ("panels.pdf", "metadata.json", "binding.json")]
    source.write_bytes(b"synthetic panel bytes")
    metadata.write_bytes(b'{"score":1.0}')
    cases = ("synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d")
    record = {"source_pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(), "case_ids": cases}
    binding.write_text(json.dumps(record), encoding="utf-8")
    binding_module.verify_case_binding(source, metadata, binding, cases)
    if change == "source":
        source.write_bytes(b"different synthetic panels")
    elif change == "metadata":
        metadata.write_bytes(b'{"score":2.0}')
    else:
        cases = tuple(reversed(cases))
    with pytest.raises(ValueError):
        binding_module.verify_case_binding(source, metadata, binding, cases)
