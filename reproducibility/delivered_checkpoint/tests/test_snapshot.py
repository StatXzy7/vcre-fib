from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from sfibai_b.snapshot import (
    create_source_snapshot,
    runtime_environment_fingerprint,
    validate_runtime_environment,
    verify_source_snapshot,
)


def test_snapshot_copies_whitelist_hashes_every_file_and_is_immutable(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "src" / "pkg").mkdir(parents=True)
    (source / "configs").mkdir()
    (source / "scripts").mkdir()
    (source / "src" / "pkg" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "configs" / "formal.yaml").write_text("epochs: 120\n", encoding="utf-8")
    (source / "scripts" / "run.py").write_text("print('run')\n", encoding="utf-8")
    (source / "ignored.bin").write_bytes(b"not in whitelist")
    runtime = tmp_path / "runtime_selection.json"
    benchmark = tmp_path / "benchmark_results.json"
    benchmark.write_text(
        json.dumps(
            {
                "protocol": {"measured_batches": 300},
                "workers": {
                    "4": {"stable": True, "images_per_second": 1.0},
                    "6": {"stable": True, "images_per_second": 2.0},
                    "8": {"stable": True, "images_per_second": 1.5},
                },
                "backends": {
                    "eager": {"stable": True, "images_per_second": 2.0},
                    "compile": {
                        "stable": False,
                        "numerical_audit_ok": False,
                        "graph_breaks": -1,
                    },
                },
                "selection": {"num_workers": 6, "backend": "eager"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_sha256 = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    runtime.write_text(
        json.dumps(
            {
                "num_workers": 6,
                "backend": "eager",
                "benchmark_sha256": benchmark_sha256,
                "benchmark_results": str(benchmark),
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "round_snapshot"

    manifest = create_source_snapshot(
        source_root=source,
        destination=destination,
        runtime_selection_path=runtime,
        data_audit={"manifest_sha256": "b" * 64},
        environment={"python": "test"},
    )

    assert manifest["file_count"] == 5
    assert (destination / "src" / "pkg" / "model.py").is_file()
    assert (destination / "runtime_benchmark_results.json").is_file()
    assert not (destination / "ignored.bin").exists()
    verify_source_snapshot(destination)
    (destination / "src" / "pkg" / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_source_snapshot(destination)
    with pytest.raises(FileExistsError):
        create_source_snapshot(
            source_root=source,
            destination=destination,
            runtime_selection_path=runtime,
            data_audit={},
            environment={},
        )


def test_runtime_environment_fingerprint_is_order_stable_and_strict() -> None:
    expected = {
        "python": "3.10",
        "python_executable": "python.exe",
        "torch": "2.0",
        "torchvision": "0.15",
        "cuda": "12.8",
        "gpu": "RTX",
        "cudnn": 9000,
        "pip_freeze": ["b==2", "a==1"],
        "git_head": "ignored",
    }
    actual = dict(expected)
    actual["pip_freeze"] = ["a==1", "b==2"]

    assert runtime_environment_fingerprint(expected) == runtime_environment_fingerprint(actual)
    validate_runtime_environment(expected, actual=actual)
    actual["torch"] = "2.1"
    with pytest.raises(ValueError, match="environment fingerprint"):
        validate_runtime_environment(expected, actual=actual)
