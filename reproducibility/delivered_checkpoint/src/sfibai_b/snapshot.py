from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


RUNTIME_ENVIRONMENT_KEYS = (
    "python",
    "python_executable",
    "torch",
    "torchvision",
    "cuda",
    "gpu",
    "cudnn",
    "pip_freeze",
)


SNAPSHOT_DIRECTORIES = ("src", "configs", "scripts", "tests", "docs")
SNAPSHOT_ROOT_FILES = (
    ".gitignore",
    "AGENTS.md",
    "ALGORITHM_EVALUATION_STANDARD.md",
    "EVALUATION_POLICY.md",
    "README.md",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
)
SKIP_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_FILE_SUFFIXES = {".pyc", ".pyo", ".tmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = _sha256(source)
    target_hash = _sha256(target)
    if source_hash != target_hash:
        raise RuntimeError(f"Snapshot copy SHA-256 mismatch: {source} -> {target}")
    return {
        "path": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": target_hash,
    }


def _source_files(source_root: Path):
    for directory_name in SNAPSHOT_DIRECTORIES:
        directory = source_root / directory_name
        if not directory.is_dir():
            continue
        for root, dirs, files in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRECTORY_NAMES]
            root_path = Path(root)
            for name in files:
                path = root_path / name
                if path.suffix.lower() not in SKIP_FILE_SUFFIXES:
                    yield path, path.relative_to(source_root)
    for name in SNAPSHOT_ROOT_FILES:
        path = source_root / name
        if path.is_file():
            yield path, Path(name)


def create_source_snapshot(
    *,
    source_root: str | Path,
    destination: str | Path,
    runtime_selection_path: str | Path,
    data_audit: Mapping[str, Any],
    environment: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    target = Path(destination).resolve()
    runtime_selection = Path(runtime_selection_path).resolve()
    if target.exists():
        raise FileExistsError(f"Snapshot destination already exists: {target}")
    if not source.is_dir() or not runtime_selection.is_file():
        raise FileNotFoundError("Snapshot source or runtime selection is missing")
    runtime_payload = json.loads(runtime_selection.read_text(encoding="utf-8"))
    benchmark_results = Path(runtime_payload.get("benchmark_results", "")).resolve()
    if not benchmark_results.is_file():
        raise FileNotFoundError("Runtime selection must reference benchmark_results.json")
    if _sha256(benchmark_results) != runtime_payload.get("benchmark_sha256"):
        raise ValueError("Runtime benchmark SHA-256 differs from runtime selection")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Incomplete snapshot already exists: {temporary}")
    temporary.mkdir()
    records: list[dict[str, Any]] = []
    try:
        for path, relative in _source_files(source):
            record = _copy_verified(path, temporary / relative)
            record["path"] = relative.as_posix()
            records.append(record)
        runtime_record = _copy_verified(
            runtime_selection, temporary / "runtime_selection.json"
        )
        runtime_record["path"] = "runtime_selection.json"
        records.append(runtime_record)
        benchmark_record = _copy_verified(
            benchmark_results, temporary / "runtime_benchmark_results.json"
        )
        benchmark_record["path"] = "runtime_benchmark_results.json"
        records.append(benchmark_record)
        records.sort(key=lambda row: row["path"])
        manifest = {
            "format_version": 1,
            "source_root": str(source),
            "destination": str(target),
            "file_count": len(records),
            "total_bytes": sum(int(record["bytes"]) for record in records),
            "files": records,
            "data_audit": dict(data_audit),
            "environment": dict(environment),
            "protocol": dict(protocol or {}),
        }
        (temporary / "snapshot_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, target)
        verify_source_snapshot(target)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_source_snapshot(destination: str | Path) -> dict[str, Any]:
    root = Path(destination)
    manifest_path = root / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            raise ValueError(f"Snapshot file is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Snapshot size mismatch: {path}")
        actual_hash = _sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"Snapshot SHA-256 mismatch: {path}")
    runtime = json.loads((root / "runtime_selection.json").read_text(encoding="utf-8"))
    benchmark = root / "runtime_benchmark_results.json"
    if _sha256(benchmark) != runtime.get("benchmark_sha256"):
        raise ValueError("Snapshot runtime benchmark provenance mismatch")
    benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
    if int(benchmark_payload.get("protocol", {}).get("measured_batches", 0)) < 300:
        raise ValueError("Snapshot runtime benchmark is shorter than 300 measured batches")
    from sfibai_b.benchmark import select_runtime

    selected = select_runtime(benchmark_payload)
    if selected != {
        "num_workers": int(runtime["num_workers"]),
        "backend": str(runtime["backend"]),
    }:
        raise ValueError("Snapshot runtime selection cannot be reproduced from benchmark")
    return manifest


def capture_runtime_environment() -> dict[str, Any]:
    import torch
    import torchvision

    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(),
        "pip_freeze": sorted(pip_freeze),
    }


def runtime_environment_fingerprint(environment: Mapping[str, Any]) -> str:
    payload = {key: environment.get(key) for key in RUNTIME_ENVIRONMENT_KEYS}
    payload["pip_freeze"] = sorted(payload.get("pip_freeze") or [])
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_runtime_environment(
    expected: Mapping[str, Any], *, actual: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    observed = dict(actual or capture_runtime_environment())
    expected_fingerprint = runtime_environment_fingerprint(expected)
    actual_fingerprint = runtime_environment_fingerprint(observed)
    if actual_fingerprint != expected_fingerprint:
        mismatches = [
            key
            for key in RUNTIME_ENVIRONMENT_KEYS
            if (sorted(expected.get(key) or []) if key == "pip_freeze" else expected.get(key))
            != (sorted(observed.get(key) or []) if key == "pip_freeze" else observed.get(key))
        ]
        raise ValueError(
            "Runtime environment fingerprint differs from round snapshot: "
            + ", ".join(mismatches)
        )
    return {"sha256": actual_fingerprint, "environment": observed}


def capture_environment(repository_root: str | Path) -> dict[str, Any]:
    repository = Path(repository_root)
    runtime = capture_runtime_environment()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    return {
        **runtime,
        "runtime_environment_sha256": runtime_environment_fingerprint(runtime),
        "git_head": git_head,
        "git_status": git_status,
    }
