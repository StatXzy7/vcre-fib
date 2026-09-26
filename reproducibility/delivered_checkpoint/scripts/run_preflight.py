from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.protocol import EXPERIMENT_ROOT  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=SOURCE_ROOT,
        check=True,
    )


def main() -> None:
    root = EXPERIMENT_ROOT.resolve()
    preflight = root / "preflight"
    completion = preflight / "PREFLIGHT_COMPLETE.json"
    if completion.exists() or (root / "round_snapshot").exists():
        raise FileExistsError("AE_COR_v2 preflight/snapshot already exists")
    preflight.mkdir(parents=True, exist_ok=True)

    protocol_output = preflight / "protocol_validation.json"
    _run("scripts/validate_protocol.py", "--output", str(protocol_output))
    _run("scripts/analyze_preprocessing.py")
    _run(
        "-m",
        "pytest",
        "tests/test_data_protocol.py::test_max_grade_union_mask_follows_stretch_resize_and_rotation",
        "tests/test_models.py",
        "tests/test_loss_and_gradients.py",
        "tests/test_runner_contract.py",
        "tests/test_tensorboard_schema.py",
        "-q",
    )

    benchmark_root = preflight / "runtime_benchmark"
    _run("scripts/benchmark_runtime.py", "--output-dir", str(benchmark_root))
    _run("scripts/analyze_model_cost.py")
    runtime_selection = benchmark_root / "runtime_selection.json"
    smoke_root = preflight / "smoke_matrix"
    _run(
        "scripts/run_smoke_matrix.py",
        "--runtime-selection",
        str(runtime_selection),
        "--output-dir",
        str(smoke_root),
    )

    payload = {
        "status": "PASS",
        "round": "AE_COR_v2",
        "protocol_validation": str(protocol_output),
        "preprocessing_geometry": str(preflight / "preprocessing_geometry"),
        "runtime_selection": str(runtime_selection),
        "runtime_selection_sha256": _sha256(runtime_selection),
        "smoke_matrix": str(smoke_root / "SMOKE_MATRIX_COMPLETE.json"),
        "model_cost": str(preflight / "model_cost" / "model_cost.json"),
        "scientific_geometry_mismatch": False,
        "boundary_tolerance": "one pixel or one attention cell warns only",
    }
    completion.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
