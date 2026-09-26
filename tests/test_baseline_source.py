"""Verify the pinned baseline imports using only the distributed source tree."""
import os
from pathlib import Path
import subprocess
import sys


def test_pinned_baseline_imports_without_weights_or_data():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(root / name) for name in (
        "code/src", "research/autosearch/src", "research/main_experiment",
    ))
    subprocess.run([
        sys.executable, "-c",
        "from runtime import upstream_module; "
        "module = upstream_module('third_party/SFibAI'); "
        "assert callable(module.calculate_loss); assert callable(module.create_model)",
    ], cwd=root, env=environment, check=True, timeout=120)
