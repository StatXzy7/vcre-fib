from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.protocol import ARMS  # noqa: E402
from sfibai_b.provenance import audit_seed_initialization  # noqa: E402
from sfibai_b.runner import RunLimits, run_task  # noqa: E402
from sfibai_b.training import RuntimeSelection, TaskSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A-E real-data one-batch preflight")
    parser.add_argument("--runtime-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"Smoke output must be new: {root}")
    runtime = RuntimeSelection.from_json(args.runtime_selection)
    results = {}
    for arm in ARMS:
        task = TaskSpec(
            arm=arm,
            seed=2026,
            output_dir=root / arm,
            runtime=runtime,
            epochs=1,
            batch_size=24,
            pretrained=True,
            formal=False,
        )
        results[arm] = run_task(
            task,
            limits=RunLimits(
                max_train_samples=24,
                max_val_samples=24,
                max_test_samples=24,
                limit_train_batches=1,
                limit_eval_batches=1,
                image_size=args.image_size,
            ),
            device_name="cuda",
        )
    pairing = audit_seed_initialization(root, seed=2026)
    report = {
        "status": "PASS",
        "arms": {arm: {"best_epoch": results[arm]["best_epoch"]} for arm in ARMS},
        "paired_initialization": pairing,
    }
    (root / "SMOKE_MATRIX_COMPLETE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
