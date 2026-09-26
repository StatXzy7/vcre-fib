from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.runner import run_task  # noqa: E402
from sfibai_b.training import RuntimeSelection, TaskSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one immutable AE_COR_v2 task")
    parser.add_argument("--arm", choices=list("ABCDE"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-selection", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runtime = RuntimeSelection.from_json(args.runtime_selection)
    task = TaskSpec.for_formal_run(arm=args.arm, seed=args.seed, runtime=runtime)
    result = run_task(task, resume=args.resume)
    print(json.dumps({"status": result["status"], "best_epoch": result["best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
