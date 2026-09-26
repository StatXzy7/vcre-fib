from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AE_COR_v2 local runtime")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measured-batches", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        output.parent / f"{output.name}_torchinductor_cache"
    )
    from sfibai_b.benchmark import run_runtime_benchmark

    payload = run_runtime_benchmark(
        output_dir=output,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
    )
    print(json.dumps(payload["runtime_selection"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
