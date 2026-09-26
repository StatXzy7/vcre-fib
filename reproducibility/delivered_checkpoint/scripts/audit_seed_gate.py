from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.gate import audit_seed_gate  # noqa: E402
from sfibai_b.protocol import EXPERIMENT_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the seed-2026 hard release gate")
    parser.add_argument("--experiment-root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = audit_seed_gate(args.experiment_root, seed=args.seed)
    output = Path(args.experiment_root) / f"SEED_{args.seed}_GATE.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "seed": result["seed"]}, indent=2))


if __name__ == "__main__":
    main()
