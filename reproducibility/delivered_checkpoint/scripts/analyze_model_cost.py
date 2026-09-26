from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.model_cost import write_model_cost_report  # noqa: E402
from sfibai_b.protocol import EXPERIMENT_ROOT  # noqa: E402


def main() -> None:
    output = EXPERIMENT_ROOT / "preflight" / "model_cost"
    payload = write_model_cost_report(output)
    print(
        json.dumps(
            {"status": "PASS", "arms": len(payload["arms"]), "output": str(output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
