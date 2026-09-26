from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.queue import run_formal_queue  # noqa: E402


def main() -> None:
    argparse.ArgumentParser(
        description="Run the fixed-root AE_COR_v2 sequential 5090D queue"
    ).parse_args()
    result = run_formal_queue()
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
