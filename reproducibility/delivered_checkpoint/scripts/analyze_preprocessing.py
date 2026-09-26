from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.preprocessing_analysis import write_preprocessing_analysis  # noqa: E402
from sfibai_b.protocol import EXPERIMENT_ROOT, IMAGES_CSV  # noqa: E402


def main() -> None:
    payload = write_preprocessing_analysis(
        IMAGES_CSV,
        EXPERIMENT_ROOT / "preflight" / "preprocessing_geometry",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "images": payload["all"]["images"],
                "output": str(
                    EXPERIMENT_ROOT / "preflight" / "preprocessing_geometry"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
