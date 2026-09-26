from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.data import preprocessing_for_arm, validate_data_v4  # noqa: E402
from sfibai_b.model import ARM_BRANCHES, RESNET50_PRETRAINED_WEIGHTS  # noqa: E402
from sfibai_b.protocol import (  # noqa: E402
    ANNOTATIONS_JSONL,
    ARMS,
    BEST_SELECTION_START_EPOCH,
    DATASET_ROOT,
    EPOCHS,
    EXPECTED_SPLITS,
    IMAGES_CSV,
    SEEDS,
)
from sfibai_b.training import learning_rate_for_epoch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the locked AE_COR_v2 protocol")
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = {
        "status": "PASS",
        "data": validate_data_v4(DATASET_ROOT, IMAGES_CSV, ANNOTATIONS_JSONL),
        "arms": {arm: ARM_BRANCHES[arm] for arm in ARMS},
        "preprocessing": {
            arm: preprocessing_for_arm(arm).resize_mode for arm in ARMS
        },
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "best_selection_start_epoch": BEST_SELECTION_START_EPOCH,
        "learning_rate_by_phase": {
            "1": learning_rate_for_epoch(1),
            "16": learning_rate_for_epoch(16),
            "31": learning_rate_for_epoch(31),
            "46": learning_rate_for_epoch(46),
            "60": learning_rate_for_epoch(60),
            "61": learning_rate_for_epoch(61),
            "106": learning_rate_for_epoch(106),
            "120": learning_rate_for_epoch(120),
        },
        "weights": str(RESNET50_PRETRAINED_WEIGHTS),
        "expected_splits": EXPECTED_SPLITS,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
