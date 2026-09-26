from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from sfibai_b.data import validate_data_v4  # noqa: E402
from sfibai_b.protocol import (  # noqa: E402
    ANNOTATIONS_JSONL,
    DATASET_ROOT,
    EXPERIMENT_ROOT,
    IMAGES_CSV,
)
from sfibai_b.snapshot import capture_environment, create_source_snapshot  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the immutable AE_COR_v2 round snapshot")
    parser.add_argument("--runtime-selection", required=True)
    args = parser.parse_args()
    root = EXPERIMENT_ROOT.resolve()
    environment = capture_environment(SOURCE_ROOT)
    if environment["git_status"]:
        raise RuntimeError("Round snapshot requires a clean committed Git worktree")
    protocol_path = SOURCE_ROOT / "configs" / "ae_cor_v2_protocol.yaml"
    manifest = create_source_snapshot(
        source_root=SOURCE_ROOT,
        destination=root / "round_snapshot",
        runtime_selection_path=args.runtime_selection,
        data_audit=validate_data_v4(DATASET_ROOT, IMAGES_CSV, ANNOTATIONS_JSONL),
        environment=environment,
        protocol={
            "path": "configs/ae_cor_v2_protocol.yaml",
            "sha256": _sha256(protocol_path),
        },
    )
    print(json.dumps({"status": "PASS", "file_count": manifest["file_count"]}, indent=2))


if __name__ == "__main__":
    main()
