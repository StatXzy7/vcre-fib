from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sfibai_b.protocol import ARMS, ROUND_NAME, SEEDS


PACK_ROOTS = (
    "round_snapshot",
    "preflight",
    "final_ranking",
    "selection_trajectories",
    "paper_plot_packs",
    f"seed_{SEEDS[0]}",
)
PACK_ROOT_FILES = ("QUEUE_STATE.json", f"SEED_{SEEDS[0]}_GATE.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reviewer_pack_entries(experiment_root: str | Path) -> list[str]:
    root = Path(experiment_root)
    paths = []
    for directory_name in PACK_ROOTS:
        directory = root / directory_name
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    paths.extend(
        root / name for name in PACK_ROOT_FILES if (root / name).is_file()
    )
    return sorted(path.relative_to(root).as_posix() for path in paths)


def write_reviewer_pack(experiment_root: str | Path) -> dict[str, Any]:
    root = Path(experiment_root)
    entries = reviewer_pack_entries(root)
    required = [
        "final_ranking/test_ranking.json",
        "final_ranking/paired_patient_cluster_bootstrap.json",
        "preflight/PREFLIGHT_COMPLETE.json",
        "preflight/preprocessing_geometry/geometry_summary.json",
        "preflight/model_cost/model_cost.json",
        "round_snapshot/snapshot_manifest.json",
        f"SEED_{SEEDS[0]}_GATE.json",
    ]
    missing = [path for path in required if path not in entries]
    for arm in ARMS:
        for relative in (
            f"seed_{SEEDS[0]}/{arm}/RUN_COMPLETE.json",
            f"seed_{SEEDS[0]}/{arm}/checkpoints/best.pt",
            f"seed_{SEEDS[0]}/{arm}/checkpoints/last.pt",
            f"selection_trajectories/seed_{SEEDS[0]}_{arm}.png",
            f"paper_plot_packs/seed_{SEEDS[0]}/{arm}/best_test/plot_manifest.json",
        ):
            if relative not in entries:
                missing.append(relative)
    if missing:
        raise FileNotFoundError("Reviewer pack evidence is incomplete: " + ", ".join(missing))

    records = [
        {
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": _sha256(root / relative),
        }
        for relative in entries
    ]
    payload = {
        "round": ROUND_NAME,
        "storage_mode": "reference_index_no_duplicate_copies",
        "artifact_count": len(records),
        "artifacts": records,
    }
    output = root / "paper_reviewer_pack"
    output.mkdir(parents=True, exist_ok=False)
    (output / "artifact_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        f"# {ROUND_NAME} paper / reviewer response pack",
        "",
        "本目录不复制大体积 prediction/checkpoint；`artifact_index.json` 以相对路径、大小和 SHA-256 索引全部证据。",
        "",
        "优先入口：",
        "",
        "- `../final_ranking/TEST_SET_MAIN_RESULTS.md`：test-first 主表与 paired bootstrap。",
        "- `../preflight/preprocessing_geometry/`：全量宽高比与 letterbox padding 证据。",
        "- `../preflight/model_cost/`：参数量、FLOPs、延迟和相对开销。",
        "- `../selection_trajectories/`：epoch 1–20 burn-in 与正式 best 选择轨迹。",
        "- `../paper_plot_packs/`：ROC/PR/校准/混淆/患者/中心绘图数据与总览图。",
        f"- `../seed_{SEEDS[0]}/<arm>/{{best,last}}/{{val,test}}/`：完整逐图预测和 canonical metrics。",
    ]
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return payload
