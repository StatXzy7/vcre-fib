from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from sfibai_b.evaluation import flatten_metrics
from sfibai_b.protocol import ARMS, SEEDS


CONTRASTS: tuple[tuple[str, str], ...] = (
    ("A", "B"),
    ("C", "A"),
    ("D", "A"),
    ("E", "A"),
    ("E", "C"),
    ("E", "D"),
)


def _arm_metrics(completion: Mapping[str, Any]) -> dict[str, float]:
    metrics = completion["best_test"]
    return {
        "image_cor": float(metrics["image"]["cor"]),
        "image_mae": float(metrics["image"]["mae"]),
        "patient_max_cor": float(metrics["patient_max"]["cor"]),
        "patient_median_cor": float(metrics["patient_median"]["cor"]),
        "center_balanced_patient_max_cor": float(
            metrics["center_balanced_patient_max"]["cor"]
        ),
    }


def rank_test_results(
    completions: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if len(SEEDS) != 1:
        raise ValueError("AE_COR_v2 ranking is locked to exactly one seed")
    seed = SEEDS[0]
    expected = {(arm, seed) for arm in ARMS}
    if set(completions) != expected:
        missing = sorted(expected - set(completions))
        extra = sorted(set(completions) - expected)
        raise ValueError(
            "Ranking requires complete seed 2026 A-E matrix; "
            f"missing={missing}, extra={extra}"
        )

    r_final = {
        arm: float(completions[(arm, seed)]["best_test"]["r_final"])
        for arm in ARMS
    }
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        completion = completions[(arm, seed)]
        value = r_final[arm]
        baseline = r_final["A"]
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "r_final": value,
                "last_test_r_final": float(completion["last_test"]["r_final"]),
                "candidate_minus_a": value - baseline,
                "relative_difference_vs_a": (
                    (value - baseline) / baseline if baseline != 0.0 else None
                ),
                "best_epoch": int(completion["best_epoch"]),
                "checkpoint_sha256": {
                    "best": str(completion["best_checkpoint_sha256"]),
                    "last": str(completion["last_checkpoint_sha256"]),
                },
                "best_test_metrics": dict(completion["best_test"]),
                "last_test_metrics": dict(completion["last_test"]),
                **_arm_metrics(completion),
            }
        )
    rows.sort(key=lambda row: (row["r_final"], row["image_cor"], row["arm"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    contrasts = []
    for candidate, reference in CONTRASTS:
        reference_value = r_final[reference]
        delta = r_final[candidate] - reference_value
        contrasts.append(
            {
                "name": f"{candidate}-{reference}",
                "candidate": candidate,
                "reference": reference,
                "candidate_minus_reference": delta,
                "relative_difference": (
                    delta / reference_value if reference_value != 0.0 else None
                ),
                "favorable": delta < 0.0,
            }
        )
    return {
        "round": "AE_COR_v2",
        "single_seed": seed,
        "primary_endpoint": "test best-checkpoint r_final",
        "direction": "lower_is_better",
        "official_ranking_uses": "best_test_only",
        "ranking": rows,
        "contrasts": contrasts,
    }


def load_and_rank(experiment_root: str | Path) -> dict[str, Any]:
    root = Path(experiment_root)
    seed = SEEDS[0]
    completions = {
        (arm, seed): json.loads(
            (root / f"seed_{seed}" / arm / "RUN_COMPLETE.json").read_text(
                encoding="utf-8"
            )
        )
        for arm in ARMS
    }
    return rank_test_results(completions)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_ranking_artifacts(experiment_root: str | Path) -> dict[str, Any]:
    root = Path(experiment_root)
    report = load_and_rank(root)
    output = root / "final_ranking"
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output / "test_ranking.json")

    ranking_rows = []
    metric_rows = []
    checkpoint_rows = []
    for row in report["ranking"]:
        ranking_rows.append(
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, dict)
            }
        )
        for checkpoint, checksum in row["checkpoint_sha256"].items():
            checkpoint_rows.append(
                {
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "checkpoint": checkpoint,
                    "sha256": checksum,
                    "best_epoch": row["best_epoch"],
                }
            )
        for checkpoint_name in ("best_test_metrics", "last_test_metrics"):
            for metric, value in flatten_metrics(
                row[checkpoint_name], separator="."
            ).items():
                metric_rows.append(
                    {
                        "arm": row["arm"],
                        "seed": row["seed"],
                        "checkpoint": checkpoint_name.removesuffix("_metrics"),
                        "metric": metric,
                        "value": value,
                    }
                )
    pd.DataFrame(ranking_rows).to_csv(
        output / "test_ranking.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(report["contrasts"]).to_csv(
        output / "observed_contrasts.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(metric_rows).to_csv(
        output / "test_metrics_long.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(checkpoint_rows).to_csv(
        output / "checkpoint_identity.csv", index=False, encoding="utf-8"
    )

    lines = [
        "# Test set 主结果",
        "",
        "预锁定主终点为 validation 选出的 best checkpoint 在独立 test 上的 R_final；越低越好。正式排名只使用 best test。",
        "",
        "> 本轮严格为单 seed（2026）完整比较，不将单次观测写成多 seed 稳定性结论。",
        "",
        "| 排名 | Arm | best epoch | best test R_final | last test R_final | vs A | image COR | patient-max COR | patient-median COR | center-balanced patient-max COR |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ranking"]:
        lines.append(
            f"| {row['rank']} | {row['arm']} | {row['best_epoch']} | "
            f"{row['r_final']:.6f} | {row['last_test_r_final']:.6f} | "
            f"{row['candidate_minus_a']:+.6f} | {row['image_cor']:.6f} | "
            f"{row['patient_max_cor']:.6f} | {row['patient_median_cor']:.6f} | "
            f"{row['center_balanced_patient_max_cor']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 预注册 contrasts（单 seed 观测差）",
            "",
            "candidate-reference 为负表示 candidate 的 R_final 更低。",
            "",
            "| Contrast | candidate-reference | 相对差 | 方向有利 |",
            "|:---:|---:|---:|:---:|",
        ]
    )
    for contrast in report["contrasts"]:
        relative = contrast["relative_difference"]
        relative_text = "N/A" if relative is None else f"{relative:.3%}"
        lines.append(
            f"| {contrast['name']} | {contrast['candidate_minus_reference']:+.6f} | "
            f"{relative_text} | {'是' if contrast['favorable'] else '否'} |"
        )
    bootstrap_path = output / "paired_patient_cluster_bootstrap.json"
    if bootstrap_path.is_file():
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Test paired bootstrap",
                "",
                "10,000 次 center-stratified patient-cluster paired bootstrap；共享抽样。95% CI 为 percentile interval。",
                "",
                "| Contrast | observed delta | bootstrap mean ± SD | 95% CI | P(candidate better) |",
                "|:---:|---:|---:|---:|---:|",
            ]
        )
        for item in bootstrap["results"]:
            lines.append(
                f"| {item['contrast']} | {item['observed_candidate_minus_reference']:+.6f} | "
                f"{item['bootstrap_mean']:+.6f} ± {item['bootstrap_sample_sd']:.6f} | "
                f"[{item['ci_95'][0]:+.6f}, {item['ci_95'][1]:+.6f}] | "
                f"{item['probability_candidate_better']:.3f} |"
            )
    lines.extend(
        [
            "",
            "完整 best/last test 标量见 `test_metrics_long.csv`；checkpoint 身份见 `checkpoint_identity.csv`；配对 patient-cluster bootstrap 在队列收尾阶段另行生成。",
        ]
    )
    (output / "TEST_SET_MAIN_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return report
