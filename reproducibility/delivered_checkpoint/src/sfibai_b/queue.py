from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sfibai_b.bootstrap import write_paired_bootstrap_artifacts
from sfibai_b.figures import (
    generate_plot_pack,
    generate_selection_trajectory,
    verify_plot_pack,
)
from sfibai_b.gate import audit_completed_task, audit_seed_gate
from sfibai_b.protocol import ARMS, EXPERIMENT_ROOT, SEEDS
from sfibai_b.ranking import write_ranking_artifacts
from sfibai_b.reviewer_pack import write_reviewer_pack
from sfibai_b.snapshot import verify_source_snapshot


def formal_task_order() -> list[tuple[str, int]]:
    return [(arm, seed) for seed in SEEDS for arm in ARMS]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _run_one(*, snapshot: Path, root: Path, arm: str, seed: int) -> None:
    task_root = root / f"seed_{seed}" / arm
    complete = task_root / "RUN_COMPLETE.json"
    if complete.is_file():
        audit_completed_task(task_root, arm=arm, seed=seed)
        return
    resume = task_root.exists() and any(task_root.iterdir())
    if resume and not (task_root / "checkpoints" / "last.pt").is_file():
        raise FileNotFoundError(
            f"Non-empty task directory cannot strictly resume without last.pt: {task_root}"
        )
    command = [
        sys.executable,
        str(snapshot / "scripts" / "run_task.py"),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--runtime-selection",
        str(snapshot / "runtime_selection.json"),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(snapshot / "src")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(root / "runtime_cache" / "torchinductor")
    recovery_marker = task_root / "AUTO_STRICT_RESUME_USED.json"
    if resume:
        if recovery_marker.is_file():
            raise RuntimeError(
                f"Automatic strict resume was already used for {arm} seed {seed}"
            )
        _atomic_json(
            {"arm": arm, "seed": seed, "reason": "preexisting_incomplete_task", "used_at": _now()},
            recovery_marker,
        )
        command.append("--resume")
        subprocess.run(command, cwd=snapshot, env=environment, check=True)
    else:
        try:
            subprocess.run(command, cwd=snapshot, env=environment, check=True)
        except subprocess.CalledProcessError:
            last_checkpoint = task_root / "checkpoints" / "last.pt"
            if not last_checkpoint.is_file() or recovery_marker.is_file():
                raise
            _atomic_json(
                {"arm": arm, "seed": seed, "reason": "transient_process_exit", "used_at": _now()},
                recovery_marker,
            )
            subprocess.run(
                [*command, "--resume"],
                cwd=snapshot,
                env=environment,
                check=True,
            )
    audit_completed_task(task_root, arm=arm, seed=seed)


def run_formal_queue() -> dict[str, Any]:
    root = EXPERIMENT_ROOT.resolve()
    snapshot = root / "round_snapshot"
    verify_source_snapshot(snapshot)
    state_path = root / "QUEUE_STATE.json"
    tasks = formal_task_order()
    state: dict[str, Any] = {
        "status": "RUNNING",
        "round": "AE_COR_v2",
        "total_tasks": len(tasks),
        "completed": [],
        "current": None,
        "updated_at": _now(),
    }
    _atomic_json(state, state_path)
    try:
        for arm, seed in tasks:
            state["current"] = {"arm": arm, "seed": seed, "started_at": _now()}
            state["updated_at"] = _now()
            _atomic_json(state, state_path)
            _run_one(snapshot=snapshot, root=root, arm=arm, seed=seed)
            state["completed"].append({"arm": arm, "seed": seed})
            state["current"] = None
            state["updated_at"] = _now()
            _atomic_json(state, state_path)
        gate = audit_seed_gate(root, seed=2026)
        _atomic_json(gate, root / "SEED_2026_GATE.json")
        state["seed_2026_gate"] = "PASS"
        state["updated_at"] = _now()
        _atomic_json(state, state_path)
        bootstrap = write_paired_bootstrap_artifacts(root)
        report = write_ranking_artifacts(root)
        for arm in ARMS:
            for seed in SEEDS:
                completion = json.loads(
                    (root / f"seed_{seed}" / arm / "RUN_COMPLETE.json").read_text(
                        encoding="utf-8"
                    )
                )
                generate_selection_trajectory(
                    root / f"seed_{seed}" / arm / "history.csv",
                    root / "selection_trajectories" / f"seed_{seed}_{arm}.png",
                    best_epoch=int(completion["best_epoch"]),
                )
                prediction = root / f"seed_{seed}" / arm / "best" / "test" / "predictions_full.csv.gz"
                plot_root = root / "paper_plot_packs" / f"seed_{seed}" / arm / "best_test"
                if not plot_root.exists():
                    generate_plot_pack(prediction, plot_root)
                else:
                    verify_plot_pack(plot_root)
        reviewer_pack = write_reviewer_pack(root)
        state["status"] = "COMPLETE"
        state["final_ranking"] = report
        state["paired_bootstrap"] = bootstrap
        state["reviewer_pack"] = {
            "artifact_count": reviewer_pack["artifact_count"],
            "path": str(root / "paper_reviewer_pack"),
        }
        state["updated_at"] = _now()
        _atomic_json(state, state_path)
        return state
    except Exception as exc:
        state["status"] = "FAILED"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        state["updated_at"] = _now()
        _atomic_json(state, state_path)
        raise
