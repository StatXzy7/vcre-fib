"""Portable launcher for the released regional implementation; never runs on import."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "vcre": ("main_experiment", "SYNAP"),
    "sfibai": ("main_experiment", "SFIBAI"),
    "wo-view": ("re_ablation", "RE_WO_VIEW"),
    "wo-weakloc": ("re_ablation", "RE_WO_WEAKLOC"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "resources"))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--epochs", type=int, choices=range(21, 121), default=90, metavar="21..120")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--round", default="vcre_fib_seed2026")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.mode == "resources" and args.gpus != 1:
        parser.error("Resource profiling requires one idle GPU")
    folder, method = METHODS[args.method]
    script = ROOT / "research" / folder / ("profile_resources.py" if args.mode == "resources" else "runtime.py")
    command = [sys.executable]
    if args.mode != "resources":
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}"]
    command.append(str(script))
    if args.mode != "resources":
        command.append(args.mode)
    command += ["--root", str(args.workspace.resolve()), "--output", str(args.output.resolve()), "--method", method]
    if args.mode != "resources":
        command += ["--round", args.round, "--workers", str(args.workers), "--epochs", str(args.epochs)]
        if args.resume:
            command.append("--resume")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "code/src"), str(ROOT / "research/autosearch/src")])
    env["TORCH_HOME"] = str(args.workspace.resolve() / "cache/torch")
    env["PYTHONUTF8"] = "1"
    if args.dry_run:
        print("Arguments:", command)
        return
    if not sys.platform.startswith("linux"):
        parser.error("Formal CUDA/NCCL execution is supported on Linux; use --dry-run elsewhere")
    subprocess.run(command, env=env, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
