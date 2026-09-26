"""Measure only frozen E on the local GPU, with recorded batch latency samples."""

from __future__ import annotations

import gc
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from package_common import PACK, read_json, sha256, write_json, write_table


def main() -> None:
    output = PACK / "resources"
    output.mkdir(exist_ok=True)
    if (output / "resources.json").exists():
        raise FileExistsError("Resource measurements already exist")
    sys.path.insert(0, str(PACK / "source/round_snapshot/src"))
    from sfibai_b.model import build_model
    from sfibai_b.model_cost import count_conv_linear_flops

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    checkpoint = PACK / "model/checkpoints/best.pt"
    frozen = read_json(PACK / "provenance/RUN_COMPLETE.json")
    if sha256(checkpoint) != frozen["best_checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != 103:
        raise ValueError("Unexpected epoch")
    model = build_model(arm="E", seed=2026, pretrained=False).eval()
    model.load_state_dict(state["model"], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.set_num_threads(4)
    flops = count_conv_linear_flops(model, torch.zeros(1, 3, 512, 512))
    del state
    model = model.cuda()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, raw = [], []
    gpu_before = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    for batch in [1, 8]:
        inputs = torch.zeros(batch, 3, 512, 512, device="cuda")
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            for _ in range(50):
                model(inputs)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            event_values, wall_values = [], []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for iteration in range(200):
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                start.record()
                result = model(inputs)
                end.record()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000
                event_ms = float(start.elapsed_time(end))
                event_values.append(event_ms)
                wall_values.append(wall_ms)
                raw.append(
                    {
                        "batch": batch,
                        "iteration": iteration,
                        "cuda_event_ms": event_ms,
                        "synchronized_wall_ms": wall_ms,
                    }
                )
                del result
            peak = torch.cuda.max_memory_allocated()
        rows.append(
            {
                "model": "SynAP-Fib E",
                "batch": batch,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "GMACs_per_image": flops / 2e9,
                "GFLOPs_per_image": flops / 1e9,
                "latency_median_ms": float(np.median(event_values)),
                "latency_p95_ms": float(np.percentile(event_values, 95)),
                "wall_latency_median_ms": float(np.median(wall_values)),
                "wall_latency_p95_ms": float(np.percentile(wall_values, 95)),
                "peak_allocated_MiB": peak / 2**20,
                "baseline_allocated_MiB": baseline / 2**20,
                "incremental_peak_MiB": (peak - baseline) / 2**20,
            }
        )
        del inputs
        torch.cuda.empty_cache()
        print(f"Batch {batch} measured: {rows[-1]}", flush=True)
    payload = {
        "status": "MEASURED",
        "model": "SynAP-Fib E",
        "arm": "E",
        "epoch": 103,
        "checkpoint_sha256": frozen["best_checkpoint_sha256"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input": ["batch", 3, 512, 512],
        "input_values": "synthetic all-zero tensor for model-only resource measurement",
        "cpu_threads": 4,
        "precision": "CUDA autocast FP16; frozen lesion head FP32 exception preserved",
        "backend": "eager",
        "tf32": False,
        "cudnn_benchmark": True,
        "warmup": 50,
        "timed_iterations_per_batch": 200,
        "latency_scope": "model forward only; CUDA events; batch latency; excludes I/O and preprocessing",
        "memory_scope": "PyTorch peak allocated including model, input, outputs; not nvidia-smi reserved memory",
        "mac_scope": "Conv2d and Linear MACs only; excludes elementwise, normalization, activation and pooling",
        "flops_convention": "2 FLOPs per MAC",
        "gpu_before": gpu_before,
        "measurements": rows,
    }
    write_table(pd.DataFrame(raw), output / "raw_latency_samples.csv")
    write_table(pd.DataFrame(rows), output / "resources.csv")
    write_json(output / "resources.json", payload)


if __name__ == "__main__":
    main()
