from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sfibai_b.model import build_model
from sfibai_b.protocol import ARMS


def count_conv_linear_flops(
    model: torch.nn.Module, inputs: torch.Tensor
) -> int:
    """Count Conv2d/Linear FLOPs with two operations per multiply-accumulate."""

    total = 0

    def hook(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        nonlocal total
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            return
        if isinstance(module, torch.nn.Conv2d):
            kernel_operations = (
                module.in_channels
                // module.groups
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
            total += int(tensor.numel() * kernel_operations * 2)
        elif isinstance(module, torch.nn.Linear):
            total += int(tensor.numel() * module.in_features * 2)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    try:
        model.eval()
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return total


def _cuda_latency_ms(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    warmup: int = 10,
    measured: int = 30,
) -> dict[str, float]:
    if inputs.device.type != "cuda":
        raise ValueError("Latency profiling requires CUDA")
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for _ in range(warmup):
            model(inputs)
        torch.cuda.synchronize()
        values = []
        for _ in range(measured):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(inputs)
            end.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
    return {
        "latency_batch1_median_ms": float(np.median(values)),
        "latency_batch1_p90_ms": float(np.percentile(values, 90)),
    }


def write_model_cost_report(output_dir: str | Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Model cost latency report requires CUDA")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for arm in ARMS:
        cpu_model = build_model(arm=arm, seed=2026, pretrained=False)
        parameters = sum(parameter.numel() for parameter in cpu_model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in cpu_model.parameters()
            if parameter.requires_grad
        )
        flops = count_conv_linear_flops(
            cpu_model, torch.zeros(1, 3, 512, 512)
        )
        del cpu_model
        gc.collect()

        device = torch.device("cuda")
        gpu_model = build_model(arm=arm, seed=2026, pretrained=False).to(device)
        latency = _cuda_latency_ms(
            gpu_model, torch.zeros(1, 3, 512, 512, device=device)
        )
        rows.append(
            {
                "arm": arm,
                "parameters": int(parameters),
                "trainable_parameters": int(trainable),
                "conv_linear_flops_batch1": int(flops),
                "conv_linear_gflops_batch1": float(flops / 1e9),
                **latency,
            }
        )
        del gpu_model
        gc.collect()
        torch.cuda.empty_cache()
    baseline = next(row for row in rows if row["arm"] == "A")
    for row in rows:
        row["extra_parameters_vs_a"] = int(
            row["parameters"] - baseline["parameters"]
        )
        row["extra_flops_fraction_vs_a"] = float(
            row["conv_linear_flops_batch1"] / baseline["conv_linear_flops_batch1"]
            - 1.0
        )
        row["extra_latency_fraction_vs_a"] = float(
            row["latency_batch1_median_ms"]
            / baseline["latency_batch1_median_ms"]
            - 1.0
        )
    payload = {
        "round": "AE_COR_v2",
        "input": "batch=1, channels=3, height=512, width=512",
        "flop_definition": "Conv2d and Linear only; two FLOPs per MAC",
        "latency": "CUDA AMP eager, 10 warmup and 30 measured iterations",
        "arms": rows,
    }
    (output / "model_cost.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    pd.DataFrame(rows).to_csv(
        output / "model_cost.csv", index=False, encoding="utf-8"
    )
    return payload
