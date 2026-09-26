from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping


COMPILE_MINIMUM_SPEEDUP = 0.08


def select_runtime(results: Mapping[str, Any]) -> dict[str, int | str]:
    stable_workers = [
        (int(worker), float(values["images_per_second"]))
        for worker, values in results["workers"].items()
        if bool(values.get("stable", False)) and int(worker) in {4, 6, 8}
    ]
    if not stable_workers:
        raise ValueError("No stable num_workers candidate completed the benchmark")
    selected_workers, eager_throughput = max(stable_workers, key=lambda item: item[1])
    eager = results["backends"]["eager"]
    if not bool(eager.get("stable", False)):
        raise ValueError("Eager backend must be stable")
    eager_throughput = float(eager["images_per_second"])
    compile_result = results["backends"].get("compile", {})
    compile_ok = (
        bool(compile_result.get("stable", False))
        and bool(compile_result.get("numerical_audit_ok", False))
        and int(compile_result.get("graph_breaks", -1)) == 0
    )
    compile_throughput = float(compile_result.get("images_per_second", 0.0))
    speedup = compile_throughput / eager_throughput - 1.0
    backend = "compile" if compile_ok and speedup >= COMPILE_MINIMUM_SPEEDUP else "eager"
    return {"num_workers": selected_workers, "backend": backend}


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_break_count() -> int:
    import torch

    return int(sum(torch._dynamo.utils.counters["graph_break"].values()))


def _reset_graph_breaks() -> None:
    import torch

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


def _benchmark_task(num_workers: int):
    from sfibai_b.training import RuntimeSelection, TaskSpec

    return TaskSpec(
        arm="E",
        seed=2026,
        output_dir=Path("benchmark-only"),
        runtime=RuntimeSelection(num_workers, "eager", "0" * 64),
        epochs=1,
        batch_size=24,
        pretrained=True,
        formal=False,
    )


def _build_benchmark_loader(num_workers: int):
    from sfibai_b.data import FormalImageDataset
    from sfibai_b.protocol import ANNOTATIONS_JSONL, DATASET_ROOT, IMAGES_CSV
    from sfibai_b.runner import _make_loader

    task = _benchmark_task(num_workers)
    dataset = FormalImageDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=IMAGES_CSV,
        annotations_path=ANNOTATIONS_JSONL,
        split="train",
        arm="E",
        seed=2026,
        training=True,
        image_size=512,
    )
    loader, sampler = _make_loader(dataset=dataset, task=task, training=True)
    assert sampler is not None
    sampler.set_epoch(1)
    return loader


def _fixed_batch_numerical_audit(num_workers: int) -> dict[str, Any]:
    import torch

    from sfibai_b.loss import build_objective
    from sfibai_b.model import build_model
    from sfibai_b.runner import _move_tensor_batch

    device = torch.device("cuda")
    loader = _build_benchmark_loader(num_workers)
    cpu_batch = next(iter(loader))
    batch = _move_tensor_batch(cpu_batch, device)
    eager_model = build_model(arm="E", seed=2026, pretrained=True).to(device)
    compile_model = build_model(arm="E", seed=2026, pretrained=True).to(device)
    objective = build_objective("E").to(device)
    eager_model.train()
    compile_model.train()

    with torch.amp.autocast("cuda", enabled=True):
        eager_outputs = eager_model(batch["image"])
        eager_loss = objective(eager_outputs, batch)
    eager_loss.backward()
    eager_logits = eager_outputs["logits"].detach().cpu().float()
    eager_gradients = {
        name: parameter.grad.detach().cpu().float().clone()
        for name, parameter in eager_model.named_parameters()
        if parameter.grad is not None
    }

    _reset_graph_breaks()
    compiled = torch.compile(compile_model, fullgraph=True)
    with torch.amp.autocast("cuda", enabled=True):
        compiled_outputs = compiled(batch["image"])
        compiled_loss = objective(compiled_outputs, batch)
    compiled_loss.backward()
    graph_breaks = _graph_break_count()

    output_max_abs = float(
        torch.max(torch.abs(eager_logits - compiled_outputs["logits"].detach().cpu().float()))
    )
    loss_abs = abs(float(eager_loss.detach().cpu()) - float(compiled_loss.detach().cpu()))
    gradient_max_abs = 0.0
    gradient_max_rel = 0.0
    audit_ok = graph_breaks == 0
    failures: list[str] = []
    try:
        torch.testing.assert_close(
            compiled_outputs["logits"].detach().cpu().float(),
            eager_logits,
            rtol=5e-3,
            atol=5e-4,
        )
        if loss_abs > 1e-4:
            raise AssertionError(f"loss abs diff {loss_abs} exceeds 1e-4")
        compiled_named = dict(compile_model.named_parameters())
        if set(eager_gradients) != {
            name for name, parameter in compiled_named.items() if parameter.grad is not None
        }:
            raise AssertionError("compiled and eager gradient key sets differ")
        for name, eager_gradient in eager_gradients.items():
            compiled_gradient = compiled_named[name].grad.detach().cpu().float()
            difference = torch.abs(eager_gradient - compiled_gradient)
            gradient_max_abs = max(gradient_max_abs, float(difference.max()))
            denominator = torch.maximum(
                torch.abs(eager_gradient), torch.full_like(eager_gradient, 1e-6)
            )
            gradient_max_rel = max(
                gradient_max_rel, float((difference / denominator).max())
            )
            torch.testing.assert_close(
                compiled_gradient,
                eager_gradient,
                rtol=5e-3,
                atol=5e-4,
            )
    except (AssertionError, RuntimeError) as exc:
        audit_ok = False
        failures.append(str(exc))
    if graph_breaks:
        failures.append(f"graph_breaks={graph_breaks}")
    del compiled, compile_model, eager_model, objective, batch, cpu_batch, loader
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "numerical_audit_ok": audit_ok,
        "graph_breaks": graph_breaks,
        "output_max_abs": output_max_abs,
        "loss_abs": loss_abs,
        "gradient_max_abs": gradient_max_abs,
        "gradient_max_rel": gradient_max_rel,
        "failures": failures,
    }


def _measure_training_throughput(
    *, num_workers: int, backend: str, warmup_batches: int, measured_batches: int
) -> dict[str, Any]:
    import torch

    from sfibai_b.loss import build_objective
    from sfibai_b.model import build_model
    from sfibai_b.runner import _move_tensor_batch

    if backend not in {"eager", "compile"}:
        raise ValueError("backend must be eager or compile")
    device = torch.device("cuda")
    loader = _build_benchmark_loader(num_workers)
    model = build_model(arm="E", seed=2026, pretrained=True).to(device)
    forward_model = torch.compile(model, fullgraph=True) if backend == "compile" else model
    objective = build_objective("E").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    forward_model.train()
    iterator = iter(loader)
    elapsed_start = 0.0
    images = 0
    try:
        for batch_index in range(warmup_batches + measured_batches):
            cpu_batch = next(iterator)
            batch = _move_tensor_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                outputs = forward_model(batch["image"])
                loss = objective(outputs, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite benchmark loss")
            scaler.scale(loss).backward()
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < scale_before:
                raise FloatingPointError("AMP overflow skipped a benchmark optimizer step")
            if batch_index + 1 == warmup_batches:
                torch.cuda.synchronize()
                elapsed_start = time.perf_counter()
            if batch_index >= warmup_batches:
                images += int(batch["image"].shape[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - elapsed_start
        result = {
            "stable": True,
            "images": images,
            "seconds": elapsed,
            "images_per_second": images / elapsed,
            "warmup_batches": warmup_batches,
            "measured_batches": measured_batches,
        }
    except Exception as exc:
        result = {
            "stable": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "warmup_batches": warmup_batches,
            "measured_batches": measured_batches,
        }
    del iterator, loader, forward_model, model, objective, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_runtime_benchmark(
    *, output_dir: str | Path, warmup_batches: int = 20, measured_batches: int = 300
) -> dict[str, Any]:
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("Runtime benchmark requires CUDA")
    if warmup_batches <= 0:
        raise ValueError("Runtime benchmark requires at least one warmup batch")
    if measured_batches < 300:
        raise ValueError("Formal runtime selection requires at least 300 measured batches")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = True
    worker_results: dict[str, Any] = {}
    for workers in (4, 6, 8):
        print(f"BENCHMARK workers={workers} backend=eager START", flush=True)
        worker_results[str(workers)] = _measure_training_throughput(
            num_workers=workers,
            backend="eager",
            warmup_batches=warmup_batches,
            measured_batches=measured_batches,
        )
        _atomic_json(
            {"stage": "workers", "completed": worker_results},
            output / "benchmark_progress.json",
        )
        print(
            f"BENCHMARK workers={workers} backend=eager DONE "
            f"stable={worker_results[str(workers)].get('stable')}",
            flush=True,
        )
    stable_worker_results = {
        key: value for key, value in worker_results.items() if value.get("stable")
    }
    if not stable_worker_results:
        raise RuntimeError("All num_workers benchmark candidates failed")
    best_workers = int(
        max(stable_worker_results, key=lambda key: stable_worker_results[key]["images_per_second"])
    )
    eager_result = dict(worker_results[str(best_workers)])
    numerical_audit: dict[str, Any]
    print(f"BENCHMARK workers={best_workers} compile_numerical_audit START", flush=True)
    try:
        numerical_audit = _fixed_batch_numerical_audit(best_workers)
    except Exception as exc:
        numerical_audit = {
            "numerical_audit_ok": False,
            "graph_breaks": -1,
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    print(
        "BENCHMARK compile_numerical_audit DONE "
        f"ok={numerical_audit.get('numerical_audit_ok')}",
        flush=True,
    )
    if numerical_audit["numerical_audit_ok"]:
        print(f"BENCHMARK workers={best_workers} backend=compile START", flush=True)
        compile_result = _measure_training_throughput(
            num_workers=best_workers,
            backend="compile",
            warmup_batches=warmup_batches,
            measured_batches=measured_batches,
        )
        print(
            "BENCHMARK backend=compile DONE "
            f"stable={compile_result.get('stable')}",
            flush=True,
        )
    else:
        compile_result = {"stable": False, "error": "numerical audit failed"}
    compile_result.update(numerical_audit)
    results = {
        "protocol": {
            "arm": "E",
            "batch_size": 24,
            "warmup_batches": warmup_batches,
            "measured_batches": measured_batches,
            "worker_candidates": [4, 6, 8],
            "compile_minimum_speedup": COMPILE_MINIMUM_SPEEDUP,
            "amp": True,
            "cudnn_benchmark": True,
        },
        "environment": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "workers": worker_results,
        "backends": {"eager": eager_result, "compile": compile_result},
    }
    selected = select_runtime(results)
    results["selection"] = selected
    benchmark_path = output / "benchmark_results.json"
    _atomic_json(results, benchmark_path)
    benchmark_sha256 = _sha256(benchmark_path)
    selection_payload = {
        **selected,
        "benchmark_sha256": benchmark_sha256,
        "benchmark_results": str(benchmark_path.resolve()),
    }
    _atomic_json(selection_payload, output / "runtime_selection.json")
    _atomic_json(
        {"stage": "complete", "runtime_selection": selection_payload},
        output / "benchmark_progress.json",
    )
    return {"results": results, "runtime_selection": selection_payload}
