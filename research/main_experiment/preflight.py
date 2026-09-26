"""Bounded GPU correctness and throughput checks; never writes training weights."""
import argparse
import json
import faulthandler
import os
import sys
import traceback
import time
from pathlib import Path

import torch
from torch import distributed as dist
import runtime as rt


def objective_parts(objective, method, out, batch):
    return objective.loss_components(out, batch) if method == "SYNAP" else objective(out, batch)


def equivalence(root, method, rank, world, device):
    print(f"rank={rank} stage=build", flush=True)
    model, _, objective = rt.build(method, root, device, world, distributed=False)
    # Test algebra in high precision: FP32 SyncBN/reduction order can move
    # activations across ReLU boundaries in the deep pretrained backbone.
    model.double()
    if method == "SYNAP":
        model.localization_head.float()  # RE explicitly evaluates this head in FP32.
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if rank == 0 else None
    generator = torch.Generator().manual_seed(5726)
    images = torch.randn(10, 3, 64, 64, generator=generator, dtype=torch.float64)
    masks = torch.zeros(10, 1, 64, 64); masks[:, :, 5:50, 10:54] = 1
    full = {"image": images, "label_bin": torch.tensor([0, 1, 2, 5, 10, 15, 20, 25, 30, 35]),
            "position_norm": torch.arange(10) % 6 + 1, "lesion_mask": masks,
            "lesion_box_valid": torch.tensor([False, True, True, False, True, True, False, True, True, True])}
    batch = {k: v[rank::world].to(device) for k, v in full.items()}
    wrapped.train()
    print(f"rank={rank} stage=forward", flush=True)
    out, targets, _ = rt.global_loss_inputs(wrapped(batch["image"]), batch, world, method)
    loss = objective_parts(objective, method, out, targets)["total"]
    loss.backward()
    print(f"rank={rank} stage=distributed_backward_done", flush=True)
    distributed_grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None} if rank == 0 else None
    distributed_buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()} if rank == 0 else None
    rt.barrier(world)
    result = None
    if rank == 0:
        print("rank=0 stage=reference_build", flush=True)
        reference, _, ref_objective = rt.build(method, root, device, 1)
        reference.double()
        if method == "SYNAP":
            reference.localization_head.float()
        reference.load_state_dict(initial); reference.train()
        print("rank=0 stage=reference_forward", flush=True)
        full = rt.move(full, device)
        reference_loss = objective_parts(ref_objective, method, reference(full["image"]), full)["total"]
        reference_loss.backward()
        print("rank=0 stage=reference_backward_done", flush=True)
        torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-7)
        max_error = 0.
        for name, p in reference.named_parameters():
            if p.grad is None:
                if name in distributed_grads:
                    raise AssertionError("Unexpected distributed gradient: " + name)
                continue
            grad = distributed_grads[name].to(device)
            # Relative L2 avoids meaningless elementwise relative errors at zero.
            error = float(torch.linalg.vector_norm(grad - p.grad) / torch.linalg.vector_norm(p.grad).clamp_min(1e-7))
            max_error = max(max_error, error)
            if error > 1e-5 and float((grad-p.grad).abs().max()) > 1e-7:
                raise AssertionError(f"Gradient differs for {name}: relative L2={error}")
        for name, value in reference.named_buffers():
            torch.testing.assert_close(value.cpu(), distributed_buffers[name], rtol=2e-4, atol=2e-5)
        result = {"status": "PASS", "global_batch": 10, "input_size": 64, "uneven_shards": world > 2,
                  "precision": "fp64_backbone_and_heads_except_RE_fp32_localization",
                  "loss": float(loss.detach()), "reference_loss": float(reference_loss.detach()),
                  "max_gradient_relative_l2": max_error}
        del reference
    del wrapped, model, objective, out, targets, batch, initial, distributed_grads
    torch.cuda.empty_cache(); rt.barrier(world)
    print(f"rank={rank} stage=equivalence_done", flush=True)
    return result


def throughput(root, method, rank, world, device, workers):
    torch.cuda.reset_peak_memory_stats(device)
    ds = rt.dataset(root, "train", method)
    dl, sampler = rt.loader(ds, rank, world, workers, True)
    model, wrapped, objective = rt.build(method, root, device, world)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    wrapped.train(); iterator = iter(dl)
    elapsed = []
    for step in range(16):
        tick = time.perf_counter()
        batch = rt.move(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, targets, _ = rt.global_loss_inputs(wrapped(batch["image"]), batch, world, method)
            loss = objective_parts(objective, method, out, targets)["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite preflight loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
        optimizer.step(); torch.cuda.synchronize(); rt.barrier(world)
        if step >= 4:
            elapsed.append(time.perf_counter() - tick)
    values = rt.objects({"mean_step_seconds": sum(elapsed)/len(elapsed), "peak_bytes": torch.cuda.max_memory_allocated()}, rank, world)
    return {"steps": 12, "global_batch": 24, "ranks": values,
            "images_per_second": 24 / max(v["mean_step_seconds"] for v in values)} if rank == 0 else None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--method", choices=["SYNAP", "SFIBAI"], required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(); faulthandler.dump_traceback_later(90, repeat=True)
    rank, world, device = rt.setup()
    try:
        correct = equivalence(args.root, args.method, rank, world, device)
        speed = throughput(args.root, args.method, rank, world, device, args.workers)
        if rank == 0:
            result = {"method": args.method, "world": world, "correctness": correct, "throughput": speed,
                      "formal_training": False, "checkpoint_saved": False}
            rt.write_json(args.output, result); print(json.dumps(result), flush=True)
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        # Other ranks can be waiting in a collective; a collective destructor
        # here would hide the original error until NCCL times out.
        os._exit(1)
    else:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__": main()
