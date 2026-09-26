"""Bounded structural, gradient, initialization and train-only DDP smoke checks."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn import functional as F
import runtime as rt
from model import DeletionModel, VARIANTS, tensor_hashes
from objective import DeletionObjective
from synap_search.re_model import build_re_model


def structural(root, destination):
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    full = build_re_model(seed=2026, pretrained=True).eval()
    hashes = tensor_hashes(full.state_dict())
    x = torch.randn(2, 3, 64, 64)
    compatibility = DeletionModel("FULL", pretrained=True).eval()
    with torch.no_grad():
        assert torch.equal(full(x)["probabilities"], compatibility(x)["probabilities"])
    del compatibility
    report = {}
    for variant, count in zip(VARIANTS, (25311562, 24832975)):
        model = DeletionModel(variant, pretrained=True).eval()
        assert model.full_initial_hashes == hashes
        assert all(hashes[k] == v for k, v in model.initial_hashes.items())
        assert sum(p.numel() for p in model.parameters()) == count
        assert hasattr(model, "view_head") == (variant != "RE_WO_VIEW")
        assert hasattr(model, "view_adapters") == (variant != "RE_WO_VIEW")
        assert hasattr(model, "localization_head") == (variant != "RE_WO_WEAKLOC")
        out = model(x)
        assert torch.allclose(out["probabilities"].sum(1), torch.ones(2), atol=1e-6)
        assert bool(((out["score"] >= 0) & (out["score"] <= 3.5)).all())
        assert ("view_logits" in out) == (variant != "RE_WO_VIEW")
        assert ("attention" in out) == (variant != "RE_WO_WEAKLOC")
        batch = {"label_bin": torch.tensor([12, 25]), "position_norm": torch.tensor([1, 5]),
                 "lesion_mask": torch.ones(2, 1, 64, 64), "lesion_box_valid": torch.ones(2, dtype=torch.bool)}
        parts = DeletionObjective(variant).loss_components(out, batch)
        parts["grading"].backward(retain_graph=True)
        for name in ("evidence_mlp", "support_head", "context_head"):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters())
        isolated = model.localization_head if variant == "RE_WO_VIEW" else model.view_head
        assert all(p.grad is None for p in isolated.parameters())
        model.zero_grad(set_to_none=True)
        parts["total"].backward()
        assert all(p.grad is not None for p in model.parameters())
        assert any(p.grad.abs().sum() > 0 for p in isolated.parameters())
        # Check the actual deletion equation, independently of the model's compose call.
        with torch.no_grad():
            layer, features = model.backbone(x)
            z = model.local_projection(layer); support = F.softplus(model.support_head(z))
            if variant == "RE_WO_VIEW":
                attention = model.localization_head(layer.float()).sigmoid()
                scores = model.evidence_mlp(z.permute(0, 2, 3, 1))
                region = (scores * (support * attention).permute(0, 2, 3, 1)).sum((1, 2)) / support.sum((2, 3))
                expected = (model.context_head(features) + region).softmax(-1)
            else:
                scores = model._local_scores(z, model.view_adapters, model.evidence_mlp)
                region = (scores * support.permute(0, 2, 3, 1)[:, None]).sum((2, 3)) / support.sum((2, 3))[:, None]
                expected = ((model.context_head(features)[:, None] + region).softmax(-1)
                            * model.view_head(features).softmax(-1)[:, :, None]).sum(1)
            assert torch.allclose(expected, out["probabilities"], atol=1e-6)
        restored = DeletionModel(variant, pretrained=True).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        assert tensor_hashes(restored.state_dict()) == tensor_hashes(model.state_dict())
        report[variant] = {"parameters": count, "retained_initial_hashes_equal_full": True,
                           "deleted_modules_absent": True, "gradient_boundaries": "PASS", "formula": "PASS"}
        del model, restored, out, parts
    rt.write_json(destination / "STRUCTURAL_PASS.json", report)


def smoke(args):
    rank, world, device = rt.setup()
    assert world == 4
    model, wrapped, objective = rt.build(args.method, args.root, device, world)
    ds = rt.dataset(args.root, "train", args.method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, .6)
    report = []
    wrapped.train()
    for step, n in enumerate((24, 10)):
        # Fixed train-only samples; second step reproduces the 3/3/2/2 tail.
        indices = list(range(step * 24, step * 24 + n))[rank::world]
        batch = rt.move(torch.utils.data.default_collate([ds[(i, 1)] for i in indices]), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, target, global_n = rt.global_loss_inputs(wrapped(batch["image"]), batch, world, args.method)
            parts = objective.loss_components(out, target)
        assert global_n == n and torch.isfinite(parts["total"])
        # All ranks compute the same global objective, including the unequal tail.
        losses = [torch.empty_like(parts["total"]) for _ in range(world)]
        dist.all_gather(losses, parts["total"].detach())
        assert all(torch.equal(v, losses[0]) for v in losses)
        parts["total"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step(); scheduler.step()
        # Exact state round-trip, including optimizer/scheduler and per-rank RNG.
        state = {"variant": args.method, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": rt.objects(torch.cuda.get_rng_state(device), rank, world)}
        if rank == 0:
            rt.save_checkpoint(args.output / (args.method + "_smoke.pt"), state)
        rt.barrier(world)
        loaded = torch.load(args.output / (args.method + "_smoke.pt"), map_location="cpu", weights_only=False)
        assert loaded["variant"] == args.method
        model.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"]); scheduler.load_state_dict(loaded["scheduler"])
        torch.cuda.set_rng_state(loaded["rng"][rank], device)
        report.append({"global_batch": n, "loss": float(parts["total"].detach()), "round_trip": "PASS"})
    # Independent auxiliary evaluation branches on a train-only sample subset.
    ds.training = False  # Train split only; deterministic evaluation preprocessing.
    class SmallDataset(torch.utils.data.Dataset):
        frame = ds.frame.iloc[:8].copy()
        def __len__(self): return 8
        def __getitem__(self, i): return ds[i]
    small = SmallDataset()
    dl, _ = rt.loader(small, rank, world, 0)
    metrics, frame = rt.evaluate(model, dl, small, args.method, device, rank, world)
    if rank == 0:
        assert metrics["position"]["available"] == (args.method != "RE_WO_VIEW")
        assert metrics["lesion"]["available"] == (args.method != "RE_WO_WEAKLOC")
        rt.write_json(args.output / (args.method + "_SMOKE_PASS.json"), {"steps": report, "metrics": metrics})
    rt.barrier(world); dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["structural", "smoke"])
    p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=VARIANTS)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "structural": structural(args.root, args.output)
    else: smoke(args)
