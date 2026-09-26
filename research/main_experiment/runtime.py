"""Protocol-preserving distributed execution for the paper's two main methods."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
from torch import distributed as dist, nn
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.tensorboard import SummaryWriter

from sfibai_b.data import EpochShuffleSampler, FormalImageDataset
from sfibai_b.evaluation import checkpoint_key, evaluate_predictions, flatten_metrics, score_to_grade
from synap_search.io import environment, save_checkpoint, sha256, write_json
from synap_search.re_model import build_re_model
from synap_search.re_training import REObjective


class GlobalBatchShard(Sampler):
    """No padding, repeats or dropped final samples; global batch stays 24."""
    def __init__(self, size, seed, rank=0, world=1, batch=24):
        self.base = EpochShuffleSampler(size=size, seed=seed)
        self.rank, self.world, self.batch = rank, world, batch
        if batch % world or (size % batch and size % batch < world):
            raise ValueError("Global/final batch cannot be split over these ranks")

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __iter__(self):
        items = list(self.base)
        for start in range(0, len(items), self.batch):
            yield items[start:start + self.batch][self.rank::self.world]

    def __len__(self):
        return math.ceil(len(self.base) / self.batch)


def setup():
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    return rank, world, device


def barrier(world):
    if world > 1:
        dist.barrier()


def objects(value, rank, world):
    if world == 1:
        return [value]
    result = [None] * world if rank == 0 else None
    dist.gather_object(value, result, dst=0)
    return result


def gather_tensor(value, sizes, differentiable=False):
    if len(sizes) == 1:
        return value
    padding = value.new_zeros((max(sizes) - len(value), *value.shape[1:]))
    padded = torch.cat([value, padding], dim=0).contiguous()
    if differentiable:
        parts = all_gather(padded)
    else:
        parts = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(parts, padded)
    return torch.cat([part[:size] for part, size in zip(parts, sizes)], dim=0)


def global_loss_inputs(output, batch, world, method):
    size = torch.tensor([len(batch["image"])], device=batch["image"].device)
    if world > 1:
        size_parts = [torch.empty_like(size) for _ in range(world)]
        dist.all_gather(size_parts, size)
        sizes = [int(x.item()) for x in size_parts]
    else:
        sizes = [int(size.item())]
    keys = ("probabilities", "view_logits", "localization_logits") if method == "SYNAP" else ("logits",)
    out = {key: gather_tensor(output[key], sizes, True) for key in keys}
    target = {"label_bin": gather_tensor(batch["label_bin"], sizes)}
    if method == "SYNAP":
        mask = F.interpolate(batch["lesion_mask"].float(), size=output["localization_logits"].shape[-2:], mode="area")
        target.update(position_norm=gather_tensor(batch["position_norm"], sizes),
                      lesion_mask=gather_tensor(mask, sizes),
                      lesion_box_valid=gather_tensor(batch["lesion_box_valid"], sizes))
    return out, target, sum(sizes)


def upstream_module(root):
    root = Path(root)
    snapshot = json.loads((Path(__file__).resolve().parents[2] / "provenance/SFIBAI_SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    for item in snapshot["files"]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError("Upstream SFibAI source differs from manuscript: " + item["path"])
    sys.path.insert(0, str(root / "src/sfibai"))
    spec = importlib.util.spec_from_file_location("released_sfibai_train", root / "src/sfibai/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleasedSFibAI(nn.Module):
    def __init__(self, module, weights):
        super().__init__()
        self.net = module.create_model("resnet50", num_classes=36, checkpoint_path=weights)

    def forward(self, images):
        logits = self.net(images)
        probabilities = logits.float().softmax(1)
        return {"logits": logits, "probabilities": probabilities,
                "score": (probabilities * torch.arange(36, device=images.device)).sum(1) / 10}


def build(method, root, device, world, distributed=True):
    if method == "SYNAP":
        model, objective = build_re_model(seed=2026, pretrained=True), REObjective()
    else:
        module = upstream_module(Path(__file__).resolve().parents[2] / "third_party/SFibAI")
        model = ReleasedSFibAI(module, root / "cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")
        objective = lambda out, batch: {"total": module.calculate_loss(
            out["logits"], batch["label_bin"], "hybrid", device,
            SimpleNamespace(alpha=1.0, beta=0.02, gamma=0.02))}
    if world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 and distributed else model
    return model, wrapped, objective


def dataset(root, split, method):
    # Only construct test from the final-evaluation stage after verifying freeze.
    manifest = root / ("data/schisto_2024_clean_v4/manifests/images.csv" if split == "test" else "development/images.csv")
    annotations = root / ("data/schisto_2024_clean_v4/manifests/annotations.jsonl" if split == "test" else f"development/native_{split}.jsonl")
    ds = FormalImageDataset(dataset_root=root / "data/schisto_2024_clean_v4", manifest_path=manifest,
                            annotations_path=annotations, split=split, arm="E" if method == "SYNAP" else "A",
                            seed=2026, training=split == "train", image_size=512)
    expected = {"train": (83722, 4906), "val": (20880, 1227), "test": (4107, 240)}
    if (len(ds), ds.frame.patient_uid.nunique()) != expected[split]:
        raise ValueError("Native dataset count mismatch")
    return ds


def loader(ds, rank, world, workers, training=False):
    kwargs = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    if training:
        sampler = GlobalBatchShard(len(ds), 2026, rank, world)
        return DataLoader(ds, batch_sampler=sampler, **kwargs), sampler
    return DataLoader(Subset(ds, range(rank, len(ds), world)), batch_size=24, **kwargs), None


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def lesion_rows(output, batch):
    attention = output["attention"].detach().float().cpu()
    target = F.interpolate(batch["lesion_mask"].detach().float().cpu(), size=attention.shape[-2:], mode="nearest") > .5
    valid = batch["lesion_box_valid"].cpu().bool() & (batch["label_bin"].cpu() > 0)
    rows = []
    for i, heat in enumerate(attention[:, 0]):
        inside = target[i, 0]
        if not (valid[i] and inside.any()):
            rows.append({"lesion_valid": False})
            continue
        hard = heat >= .5; intersection = (hard & inside).sum().float()
        outside = ~inside
        outside_mean = heat[outside].mean() if outside.any() else torch.tensor(0.)
        rows.append({"lesion_valid": True, "inside_attention": float(heat[inside].mean()),
                     "outside_ratio": float(heat[outside].sum() / heat.sum().clamp_min(1e-8)),
                     "inside_outside_ratio": float(heat[inside].mean() / outside_mean.clamp_min(1e-8)),
                     "dice_at_0_5": float(2 * intersection / (hard.sum() + inside.sum()).clamp_min(1)),
                     "iou_at_0_5": float(intersection / (hard | inside).sum().clamp_min(1))})
    return rows


def extended_metrics(frame):
    metrics = evaluate_predictions(frame)
    from sklearn.metrics import confusion_matrix
    metrics["grade_confusion_matrix"] = confusion_matrix(score_to_grade(frame.true_score), score_to_grade(frame.pred_score), labels=range(4)).tolist()
    if "position_pred" in frame:
        metrics["position"]["confusion_matrix"] = confusion_matrix(frame.position_true, frame.position_pred, labels=range(1, 7)).tolist()
    metrics["per_center"] = {str(center): evaluate_predictions(group) for center, group in frame.groupby("center_id")}
    for key in ("image", "patient_max", "patient_median"):
        metrics[key]["clinical_risk"] = metrics[key]["cor"]
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, ds, method, device, rank, world):
    model.eval(); rows = []
    for batch in data_loader:
        output = model(batch["image"].to(device, non_blocking=True))  # FP32 evaluation, fixed before launch.
        probabilities = output["probabilities"].float().cpu().numpy()
        scores = output["score"].float().cpu().numpy()
        grade = np.stack([probabilities[:, :5].sum(1), probabilities[:, 5:15].sum(1),
                          probabilities[:, 15:25].sum(1), probabilities[:, 25:].sum(1)], axis=1)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
            raise FloatingPointError("Invalid probabilities")
        lesion = lesion_rows(output, batch) if method == "SYNAP" else None
        position = output["view_logits"].float().softmax(1).cpu().numpy() if method == "SYNAP" else None
        for i, score in enumerate(scores):
            row = {k: str(batch[k][i]) for k in ("image_uid", "patient_uid", "center_id")}
            # Match the existing canonical predictor's float32 target serialization.
            row.update(true_score=float(batch["true_score"][i]), pred_score=float(score))
            row.update({f"prob_f{k}": float(grade[i, k]) for k in range(4)})
            if position is not None:
                row.update(position_true=int(batch["position_norm"][i]), position_pred=int(position[i].argmax()) + 1)
                row.update({f"position_prob_{k+1}": float(position[i, k]) for k in range(6)})
                row.update(lesion[i])
            rows.append(row)
    gathered = objects(rows, rank, world)
    if rank != 0:
        return None, None
    frame = pd.DataFrame([row for part in gathered for row in part])
    if len(frame) != len(ds) or frame.image_uid.duplicated().any() or set(frame.image_uid) != set(ds.frame.image_uid):
        raise ValueError("Evaluation sample coverage differs from native split")
    frame = frame.set_index("image_uid").loc[ds.frame.image_uid].reset_index()
    return extended_metrics(frame), frame


def source_identity(root, cfg):
    import sfibai_b, synap_search
    sources = {}
    for base in (Path(__file__).parent, Path(sfibai_b.__file__).parent, Path(synap_search.__file__).parent):
        sources.update({str(p): sha256(p) for p in sorted(base.glob("*.py"))})
    for p in sorted((Path(__file__).resolve().parents[2] / "third_party/SFibAI/src/sfibai").rglob("*.py")):
        sources[str(p)] = sha256(p)
    data = {name: sha256(root / "data/schisto_2024_clean_v4/manifests" / name) for name in ("images.csv", "annotations.jsonl")}
    if data != {"images.csv": "73a14221ea1a67f3248351a4ed5d08574f904175e69d9129ab2e705a0bb86a40", "annotations.jsonl": "6684cb633ad055eb55f399a6383f531b5e8cf4db0a3ef828fb6967c16150d167"}:
        raise ValueError("Data V4 manifest identity changed")
    # These are the files actually consumed during train/val, verified against
    # the canonical native split in the preparation receipt, not just metadata.
    expected_development = {
        "images.csv": "d3a3d04b9adb947b6c3905a2d18d256f6a5ca28586970dc1910881786f48d7cb",
        "native_train.jsonl": "531347f450bb8473274cd5bcd7875586a05ec01d69ddcd2d9aa0a7cf6d874c68",
        "native_val.jsonl": "4e61622f2064582a29a6ce7032e32d82328365ba599688a41c7497326093bd22",
    }
    development = {name: sha256(root / "development" / name) for name in expected_development}
    if development != expected_development:
        raise ValueError("Effective native train/val inputs changed")
    return {"config": cfg, "source_hashes": sources, "data_hashes": data,
            "development_hashes": development,
            "initial_weights": sha256(root / "cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"), "environment": environment()}


def train(args, rank, world, device):
    root, output = args.root, args.output
    cfg = {"round": args.round, "method": args.method, "seed": 2026, "epochs": args.epochs, "global_batch": 24,
           "image_size": 512, "precision": "bf16", "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4,
           "scheduler": [15, .6], "world_size": world, "workers_per_rank": args.workers,
           "selection": f"val_R_final_then_image_COR_then_earlier_epoch_21_to_{args.epochs}", "checkpoint_every": 10,
           "evaluation_precision": "fp32", "batchnorm": "SyncBatchNorm" if world > 1 else "BatchNorm"}
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError("Formal output must be new; use explicit validated resume")
        output.mkdir(parents=True, exist_ok=True)
    barrier(world)
    identity = source_identity(root, cfg)
    model, wrapped, objective = build(args.method, root, device, world)
    train_ds, val_ds = dataset(root, "train", args.method), dataset(root, "val", args.method)
    if set(train_ds.frame.patient_uid) & set(val_ds.frame.patient_uid):
        raise ValueError("Train/val patient overlap")
    train_loader, sampler = loader(train_ds, rank, world, args.workers, True)
    val_loader, _ = loader(val_ds, rank, world, args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.6)
    best_key, start_epoch = None, 1
    if args.resume:
        previous = json.loads((output / "IDENTITY.json").read_text(encoding="utf-8"))
        if identity != previous:
            raise ValueError("Strict resume identity mismatch")
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Checkpoint does not belong to this frozen identity")
        history = [json.loads(line) for line in (output / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        if not history or history[-1]["epoch"] != state["epoch"]:
            raise ValueError("History/checkpoint mismatch; no automatic recovery")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        best_key = tuple(state["best_key"]) if state["best_key"] is not None else None
        if best_key is not None:
            best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            if best["identity"] != identity or best["epoch"] != best_key[2] or best["epoch"] > state["epoch"]:
                raise ValueError("Resume best checkpoint lineage mismatch")
            del best
        start_epoch = state["epoch"] + 1
        rng = state["rank_rng"][rank]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"], device)
    if rank == 0:
        write_json(output / "IDENTITY.json", identity)
        writer = SummaryWriter(str(output / "tensorboard"))
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        tick = time.time(); wrapped.train(); sampler.set_epoch(epoch)
        sums, count = {}, 0
        for step, batch in enumerate(train_loader, 1):
            batch = move(batch, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, target, global_n = global_loss_inputs(wrapped(batch["image"]), batch, world, args.method)
                parts = objective.loss_components(out, target) if args.method == "SYNAP" else objective(out, target)
                loss = parts["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            count += global_n
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.) + float(value.detach()) * global_n
            if rank == 0 and (step == 1 or step % 50 == 0):
                status = {"status": "TRAINING", "method": args.method, "epoch": epoch, "epochs": args.epochs,
                          "step": step, "steps": len(train_loader), "images_seen": count,
                          "elapsed_epoch_seconds": time.time() - tick, "loss": float(loss.detach()), "updated_at": time.time()}
                write_json(output / "STATUS.json", status)
                print(json.dumps(status), flush=True)
        if count != 83722:
            raise ValueError("Training coverage mismatch")
        scheduler.step()
        val_metrics, val_frame = evaluate(model, val_loader, val_ds, args.method, device, rank, world)
        rngs = objects({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}, rank, world)
        if rank == 0:
            key = checkpoint_key(val_metrics, epoch)
            improved = epoch >= 21 and (best_key is None or key < best_key)
            if improved:
                best_key = key
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "identity": identity, "best_key": best_key, "rank_rng": rngs,
                     "val_metrics": val_metrics, "seed": 2026}
            save_checkpoint(output / "last.pt", state)
            if epoch % 10 == 0:
                save_checkpoint(output / f"epoch_{epoch:03d}.pt", state)
            if improved:
                save_checkpoint(output / "best.pt", state)
                val_frame.to_csv(output / "best_val_predictions.csv.gz", index=False)
                write_json(output / "best_val_metrics.json", val_metrics)
            train_metrics = {key: val / count for key, val in sums.items()}
            entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "best_key": best_key,
                     "epoch_seconds": time.time() - tick, "updated_at": time.time()}
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
            write_json(output / "validation" / f"epoch_{epoch:03d}.json", val_metrics)
            for key, val in flatten_metrics(val_metrics).items():
                if isinstance(val, (float, int)):
                    writer.add_scalar("val/" + key, val, epoch)
            for key, val in train_metrics.items():
                writer.add_scalar("train/" + key, val, epoch)
            writer.flush()
            write_json(output / "STATUS.json", {"status": "EPOCH_COMPLETE", **entry})
        barrier(world)
    if rank == 0:
        if best_key is None:
            raise RuntimeError("No eligible best checkpoint")
        freeze = {"method": args.method, "seed": 2026, "epoch": best_key[2], "selection_key": best_key,
                  "checkpoint": str(output / "best.pt"), "checkpoint_sha256": sha256(output / "best.pt"),
                  "identity_sha256": sha256(output / "IDENTITY.json"), "test_opened": False}
        write_json(output / "CHECKPOINT_FREEZE.json", freeze)
        write_json(output / "TRAINING_COMPLETE.json", {"epochs": args.epochs, "elapsed_seconds": time.time()-started, "test_opened": False})
        write_json(output / "STATUS.json", {"status": "TRAIN_VAL_COMPLETE_TEST_PENDING", **freeze})
        writer.close()
    barrier(world)


def final_evaluate(args, rank, world, device):
    output = args.output
    freeze = json.loads((output / "CHECKPOINT_FREEZE.json").read_text(encoding="utf-8"))
    identity = json.loads((output / "IDENTITY.json").read_text(encoding="utf-8"))
    if identity["config"]["method"] != args.method:
        raise ValueError("Evaluation method does not match frozen identity")
    if not (output / "TRAINING_COMPLETE.json").is_file() or freeze["checkpoint_sha256"] != sha256(output / "best.pt"):
        raise ValueError("Training/freeze verification failed")
    if freeze["identity_sha256"] != sha256(output / "IDENTITY.json") or source_identity(args.root, identity["config"]) != identity:
        raise ValueError("Source/environment identity changed before final evaluation")
    if (output / "EVALUATION_COMPLETE.json").exists():
        raise FileExistsError("Evaluation already complete; do not overwrite")
    model, _, _ = build(args.method, args.root, device, world, distributed=False)
    state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    for split in ("val", "test"):
        ds = dataset(args.root, split, args.method)
        dl, _ = loader(ds, rank, world, args.workers)
        metrics, frame = evaluate(model, dl, ds, args.method, device, rank, world)
        if rank == 0:
            if split == "val" and not np.allclose(
                [metrics["r_final"], metrics["image"]["cor"]], freeze["selection_key"][:2], rtol=0, atol=1e-10
            ):
                raise ValueError("Frozen validation result differs from checkpoint selection record")
            destination = output / "evaluation" / split
            destination.mkdir(parents=True, exist_ok=False)
            frame.to_csv(destination / "predictions.csv.gz", index=False)
            patient_hashes = {}
            for aggregation in ("max", "median"):
                patient = frame.groupby("patient_uid", sort=True, as_index=False).agg(
                    center_id=("center_id", "first"), true_score=("true_score", aggregation),
                    pred_score=("pred_score", aggregation), image_count=("image_uid", "size"))
                patient_path = destination / f"patient_{aggregation}_predictions.csv.gz"
                patient.to_csv(patient_path, index=False)
                patient_hashes[aggregation] = sha256(patient_path)
            write_json(destination / "metrics.json", metrics)
            write_json(destination / "provenance.json", {"freeze": freeze, "split": split, "images": len(ds),
                       "patients": int(ds.frame.patient_uid.nunique()), "centers": int(ds.frame.center_id.nunique()),
                       "predictions_sha256": sha256(destination / "predictions.csv.gz"), "metrics_sha256": sha256(destination / "metrics.json"),
                       "patient_predictions_sha256": patient_hashes})
        barrier(world)
    if rank == 0:
        write_json(output / "EVALUATION_COMPLETE.json", {"status": "PASS", "splits": ["val", "test"], "freeze": freeze})
        write_json(output / "STATUS.json", {"status": "EVALUATION_COMPLETE", "method": args.method})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["SFIBAI", "SYNAP"], required=True)
    parser.add_argument("--round", default="zgc_main_seed2026_20260925")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, choices=range(21, 121), default=90, metavar="21..120")
    args = parser.parse_args()
    rank, world, device = setup()
    try:
        (train if args.mode == "train" else final_evaluate)(args, rank, world, device)
    except Exception as exc:
        if rank == 0 and args.output.exists():
            write_json(args.output / "FAILED.json", {"mode": args.mode, "error": repr(exc), "time": time.time()})
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)  # Let torchrun stop peer ranks instead of blocking on NCCL cleanup.
    else:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
