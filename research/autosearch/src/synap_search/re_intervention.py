"""Quantification and interventions for the SynAP-Fib-RE interface."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

import torch

from .re_model import SynAPFibRE, _validate_map, _normalize_probabilities


def _hash_tensor(value: torch.Tensor) -> str:
    data = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(data.shape)).encode("ascii"))
    digest.update(data.numpy().tobytes(order="C"))
    return digest.hexdigest()


def region_masks_from_boxes(
    boxes: Sequence[Mapping[str, float] | Sequence[float]],
    *,
    feature_hw: tuple[int, int],
    coordinate_frame: str = "normalized",
    source_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Rasterize normalized or source-pixel xyxy boxes to a feature-grid mask.

    ``normalized`` means ROI-normalized coordinates in [0,1].  ``roi_pixel``
    requires ``source_hw`` and performs the explicit ROI-to-feature transform.
    Boxes are clipped; malformed or empty boxes raise instead of silently
    producing a display-only overlay.
    """
    height, width = map(int, feature_hw)
    if height <= 0 or width <= 0:
        raise ValueError("feature_hw must be positive")
    if coordinate_frame not in {"normalized", "roi_pixel"}:
        raise ValueError("coordinate_frame must be normalized or roi_pixel")
    if coordinate_frame == "roi_pixel" and (source_hw is None or min(source_hw) <= 0):
        raise ValueError("source_hw is required for roi_pixel boxes")
    masks = torch.zeros((len(boxes), height, width), dtype=torch.float32)
    for index, box in enumerate(boxes):
        if isinstance(box, Mapping):
            values = [box[k] for k in ("x_min", "y_min", "x_max", "y_max")]
        else:
            values = list(box)
        if len(values) != 4:
            raise ValueError("each box must contain xyxy")
        x1, y1, x2, y2 = (float(v) for v in values)
        if coordinate_frame == "roi_pixel":
            source_h, source_w = source_hw  # type: ignore[misc]
            x1, x2 = x1 / source_w, x2 / source_w
            y1, y2 = y1 / source_h, y2 / source_h
        if not all(torch.isfinite(torch.tensor(v)) for v in (x1, y1, x2, y2)):
            raise ValueError("box coordinates must be finite")
        x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
        y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
        if not (x1 < x2 and y1 < y2):
            raise ValueError("box is empty after coordinate transform")
        left = max(0, min(width - 1, int(x1 * width)))
        right = max(left + 1, min(width, int(x2 * width + 0.999999)))
        top = max(0, min(height - 1, int(y1 * height)))
        bottom = max(top + 1, min(height, int(y2 * height + 0.999999)))
        masks[index, top:bottom, left:right] = 1.0
    return masks


def _partition_masks(
    region_masks: torch.Tensor, *, batch: int, height: int, width: int
) -> torch.Tensor:
    if region_masks.ndim == 2:
        region_masks = region_masks.unsqueeze(0)
    if region_masks.ndim == 3:
        region_masks = region_masks.unsqueeze(0).expand(batch, -1, -1, -1)
    if (
        region_masks.ndim != 4
        or region_masks.shape[0] != batch
        or tuple(region_masks.shape[2:]) != (height, width)
    ):
        raise ValueError("region_masks has incompatible feature-grid shape")
    if not torch.isfinite(region_masks).all() or (region_masks < 0).any():
        raise ValueError("region_masks must be finite and non-negative")
    clipped = region_masks.clamp(0, 1)
    union = clipped.sum(dim=1).clamp_max(1.0)
    # Overlap is split between named regions. Background makes the groups a
    # complete partition, so path contributions can be audited for completeness.
    overlap_count = clipped.sum(dim=1).clamp_min(1.0)
    named = clipped / overlap_count.unsqueeze(1)
    background = (1.0 - union).clamp_min(0.0).unsqueeze(1)
    return torch.cat([named, background], dim=1)


def path_integral_contributions(
    outputs: Mapping[str, torch.Tensor],
    *,
    region_masks: torch.Tensor | None = None,
    steps: int = 256,
) -> dict[str, torch.Tensor | float | list[str]]:
    """Integrate output-score sensitivity along the regional logit path.

    The returned ``A_R`` values are model-output path contributions.  They are
    additive by construction up to the reported quadrature error and include a
    background group for uncovered pixels.  They are not clinical causal
    effects or independent region scores.
    """
    if steps < 8:
        raise ValueError("steps must be at least 8")
    required = {"context_logits", "context_score", "view_probs", "regional_logits", "score", "attention"}
    missing = sorted(required - set(outputs))
    if missing:
        raise KeyError(f"outputs missing quantification tensors: {missing}")
    context = outputs["context_logits"].double()
    q = outputs["view_probs"].double().detach()
    q = _normalize_probabilities(q, expected_shape=tuple(q.shape), name="view_probs").double()
    regional = outputs["regional_logits"].double()
    batch, views, classes = regional.shape
    height, width = outputs["attention"].shape[-2:]
    if region_masks is None:
        region_masks = torch.zeros((batch, 0, height, width), dtype=regional.dtype, device=regional.device)
    region_masks = region_masks.to(device=regional.device, dtype=regional.dtype)
    partition = _partition_masks(region_masks, batch=batch, height=height, width=width)
    # Reconstruct each group's D_v from the original spatial factors.  If the
    # caller supplies only aggregate logits, a single full-path group remains
    # valid and preserves the exact score difference.
    if "local_scores" in outputs and "support" in outputs:
        local_scores = outputs["local_scores"].double()
        support = outputs["support"].double()
        attention = outputs["attention"].double().detach()
        base = support * attention
        denom = support.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)
        norm = base.flatten(1) / denom
        norm = torch.where(support.flatten(1).sum(dim=1, keepdim=True) > 1e-8, norm, torch.zeros_like(norm))
        raw = torch.stack(
            [torch.einsum("bvhwk,bhw->bvk", local_scores, norm.view(batch, height, width) * group)
             for group in partition.unbind(dim=1)], dim=1
        )
        scale_value = outputs.get(
            "regional_lambda",
            torch.ones((), dtype=regional.dtype, device=regional.device),
        ).to(dtype=regional.dtype, device=regional.device)
        if scale_value.numel() != 1 or not torch.isfinite(scale_value):
            raise ValueError("regional_lambda must be one finite scalar")
        group_logits = raw * scale_value
        decomposition_error = (regional - group_logits.sum(dim=1)).abs().amax()
    else:
        group_logits = torch.zeros((batch, partition.shape[1], views, classes), dtype=regional.dtype, device=regional.device)
        group_logits[:, -1] = regional
        decomposition_error = torch.zeros((), dtype=regional.dtype, device=regional.device)
    total = group_logits.sum(dim=1)
    grades = torch.arange(classes, device=regional.device, dtype=regional.dtype) / 10.0
    t_values = torch.linspace(0.0, 1.0, steps + 1, device=regional.device, dtype=regional.dtype)
    derivatives = []
    for t in t_values:
        probabilities = torch.softmax(context[:, None, :] + t * total, dim=-1)
        expected = (probabilities[:, None, :, :] * group_logits).sum(dim=-1, keepdim=True)
        # B x R x V x K, then weighted mixture over V and score bins.
        delta = group_logits - expected
        derivative = (q[:, None, :, None] * probabilities[:, None, :, :] * delta * grades).sum(dim=(2, 3))
        derivatives.append(derivative)
    derivative_stack = torch.stack(derivatives, dim=0)
    contributions = torch.trapz(derivative_stack, t_values, dim=0)
    context_score = outputs["context_score"].double()
    full_score = outputs["score"].double()
    error = full_score - context_score - contributions.sum(dim=1)
    return {
        "A_R": contributions.float(),
        "context_score": context_score.float(),
        "full_score": full_score.float(),
        "completeness_error": float(error.abs().max()),
        "decomposition_error": float(decomposition_error),
        "region_names": [f"region_{i}" for i in range(partition.shape[1] - 1)] + ["background"],
    }


def intervene(
    model: SynAPFibRE,
    outputs: Mapping[str, torch.Tensor],
    *,
    view_probs: torch.Tensor | None = None,
    support: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
    close_region: bool = False,
    close_mask: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Consume an override in the actual RE grading path.

    This is a fixed-feature intervention: image features are held fixed and
    the named interface values are recomputed by ``model.recompose``.  It is
    distinct from a pixel perturbation or a concept-replacement retraining run.
    """
    if model.training:
        raise RuntimeError("RE interventions require model.eval()")
    with torch.no_grad():
        result = model.recompose(
            outputs,
            view_override=view_probs,
            support_override=support,
            attention_override=attention,
            region_keep_mask=None if close_mask is None else 1.0 - close_mask,
            close_region=close_region,
        )
    log: dict[str, Any] = {
        "view_override": view_probs is not None,
        "support_override": support is not None,
        "attention_override": attention is not None,
        "close_region": bool(close_region),
        "close_mask": close_mask is not None,
        "consumed": {},
        "fixed_feature_intervention": True,
        "pixel_perturbation": False,
    }
    for name, value in (("view_probs", result["view_probs"]), ("support", result["support"]), ("attention", result["attention"])):
        log["consumed"][name + "_sha256"] = _hash_tensor(value)
    if "score" in outputs:
        log["score_delta_closed_minus_original"] = (result["score"] - outputs["score"]).detach().cpu().tolist()
    return result, log


def delta_off(
    model: SynAPFibRE,
    outputs: Mapping[str, torch.Tensor],
    *,
    close_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the actual closed-path score minus the open-path score."""
    result, _ = intervene(model, outputs, close_region=close_mask is None, close_mask=close_mask)
    return result["score"] - outputs["score"]


__all__ = ["delta_off", "intervene", "path_integral_contributions", "region_masks_from_boxes"]
