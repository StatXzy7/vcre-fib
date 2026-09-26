"""Inference-only concept interventions for :class:`SFibAIModel`.

The helper in this module operates on the tensors produced by a normal image
forward pass.  It never accepts labels, boxes, patient identifiers, or centre
metadata.  Network parameters and the image-derived backbone tensors are kept
unchanged; only the predicted concept consumed by the downstream residual path
is replaced or disabled.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch

from sfibai_b.model import SFibAIModel


def _tensor_sha256(value: torch.Tensor) -> str:
    """Return a stable hash of a detached CPU tensor, including its shape."""
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checked_position_probs(
    value: torch.Tensor, *, expected: torch.Tensor
) -> torch.Tensor:
    if value.shape != expected.shape:
        raise ValueError(
            f"position_probs shape {tuple(value.shape)} does not match "
            f"{tuple(expected.shape)}"
        )
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError("position_probs must be finite and non-negative")
    total = value.sum(dim=1, keepdim=True)
    if (total <= 0).any():
        raise ValueError("position_probs must have positive row sums")
    return (value / total).to(device=expected.device, dtype=expected.dtype)


def _checked_attention(
    value: torch.Tensor, *, expected: torch.Tensor
) -> torch.Tensor:
    if value.shape != expected.shape:
        raise ValueError(
            f"lesion_attention shape {tuple(value.shape)} does not match "
            f"{tuple(expected.shape)}"
        )
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("lesion_attention must be finite and lie in [0, 1]")
    return value.to(device=expected.device, dtype=expected.dtype)


def _checked_gate(value: torch.Tensor, *, expected: torch.Tensor) -> torch.Tensor:
    """Validate an original gate before using it as an analysis control."""
    if value.shape != expected.shape:
        raise ValueError(
            f"gate shape {tuple(value.shape)} does not match {tuple(expected.shape)}"
        )
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError("gate must be finite and non-negative")
    return value.to(device=expected.device, dtype=expected.dtype)


def intervene(
    model: SFibAIModel,
    outputs: dict[str, torch.Tensor],
    *,
    position_probs: torch.Tensor | None = None,
    lesion_attention: torch.Tensor | None = None,
    close_position: bool = False,
    close_lesion: bool = False,
    fixed_original_gates: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Recompute grading after a controlled concept replacement.

    ``outputs`` must be from the same ``model`` and one image batch.  A
    replacement is consumed by the actual residual, gate, and lesion pooling
    path; it is not merely recorded for visualisation.  The returned log hashes
    both the original and consumed tensors so that a diagnostic run can prove
    which values reached the downstream computation.  The caller must provide
    an evaluated model under ``torch.no_grad()``; when a concept is replaced,
    its raw branch logits are omitted from the returned mapping because they
    would describe the original, not the consumed, concept.
    """
    if not isinstance(model, SFibAIModel):
        raise TypeError("intervene currently supports SFibAIModel only")
    if model.training:
        raise RuntimeError("intervene requires model.eval()")
    if torch.is_grad_enabled():
        raise RuntimeError("intervene requires torch.no_grad()")
    if "global_features" not in outputs or "logits" not in outputs:
        raise KeyError("outputs must come from a complete SFibAIModel forward")
    if not model.use_position and (position_probs is not None or close_position):
        raise ValueError("position intervention requested for an inactive branch")
    if not model.use_lesion and (lesion_attention is not None or close_lesion):
        raise ValueError("lesion intervention requested for an inactive branch")

    fused_features = outputs["global_features"]
    result = dict(outputs)
    log: dict[str, Any] = {
        "position_replaced": position_probs is not None,
        "lesion_replaced": lesion_attention is not None,
        "position_closed": bool(close_position),
        "lesion_closed": bool(close_lesion),
        "fixed_original_gates": bool(fixed_original_gates),
        "consumed": {},
    }

    # Preserve exact forward output for the default no-op.  This is stronger
    # than numerical closeness and makes the control usable as a regression
    # identity check when the intervention is inserted into an evaluator.
    if (
        position_probs is None
        and lesion_attention is None
        and not close_position
        and not close_lesion
        and not fixed_original_gates
    ):
        if model.use_position:
            log.setdefault("consumed", {})["position_probs_sha256"] = _tensor_sha256(
                outputs["position_probs"]
            )
            log.setdefault("original", {})["position_probs_sha256"] = _tensor_sha256(
                outputs["position_probs"]
            )
        if model.use_lesion:
            log.setdefault("consumed", {})["lesion_attention_sha256"] = _tensor_sha256(
                outputs["lesion_attention"]
            )
            log.setdefault("original", {})["lesion_attention_sha256"] = _tensor_sha256(
                outputs["lesion_attention"]
            )
        log["fused_features_sha256"] = _tensor_sha256(outputs["fused_features"])
        return result, log

    if model.use_position:
        original = outputs["position_probs"]
        consumed = (
            _checked_position_probs(position_probs, expected=original)
            if position_probs is not None
            else original
        ).detach()
        entropy, maximum = model._position_statistics(consumed)
        residual = model.position_residual(consumed)
        dynamic_gate = model.gate_max * torch.sigmoid(
            model.position_gate(torch.cat([consumed, entropy, maximum], dim=1))
        )
        original_gate = _checked_gate(outputs["position_gate"], expected=dynamic_gate)
        gate = original_gate if fixed_original_gates else dynamic_gate
        if not close_position and model.inject_residuals:
            fused_features = fused_features + gate * residual
        result.update(
            {
                "position_probs": consumed,
                "position_residual": residual,
                "position_gate": gate,
                "position_gate_dynamic": dynamic_gate,
                "position_gate_used": gate,
            }
        )
        if position_probs is not None:
            result.pop("position_logits", None)
        log["consumed"]["position_probs_sha256"] = _tensor_sha256(consumed)
        log.setdefault("original", {})
        log["original"]["position_probs_sha256"] = _tensor_sha256(original)
        log["consumed"]["position_gate_sha256"] = _tensor_sha256(gate)
        log["original"]["position_gate_sha256"] = _tensor_sha256(original_gate)

    if model.use_lesion:
        original = outputs["lesion_attention"]
        consumed = (
            _checked_attention(lesion_attention, expected=original)
            if lesion_attention is not None
            else original
        ).detach()
        normalized = consumed / consumed.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        lesion_features = (normalized * outputs["layer3_features"]).sum(dim=(2, 3))
        residual = model.lesion_residual(lesion_features)
        dynamic_gate = model.gate_max * torch.sigmoid(
            model.lesion_gate(
                torch.cat(
                    [lesion_features, model._attention_statistics(consumed)], dim=1
                )
            )
        )
        original_gate = _checked_gate(outputs["lesion_gate"], expected=dynamic_gate)
        gate = original_gate if fixed_original_gates else dynamic_gate
        if not close_lesion and model.inject_residuals:
            fused_features = fused_features + gate * residual
        result.update(
            {
                "lesion_attention": consumed,
                "lesion_features": lesion_features,
                "lesion_residual": residual,
                "lesion_gate": gate,
                "lesion_gate_dynamic": dynamic_gate,
                "lesion_gate_used": gate,
            }
        )
        if lesion_attention is not None:
            result.pop("lesion_logits", None)
        log["consumed"]["lesion_attention_sha256"] = _tensor_sha256(consumed)
        log.setdefault("original", {})
        log["original"]["lesion_attention_sha256"] = _tensor_sha256(original)
        log["consumed"]["lesion_gate_sha256"] = _tensor_sha256(gate)
        log["original"]["lesion_gate_sha256"] = _tensor_sha256(original_gate)

    result["fused_features"] = fused_features
    result["logits"] = model.grading_head(fused_features)
    log["fused_features_sha256"] = _tensor_sha256(fused_features)
    return result, log


def concept_override(
    model: SFibAIModel,
    outputs: dict[str, torch.Tensor],
    **kwargs: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Named public alias for the fixed-checkpoint concept intervention API."""
    return intervene(model, outputs, **kwargs)


__all__ = ["concept_override", "intervene"]
