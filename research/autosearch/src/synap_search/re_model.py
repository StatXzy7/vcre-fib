"""SynAP-Fib-RE: view-conditioned regional evidence grader.

This module is deliberately separate from ``sfibai_b.model`` and the M0--M4
search models.  It implements the new research round only; the historical
A--G definitions and their checkpoints are not changed by importing it.

The grading path is

    c_vu = lambda * V_u * stopgrad(a_u) * s_vu / sum(V)
    D_v = sum_u c_vu
    ell_v = context_logits + D_v
    p = sum_v stopgrad(q_v) * softmax(ell_v)

where ``s_vu`` is produced before any spatial pooling by six view adapters and
a shared 128 -> 64 -> 36 MLP.  The context head remains available when the
regional scale is zero, which is the RE control condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from sfibai_b.model import ResNet50SpatialBackbone, _component_seed, _grading_head


@dataclass(frozen=True)
class REConfig:
    """Structural knobs frozen for the SynAP-Fib-RE round."""

    num_views: int = 6
    num_classes: int = 36
    local_dim: int = 128
    evidence_hidden_dim: int = 64
    regional_lambda: float = 1.0
    support_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_views != 6 or self.num_classes != 36:
            raise ValueError("RE fixes six views and 36 grading bins")
        if self.local_dim != 128 or self.evidence_hidden_dim != 64:
            raise ValueError("RE fixes the 128 -> 64 -> 36 local evidence MLP")
        if self.regional_lambda < 0 or not math.isfinite(self.regional_lambda):
            raise ValueError("regional_lambda must be finite and non-negative")
        if self.support_eps <= 0:
            raise ValueError("support_eps must be positive")


def _require_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("RE forward accepts only a B x 3 x H x W image tensor")
    if not torch.isfinite(image).all():
        raise ValueError("image must be finite")


def _normalize_probabilities(
    value: torch.Tensor, *, expected_shape: tuple[int, ...], name: str
) -> torch.Tensor:
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {expected_shape}")
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    total = value.sum(dim=1, keepdim=True)
    if (total <= 0).any():
        raise ValueError(f"{name} must have positive row sums")
    return value / total


def _validate_map(
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, int, int, int],
    name: str,
    upper: float | None = None,
) -> torch.Tensor:
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {expected_shape}")
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    if upper is not None and (value > upper).any():
        raise ValueError(f"{name} must be <= {upper}")
    return value


class SynAPFibRE(nn.Module):
    """Independent RE model with image-only inference and replaceable concepts."""

    def __init__(
        self,
        *,
        seed: int = 34001,
        pretrained: bool = False,
        config: REConfig | None = None,
        context_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.config = config or REConfig()
        self.seed = int(seed)
        self.backbone = ResNet50SpatialBackbone(
            pretrained=pretrained, initialization_seed=self.seed + 101
        )
        with _component_seed(self.seed + 202):
            self.context_head = _grading_head(
                feature_dim=self.backbone.feature_dim,
                hidden_dim=context_hidden_dim,
                num_classes=self.config.num_classes,
                dropout=0.0,
            )
        with _component_seed(self.seed + 10_001):
            self.view_head = nn.Linear(self.backbone.feature_dim, self.config.num_views)
        with _component_seed(self.seed + 20_001):
            self.localization_head = nn.Sequential(
                nn.Conv2d(self.backbone.spatial_dim, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
        with _component_seed(self.seed + 30_001):
            self.local_projection = nn.Conv2d(
                self.backbone.spatial_dim, self.config.local_dim, 1
            )
            self.view_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.config.local_dim, self.config.local_dim, 1),
                        nn.GELU(),
                    )
                    for _ in range(self.config.num_views)
                ]
            )
            self.evidence_mlp = nn.Sequential(
                nn.Linear(self.config.local_dim, self.config.evidence_hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.evidence_hidden_dim, self.config.num_classes),
            )
            self.support_head = nn.Conv2d(self.config.local_dim, 1, 1)

    @property
    def num_views(self) -> int:
        return self.config.num_views

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @staticmethod
    def _score(probabilities: torch.Tensor) -> torch.Tensor:
        bins = torch.arange(
            probabilities.shape[-1], device=probabilities.device, dtype=probabilities.dtype
        )
        return (probabilities * bins).sum(dim=-1) / 10.0

    @staticmethod
    def _local_scores(
        projected: torch.Tensor, adapters: nn.ModuleList, evidence_mlp: nn.Module
    ) -> torch.Tensor:
        # B x V x H x W x K.  No spatial pooling occurs before the evidence MLP.
        scores = []
        for adapter in adapters:
            local = adapter(projected).permute(0, 2, 3, 1)
            scores.append(evidence_mlp(local))
        return torch.stack(scores, dim=1)

    def _compose(
        self,
        *,
        context_logits: torch.Tensor,
        view_logits: torch.Tensor,
        view_probs: torch.Tensor,
        attention: torch.Tensor,
        support: torch.Tensor,
        local_scores: torch.Tensor,
        region_keep_mask: torch.Tensor | None = None,
        close_region: bool = False,
        regional_lambda: float | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, _, height, width = attention.shape
        q = _normalize_probabilities(
            view_probs,
            expected_shape=(batch, self.num_views),
            name="view_probs",
        ).to(dtype=context_logits.dtype)
        _validate_map(
            attention,
            expected_shape=(batch, 1, height, width),
            name="attention",
            upper=1.0,
        )
        _validate_map(
            support,
            expected_shape=(batch, 1, height, width),
            name="support",
        )
        if tuple(local_scores.shape) != (batch, self.num_views, height, width, self.num_classes):
            raise ValueError("local_scores has an incompatible view/spatial/class shape")
        if region_keep_mask is not None:
            _validate_map(
                region_keep_mask,
                expected_shape=(batch, 1, height, width),
                name="region_keep_mask",
                upper=1.0,
            )
            attention = attention * region_keep_mask
        if close_region:
            attention = torch.zeros_like(attention)
        weights = support * attention.detach()
        denominator = support.flatten(1).sum(dim=1, keepdim=True)
        normalized = weights.flatten(1) / denominator.clamp_min(self.config.support_eps)
        # Empty support maps are a valid control and produce a zero residual.
        normalized = torch.where(denominator > self.config.support_eps, normalized, torch.zeros_like(normalized))
        regional = torch.einsum(
            "bvhwk,bhw->bvk", local_scores, normalized.view(batch, height, width)
        )
        scale = self.config.regional_lambda if regional_lambda is None else float(regional_lambda)
        if scale < 0 or not math.isfinite(scale):
            raise ValueError("regional_lambda must be finite and non-negative")
        regional = regional * scale
        ell = context_logits[:, None, :] + regional
        context_probabilities = F.softmax(context_logits.float(), dim=-1)
        if scale == 0.0 or bool(torch.count_nonzero(regional) == 0):
            # Exact context control: avoid a second softmax and a floating-point
            # reduction over six identical view copies.
            p_views = context_probabilities[:, None, :].expand(-1, self.num_views, -1)
            p = context_probabilities
        else:
            p_views = F.softmax(ell.float(), dim=-1)
            p = (q.detach().float()[:, :, None] * p_views).sum(dim=1)
        return {
            "context_logits": context_logits,
            "context_probabilities": context_probabilities,
            "context_score": self._score(context_probabilities),
            "view_logits": view_logits,
            "view_probs": q,
            "view_probs_for_grade": q.detach(),
            "attention": attention,
            "support": support,
            "local_scores": local_scores,
            "regional_logits": regional,
            "regional_lambda": context_logits.new_tensor(scale),
            "view_conditioned_logits": ell,
            "view_conditioned_probabilities": p_views,
            "probabilities": p,
            "score": self._score(p),
            "grade": torch.bucketize(
                self._score(p),
                torch.tensor([0.5, 1.5, 2.5], device=p.device, dtype=p.dtype),
                right=True,
            ),
        }

    def forward(
        self,
        image: torch.Tensor,
        *,
        view_override: torch.Tensor | None = None,
        support_override: torch.Tensor | None = None,
        attention_override: torch.Tensor | None = None,
        region_keep_mask: torch.Tensor | None = None,
        close_region: bool = False,
        regional_lambda: float | None = None,
    ) -> dict[str, torch.Tensor]:
        _require_image(image)
        layer3, global_features = self.backbone(image)
        context_logits = self.context_head(global_features)
        view_logits = self.view_head(global_features)
        native_q = F.softmax(view_logits.float(), dim=1).to(view_logits.dtype)
        projected = self.local_projection(layer3)
        local_scores = self._local_scores(projected, self.view_adapters, self.evidence_mlp)
        support = F.softplus(self.support_head(projected))
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            localization_logits = self.localization_head(layer3.float())
        attention = localization_logits.sigmoid()
        q = native_q if view_override is None else _normalize_probabilities(
            view_override,
            expected_shape=(image.shape[0], self.num_views),
            name="view_override",
        ).to(native_q)
        support = support if support_override is None else _validate_map(
            support_override,
            expected_shape=tuple(support.shape),
            name="support_override",
        ).to(support)
        attention = attention if attention_override is None else _validate_map(
            attention_override,
            expected_shape=tuple(attention.shape),
            name="attention_override",
            upper=1.0,
        ).to(attention)
        result = self._compose(
            context_logits=context_logits,
            view_logits=view_logits,
            view_probs=q,
            attention=attention,
            support=support,
            local_scores=local_scores,
            region_keep_mask=region_keep_mask,
            close_region=close_region,
            regional_lambda=regional_lambda,
        )
        result.update(
            {
                "layer3_features": layer3,
                "global_features": global_features,
                "projected_features": projected,
                "localization_logits": localization_logits,
                "native_view_probs": native_q,
            }
        )
        return result

    def recompose(
        self,
        outputs: Mapping[str, torch.Tensor],
        *,
        view_override: torch.Tensor | None = None,
        support_override: torch.Tensor | None = None,
        attention_override: torch.Tensor | None = None,
        region_keep_mask: torch.Tensor | None = None,
        close_region: bool = False,
        regional_lambda: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Recompute the consumed grading path from a fixed image forward pass."""
        required = {
            "context_logits",
            "view_logits",
            "view_probs",
            "attention",
            "support",
            "local_scores",
        }
        missing = sorted(required - set(outputs))
        if missing:
            raise KeyError(f"outputs missing RE tensors: {missing}")
        attention = outputs["attention"] if attention_override is None else attention_override
        support = outputs["support"] if support_override is None else support_override
        view_probs = outputs["view_probs"] if view_override is None else view_override
        if regional_lambda is None and "regional_lambda" in outputs:
            regional_lambda = float(outputs["regional_lambda"].detach().cpu().item())
        result = self._compose(
            context_logits=outputs["context_logits"],
            view_logits=outputs["view_logits"],
            view_probs=view_probs,
            attention=attention,
            support=support,
            local_scores=outputs["local_scores"],
            region_keep_mask=region_keep_mask,
            close_region=close_region,
            regional_lambda=regional_lambda,
        )
        for key in ("layer3_features", "global_features", "projected_features", "localization_logits", "native_view_probs"):
            if key in outputs:
                result[key] = outputs[key]
        # A replacement must not leave stale branch logits looking like the
        # values consumed by grading.  Preserve them under explicit raw names.
        if view_override is not None:
            result["raw_view_logits"] = result.pop("view_logits")
        if attention_override is not None or close_region or region_keep_mask is not None:
            if "localization_logits" in result:
                result["raw_localization_logits"] = result.pop("localization_logits")
        return result


def build_re_model(
    *, seed: int = 34001, pretrained: bool = False, config: REConfig | None = None
) -> SynAPFibRE:
    """Factory kept separate from ``sfibai_b.model.build_model`` on purpose."""
    return SynAPFibRE(seed=seed, pretrained=pretrained, config=config)


__all__ = ["REConfig", "SynAPFibRE", "build_re_model"]
