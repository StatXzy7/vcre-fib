from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


ARM_LOSS_WEIGHTS: dict[str, tuple[float, float]] = {
    "A": (0.0, 0.0),
    "B": (0.0, 0.0),
    "C": (0.1, 0.0),
    "D": (0.0, 0.1),
    "E": (0.1, 0.1),
}


def _soft_labels(labels: torch.Tensor, classes: int, std: float) -> torch.Tensor:
    indices = torch.arange(classes, device=labels.device, dtype=torch.float32).unsqueeze(0)
    centers = labels.to(torch.float32).unsqueeze(1)
    weights = torch.exp(-0.5 * torch.square((indices - centers) / std))
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


class SFibAIObjective(nn.Module):
    """Original SFibAI hybrid grading loss plus locked auxiliary objectives."""

    def __init__(
        self,
        *,
        lambda_position: float,
        lambda_box: float,
        num_classes: int = 36,
        num_positions: int = 6,
        alpha: float = 1.0,
        beta: float = 0.02,
        gamma: float = 0.02,
        soft_label_std: float = 1.0,
        box_inside_weight: float = 1.0,
        box_outside_weight: float = 1.0,
        inside_pool_temperature: float = 10.0,
    ) -> None:
        super().__init__()
        values = (
            lambda_position,
            lambda_box,
            alpha,
            beta,
            gamma,
            box_inside_weight,
            box_outside_weight,
        )
        if any(value < 0 for value in values):
            raise ValueError("Loss weights must be non-negative")
        if soft_label_std <= 0 or inside_pool_temperature <= 0:
            raise ValueError("Loss temperatures must be positive")
        self.lambda_position = float(lambda_position)
        self.lambda_box = float(lambda_box)
        self.num_classes = int(num_classes)
        self.num_positions = int(num_positions)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.soft_label_std = float(soft_label_std)
        self.box_inside_weight = float(box_inside_weight)
        self.box_outside_weight = float(box_outside_weight)
        self.inside_pool_temperature = float(inside_pool_temperature)

    def _grading_components(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        soft = _soft_labels(labels, self.num_classes, self.soft_label_std)
        kl_all = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="none")
        offsets = torch.arange(-2, 3, device=labels.device)
        nearby = (labels.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            0, self.num_classes - 1
        )
        kl = kl_all.gather(1, nearby).sum(dim=1).mean()

        bins = torch.arange(
            self.num_classes, device=logits.device, dtype=logits.dtype
        )
        expected_bin = (logits.softmax(dim=1) * bins).sum(dim=1)
        mse = F.mse_loss(expected_bin, labels.to(expected_bin.dtype))
        boundaries = expected_bin.new_tensor([5.0, 15.0, 25.0])
        predicted_grade = torch.bucketize(expected_bin, boundaries, right=True)
        true_grade = torch.bucketize(labels.to(expected_bin.dtype), boundaries, right=True)
        boundary = (
            torch.abs(expected_bin - labels.to(expected_bin.dtype))
            * (predicted_grade != true_grade).to(expected_bin.dtype)
        ).mean()
        grading = self.alpha * kl + self.beta * mse + self.gamma * boundary
        return {
            "grading": grading,
            "grading_kl": kl,
            "grading_mse": mse,
            "grading_boundary": boundary,
        }

    def _position_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> torch.Tensor:
        if self.lambda_position == 0.0:
            return outputs["logits"].sum() * 0.0
        if "position_logits" not in outputs or "position_norm" not in batch:
            raise KeyError("Position-enabled objective requires logits and position_norm")
        labels = batch["position_norm"].to(outputs["logits"].device).long()
        if not ((labels >= 1) & (labels <= self.num_positions)).all():
            raise ValueError("position_norm must contain only values 1..6")
        return F.cross_entropy(outputs["position_logits"], labels - 1)

    def _box_components(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        zero = outputs["logits"].sum() * 0.0
        if self.lambda_box == 0.0:
            return {
                "box_inside": zero,
                "box_outside": zero,
                "box_raw": zero,
                "valid_box_count": zero.detach(),
                "grade0_ignored_box_count": zero.detach(),
                "missing_box_count": zero.detach(),
            }
        required_outputs = {"lesion_logits"}
        required_batch = {"lesion_mask", "lesion_box_valid"}
        if not required_outputs.issubset(outputs) or not required_batch.issubset(batch):
            raise KeyError("Lesion-enabled objective requires logits, mask, and validity")

        lesion_logits = outputs["lesion_logits"]
        target = batch["lesion_mask"].to(
            device=lesion_logits.device, dtype=lesion_logits.dtype
        )
        if target.shape[-2:] != lesion_logits.shape[-2:]:
            target = F.interpolate(target, size=lesion_logits.shape[-2:], mode="area")
        target = target.clamp(0.0, 1.0)
        has_box = batch["lesion_box_valid"].to(lesion_logits.device).bool()
        positive_grade = labels > 0
        nonempty = target.flatten(1).sum(dim=1) > 0
        valid = has_box & positive_grade & nonempty
        grade0_ignored = has_box & ~positive_grade
        missing = ~has_box
        if not valid.any():
            lesion_zero = lesion_logits.sum() * 0.0
            return {
                "box_inside": lesion_zero,
                "box_outside": lesion_zero,
                "box_raw": lesion_zero,
                "valid_box_count": valid.sum().to(lesion_logits.dtype),
                "grade0_ignored_box_count": grade0_ignored.sum().to(
                    lesion_logits.dtype
                ),
                "missing_box_count": missing.sum().to(lesion_logits.dtype),
            }

        attention = torch.sigmoid(lesion_logits[valid]).flatten(1)
        inside = target[valid].flatten(1) > 0
        inside_count = inside.sum(dim=1).to(attention.dtype)
        temperature = self.inside_pool_temperature
        inside_pool = (
            torch.logsumexp(
                (temperature * attention).masked_fill(~inside, -torch.inf), dim=1
            )
            - inside_count.log()
        ) / temperature
        inside_loss = (1.0 - inside_pool.clamp(0.0, 1.0)).mean()
        outside_mass = (attention * (~inside).to(attention.dtype)).sum(dim=1)
        outside_ratio = outside_mass / attention.sum(dim=1).clamp_min(1e-8)
        outside_loss = outside_ratio.mean()
        raw = (
            self.box_inside_weight * inside_loss
            + self.box_outside_weight * outside_loss
        )
        return {
            "box_inside": inside_loss,
            "box_outside": outside_loss,
            "box_raw": raw,
            "valid_box_count": valid.sum().to(lesion_logits.dtype),
            "grade0_ignored_box_count": grade0_ignored.sum().to(
                lesion_logits.dtype
            ),
            "missing_box_count": missing.sum().to(lesion_logits.dtype),
        }

    def loss_components(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        if "logits" not in outputs or "label_bin" not in batch:
            raise KeyError("Objective requires logits and label_bin")
        labels = batch["label_bin"].to(outputs["logits"].device).long()
        grading = self._grading_components(outputs["logits"], labels)
        position = self._position_loss(outputs, batch)
        box = self._box_components(outputs, labels, batch)
        weighted_position = self.lambda_position * position
        weighted_box = self.lambda_box * box["box_raw"]
        return {
            **grading,
            "position": position,
            "weighted_position": weighted_position,
            **box,
            "weighted_box": weighted_box,
            "total": grading["grading"] + weighted_position + weighted_box,
        }

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> torch.Tensor:
        return self.loss_components(outputs, batch)["total"]


def build_objective(arm: str) -> SFibAIObjective:
    normalized_arm = arm.upper()
    if normalized_arm not in ARM_LOSS_WEIGHTS:
        raise ValueError(f"Unknown ablation arm: {arm!r}")
    position_weight, box_weight = ARM_LOSS_WEIGHTS[normalized_arm]
    return SFibAIObjective(
        lambda_position=position_weight,
        lambda_box=box_weight,
        box_inside_weight=1.0,
        box_outside_weight=1.0,
    )
