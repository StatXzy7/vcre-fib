"""Training adapter for SynAP-Fib-RE.

The adapter reuses the historical hybrid grading and weak-box objectives while
keeping the RE probability mixture explicit.  It is not imported by the old
A--G runner and does not open a data loader or test set.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sfibai_b.loss import SFibAIObjective


class REObjective(nn.Module):
    """Legacy-compatible losses with RE-specific gradient boundaries."""

    def __init__(self, *, lambda_position: float = 0.1, lambda_box: float = 0.1) -> None:
        super().__init__()
        self.base = SFibAIObjective(
            lambda_position=lambda_position,
            lambda_box=lambda_box,
            box_inside_weight=1.0,
            box_outside_weight=1.0,
        )

    def set_epoch(self, epoch: int) -> float:
        return self.base.set_epoch(epoch)

    @staticmethod
    def _adapt(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"probabilities", "view_logits", "localization_logits"}
        missing = sorted(required - set(outputs))
        if missing:
            raise KeyError(f"RE objective outputs missing {missing}")
        # log(p) is the effective output-layer distribution.  This preserves
        # the probability mixture instead of averaging logits and softmaxing.
        return {
            "logits": outputs["probabilities"].float().clamp_min(1e-8).log(),
            "position_logits": outputs["view_logits"],
            "lesion_logits": outputs["localization_logits"],
        }

    def loss_components(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        return self.base.loss_components(self._adapt(outputs), batch)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        return self.loss_components(outputs, batch)["total"]


__all__ = ["REObjective"]
