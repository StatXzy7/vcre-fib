from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sfibai_b.loss import SFibAIObjective, build_objective
from sfibai_b.model import build_model


class TinyBackbone(nn.Module):
    spatial_dim = 1024
    feature_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.probe(x)
        local = F.adaptive_avg_pool2d(base, (2, 2)).expand(-1, 1024, -1, -1)
        global_features = base.mean(dim=(2, 3)).expand(-1, 2048)
        return local, global_features


def gradient_sum(module: nn.Module) -> float:
    return float(
        sum(
            parameter.grad.detach().abs().sum()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def zero_grad(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def full_model() -> nn.Module:
    model = build_model(arm="E", seed=2026, pretrained=False)
    model.backbone = TinyBackbone()
    return model


def test_arm_losses_have_locked_auxiliary_weights() -> None:
    expected = {
        "A": (0.0, 0.0),
        "B": (0.0, 0.0),
        "C": (0.1, 0.0),
        "D": (0.0, 0.1),
        "E": (0.1, 0.1),
    }
    for arm, weights in expected.items():
        objective = build_objective(arm)
        assert (objective.lambda_position, objective.lambda_box) == weights
        assert objective.box_inside_weight == 1.0
        assert objective.box_outside_weight == 1.0


def test_weak_box_uses_positive_grade_with_box_only() -> None:
    objective = SFibAIObjective(lambda_position=0.0, lambda_box=0.1)
    outputs = {
        "logits": torch.zeros(3, 36, requires_grad=True),
        "lesion_logits": torch.zeros(3, 1, 4, 4, requires_grad=True),
    }
    mask = torch.zeros(3, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 1
    batch = {
        "label_bin": torch.tensor([0, 10, 20]),
        "lesion_mask": mask,
        "lesion_box_valid": torch.tensor([True, True, False]),
    }

    parts = objective.loss_components(outputs, batch)

    assert parts["valid_box_count"].item() == 1
    assert parts["grade0_ignored_box_count"].item() == 1
    assert parts["missing_box_count"].item() == 1
    assert torch.allclose(parts["weighted_box"], 0.1 * parts["box_raw"])


def test_position_loss_updates_head_not_backbone() -> None:
    model = full_model()
    outputs = model(torch.randn(2, 3, 8, 8))

    F.cross_entropy(outputs["position_logits"], torch.tensor([0, 1])).backward()

    assert gradient_sum(model.position_head) > 0
    assert gradient_sum(model.backbone) == 0


def test_box_loss_updates_lesion_head_not_backbone() -> None:
    model = full_model()
    outputs = model(torch.randn(2, 3, 8, 8))
    objective = SFibAIObjective(lambda_position=0.0, lambda_box=1.0)
    mask = torch.zeros(2, 1, 2, 2)
    mask[:, :, 0, 0] = 1
    batch = {
        "label_bin": torch.tensor([10, 20]),
        "lesion_mask": mask,
        "lesion_box_valid": torch.tensor([True, True]),
    }

    objective.loss_components(outputs, batch)["box_raw"].backward()

    assert gradient_sum(model.lesion_head) > 0
    assert gradient_sum(model.backbone) == 0


def test_grading_updates_residual_paths_but_not_auxiliary_heads() -> None:
    model = full_model()
    outputs = model(torch.randn(2, 3, 8, 8))

    outputs["logits"].square().mean().backward()

    assert gradient_sum(model.position_head) == 0
    assert gradient_sum(model.lesion_head) == 0
    assert gradient_sum(model.position_residual) > 0
    assert gradient_sum(model.position_gate) > 0
    assert gradient_sum(model.lesion_residual) > 0
    assert gradient_sum(model.lesion_gate) > 0
    assert gradient_sum(model.backbone) > 0
