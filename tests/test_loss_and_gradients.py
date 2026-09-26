from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sfibai_b.loss import SFibAIObjective, aux_scale_for_epoch, build_objective
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
        "F": (0.1, 0.1),
        "G": (0.1, 0.1),
    }
    for arm, weights in expected.items():
        objective = build_objective(arm)
        assert (objective.lambda_position, objective.lambda_box) == weights
        assert objective.box_inside_weight == 1.0
        assert objective.box_outside_weight == 1.0


def test_aux_schedule_is_constant_for_frozen_arms_and_warmup_for_g() -> None:
    # Frozen arms: full auxiliary weight at every epoch
    for arm in "ABCDEF":
        for epoch in (1, 20, 21, 39, 40, 120):
            assert aux_scale_for_epoch(arm, epoch) == 1.0
    # Arm G: zero through burn-in, linear ramp over epochs 21-39, full from 40
    assert aux_scale_for_epoch("G", 1) == 0.0
    assert aux_scale_for_epoch("G", 20) == 0.0
    assert aux_scale_for_epoch("G", 21) == pytest.approx(1 / 20)
    assert aux_scale_for_epoch("G", 30) == pytest.approx(10 / 20)
    assert aux_scale_for_epoch("G", 39) == pytest.approx(19 / 20)
    assert aux_scale_for_epoch("G", 40) == 1.0
    assert aux_scale_for_epoch("G", 120) == 1.0


def test_arm_g_objective_scales_auxiliary_losses_by_epoch() -> None:
    model = build_model(arm="G", seed=2026, pretrained=False)
    model.backbone = TinyBackbone()
    outputs = model(torch.randn(2, 3, 8, 8))
    objective = build_objective("G")
    mask = torch.zeros(2, 1, 2, 2)
    mask[:, :, 0, 0] = 1
    batch = {
        "label_bin": torch.tensor([10, 20]),
        "position_norm": torch.tensor([1, 6]),
        "lesion_mask": mask,
        "lesion_box_valid": torch.tensor([True, True]),
    }

    objective.set_epoch(5)  # burn-in: auxiliary weights effectively zero
    parts = objective.loss_components(outputs, batch)
    assert float(parts["aux_scale"]) == 0.0
    assert float(parts["weighted_position"]) == 0.0
    assert float(parts["weighted_box"]) == 0.0
    assert float(parts["total"]) == pytest.approx(float(parts["grading"]))

    objective.set_epoch(40)  # full auxiliary weight
    parts = objective.loss_components(outputs, batch)
    assert float(parts["aux_scale"]) == 1.0
    assert float(parts["weighted_position"]) == pytest.approx(0.1 * float(parts["position"]))
    assert float(parts["weighted_box"]) == pytest.approx(0.1 * float(parts["box_raw"]))


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


def test_arm_f_grading_is_decoupled_from_all_auxiliary_modules() -> None:
    model = build_model(arm="F", seed=2026, pretrained=False)
    model.backbone = TinyBackbone()
    outputs = model(torch.randn(2, 3, 8, 8))

    outputs["logits"].square().mean().backward()

    # Without injection the grading loss has no path into any auxiliary module.
    assert gradient_sum(model.position_head) == 0
    assert gradient_sum(model.lesion_head) == 0
    assert gradient_sum(model.position_residual) == 0
    assert gradient_sum(model.position_gate) == 0
    assert gradient_sum(model.lesion_residual) == 0
    assert gradient_sum(model.lesion_gate) == 0
    assert gradient_sum(model.backbone) > 0
    assert gradient_sum(model.grading_head) > 0


def test_arm_f_auxiliary_losses_still_train_their_heads() -> None:
    model = build_model(arm="F", seed=2026, pretrained=False)
    model.backbone = TinyBackbone()
    outputs = model(torch.randn(2, 3, 8, 8))
    mask = torch.zeros(2, 1, 2, 2)
    mask[:, :, 0, 0] = 1
    batch = {
        "label_bin": torch.tensor([10, 20]),
        "position_norm": torch.tensor([1, 6]),
        "lesion_mask": mask,
        "lesion_box_valid": torch.tensor([True, True]),
    }

    F.cross_entropy(outputs["position_logits"], torch.tensor([0, 1])).backward()
    assert gradient_sum(model.position_head) > 0
    assert gradient_sum(model.backbone) == 0
    zero_grad(model)

    objective = SFibAIObjective(lambda_position=0.0, lambda_box=1.0)
    objective.loss_components(outputs, batch)["box_raw"].backward()
    assert gradient_sum(model.lesion_head) > 0
    assert gradient_sum(model.backbone) == 0
