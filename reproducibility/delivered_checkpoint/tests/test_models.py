from __future__ import annotations

import torch
from torchvision import models

from sfibai_b.model import (
    RESNET50_PRETRAINED_WEIGHTS,
    SFibAIModel,
    build_model,
)


def test_pretrained_weight_is_explicit_imagenet1k_v2() -> None:
    assert RESNET50_PRETRAINED_WEIGHTS is models.ResNet50_Weights.IMAGENET1K_V2


def test_a_to_e_use_true_structure_ablation() -> None:
    expected = {
        "A": (False, False),
        "B": (False, False),
        "C": (True, False),
        "D": (False, True),
        "E": (True, True),
    }
    for arm, (has_position, has_lesion) in expected.items():
        model = build_model(arm=arm, seed=2026, pretrained=False)
        assert hasattr(model, "position_head") is has_position
        assert hasattr(model, "lesion_head") is has_lesion
        assert model.use_position is has_position
        assert model.use_lesion is has_lesion


def test_shared_modules_are_bitwise_equal_for_same_seed() -> None:
    baseline = build_model(arm="A", seed=2026, pretrained=False)
    full = build_model(arm="E", seed=2026, pretrained=False)

    for name, tensor in baseline.backbone.state_dict().items():
        assert torch.equal(tensor, full.backbone.state_dict()[name])
    for name, tensor in baseline.grading_head.state_dict().items():
        assert torch.equal(tensor, full.grading_head.state_dict()[name])


def test_inactive_branches_do_not_emit_placeholder_outputs() -> None:
    model = SFibAIModel(
        arm="B",
        seed=2026,
        pretrained=False,
        use_position=False,
        use_lesion=False,
    ).eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 64, 64))

    assert set(outputs) == {"logits", "global_features", "fused_features"}
    assert outputs["logits"].shape == (1, 36)
