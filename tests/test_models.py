from __future__ import annotations

import pytest
import torch
from torchvision import models

from sfibai_b.model import (
    ARM_RESIDUAL_INJECTION,
    RESNET50_PRETRAINED_WEIGHTS,
    SFibAIModel,
    build_model,
)


def test_pretrained_weight_is_explicit_imagenet1k_v2() -> None:
    assert RESNET50_PRETRAINED_WEIGHTS is models.ResNet50_Weights.IMAGENET1K_V2


def test_a_to_f_use_true_structure_ablation() -> None:
    expected = {
        "A": (False, False),
        "B": (False, False),
        "C": (True, False),
        "D": (False, True),
        "E": (True, True),
        "F": (True, True),
        "G": (True, True),
    }
    for arm, (has_position, has_lesion) in expected.items():
        model = build_model(arm=arm, seed=2026, pretrained=False)
        assert hasattr(model, "position_head") is has_position
        assert hasattr(model, "lesion_head") is has_lesion
        assert model.use_position is has_position
        assert model.use_lesion is has_lesion
    assert ARM_RESIDUAL_INJECTION == {
        "A": False,
        "B": False,
        "C": True,
        "D": True,
        "E": True,
        "F": False,
        "G": True,
    }


def test_arm_f_keeps_e_modules_and_losses_but_never_injects() -> None:
    model = build_model(arm="F", seed=2026, pretrained=False).eval()

    assert model.inject_residuals is False
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))

    # F emits the same output keys as E so the prediction schema is shared.
    assert "position_gate" in outputs and "lesion_gate" in outputs
    # The grading pathway must be exactly the un-augmented backbone path.
    assert torch.equal(outputs["fused_features"], outputs["global_features"])


def test_arm_f_rejects_injection_flag_mismatch() -> None:
    with pytest.raises(ValueError, match="inject_residuals"):
        SFibAIModel(
            arm="F",
            seed=2026,
            pretrained=False,
            use_position=True,
            use_lesion=True,
            inject_residuals=True,
        )


def test_arm_f_shares_all_auxiliary_tensors_with_e_bitwise() -> None:
    full = build_model(arm="E", seed=2026, pretrained=False)
    aux_only = build_model(arm="F", seed=2026, pretrained=False)

    for prefix in ("backbone.", "grading_head.", "position_", "lesion_"):
        left = {
            name: tensor
            for name, tensor in full.state_dict().items()
            if name.startswith(prefix)
        }
        right = {
            name: tensor
            for name, tensor in aux_only.state_dict().items()
            if name.startswith(prefix)
        }
        assert set(left) == set(right)
        for name, tensor in left.items():
            assert torch.equal(tensor, right[name]), name


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
        inject_residuals=False,
    ).eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 64, 64))

    assert set(outputs) == {"logits", "global_features", "fused_features"}
    assert outputs["logits"].shape == (1, 36)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lesion_head_stays_finite_when_outer_amp_would_overflow() -> None:
    model = build_model(arm="D", seed=2026, pretrained=False).cuda().eval()
    first_conv = model.lesion_head[0]
    final_conv = model.lesion_head[2]
    with torch.no_grad():
        first_conv.weight.fill_(1.0)
        first_conv.bias.zero_()
        final_conv.weight.fill_(1e-6)
        final_conv.bias.zero_()
        features = torch.full((1, 1024, 4, 4), 1000.0, device="cuda")

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            amp_logits = model.lesion_head(features)
            stable_logits = model._lesion_logits_fp32(features)

    assert not torch.isfinite(amp_logits).all()
    assert stable_logits.dtype is torch.float32
    assert torch.isfinite(stable_logits).all()
