from __future__ import annotations

import io

import pytest
import torch
import torch.nn.functional as F

from synap_search.re_intervention import (
    delta_off,
    intervene,
    path_integral_contributions,
    region_masks_from_boxes,
)
from synap_search.re_model import REConfig, build_re_model
from synap_search.re_training import REObjective


@pytest.fixture()
def model() -> torch.nn.Module:
    torch.set_num_threads(2)
    return build_re_model(seed=34001, pretrained=False).eval()


@pytest.fixture()
def batch() -> torch.Tensor:
    return torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(12))


def test_re_shapes_probability_mixture_and_score(model, batch):
    with torch.no_grad():
        output = model(batch)
    assert output["layer3_features"].shape[1] == 1024
    assert output["projected_features"].shape[1] == 128
    assert output["local_scores"].shape[1:] == (6, 4, 4, 36)
    assert output["view_probs"].shape == (2, 6)
    assert output["probabilities"].shape == (2, 36)
    assert torch.allclose(output["probabilities"].sum(1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(output["score"]).all()
    assert ((output["score"] >= 0) & (output["score"] <= 3.5)).all()
    expected = (output["view_probs"][:, :, None] * output["view_conditioned_probabilities"]).sum(1)
    assert torch.allclose(expected, output["probabilities"], atol=1e-6)


def test_zero_regional_residual_recovers_context_prediction(batch):
    model = build_re_model(
        seed=34001, pretrained=False, config=REConfig(regional_lambda=0.0)
    ).eval()
    with torch.no_grad():
        output = model(batch)
    assert torch.equal(output["probabilities"], output["context_probabilities"])
    assert torch.equal(output["score"], output["context_score"])
    assert torch.equal(output["regional_logits"], torch.zeros_like(output["regional_logits"]))


def test_hard_view_override_consumes_the_selected_view(model, batch):
    with torch.no_grad():
        output = model(batch)
        hard = F.one_hot(torch.tensor([2, 5]), num_classes=6).float()
        replaced, record = intervene(model, output, view_probs=hard)
    assert torch.allclose(replaced["probabilities"], replaced["view_conditioned_probabilities"][:, [2, 5], :].diagonal(dim1=0, dim2=1).T, atol=1e-6)
    assert record["view_override"] is True
    assert record["consumed"]["view_probs_sha256"]


def test_common_view_relabeling_is_invariant(model, batch):
    permutation = torch.tensor([2, 5, 1, 3, 0, 4])
    with torch.no_grad():
        output = model(batch)
        relabeled = dict(output)
        relabeled["view_probs"] = output["view_probs"][:, permutation]
        relabeled["local_scores"] = output["local_scores"][:, permutation]
        relabeled_output = model.recompose(relabeled)
    assert torch.allclose(output["probabilities"], relabeled_output["probabilities"], atol=1e-6)
    assert torch.allclose(output["score"], relabeled_output["score"], atol=1e-6)


def test_empty_support_and_invalid_controls_are_explicit(model, batch):
    with torch.no_grad():
        output = model(batch)
        empty = model(batch, support_override=torch.zeros_like(output["support"]))
    assert torch.isfinite(empty["probabilities"]).all()
    assert torch.equal(empty["regional_logits"], torch.zeros_like(empty["regional_logits"]))
    with pytest.raises(ValueError, match="view_override"):
        model(batch, view_override=torch.tensor([[1.0, -1.0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]))
    with pytest.raises(ValueError, match="attention_override"):
        model(batch, attention_override=torch.full_like(output["attention"], 2.0))


def test_region_grouping_overlap_and_coordinate_transform():
    normalized = region_masks_from_boxes(
        [{"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5}],
        feature_hw=(4, 4),
    )
    pixels = region_masks_from_boxes(
        [(0.0, 0.0, 32.0, 32.0)],
        feature_hw=(4, 4),
        coordinate_frame="roi_pixel",
        source_hw=(64, 64),
    )
    assert torch.equal(normalized, pixels)
    overlap = region_masks_from_boxes(
        [(0.0, 0.0, 0.75, 0.75), (0.25, 0.25, 1.0, 1.0)], feature_hw=(4, 4)
    )
    assert overlap.shape == (2, 4, 4)
    assert overlap[:, 1, 1].sum() == 2


def test_path_integral_completeness_and_empty_groups(model, batch):
    with torch.no_grad():
        output = model(batch)
    regions = region_masks_from_boxes(
        [(0.0, 0.0, 0.6, 0.6), (0.4, 0.4, 1.0, 1.0)], feature_hw=output["attention"].shape[-2:]
    )
    quantized = path_integral_contributions(output, region_masks=regions, steps=96)
    assert quantized["A_R"].shape == (2, 3)
    assert float(quantized["completeness_error"]) < 1e-5
    assert float(quantized["decomposition_error"]) < 1e-5
    assert torch.allclose(
        quantized["A_R"].sum(1) + quantized["context_score"], output["score"], atol=2e-5
    )
    empty = path_integral_contributions(output, steps=32)
    assert empty["A_R"].shape == (2, 1)
    assert float(empty["completeness_error"]) < 1e-4


def test_per_sample_region_groups_and_actual_map_override(model, batch):
    with torch.no_grad():
        output = model(batch)
        masks = torch.zeros(2, 1, 4, 4)
        masks[0, 0, :2, :2] = 1
        masks[1, 0, 2:, 2:] = 1
        quantized = path_integral_contributions(output, region_masks=masks, steps=64)
        overridden, record = intervene(
            model,
            output,
            attention=torch.zeros_like(output["attention"]),
            support=torch.ones_like(output["support"]),
        )
    assert quantized["A_R"].shape == (2, 2)
    assert float(quantized["completeness_error"]) < 1e-4
    assert torch.equal(overridden["regional_logits"], torch.zeros_like(output["regional_logits"]))
    assert record["attention_override"] and record["support_override"]
    assert "raw_localization_logits" in overridden


def test_actual_region_close_is_distinct_from_additive_quantification(model, batch):
    with torch.no_grad():
        output = model(batch)
        closed, record = intervene(model, output, close_region=True)
    assert torch.equal(closed["regional_logits"], torch.zeros_like(closed["regional_logits"]))
    assert torch.allclose(closed["probabilities"], output["context_probabilities"], atol=1e-6)
    assert torch.allclose(delta_off(model, output), closed["score"] - output["score"], atol=1e-7)
    assert record["fixed_feature_intervention"] is True
    assert record["pixel_perturbation"] is False


def test_recompose_inherits_nondefault_forward_lambda(batch):
    model = build_re_model(seed=34001, pretrained=False).eval()
    with torch.no_grad():
        output = model(batch, regional_lambda=0.23)
        recomposed, _ = intervene(model, output)
    assert torch.equal(output["regional_lambda"], recomposed["regional_lambda"])
    assert torch.allclose(output["probabilities"], recomposed["probabilities"], atol=1e-6)


def test_single_region_close_mask_is_consumed(model, batch):
    with torch.no_grad():
        output = model(batch)
        close_mask = torch.zeros_like(output["attention"])
        close_mask[:, :, :2, :2] = 1
        closed, record = intervene(model, output, close_mask=close_mask)
    assert torch.allclose(closed["attention"], output["attention"] * (1 - close_mask))
    assert record["close_mask"] is True
    assert torch.isfinite(closed["score"]).all()


def test_gradient_routes_auxiliary_to_backbone_but_not_through_q_or_a(model, batch):
    model.train()
    output = model(batch)
    model.zero_grad(set_to_none=True)
    output["score"].sum().backward()
    view_grad = sum((p.grad.abs().sum() for p in model.view_head.parameters() if p.grad is not None), torch.tensor(0.0))
    map_grad = sum((p.grad.abs().sum() for p in model.localization_head.parameters() if p.grad is not None), torch.tensor(0.0))
    backbone_grad = sum((p.grad.abs().sum() for p in model.backbone.parameters() if p.grad is not None), torch.tensor(0.0))
    local_projection_grad = sum((p.grad.abs().sum() for p in model.local_projection.parameters() if p.grad is not None), torch.tensor(0.0))
    local_evidence_grad = sum((p.grad.abs().sum() for p in model.evidence_mlp.parameters() if p.grad is not None), torch.tensor(0.0))
    assert float(view_grad) == 0.0
    assert float(map_grad) == 0.0
    assert float(backbone_grad) > 0.0
    assert float(local_projection_grad) > 0.0
    assert float(local_evidence_grad) > 0.0

    model.zero_grad(set_to_none=True)
    output = model(batch)
    F.cross_entropy(output["view_logits"], torch.tensor([0, 5])).backward()
    assert sum((p.grad.abs().sum() for p in model.view_head.parameters() if p.grad is not None), torch.tensor(0.0)) > 0
    assert sum((p.grad.abs().sum() for p in model.backbone.parameters() if p.grad is not None), torch.tensor(0.0)) > 0


def test_re_objective_keeps_probability_mixture_and_auxiliary_backbone_routes(model, batch):
    model.train()
    objective = REObjective()
    batch_targets = {
        "label_bin": torch.tensor([4, 25]),
        "position_norm": torch.tensor([1, 6]),
        "lesion_mask": torch.zeros(2, 1, 4, 4),
        "lesion_box_valid": torch.ones(2, dtype=torch.bool),
    }
    batch_targets["lesion_mask"][:, :, 1:3, 1:3] = 1
    output = model(batch)
    model.zero_grad(set_to_none=True)
    objective(output, batch_targets).backward()
    assert sum((p.grad.abs().sum() for p in model.backbone.parameters() if p.grad is not None), torch.tensor(0.0)) > 0
    assert sum((p.grad.abs().sum() for p in model.view_head.parameters() if p.grad is not None), torch.tensor(0.0)) > 0
    assert sum((p.grad.abs().sum() for p in model.localization_head.parameters() if p.grad is not None), torch.tensor(0.0)) > 0

    model.zero_grad(set_to_none=True)
    output = model(batch)
    F.binary_cross_entropy(output["localization_logits"].sigmoid(), torch.zeros_like(output["localization_logits"])).backward()
    assert sum((p.grad.abs().sum() for p in model.localization_head.parameters() if p.grad is not None), torch.tensor(0.0)) > 0
    assert sum((p.grad.abs().sum() for p in model.backbone.parameters() if p.grad is not None), torch.tensor(0.0)) > 0


def test_checkpoint_roundtrip_preserves_re_forward(model, batch):
    with torch.no_grad():
        before = model(batch)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "config": model.config.__dict__}, buffer)
    restored = build_re_model(seed=34001, pretrained=False).eval()
    payload = torch.load(io.BytesIO(buffer.getvalue()), weights_only=False)
    restored.load_state_dict(payload["model"])
    with torch.no_grad():
        after = restored(batch)
    for key in ("probabilities", "score", "view_probs", "attention", "support"):
        assert torch.equal(before[key], after[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_forward_and_probability_outputs_are_finite():
    model = build_re_model(seed=34001, pretrained=False).cuda().eval()
    image = torch.randn(2, 3, 64, 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(19))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(image)
    for key in ("probabilities", "view_probs", "attention", "support", "score"):
        assert torch.isfinite(output[key]).all(), key
    assert torch.allclose(output["probabilities"].sum(1), torch.ones(2, device="cuda"), atol=1e-5)
