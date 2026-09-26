"""Unchanged RE grading loss, with independent enabled auxiliary tasks."""
from torch import nn
from sfibai_b.loss import SFibAIObjective
from model import VARIANTS


class DeletionObjective(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.view = variant != "RE_WO_VIEW"
        self.weakloc = variant != "RE_WO_WEAKLOC"
        self.base = SFibAIObjective(lambda_position=.1 if self.view else 0.,
                                    lambda_box=.1 if self.weakloc else 0.,
                                    box_inside_weight=1., box_outside_weight=1.)

    def loss_components(self, outputs, batch):
        adapted = {"logits": outputs["probabilities"].float().clamp_min(1e-8).log()}
        if self.view:
            adapted["position_logits"] = outputs["view_logits"]
        if self.weakloc:
            adapted["lesion_logits"] = outputs["localization_logits"]
        parts = self.base.loss_components(adapted, batch)
        if not self.view:
            for key in ("position", "weighted_position"):
                parts.pop(key)
        if not self.weakloc:
            for key in ("box_inside", "box_outside", "box_raw", "weighted_box", "valid_box_count",
                        "grade0_ignored_box_count", "missing_box_count"):
                parts.pop(key)
        return parts
