"""True module deletions from the unchanged Full RE initialization."""
import hashlib

import torch
from torch.nn import functional as F
from synap_search.re_model import SynAPFibRE, _require_image

VARIANTS = ("RE_WO_VIEW", "RE_WO_WEAKLOC")


def tensor_hashes(state):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in state.items()}


class DeletionModel(SynAPFibRE):
    def __init__(self, variant, *, seed=2026, pretrained=True):
        if variant not in (*VARIANTS, "FULL"):
            raise ValueError(variant)
        super().__init__(seed=seed, pretrained=pretrained)
        assert self.config.regional_lambda == 1.0 and self.config.support_eps == 1e-8
        self.variant = variant
        self.full_initial_hashes = tensor_hashes(self.state_dict())
        if variant == "RE_WO_VIEW":
            del self.view_head
            del self.view_adapters
        elif variant == "RE_WO_WEAKLOC":
            del self.localization_head
        self.initial_hashes = tensor_hashes(self.state_dict())
        assert all(self.full_initial_hashes[k] == v for k, v in self.initial_hashes.items())

    def forward(self, image):
        if self.variant == "FULL":
            return super().forward(image)
        _require_image(image)
        layer3, features = self.backbone(image)
        context = self.context_head(features)
        projected = self.local_projection(layer3)
        support = F.softplus(self.support_head(projected))
        if self.variant == "RE_WO_VIEW":
            with torch.autocast(device_type=image.device.type, enabled=False):
                loc = self.localization_head(layer3.float())
            attention = loc.sigmoid()
            scores = self.evidence_mlp(projected.permute(0, 2, 3, 1))
            denominator = support.flatten(1).sum(1, keepdim=True)
            weights = (support * attention.detach()).flatten(1) / denominator.clamp_min(self.config.support_eps)
            weights = torch.where(denominator > self.config.support_eps, weights, torch.zeros_like(weights))
            regional = torch.einsum("bhwk,bhw->bk", scores, weights.view(image.shape[0], *attention.shape[-2:]))
            p = F.softmax((context + regional).float(), dim=-1)
            return {"probabilities": p, "score": self._score(p), "context_logits": context,
                    "regional_logits": regional, "support": support, "attention": attention,
                    "localization_logits": loc, "local_scores": scores}
        view = self.view_head(features)
        q = F.softmax(view.float(), dim=1).to(view.dtype)
        scores = self._local_scores(projected, self.view_adapters, self.evidence_mlp)
        neutral = torch.ones((image.shape[0], 1, *projected.shape[-2:]), device=image.device, dtype=torch.float32)
        result = self._compose(context_logits=context, view_logits=view, view_probs=q,
                               attention=neutral, support=support, local_scores=scores)
        # The neutral multiplier is not a learned localization output.
        result["attention_for_grade"] = result.pop("attention")
        return result
