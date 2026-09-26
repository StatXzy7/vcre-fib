"""LUA controls: shared gradients, frozen shared features, independent auxiliary net."""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torchvision import models

from sfibai_b.model import SFibAIModel, _component_seed
from sfibai_b.provenance import _tensor_sha256


FROZEN_ARMS = frozenset({"FROZEN", "INDEPENDENT"})


def grading_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith(("backbone.", "grading_head.")):
            digest.update(name.encode("utf-8"))
            digest.update(_tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


class ResNet18SpatialBackbone(nn.Module):
    spatial_dim = 256
    feature_dim = 512

    def __init__(self, *, pretrained: bool, seed: int) -> None:
        super().__init__()
        with _component_seed(seed + 30_001):
            source = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2",
                     "layer3", "layer4", "avgpool"):
            setattr(self, name, getattr(source, name))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        spatial = self.layer3(self.layer2(self.layer1(x)))
        pooled = torch.flatten(self.avgpool(self.layer4(spatial)), 1)
        return spatial, pooled


class LUAModel(SFibAIModel):
    def __init__(self, *, arm: str, seed: int, pretrained: bool) -> None:
        # Construct exactly the shared A tensors first, preserving paired initialization.
        super().__init__(arm="A", seed=seed, pretrained=pretrained,
                         use_position=False, use_lesion=False, inject_residuals=False)
        if arm not in {"MTL", *FROZEN_ARMS}:
            raise ValueError(f"Unknown LUA arm: {arm}")
        self.arm = arm
        self.use_position = self.use_lesion = True
        self.grading_frozen = arm in FROZEN_ARMS
        feature_dim, spatial_dim = 2048, 1024
        if arm == "INDEPENDENT":
            self.aux_backbone = ResNet18SpatialBackbone(pretrained=pretrained, seed=seed)
            feature_dim, spatial_dim = 512, 256
        with _component_seed(seed + 10_001):
            self.position_head = nn.Linear(feature_dim, 6)
        with _component_seed(seed + 20_001):
            self.lesion_head = nn.Sequential(
                nn.Conv2d(spatial_dim, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
        if self.grading_frozen:
            self.backbone.requires_grad_(False)
            self.grading_head.requires_grad_(False)
            self.train(self.training)

    def train(self, mode: bool = True) -> "LUAModel":
        super().train(mode)
        if getattr(self, "grading_frozen", False):
            self.backbone.eval()
            self.grading_head.eval()
        return self

    def load_grader(self, state: dict[str, torch.Tensor]) -> None:
        expected = {k for k in self.state_dict()
                    if k.startswith(("backbone.", "grading_head."))}
        if set(state) != expected:
            raise ValueError("Parent checkpoint must contain exactly A grading tensors")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or expected.intersection(result.missing_keys):
            raise ValueError("Incomplete parent grader load")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial, global_features = self.backbone(x)
        # Run the complete A path before auxiliary operations, including under AMP.
        logits = self.grading_head(global_features)
        aux_spatial, aux_global = spatial, global_features
        if self.arm == "INDEPENDENT":
            aux_spatial, aux_global = self.aux_backbone(x)
        position_logits = self.position_head(aux_global)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            lesion_logits = self.lesion_head(aux_spatial.float())
        return {
            "logits": logits, "global_features": global_features,
            "fused_features": global_features, "layer3_features": spatial,
            "position_logits": position_logits,
            "position_probs": position_logits.softmax(dim=1),
            "lesion_logits": lesion_logits,
            "lesion_attention": lesion_logits.sigmoid(),
        }
