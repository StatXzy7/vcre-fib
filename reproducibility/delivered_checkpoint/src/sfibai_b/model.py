from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


RESNET50_PRETRAINED_WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V2
ARM_BRANCHES: dict[str, tuple[bool, bool]] = {
    "A": (False, False),
    "B": (False, False),
    "C": (True, False),
    "D": (False, True),
    "E": (True, True),
}


@contextmanager
def _component_seed(seed: int) -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


class ResNet50SpatialBackbone(nn.Module):
    spatial_dim = 1024
    feature_dim = 2048

    def __init__(self, *, pretrained: bool, initialization_seed: int) -> None:
        super().__init__()
        with _component_seed(initialization_seed):
            source = models.resnet50(
                weights=RESNET50_PRETRAINED_WEIGHTS if pretrained else None
            )
        self.conv1 = source.conv1
        self.bn1 = source.bn1
        self.relu = source.relu
        self.maxpool = source.maxpool
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3
        self.layer4 = source.layer4
        self.avgpool = source.avgpool

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)
        x = self.layer4(layer3)
        pooled = torch.flatten(self.avgpool(x), 1)
        return layer3, pooled


def _grading_head(
    *, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def _small_init(layer: nn.Linear, scale: float) -> None:
    if scale <= 0:
        raise ValueError("residual_init_scale must be positive")
    nn.init.normal_(layer.weight, mean=0.0, std=scale)
    nn.init.zeros_(layer.bias)


class SFibAIModel(nn.Module):
    """ResNet-50 SFibAI grader with physically optional auxiliary branches."""

    def __init__(
        self,
        *,
        arm: str,
        seed: int,
        pretrained: bool,
        use_position: bool,
        use_lesion: bool,
        num_classes: int = 36,
        num_positions: int = 6,
        dense_hidden_dim: int = 512,
        lesion_hidden_dim: int = 64,
        position_gate_hidden_dim: int = 8,
        lesion_gate_hidden_dim: int = 32,
        gate_max: float = 0.5,
        residual_init_scale: float = 0.001,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        normalized_arm = arm.upper()
        if normalized_arm not in ARM_BRANCHES:
            raise ValueError(f"Unknown ablation arm: {arm!r}")
        if (bool(use_position), bool(use_lesion)) != ARM_BRANCHES[normalized_arm]:
            raise ValueError(
                f"Arm {normalized_arm} requires branches {ARM_BRANCHES[normalized_arm]}"
            )
        if not 0.0 < gate_max <= 1.0:
            raise ValueError("gate_max must be in (0, 1]")
        self.arm = normalized_arm
        self.seed = int(seed)
        self.use_position = bool(use_position)
        self.use_lesion = bool(use_lesion)
        self.num_positions = int(num_positions)
        self.gate_max = float(gate_max)

        self.backbone = ResNet50SpatialBackbone(
            pretrained=pretrained,
            initialization_seed=self.seed + 101,
        )
        with _component_seed(self.seed + 202):
            self.grading_head = _grading_head(
                feature_dim=self.backbone.feature_dim,
                hidden_dim=dense_hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

        if self.use_position:
            with _component_seed(self.seed + 10_001):
                self.position_head = nn.Linear(
                    self.backbone.feature_dim, self.num_positions
                )
                self.position_residual = nn.Sequential(
                    nn.Linear(self.num_positions, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 8),
                    nn.ReLU(inplace=True),
                    nn.Linear(8, self.backbone.feature_dim),
                )
                _small_init(self.position_residual[-1], residual_init_scale)
                self.position_gate = nn.Sequential(
                    nn.Linear(self.num_positions + 2, position_gate_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(position_gate_hidden_dim, 1),
                )

        if self.use_lesion:
            with _component_seed(self.seed + 20_001):
                self.lesion_head = nn.Sequential(
                    nn.Conv2d(
                        self.backbone.spatial_dim,
                        lesion_hidden_dim,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(lesion_hidden_dim, 1, kernel_size=1),
                )
                self.lesion_residual = nn.Sequential(
                    nn.Linear(self.backbone.spatial_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, self.backbone.feature_dim),
                )
                _small_init(self.lesion_residual[-1], residual_init_scale)
                self.lesion_gate = nn.Sequential(
                    nn.Linear(self.backbone.spatial_dim + 3, lesion_gate_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(lesion_gate_hidden_dim, 1),
                )

    @staticmethod
    def _position_statistics(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        entropy = -(q * q.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        maximum = q.max(dim=1, keepdim=True).values
        return entropy, maximum

    @staticmethod
    def _attention_statistics(attention: torch.Tensor) -> torch.Tensor:
        flat = attention.flatten(1)
        mass = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(mass * mass.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy_scale = math.log(float(flat.shape[1]))
        entropy = entropy / entropy_scale if entropy_scale > 0.0 else torch.zeros_like(entropy)
        return torch.cat(
            [
                flat.mean(dim=1, keepdim=True),
                flat.max(dim=1, keepdim=True).values,
                entropy,
            ],
            dim=1,
        )

    def _lesion_logits_fp32(self, layer3_features: torch.Tensor) -> torch.Tensor:
        """Run the lesion head outside AMP to avoid FP16 convolution overflow."""
        with torch.amp.autocast(
            device_type=layer3_features.device.type,
            enabled=False,
        ):
            return self.lesion_head(layer3_features.detach().float())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        layer3_features, global_features = self.backbone(x)
        fused_features = global_features
        outputs: dict[str, torch.Tensor] = {"global_features": global_features}

        if self.use_position:
            position_logits = self.position_head(global_features.detach())
            position_probs = F.softmax(position_logits, dim=1)
            q_detached = position_probs.detach()
            entropy, maximum = self._position_statistics(q_detached)
            position_residual = self.position_residual(q_detached)
            position_gate = self.gate_max * torch.sigmoid(
                self.position_gate(torch.cat([q_detached, entropy, maximum], dim=1))
            )
            fused_features = fused_features + position_gate * position_residual
            outputs.update(
                {
                    "position_logits": position_logits,
                    "position_probs": position_probs,
                    "position_residual": position_residual,
                    "position_gate": position_gate,
                }
            )

        if self.use_lesion:
            lesion_logits = self._lesion_logits_fp32(layer3_features)
            lesion_attention = torch.sigmoid(lesion_logits)
            attention_detached = lesion_attention.detach()
            attention_normalized = attention_detached / attention_detached.sum(
                dim=(2, 3), keepdim=True
            ).clamp_min(1e-8)
            lesion_features = (attention_normalized * layer3_features).sum(dim=(2, 3))
            lesion_residual = self.lesion_residual(lesion_features)
            lesion_gate = self.gate_max * torch.sigmoid(
                self.lesion_gate(
                    torch.cat(
                        [lesion_features, self._attention_statistics(attention_detached)],
                        dim=1,
                    )
                )
            )
            fused_features = fused_features + lesion_gate * lesion_residual
            outputs.update(
                {
                    "layer3_features": layer3_features,
                    "lesion_logits": lesion_logits,
                    "lesion_attention": lesion_attention,
                    "lesion_features": lesion_features,
                    "lesion_residual": lesion_residual,
                    "lesion_gate": lesion_gate,
                }
            )

        outputs["fused_features"] = fused_features
        outputs["logits"] = self.grading_head(fused_features)
        return outputs


def build_model(*, arm: str, seed: int, pretrained: bool = True) -> SFibAIModel:
    normalized_arm = arm.upper()
    if normalized_arm not in ARM_BRANCHES:
        raise ValueError(f"Unknown ablation arm: {arm!r}")
    use_position, use_lesion = ARM_BRANCHES[normalized_arm]
    return SFibAIModel(
        arm=normalized_arm,
        seed=seed,
        pretrained=pretrained,
        use_position=use_position,
        use_lesion=use_lesion,
    )
