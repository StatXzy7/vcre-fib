from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfibai_b.model import ARM_BRANCHES


class PredictionCollector:
    def __init__(self, *, arm: str, full: bool, collect_attention: bool) -> None:
        normalized_arm = arm.upper()
        if normalized_arm not in ARM_BRANCHES:
            raise ValueError(f"Unknown ablation arm: {arm!r}")
        self.arm = normalized_arm
        self.use_position, self.use_lesion = ARM_BRANCHES[normalized_arm]
        self.full = bool(full)
        self.collect_attention = bool(collect_attention)
        if self.collect_attention and not self.full:
            raise ValueError("Attention maps are retained only in full prediction bundles")
        self.rows: list[dict[str, Any]] = []
        self.attention_maps: list[np.ndarray] = []
        self.attention_uids: list[str] = []

    @staticmethod
    def _batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            return value[index].detach().cpu().item()
        return value[index]

    def add_batch(
        self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]
    ) -> None:
        logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
        probabilities = logits.softmax(dim=1)
        bins = torch.arange(36, dtype=torch.float32).unsqueeze(0)
        predicted_scores = (probabilities * bins).sum(dim=1) / 10.0
        grade_probabilities = torch.stack(
            [
                probabilities[:, :5].sum(dim=1),
                probabilities[:, 5:15].sum(dim=1),
                probabilities[:, 15:25].sum(dim=1),
                probabilities[:, 25:].sum(dim=1),
            ],
            dim=1,
        )
        batch_size = int(logits.shape[0])

        position_logits: torch.Tensor | None = None
        position_probabilities: torch.Tensor | None = None
        position_gates: torch.Tensor | None = None
        if self.use_position:
            required = {"position_logits", "position_probs", "position_gate"}
            if not required.issubset(outputs):
                raise KeyError("Position arm output is incomplete")
            position_logits = outputs["position_logits"].detach().cpu().float()
            position_probabilities = outputs["position_probs"].detach().cpu().float()
            position_gates = outputs["position_gate"].detach().cpu().flatten().float()

        lesion_attention: torch.Tensor | None = None
        lesion_gates: torch.Tensor | None = None
        lesion_stats: list[dict[str, float | bool]] | None = None
        if self.use_lesion:
            required = {"lesion_attention", "lesion_gate", "lesion_logits"}
            if not required.issubset(outputs):
                raise KeyError("Lesion arm output is incomplete")
            required_batch = {"lesion_mask", "lesion_box_valid", "label_bin"}
            if not required_batch.issubset(batch):
                raise KeyError("Lesion batch target is incomplete")
            lesion_attention = outputs["lesion_attention"].detach().cpu().float()
            lesion_gates = outputs["lesion_gate"].detach().cpu().flatten().float()
            target = batch["lesion_mask"].detach().cpu().float()
            if target.shape[-2:] != lesion_attention.shape[-2:]:
                target = F.interpolate(
                    target, size=lesion_attention.shape[-2:], mode="nearest"
                )
            target_bool = target > 0.5
            has_box = batch["lesion_box_valid"].detach().cpu().bool()
            positive_grade = batch["label_bin"].detach().cpu().long() > 0
            valid = has_box & positive_grade & (target_bool.flatten(1).sum(dim=1) > 0)
            lesion_stats = []
            for index in range(batch_size):
                if not bool(valid[index]):
                    lesion_stats.append(
                        {
                            "lesion_valid": False,
                            "inside_attention": float("nan"),
                            "outside_ratio": float("nan"),
                            "inside_outside_ratio": float("nan"),
                            "dice_at_0_5": float("nan"),
                            "iou_at_0_5": float("nan"),
                        }
                    )
                    continue
                attention = lesion_attention[index, 0]
                inside = target_bool[index, 0]
                outside = ~inside
                inside_mean = attention[inside].mean()
                outside_mean = attention[outside].mean() if outside.any() else attention.new_zeros(())
                outside_mass = attention[outside].sum() if outside.any() else attention.new_zeros(())
                outside_ratio = outside_mass / attention.sum().clamp_min(1e-8)
                hard = attention >= 0.5
                intersection = (hard & inside).sum().to(torch.float32)
                hard_count = hard.sum().to(torch.float32)
                target_count = inside.sum().to(torch.float32)
                union = (hard | inside).sum().to(torch.float32)
                lesion_stats.append(
                    {
                        "lesion_valid": True,
                        "inside_attention": float(inside_mean),
                        "outside_ratio": float(outside_ratio),
                        "inside_outside_ratio": float(
                            inside_mean / outside_mean.clamp_min(1e-8)
                        ),
                        "dice_at_0_5": float(
                            2.0 * intersection / (hard_count + target_count).clamp_min(1.0)
                        ),
                        "iou_at_0_5": float(intersection / union.clamp_min(1.0)),
                    }
                )

        for index in range(batch_size):
            row: dict[str, Any] = {
                "arm": self.arm,
                "split": str(self._batch_value(batch, "split", index)),
                "image_uid": str(self._batch_value(batch, "image_uid", index)),
                "patient_uid": str(self._batch_value(batch, "patient_uid", index)),
                "center_id": str(self._batch_value(batch, "center_id", index)),
                "true_score": float(self._batch_value(batch, "true_score", index)),
                "pred_score": float(predicted_scores[index]),
                "pred_bin": int(torch.argmax(probabilities[index]).item()),
            }
            for grade in range(4):
                row[f"prob_f{grade}"] = float(grade_probabilities[index, grade])
            if self.full:
                for bin_index in range(36):
                    row[f"logit_{bin_index:02d}"] = float(logits[index, bin_index])

            if self.use_position:
                assert position_logits is not None
                assert position_probabilities is not None
                assert position_gates is not None
                row["position_true"] = int(
                    self._batch_value(batch, "position_norm", index)
                )
                row["position_pred"] = int(
                    torch.argmax(position_probabilities[index]).item() + 1
                )
                row["position_gate"] = float(position_gates[index])
                for position in range(1, 7):
                    row[f"position_prob_{position}"] = float(
                        position_probabilities[index, position - 1]
                    )
                    if self.full:
                        row[f"position_logit_{position}"] = float(
                            position_logits[index, position - 1]
                        )

            if self.use_lesion:
                assert lesion_stats is not None
                assert lesion_gates is not None
                row.update(lesion_stats[index])
                row["lesion_gate"] = float(lesion_gates[index])
                row["lesion_box_present"] = bool(
                    self._batch_value(batch, "lesion_box_valid", index)
                )
                if self.collect_attention:
                    assert lesion_attention is not None
                    self.attention_maps.append(
                        lesion_attention[index, 0].numpy().astype(np.float16, copy=True)
                    )
                    self.attention_uids.append(row["image_uid"])
            self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def attention_payload(self) -> dict[str, np.ndarray] | None:
        if not self.collect_attention:
            return None
        return {
            "image_uid": np.asarray(self.attention_uids, dtype=str),
            "attention": np.stack(self.attention_maps).astype(np.float16, copy=False),
        }
