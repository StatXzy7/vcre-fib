from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from sfibai_b.model import ARM_BRANCHES
from sfibai_b.protocol import (
    ANNOTATIONS_SHA256,
    EXPECTED_SPLITS,
    MANIFEST_SHA256,
)


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN_RGB_8BIT = np.rint(IMAGENET_MEAN * 255.0).astype(np.uint8)


@dataclass(frozen=True)
class PreprocessingSpec:
    image_size: int
    resize_mode: str
    rot90_probability: float
    saturation_min: float
    saturation_max: float
    horizontal_flip_probability: float
    vertical_flip_probability: float


@dataclass(frozen=True)
class AugmentationParameters:
    quarter_turns: int
    apply_saturation: bool
    saturation_scale: float


@dataclass(frozen=True)
class LesionAnnotation:
    label: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def preprocessing_for_arm(arm: str) -> PreprocessingSpec:
    normalized_arm = arm.upper()
    if normalized_arm not in ARM_BRANCHES:
        raise ValueError(f"Unknown ablation arm: {arm!r}")
    return PreprocessingSpec(
        image_size=512,
        resize_mode="letterbox" if normalized_arm == "B" else "stretch",
        rot90_probability=0.7,
        saturation_min=0.75,
        saturation_max=1.25,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
    )


def augmentation_parameters(
    *, seed: int, epoch: int, image_uid: str
) -> AugmentationParameters:
    digest = hashlib.sha256(f"{int(seed)}:{int(epoch)}:{image_uid}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
    quarter_turns = int(rng.integers(1, 4)) if rng.random() < 0.7 else 0
    apply_saturation = bool(rng.random() < 0.5)
    saturation_scale = float(rng.uniform(0.75, 1.25))
    return AugmentationParameters(
        quarter_turns=quarter_turns,
        apply_saturation=apply_saturation,
        saturation_scale=saturation_scale,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data_v4(
    dataset_root: str | Path,
    manifest_path: str | Path,
    annotations_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_root)
    manifest = Path(manifest_path)
    annotations = Path(annotations_path)
    if not root.is_dir() or not manifest.is_file() or not annotations.is_file():
        raise FileNotFoundError("Data V4 root, images.csv, or annotations.jsonl is missing")
    manifest_hash = _sha256(manifest)
    annotations_hash = _sha256(annotations)
    if manifest_hash != MANIFEST_SHA256 or annotations_hash != ANNOTATIONS_SHA256:
        raise ValueError("Data V4 fingerprint does not match the locked formal protocol")
    frame = pd.read_csv(manifest, dtype=str, encoding="utf-8")
    required = {
        "image_uid",
        "patient_uid",
        "center_id",
        "image_path",
        "position_norm",
        "image_label_max",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Data V4 manifest columns are missing: {', '.join(missing)}")
    if frame["image_uid"].duplicated().any():
        raise ValueError("Data V4 image_uid values must be unique")
    patient_split_counts = frame.groupby("patient_uid", sort=False)["split"].nunique()
    if (patient_split_counts != 1).any():
        raise ValueError("Data V4 contains patient leakage across train/val/test")
    patient_center_counts = frame.groupby("patient_uid", sort=False)["center_id"].nunique()
    if (patient_center_counts != 1).any():
        raise ValueError("Data V4 contains patients assigned to multiple centers")
    splits: dict[str, dict[str, int]] = {}
    for split, group in frame.groupby("split", sort=True):
        splits[str(split)] = {
            "images": int(len(group)),
            "patients": int(group["patient_uid"].nunique()),
            "centers": int(group["center_id"].nunique()),
        }
    if splits != EXPECTED_SPLITS:
        raise ValueError(f"Data V4 split counts changed: {splits!r}")
    annotation_rows = 0
    with annotations.open("r", encoding="utf-8") as handle:
        for annotation_rows, _line in enumerate(handle, start=1):
            pass
    if annotation_rows != 109882:
        raise ValueError(f"Data V4 annotation row count changed: {annotation_rows}")
    return {
        "dataset_root": str(root.resolve()),
        "manifest_path": str(manifest.resolve()),
        "annotations_path": str(annotations.resolve()),
        "manifest_sha256": manifest_hash,
        "annotations_sha256": annotations_hash,
        "annotation_rows": annotation_rows,
        "patient_split_leakage": False,
        "patient_center_conflicts": False,
        "splits": splits,
    }


@lru_cache(maxsize=2)
def _load_annotations(path_string: str) -> dict[str, tuple[LesionAnnotation, ...]]:
    grouped: dict[str, list[LesionAnnotation]] = {}
    with Path(path_string).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid annotation JSON at line {line_number}") from exc
            if row.get("coordinate_frame") != "roi_crop":
                raise ValueError(f"Annotation line {line_number} is not in roi_crop frame")
            try:
                annotation = LesionAnnotation(
                    label=float(row["lesion_label_float"]),
                    x_min=float(row["bbox_crop_norm_x_min"]),
                    y_min=float(row["bbox_crop_norm_y_min"]),
                    x_max=float(row["bbox_crop_norm_x_max"]),
                    y_max=float(row["bbox_crop_norm_y_max"]),
                )
                image_uid = str(row["image_uid"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid annotation fields at line {line_number}") from exc
            if not (
                0.0 <= annotation.x_min < annotation.x_max <= 1.0
                and 0.0 <= annotation.y_min < annotation.y_max <= 1.0
                and math.isfinite(annotation.label)
            ):
                raise ValueError(f"Invalid annotation bounds at line {line_number}")
            grouped.setdefault(image_uid, []).append(annotation)
    return {key: tuple(values) for key, values in grouped.items()}


def _resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
    spec: PreprocessingSpec,
) -> tuple[np.ndarray, np.ndarray | None]:
    if spec.resize_mode == "stretch":
        resized_image = cv2.resize(
            image, (spec.image_size, spec.image_size), interpolation=cv2.INTER_LINEAR
        )
        resized_mask = (
            cv2.resize(
                mask,
                (spec.image_size, spec.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            if mask is not None
            else None
        )
        return resized_image, resized_mask

    height, width = image.shape[:2]
    side = max(height, width)
    square = np.empty((side, side, 3), dtype=image.dtype)
    square[...] = IMAGENET_MEAN_RGB_8BIT
    top = (side - height) // 2
    left = (side - width) // 2
    square[top : top + height, left : left + width] = image
    resized_image = cv2.resize(
        square, (spec.image_size, spec.image_size), interpolation=cv2.INTER_LINEAR
    )
    resized_mask = None
    if mask is not None:
        square_mask = np.zeros((side, side), dtype=mask.dtype)
        square_mask[top : top + height, left : left + width] = mask
        resized_mask = cv2.resize(
            square_mask,
            (spec.image_size, spec.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
    return resized_image, resized_mask


def _transform(
    image: np.ndarray,
    mask: np.ndarray | None,
    spec: PreprocessingSpec,
    augmentation: AugmentationParameters | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image, mask = _resize_image_and_mask(image, mask, spec)
    if augmentation is not None:
        if augmentation.quarter_turns:
            image = np.ascontiguousarray(np.rot90(image, augmentation.quarter_turns))
            if mask is not None:
                mask = np.ascontiguousarray(np.rot90(mask, augmentation.quarter_turns))
        if augmentation.apply_saturation:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(
                hsv[:, :, 1].astype(np.float32) * augmentation.saturation_scale,
                0,
                255,
            ).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    normalized = image.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    image_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    mask_tensor = (
        torch.from_numpy((mask > 0).astype(np.float32)[None, :, :]).float()
        if mask is not None
        else None
    )
    return image_tensor, mask_tensor


class EpochShuffleSampler(Sampler[tuple[int, int]]):
    def __init__(self, *, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("Sampler size must be positive")
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError("Epochs are one-based")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        for index in torch.randperm(self.size, generator=generator).tolist():
            yield int(index), self.epoch

    def __len__(self) -> int:
        return self.size


class FormalImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        dataset_root: str | Path,
        manifest_path: str | Path,
        annotations_path: str | Path | None,
        split: str,
        arm: str,
        seed: int,
        training: bool,
        image_size: int = 512,
        max_samples: int | None = None,
    ) -> None:
        normalized_arm = arm.upper()
        if normalized_arm not in ARM_BRANCHES:
            raise ValueError(f"Unknown ablation arm: {arm!r}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.arm = normalized_arm
        self.seed = int(seed)
        self.training = bool(training)
        self.spec = replace(preprocessing_for_arm(normalized_arm), image_size=int(image_size))
        frame = pd.read_csv(self.manifest_path, dtype=str, encoding="utf-8")
        frame = frame.loc[frame["split"] == split].copy()
        if max_samples is not None:
            frame = frame.head(int(max_samples))
        if frame.empty:
            raise ValueError(f"No manifest rows for split {split!r}")
        if frame["image_uid"].duplicated().any():
            raise ValueError("image_uid must be unique within a split")
        self.frame = frame.reset_index(drop=True)
        _use_position, self.use_lesion = ARM_BRANCHES[normalized_arm]
        if self.use_lesion:
            if annotations_path is None:
                raise ValueError("D/E datasets require annotations_jsonl")
            self.annotations = _load_annotations(str(Path(annotations_path).resolve()))
        else:
            self.annotations = {}

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _score_to_bin(value: str) -> int:
        score = float(value)
        bin_index = int(round(score * 10.0))
        if not 0 <= bin_index <= 35 or not math.isclose(
            score * 10.0, bin_index, abs_tol=1e-6
        ):
            raise ValueError(f"Invalid 0.1-resolution grade: {value!r}")
        return bin_index

    def _read_image(self, relative_path: str) -> np.ndarray:
        path = self.dataset_root / relative_path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _lesion_target(
        self, image_uid: str, shape: tuple[int, int], label_score: float
    ) -> tuple[np.ndarray, float, int, bool]:
        annotations = self.annotations.get(image_uid, ())
        if not annotations:
            return np.zeros(shape, dtype=np.uint8), float("nan"), 0, False
        max_label = max(annotation.label for annotation in annotations)
        if not math.isclose(max_label, label_score, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Maximum annotation label differs from image_label_max for {image_uid}"
            )
        selected = [
            annotation
            for annotation in annotations
            if math.isclose(annotation.label, max_label, rel_tol=0.0, abs_tol=1e-6)
        ]
        height, width = shape
        mask = np.zeros(shape, dtype=np.uint8)
        for annotation in selected:
            left = max(0, min(width - 1, int(math.floor(annotation.x_min * width))))
            top = max(0, min(height - 1, int(math.floor(annotation.y_min * height))))
            right = max(left + 1, min(width, int(math.ceil(annotation.x_max * width))))
            bottom = max(top + 1, min(height, int(math.ceil(annotation.y_max * height))))
            mask[top:bottom, left:right] = 1
        return mask, max_label, len(annotations), True

    def __getitem__(self, item: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(item, tuple):
            index, epoch = int(item[0]), int(item[1])
        else:
            index, epoch = int(item), 0
            if self.training:
                raise ValueError("Training dataset indices must include the epoch")
        row = self.frame.iloc[index]
        image_uid = str(row["image_uid"])
        image = self._read_image(str(row["image_path"]))
        label_bin = self._score_to_bin(str(row["image_label_max"]))
        label_score = label_bin / 10.0
        mask: np.ndarray | None = None
        lesion_label = float("nan")
        annotation_count = 0
        box_valid = False
        if self.use_lesion:
            mask, lesion_label, annotation_count, box_valid = self._lesion_target(
                image_uid, image.shape[:2], label_score
            )
        augmentation = (
            augmentation_parameters(seed=self.seed, epoch=epoch, image_uid=image_uid)
            if self.training
            else None
        )
        image_tensor, mask_tensor = _transform(image, mask, self.spec, augmentation)
        position = int(float(row["position_norm"]))
        if not 1 <= position <= 6:
            raise ValueError(f"position_norm outside 1..6 for {image_uid}")
        sample: dict[str, Any] = {
            "image": image_tensor,
            "label_bin": torch.tensor(label_bin, dtype=torch.long),
            "true_score": torch.tensor(label_score, dtype=torch.float32),
            "position_norm": torch.tensor(position, dtype=torch.long),
            "image_uid": image_uid,
            "patient_uid": str(row["patient_uid"]),
            "center_id": str(row["center_id"]),
            "split": self.split,
        }
        if self.use_lesion:
            assert mask_tensor is not None
            sample.update(
                {
                    "lesion_mask": mask_tensor,
                    "lesion_box_valid": torch.tensor(box_valid, dtype=torch.bool),
                    "lesion_label_max": torch.tensor(
                        lesion_label, dtype=torch.float32
                    ),
                    "lesion_annotation_count": torch.tensor(
                        annotation_count, dtype=torch.long
                    ),
                }
            )
        return sample
