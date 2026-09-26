from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import math
import cv2
import numpy as np
import pandas as pd
import torch
from sfibai_b.data import FormalImageDataset, _load_annotations, _transform, preprocessing_for_arm, augmentation_parameters


def inner_partition(frame, seed=31001, fraction=0.2):
    if set(frame["split"]) != {"train"}:
        raise ValueError("Inner folds accept native train only")
    for column in ("patient_uid", "parent_image_uid"):
        if column in frame and (frame.groupby(column)["patient_uid"].nunique() > 1).any():
            raise ValueError("Derived parent assigned to different patients")
    patients = sorted(frame.patient_uid.unique(), key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
    n = max(1, int(round(len(patients) * fraction)))
    if n >= len(patients):
        raise ValueError("Inner train and validation must both contain patients")
    validation = set(patients[:n])
    result = frame.copy()
    result["inner_split"] = np.where(result.patient_uid.isin(validation), "inner_val", "inner_train")
    return result


def safe_image_path(root, relative, split):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root / split):
        raise ValueError("Image path escapes its native split")
    return path


def geometry(shape, size, mode):
    height, width = shape
    if mode == "stretch":
        return np.asarray([[size / width, 0, 0], [0, size / height, 0], [0, 0, 1]], dtype=float)
    if mode != "letterbox":
        raise ValueError("Unknown geometry")
    side = max(height, width)
    scale = size / side
    return np.asarray([[scale, 0, ((side - width) // 2) * scale], [0, scale, ((side - height) // 2) * scale], [0, 0, 1]], dtype=float)


def transform_box(box, matrix):
    x1, y1, x2, y2 = box
    corners = np.asarray([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ np.asarray(matrix).T
    return np.asarray([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])


def predicted_regions(heatmap, inverse, input_size, roi_shape):
    """Threshold 0.5, 8-connected components, no GT hints, no fitted thresholds."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    hard = (heatmap >= 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    scale = np.diag([input_size / heatmap.shape[1], input_size / heatmap.shape[0], 1])
    boxes = []
    for index in range(1, count):
        x, y, w, h, _ = stats[index]
        box = transform_box([x, y, x + w, y + h], np.asarray(inverse) @ scale)
        box[[0, 2]] = box[[0, 2]].clip(0, roi_shape[1])
        box[[1, 3]] = box[[1, 3]].clip(0, roi_shape[0])
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append({"box_roi_xyxy": box.tolist(), "confidence": float(heatmap[labels == index].mean())})
    return boxes


def box_iou(left, right):
    left, right = np.asarray(left), np.asarray(right)
    intersection = np.maximum(0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2])).prod()
    union = np.maximum(0, left[2:] - left[:2]).prod() + np.maximum(0, right[2:] - right[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def max_grade_box_localization(regions, annotations, roi_shape):
    """Best predicted box IoU for each recorded max-grade box; no negative inference."""
    if not annotations or max(a.label for a in annotations) <= 0:
        return []
    maximum = max(a.label for a in annotations)
    h,w = roi_shape
    targets = [[a.x_min*w,a.y_min*h,a.x_max*w,a.y_max*h] for a in annotations if a.label == maximum]
    return [max((box_iou(r["box_roi_xyxy"], target) for r in regions), default=0.) for target in targets]


class DevelopmentDataset(FormalImageDataset):
    """No full native manifest, native annotation cache, or test split accepted."""
    def __init__(self, *, root, manifest, annotations, partition, config, seed, training):
        if partition not in {"inner_train", "inner_val", "native_train", "native_val"}:
            raise ValueError("Test access forbidden in development")
        frame = pd.read_csv(manifest, dtype=str, encoding="utf-8")
        if not set(frame["split"]).issubset({"train", "val"}):
            raise ValueError("Development manifest contains test or unknown rows")
        if training != (partition in {"inner_train", "native_train"}):
            raise ValueError("Only train partitions may train")
        self._initialize(root, frame, annotations, partition, config, seed, training)

    def _initialize(self, root, frame, annotations, partition, config, seed, training):
        native = "val" if partition == "native_val" else "train"
        if partition in {"inner_train", "inner_val"}:
            frame = frame.loc[frame.inner_split == partition]
        else:
            frame = frame.loc[frame.split == native]
        if frame.empty:
            raise ValueError("Empty partition")
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = Path(root)
        self.annotations = _load_annotations(str(Path(annotations).resolve()))
        if not set(self.annotations).issubset(set(self.frame.image_uid)):
            raise ValueError("Annotation file includes patients outside this partition")
        self.seed, self.training, self.split = seed, training, native
        self.spec = replace(preprocessing_for_arm("E"), image_size=config.image_size, resize_mode=config.resize)
        self.config, self.use_lesion = config, True

    def _read_image(self, relative_path):
        path = safe_image_path(self.dataset_root, relative_path, self.split)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, item):
        if self.training and not isinstance(item, tuple):
            raise ValueError("Training indices must include epoch")
        index, epoch = item if isinstance(item, tuple) else (item, 0)
        row = self.frame.iloc[index]
        image = self._read_image(row.image_path)
        uid = row.image_uid
        label = self._score_to_bin(row.image_label_max)
        mask, _, _, valid = self._lesion_target(uid, image.shape[:2], label / 10)
        # Sparse local severity: unannotated and conflicting overlaps are ignored.
        local = np.full(image.shape[:2], -1, dtype=np.float32)
        conflict = np.zeros_like(local, dtype=bool)
        h, w = local.shape
        for ann in self.annotations.get(uid, ()):
            value = self._score_to_bin(str(ann.label))
            x1, y1 = math.floor(ann.x_min * w), math.floor(ann.y_min * h)
            x2, y2 = math.ceil(ann.x_max * w), math.ceil(ann.y_max * h)
            area = local[y1:y2, x1:x2]
            conflict[y1:y2, x1:x2] |= (area >= 0) & (area != value)
            area[:] = value
        local[conflict] = -1
        inverse = np.linalg.inv(geometry((h, w), self.spec.image_size, self.spec.resize_mode))
        # Synchronize local target with the exact legacy padding/resize and rotation.
        from sfibai_b.data import _resize_image_and_mask
        _, local_shifted = _resize_image_and_mask(image, local + 1, self.spec)
        local = local_shifted - 1
        augmentation = augmentation_parameters(seed=self.seed, epoch=epoch, image_uid=uid) if self.training and self.config.augmentation == "legacy" else None
        if augmentation and augmentation.quarter_turns:
            local = np.rot90(local, augmentation.quarter_turns).copy()
        if self.training and self.config.augmentation == "conservative":
            from sfibai_b.data import _resize_image_and_mask
            image, mask = _resize_image_and_mask(image, mask, self.spec)
            seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{epoch}:{uid}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            matrix = cv2.getRotationMatrix2D((self.spec.image_size / 2, self.spec.image_size / 2), float(rng.uniform(-7, 7)), 1)
            size = (self.spec.image_size, self.spec.image_size)
            image = cv2.warpAffine(image, matrix, size, borderValue=(124, 116, 104))
            mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST)
            local = cv2.warpAffine(local, matrix, size, flags=cv2.INTER_NEAREST, borderValue=-1)
            values = (image.astype(np.float32) / 255) ** float(rng.uniform(.9, 1.1))
            values = values * float(rng.uniform(.9, 1.1)) + float(rng.uniform(-.03, .03))
            values += rng.normal(0, .005, values.shape)
            image = np.clip(values * 255, 0, 255).astype(np.uint8)
        tensor, mask_tensor = _transform(image, mask, self.spec, augmentation)
        return {"image": tensor, "label_bin": torch.tensor(label), "true_score": torch.tensor(label / 10),
                "position_norm": torch.tensor(int(row.position_norm)), "lesion_mask": mask_tensor,
                "lesion_box_valid": torch.tensor(valid), "local_target": torch.from_numpy(local[None].copy()),
                "image_uid": uid, "patient_uid": row.patient_uid, "center_id": row.center_id, "split": self.split,
                "inverse": torch.tensor(inverse), "roi_shape": torch.tensor([h, w])}
