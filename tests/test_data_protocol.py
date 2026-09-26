from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from sfibai_b.data import (
    EpochShuffleSampler,
    FormalImageDataset,
    augmentation_parameters,
    preprocessing_for_arm,
    validate_data_v4,
)
from sfibai_b.protocol import ANNOTATIONS_JSONL, DATASET_ROOT, IMAGES_CSV


@pytest.mark.skipif(
    not IMAGES_CSV.is_file(),
    reason="Data V4 lives with the active round host (nfu-hpc for seed 2028)",
)
def test_native_data_v4_identity_and_counts_are_locked() -> None:
    audit = validate_data_v4(DATASET_ROOT, IMAGES_CSV, ANNOTATIONS_JSONL)

    assert audit["manifest_sha256"] == (
        "73a14221ea1a67f3248351a4ed5d08574f904175e69d9129ab2e705a0bb86a40"
    )
    assert audit["annotations_sha256"] == (
        "6684cb633ad055eb55f399a6383f531b5e8cf4db0a3ef828fb6967c16150d167"
    )
    assert audit["splits"] == {
        "train": {"images": 83722, "patients": 4906, "centers": 35},
        "val": {"images": 20880, "patients": 1227, "centers": 33},
        "test": {"images": 4107, "patients": 240, "centers": 4},
    }
    assert audit["patient_split_leakage"] is False
    assert audit["patient_center_conflicts"] is False


def test_preprocessing_differs_only_by_locked_resize_arm() -> None:
    for arm in "ACEFG":
        assert preprocessing_for_arm(arm).resize_mode == "stretch"
    for arm in "ACEFG":
        spec = preprocessing_for_arm(arm)
        assert spec.image_size == 512
        assert spec.rot90_probability == pytest.approx(0.7)
        assert (spec.saturation_min, spec.saturation_max) == (0.75, 1.25)
        assert spec.horizontal_flip_probability == 0.0
        assert spec.vertical_flip_probability == 0.0


def test_augmentation_is_stateless_per_seed_epoch_and_image() -> None:
    first = augmentation_parameters(seed=2026, epoch=7, image_uid="image-1")
    np.random.seed(999)
    _ = np.random.random(100)
    repeated = augmentation_parameters(seed=2026, epoch=7, image_uid="image-1")
    changed_epoch = augmentation_parameters(seed=2026, epoch=8, image_uid="image-1")

    assert first == repeated
    assert first != changed_epoch


def test_epoch_sampler_replays_order_and_changes_by_epoch() -> None:
    sampler = EpochShuffleSampler(size=20, seed=2026)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    repeated = list(sampler)
    sampler.set_epoch(4)
    changed = list(sampler)

    assert first == repeated
    assert first != changed
    assert {index for index, _epoch in first} == set(range(20))
    assert {epoch for _index, epoch in first} == {3}


def test_max_grade_union_mask_follows_stretch_resize_and_rotation(tmp_path) -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), image)
    manifest = pd.DataFrame(
        [
            {
                "image_uid": "image-1",
                "patient_uid": "patient-1",
                "center_id": "center-1",
                "image_path": image_path.name,
                "position_norm": 2,
                "image_label_max": 2.0,
                "split": "train",
            }
        ]
    )
    manifest_path = tmp_path / "images.csv"
    manifest.to_csv(manifest_path, index=False)
    annotations_path = tmp_path / "annotations.jsonl"
    annotations_path.write_text(
        "\n".join(
            [
                '{"image_uid":"image-1","coordinate_frame":"roi_crop",'
                '"lesion_label_float":1.0,"bbox_crop_norm_x_min":0.0,'
                '"bbox_crop_norm_y_min":0.0,"bbox_crop_norm_x_max":0.2,'
                '"bbox_crop_norm_y_max":0.2}',
                '{"image_uid":"image-1","coordinate_frame":"roi_crop",'
                '"lesion_label_float":2.0,"bbox_crop_norm_x_min":0.25,'
                '"bbox_crop_norm_y_min":0.2,"bbox_crop_norm_x_max":0.75,'
                '"bbox_crop_norm_y_max":0.8}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = FormalImageDataset(
        dataset_root=tmp_path,
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        split="train",
        arm="D",
        seed=2026,
        training=True,
        image_size=32,
    )

    sample = dataset[(0, 1)]

    assert sample["image"].shape == (3, 32, 32)
    assert sample["lesion_mask"].shape == (1, 32, 32)
    assert sample["lesion_box_valid"].item()
    assert sample["lesion_label_max"].item() == pytest.approx(2.0)
    original_mask = np.zeros((10, 20), dtype=np.uint8)
    original_mask[2:8, 5:15] = 1
    expected = cv2.resize(original_mask, (32, 32), interpolation=cv2.INTER_NEAREST)
    augmentation = augmentation_parameters(seed=2026, epoch=1, image_uid="image-1")
    if augmentation.quarter_turns:
        expected = np.ascontiguousarray(np.rot90(expected, augmentation.quarter_turns))
    assert np.array_equal(sample["lesion_mask"].squeeze(0).numpy(), expected)
