from __future__ import annotations

from pathlib import Path
import os


WORKSPACE_ROOT = Path(os.environ.get("VCRE_WORKSPACE", Path.cwd())).resolve()
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = WORKSPACE_ROOT / "data" / "processed" / "schisto_2024_clean_v4"
IMAGES_CSV = DATASET_ROOT / "manifests" / "images.csv"
ANNOTATIONS_JSONL = DATASET_ROOT / "manifests" / "annotations.jsonl"
EXPERIMENT_ROOT = (
    WORKSPACE_ROOT
    / "research-private"
    / "experiments"
    / "SFibAI-B_AE_COR_v2"
)

ROUND_NAME = "AE_COR_v2"
ARMS = ("A", "B", "C", "D", "E")
SEEDS = (2026,)
EPOCHS = 120
BEST_SELECTION_START_EPOCH = 21
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_STEP_SIZE = 15
SCHEDULER_GAMMA = 0.6

MANIFEST_SHA256 = "73a14221ea1a67f3248351a4ed5d08574f904175e69d9129ab2e705a0bb86a40"
ANNOTATIONS_SHA256 = "6684cb633ad055eb55f399a6383f531b5e8cf4db0a3ef828fb6967c16150d167"
EXPECTED_SPLITS = {
    "train": {"images": 83722, "patients": 4906, "centers": 35},
    "val": {"images": 20880, "patients": 1227, "centers": 33},
    "test": {"images": 4107, "patients": 240, "centers": 4},
}
