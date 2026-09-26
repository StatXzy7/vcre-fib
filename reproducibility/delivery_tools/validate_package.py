"""Validate evidence identity and export structure, without a second metric evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from package_common import PACK, DATA, read_json, sha256, write_json


def require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    checks = []
    complete = read_json(PACK / "provenance/RUN_COMPLETE.json")
    for entry in read_json(PACK / "provenance/copied_sources.json"):
        require(
            sha256(PACK / entry["destination"]) == entry["sha256"],
            f"Copied file mismatch: {entry['destination']}",
        )
        require(
            sha256(Path(entry["source"])) == entry["sha256"],
            f"Original source changed: {entry['source']}",
        )
    checks.append(
        "329 copied evidence files match source hashes; original sources unchanged"
    )
    for which in ["best", "last"]:
        require(
            sha256(PACK / f"model/checkpoints/{which}.pt")
            == complete[f"{which}_checkpoint_sha256"],
            f"{which} checkpoint identity",
        )
    grading_keys = {
        "n",
        "mae",
        "rmse",
        "accuracy_within_0_3",
        "accuracy_within_0_5",
        "grade_accuracy",
        "grade_macro_f1",
        "tmae",
        "severe_error_rate",
        "cor",
        "cor_continuous",
        "cor_error_gt_0_3",
        "cor_error_gt_0_5",
        "cor_stage_distance",
        "cor_severe_stage",
    }
    grading_keys |= {f"grade_f{g}_{k}" for g in range(4) for k in ["recall", "support"]}
    probability_keys = {
        f"grade_f{g}_{k}" for g in range(4) for k in ["auroc", "auprc", "brier", "ece"]
    }
    probability_keys |= {
        "grade_macro_auroc",
        "grade_macro_auprc",
        "grade_brier",
        "grade_ece",
    }
    split_patients = {}
    for split, counts in [("test", (4107, 240, 4)), ("val", (20880, 1227, 33))]:
        root = PACK / "results" / split
        m = read_json(root / "metrics.json")
        require(m == complete[f"best_{split}"], f"Canonical metric identity: {split}")
        for level in ["image", "patient_max", "patient_median"]:
            require(
                grading_keys <= m[level].keys(),
                f"Missing grading metrics: {split}/{level}",
            )
        require(probability_keys <= m["image"].keys(), "Probability metrics incomplete")
        frame = pd.read_csv(root / "predictions.csv.gz", float_precision="round_trip")
        manifest = pd.read_csv(root / "image_manifest.csv.gz").set_index("image_uid")
        require(
            (len(frame), frame.patient_uid.nunique(), frame.center_id.nunique())
            == counts,
            f"Counts mismatch: {split}",
        )
        require(set(frame.arm) == {"E"}, "Non-E predictions included")
        require(not frame.image_uid.duplicated().any(), "Duplicate image UID")
        require(set(frame.image_uid) == set(manifest.index), "Manifest IDs mismatch")
        ordered = manifest.loc[frame.image_uid]
        require(
            ordered.patient_uid.tolist() == frame.patient_uid.tolist(),
            "Patient identity mismatch",
        )
        require(
            ordered.center_id.tolist() == frame.center_id.tolist(),
            "Center identity mismatch",
        )
        require(
            np.allclose(ordered.image_label_max, frame.true_score, rtol=0, atol=2e-7),
            "Ground truth mismatch",
        )
        require(
            ordered.position_norm.tolist() == frame.position_true.tolist(),
            "View truth mismatch",
        )
        split_patients[split] = set(frame.patient_uid)
        for reducer in ["max", "median"]:
            patient = pd.read_csv(
                root / f"patient_{reducer}_predictions.csv.gz",
                float_precision="round_trip",
            ).set_index("patient_uid")
            expected = frame.groupby("patient_uid")[["true_score", "pred_score"]].agg(
                reducer
            )
            require(
                np.array_equal(
                    patient.loc[
                        expected.index, ["true_score", "pred_score"]
                    ].to_numpy(),
                    expected.to_numpy(),
                ),
                f"Patient export aggregation mismatch: {reducer}",
            )
            require(
                int(patient.image_count.sum()) == len(frame),
                "Patient image count mismatch",
            )
        for name, size in [("grade_confusion", 4), ("view_confusion", 6)]:
            matrix = np.asarray(read_json(root / f"{name}.json")["matrix"])
            require(
                matrix.shape == (size, size) and int(matrix.sum()) == len(frame),
                "Confusion dimensions/count mismatch",
            )
        centers = pd.read_csv(root / "centers.csv")
        require(
            int(centers.images.sum()) == counts[0]
            and int(centers.patients.sum()) == counts[1],
            "Center counts mismatch",
        )
        require(
            np.isclose(centers.normalized_weight.sum(), 1),
            "Center weights do not sum to one",
        )
        require(
            np.isclose(
                centers.weighted_cor_contribution.sum(),
                m["center_balanced_patient_max"]["cor"],
            ),
            "Center table does not match saved aggregate",
        )
        with np.load(root / "lesion_attention_float16.npz", allow_pickle=False) as z:
            require(
                z["image_uid"].tolist() == frame.image_uid.tolist(),
                "Attention row identity",
            )
            require(z["attention"].shape == (counts[0], 32, 32), "Attention dimensions")
            require(
                np.isfinite(z["attention"]).all(), "Attention contains nonfinite values"
            )
        checks.append(
            f"{split}: canonical metrics identity, complete schema, predictions, patients, centers, confusion and attention PASS"
        )
    require(
        not split_patients["test"] & split_patients["val"], "Patient overlap val/test"
    )
    history = pd.read_csv(PACK / "training/history.csv")
    require(
        history.epoch.tolist() == list(range(1, 121)),
        "Training trajectory missing epochs",
    )
    require(
        len(list((PACK / "training/val_epochs").glob("epoch_*/metrics.json"))) == 120,
        "Missing val epoch metrics",
    )
    require(
        len(
            list(
                (PACK / "training/val_epochs").glob(
                    "epoch_*/predictions_compact.csv.gz"
                )
            )
        )
        == 120,
        "Missing val epoch predictions",
    )
    checks.append("120 training epochs, 120 val metrics, 120 val prediction bundles")
    sys.path.insert(0, str(PACK / "source/round_snapshot/src"))
    from sfibai_b.data import FormalImageDataset

    mapping = pd.read_csv(PACK / "provenance/case_id_mapping_PRIVATE.csv")
    for split in ["test", "val"]:
        dataset = FormalImageDataset(
            dataset_root=DATA,
            manifest_path=DATA / "manifests/images.csv",
            annotations_path=DATA / "manifests/annotations.jsonl",
            split=split,
            arm="E",
            seed=2026,
            training=False,
        )
        with np.load(
            PACK / f"results/{split}/lesion_attention_float16.npz", allow_pickle=False
        ) as z:
            indices = {uid: i for i, uid in enumerate(z["image_uid"])}
            attention = z["attention"]
            for row in mapping[mapping.case_id.str.startswith(split)].itertuples():
                directory = PACK / "cases" / split / row.case_id
                case = read_json(directory / "case.json")
                require(
                    case["doctor_interpretation"] is None
                    and case["clinical_extent_claim"] == "NOT_ESTABLISHED",
                    "Clinical review must remain pending",
                )
                require(
                    np.array_equal(
                        np.load(directory / "attention_32x32_float16.npy"),
                        attention[indices[row.image_uid]],
                    ),
                    "Case attention mismatch",
                )
                with Image.open(row.source_image_path) as im, Image.open(
                    directory / "original_roi.png"
                ) as saved:
                    original = np.asarray(im.convert("RGB"))
                    require(
                        np.array_equal(original, np.asarray(saved)),
                        "Case ROI differs from original",
                    )
                expected_mask, _, _, _ = dataset._lesion_target(
                    row.image_uid, original.shape[:2], case["reference_score"]
                )
                with Image.open(
                    directory / "highest_severity_weak_box_mask.png"
                ) as saved:
                    require(
                        np.array_equal(expected_mask, np.asarray(saved) > 0),
                        "Case box raster differs from canonical geometry",
                    )
    checks.append(
        f"{len(mapping)} cases: original pixels, frozen attention and canonical box geometry verified; doctor fields remain pending"
    )
    resources = read_json(PACK / "resources/resources.json")
    timings = pd.read_csv(PACK / "resources/raw_latency_samples.csv")
    require(
        resources["checkpoint_sha256"] == complete["best_checkpoint_sha256"],
        "Resource checkpoint mismatch",
    )
    require(
        timings.groupby("batch").size().to_dict() == {1: 200, 8: 200}, "Timing counts"
    )
    require(resources["warmup"] == 50, "Warmup protocol")
    require(
        np.isfinite(timings.cuda_event_ms).all() and (timings.cuda_event_ms > 0).all(),
        "Invalid timing",
    )
    checks.append(
        "Frozen model GPU resources: batches 1/8, 50 warmup and 200 timed iterations each"
    )
    write_json(
        PACK / "verification/VALIDATION.json",
        {
            "status": "PASS",
            "scope": "file identity, schema and derived-export consistency; not a second canonical metric evaluation",
            "checks": checks,
            "clinician_review": "PENDING_AS_REQUESTED",
            "formal_inference_rerun": False,
        },
    )
    print(
        json.dumps({"status": "PASS", "checks": checks}, ensure_ascii=False), flush=True
    )


if __name__ == "__main__":
    main()
