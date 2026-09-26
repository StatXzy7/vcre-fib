"""Stable TensorBoard tag routing for the AE_COR_v2 live dashboard."""

from __future__ import annotations


IMAGE_ACCURACY_TOKENS = (
    "accuracy",
    "macro_f1",
    "recall",
    "support",
)
PROBABILITY_TOKENS = ("auroc", "auprc", "brier", "ece")


def evaluation_tag(name: str, *, split: str = "val") -> str:
    """Route one flattened evaluator metric into one of seven fixed groups."""

    metric = str(name).strip("/")
    normalized = metric.lower()
    leaf = metric.removeprefix("image/")

    if normalized.startswith("image/") and any(
        token in normalized for token in PROBABILITY_TOKENS
    ):
        return f"04_auc_calibration/{split}/{leaf}"

    if normalized.startswith("image/") and any(
        token in normalized for token in IMAGE_ACCURACY_TOKENS
    ):
        return f"01_image_accuracy/{split}/{leaf}"

    if normalized.startswith(("patient_max/", "patient_median/")) and any(
        token in normalized for token in IMAGE_ACCURACY_TOKENS
    ):
        return f"02_patient_accuracy/{split}/{metric}"

    if normalized == "r_final" or "cor" in normalized:
        return f"03_cor_selection/{split}/{metric}"

    if normalized.startswith(("position/", "lesion/", "gates/")):
        return f"06_auxiliary/{split}/{metric}"

    return f"05_error_profile/{split}/{metric}"


def training_tag(name: str) -> str:
    """Route training health scalars into the seventh fixed group."""

    return f"07_training_health/train/{str(name).strip('/')}"


def custom_scalars_layout() -> dict[str, dict[str, list[str]]]:
    """Return a compact live dashboard without hiding the full scalar archive."""

    return {
        "01 Image accuracy": {
            "Image accuracy": [
                "Multiline",
                [
                    evaluation_tag("image/grade_accuracy"),
                    evaluation_tag("image/grade_macro_f1"),
                    evaluation_tag("image/accuracy_within_0_3"),
                    evaluation_tag("image/accuracy_within_0_5"),
                ],
            ]
        },
        "02 Patient accuracy": {
            "Patient grade accuracy": [
                "Multiline",
                [
                    evaluation_tag("patient_max/grade_accuracy"),
                    evaluation_tag("patient_median/grade_accuracy"),
                    evaluation_tag("patient_max/grade_macro_f1"),
                    evaluation_tag("patient_median/grade_macro_f1"),
                ],
            ]
        },
        "03 COR selection": {
            "COR": [
                "Multiline",
                [
                    evaluation_tag("r_final"),
                    evaluation_tag("image/cor"),
                    evaluation_tag("patient_max/cor"),
                    evaluation_tag("center_balanced_patient_max/cor"),
                ],
            ]
        },
        "04 AUC and calibration": {
            "Image probabilistic metrics": [
                "Multiline",
                [
                    evaluation_tag("image/grade_macro_auroc"),
                    evaluation_tag("image/grade_macro_auprc"),
                    evaluation_tag("image/grade_brier"),
                    evaluation_tag("image/grade_ece"),
                ],
            ]
        },
        "05 Error profile": {
            "Image errors": [
                "Multiline",
                [
                    evaluation_tag("image/mae"),
                    evaluation_tag("image/tmae"),
                    evaluation_tag("image/severe_error_rate"),
                ],
            ]
        },
        "06 Auxiliary": {
            "Position": [
                "Multiline",
                [
                    evaluation_tag("position/accuracy"),
                    evaluation_tag("position/macro_f1"),
                ],
            ],
            "Lesion": [
                "Multiline",
                [
                    evaluation_tag("lesion/inside_attention"),
                    evaluation_tag("lesion/outside_ratio"),
                    evaluation_tag("lesion/dice_at_0_5"),
                    evaluation_tag("lesion/iou_at_0_5"),
                ],
            ],
            "Fusion gates": [
                "Multiline",
                [
                    evaluation_tag("gates/position/mean"),
                    evaluation_tag("gates/lesion/mean"),
                ],
            ],
        },
        "07 Training health": {
            "Losses": [
                "Multiline",
                [
                    training_tag("total"),
                    training_tag("grade"),
                    training_tag("position"),
                    training_tag("weak_box"),
                ],
            ],
            "Learning rate": ["Multiline", [training_tag("learning_rate")]],
        },
    }
