"""Render real frozen attention and source annotations; never invent clinical review."""

from __future__ import annotations

import hashlib
import html
import json
import os
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from package_common import PACK, DATA, VIEW_NAMES, sha256, write_json, write_table

FONT_PATH = Path(os.environ.get("VCRE_FIGURE_FONT", "DejaVuSans.ttf"))
TITLES = {
    "localization_correspondence": "定位对应示例（按弱框重叠筛选）",
    "outside_box_activation": "框外激活候选（扩展区域待医生复核）",
    "low_overlap_diagnostic": "低重叠诊断示例",
    "validation_diagnostic": "Validation 补充诊断",
}


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def mask_for_annotations(rows: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize the original highest-severity boxes exactly as frozen data.py."""
    height, width = shape
    mask: np.ndarray = np.zeros(shape, np.uint8)
    highest = max(float(row["lesion_label_float"]) for row in rows)
    for row in rows:
        if not math.isclose(
            float(row["lesion_label_float"]), highest, abs_tol=1e-6, rel_tol=0
        ):
            continue
        left = max(0, min(width - 1, math.floor(row["bbox_crop_norm_x_min"] * width)))
        top = max(0, min(height - 1, math.floor(row["bbox_crop_norm_y_min"] * height)))
        right = max(
            left + 1, min(width, math.ceil(row["bbox_crop_norm_x_max"] * width))
        )
        bottom = max(
            top + 1, min(height, math.ceil(row["bbox_crop_norm_y_max"] * height))
        )
        mask[top:bottom, left:right] = 1
    return mask


def render_case(
    row: pd.Series,
    source_row: pd.Series,
    attention: np.ndarray,
    annotations: list[dict],
    split: str,
    kind: str,
    case_id: str,
) -> dict:
    output = PACK / "cases" / split / case_id
    output.mkdir(parents=True, exist_ok=False)
    source_path = DATA / str(source_row.image_path)
    source_hash = sha256(source_path)
    if source_hash != str(source_row.image_hash_v2):
        raise ValueError("Source ROI hash differs from the frozen manifest")
    with Image.open(source_path) as opened:
        original = opened.convert("RGB")
    width, height = original.size
    if (width, height) != (
        int(source_row.crop_image_width),
        int(source_row.crop_image_height),
    ):
        raise ValueError("ROI dimensions differ from manifest")
    original.save(output / "original_roi.png")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    highest = max(float(a["lesion_label_float"]) for a in annotations)
    for box in annotations:
        xy = [
            box["bbox_crop_norm_x_min"] * width,
            box["bbox_crop_norm_y_min"] * height,
            box["bbox_crop_norm_x_max"] * width,
            box["bbox_crop_norm_y_max"] * height,
        ]
        color = (
            "#ffad32"
            if math.isclose(float(box["lesion_label_float"]), highest, abs_tol=1e-6)
            else "#c0c0c0"
        )
        draw.rectangle(xy, outline=color, width=max(2, round(width / 220)))
    annotated.save(output / "doctor_boxes.png")
    reference_mask = mask_for_annotations(annotations, (height, width))
    Image.fromarray(reference_mask * 255).save(
        output / "highest_severity_weak_box_mask.png"
    )
    np.save(output / "attention_32x32_float16.npy", attention, allow_pickle=False)
    Image.fromarray(np.rint(attention.astype(float) * 65535).astype(np.uint16)).save(
        output / "attention_32x32_uint16.png"
    )
    native = cv2.resize(
        attention.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    color = np.asarray(colormaps["turbo"](native)[..., :3]) * 255
    alpha = (0.45 * native)[..., None]
    overlay = np.asarray(original) * (1 - alpha) + color * alpha
    predicted = Image.fromarray(np.rint(overlay).astype(np.uint8))
    # Threshold occurs at the native model resolution, then nearest-neighbor display.
    hard = cv2.resize(
        (attention >= 0.5).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    pred_array = np.asarray(predicted).copy()
    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(
        pred_array, contours, -1, (0, 255, 220), max(1, round(width / 400))
    )
    predicted = Image.fromarray(pred_array)
    predicted.save(output / "prediction_overlay.png")
    Image.fromarray(hard * 255).save(output / "prediction_mask_0p5.png")
    # Preserve the native ROI aspect ratio in every panel, using a common scale.
    panel_w = 510
    panel_h = round(height * panel_w / width)
    gap, margin, top, bottom = 18, 24, 112, 210
    canvas = Image.new(
        "RGB", (margin * 2 + panel_w * 3 + gap * 2, top + panel_h + bottom), "white"
    )
    d = ImageDraw.Draw(canvas)
    d.text(
        (margin, 12),
        f"{split.upper()} · {case_id} · {TITLES[kind]}",
        fill="#152d45",
        font=font(25),
    )
    titles = ["原始超声 ROI", "医生原始弱标注", "SynAP-Fib E 预测"]
    for index, (im, title) in enumerate(zip([original, annotated, predicted], titles)):
        x = margin + index * (panel_w + gap)
        d.text((x, 69), title, fill="#152d45", font=font(24))
        canvas.paste(im.resize((panel_w, panel_h), Image.Resampling.BILINEAR), (x, top))
    footer = top + panel_h + 14
    text_lines = [
        f"参考评分 {row.true_score:.1f}    预测评分 {row.pred_score:.3f}    真实视图 {int(row.position_true)} → 预测视图 {int(row.position_pred)}",
        f"真实：{VIEW_NAMES[int(row.position_true)]}",
        f"预测：{VIEW_NAMES[int(row.position_pred)]}",
        "橙框：最高严重程度弱标注；灰框：其他原始标注。热图固定 0–1；青色轮廓阈值 0.5。",
        "医生解释／框外区域影像依据：待医生复核。框外激活不等于已确认病灶或完整病灶覆盖。",
    ]
    for i, text in enumerate(text_lines):
        d.text((margin, footer + i * 35), text, fill="#384b5c", font=font(20))
    canvas.save(output / "triptych.png")
    canvas.save(output / "triptych.pdf", resolution=150)
    metadata = {
        "case_id": case_id,
        "split": split,
        "selection_category": kind,
        "true_view_id": int(row.position_true),
        "predicted_view_id": int(row.position_pred),
        "true_view_name": VIEW_NAMES[int(row.position_true)],
        "predicted_view_name": VIEW_NAMES[int(row.position_pred)],
        "reference_score": float(row.true_score),
        "prediction_score": float(row.pred_score),
        "source_image_sha256": source_hash,
        "attention_origin": "byte-identical frozen float16 attention export",
        "quantization_note": "display contour uses float16 maps; original canonical localization metrics used float32 attention before storage",
        "coordinate_frame": "roi_crop",
        "source_roi_size": [width, height],
        "attention_shape": [32, 32],
        "display": {
            "range": [0, 1],
            "colormap": "turbo",
            "opacity": "0.45 * attention",
            "contour_threshold": 0.5,
            "per_image_normalization": False,
            "attention_resize": "bilinear to original ROI",
            "hard_mask_resize": "nearest",
            "all_panels_same_field_of_view": True,
        },
        "doctor_interpretation": None,
        "doctor_review_status": "PENDING_CLINICIAN_REVIEW",
        "extended_region_image_evidence": None,
        "full_visible_lesion_reference": None,
        "clinical_extent_claim": "NOT_ESTABLISHED",
    }
    metadata["frozen_weak_box_metrics"] = {
        key: float(row[key])
        for key in [
            "inside_attention",
            "outside_ratio",
            "inside_outside_ratio",
            "dice_at_0_5",
            "iou_at_0_5",
        ]
    }
    write_json(output / "case.json", metadata)
    # Original identifying keys stay in the private provenance map, not figures.
    clean_boxes = [
        {
            k: v
            for k, v in a.items()
            if k.startswith("bbox_crop_")
            or k in {"lesion_label_float", "coordinate_frame"}
        }
        for a in annotations
    ]
    write_json(output / "doctor_annotations.json", clean_boxes)
    return metadata


def choose(
    frame: pd.DataFrame, annotations: dict[str, list[dict]]
) -> list[tuple[str, str]]:
    """Select descriptive examples by fixed rules, never clinical interpretation."""
    selected: list[tuple[str, str]] = []
    used_patients: set[str] = set()
    candidates = frame[frame.lesion_valid & (frame.true_score > 0)].copy()
    candidates = candidates[candidates.image_uid.isin(annotations)]
    if set(frame.split) == {"test"}:
        groups = [
            ("localization_correspondence", "dice_at_0_5", False, list(range(1, 7))),
            ("outside_box_activation", "outside_ratio", False, list(range(1, 7))),
            ("low_overlap_diagnostic", "dice_at_0_5", True, list(range(1, 4))),
        ]
    else:
        groups = [("validation_diagnostic", "dice_at_0_5", False, [1, 3, 5])]
    for kind, criterion, ascending, views in groups:
        for view in views:
            pool = candidates[
                (candidates.position_true == view)
                & ~candidates.patient_uid.isin(used_patients)
            ]
            if kind == "outside_box_activation":
                pool = pool[
                    (pool.inside_attention >= 0.5) & (pool.outside_ratio >= 0.6)
                ]
            if pool.empty:
                continue
            row = pool.sort_values(
                [criterion, "image_uid"], ascending=[ascending, True]
            ).iloc[0]
            selected.append((str(row.image_uid), kind))
            used_patients.add(str(row.patient_uid))
    return selected


def main() -> None:
    cases = PACK / "cases"
    cases.mkdir(exist_ok=True)
    if (cases / "case_index.csv").exists():
        raise FileExistsError("Case export already exists")
    annotations: dict[str, list[dict]] = {}
    with (PACK / "provenance/annotations_val_test.jsonl").open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            row = json.loads(line)
            annotations.setdefault(row["image_uid"], []).append(row)
    all_cases, mapping = [], []
    for split in ["test", "val"]:
        root = PACK / "results" / split
        frame = pd.read_csv(root / "predictions.csv.gz", float_precision="round_trip")
        manifest = pd.read_csv(root / "image_manifest.csv.gz").set_index("image_uid")
        frames_by_id = frame.set_index("image_uid", drop=False)
        with np.load(root / "lesion_attention_float16.npz", allow_pickle=False) as maps:
            index = {str(uid): i for i, uid in enumerate(maps["image_uid"])}
            attention = maps["attention"]
            for uid, kind in choose(frame, annotations):
                token = hashlib.sha256(
                    ("SynAP-Fib-E-case-v1:" + uid).encode()
                ).hexdigest()[:12]
                case_id = f"{split}-{token}"
                row = frames_by_id.loc[uid]
                metadata = render_case(
                    row,
                    manifest.loc[uid],
                    attention[index[uid]],
                    annotations[uid],
                    split,
                    kind,
                    case_id,
                )
                all_cases.append(
                    {
                        k: v
                        for k, v in metadata.items()
                        if k
                        in {
                            "case_id",
                            "split",
                            "selection_category",
                            "true_view_id",
                            "predicted_view_id",
                            "true_view_name",
                            "predicted_view_name",
                            "reference_score",
                            "prediction_score",
                            "doctor_review_status",
                        }
                    }
                )
                mapping.append(
                    {
                        "case_id": case_id,
                        "image_uid": uid,
                        "patient_uid": row.patient_uid,
                        "source_image_path": str(DATA / manifest.loc[uid].image_path),
                    }
                )
    write_table(pd.DataFrame(all_cases), cases / "case_index.csv")
    write_table(pd.DataFrame(mapping), PACK / "provenance/case_id_mapping_PRIVATE.csv")
    review = pd.DataFrame(all_cases)
    for key in [
        "doctor_interpretation",
        "extended_region_image_evidence",
        "full_visible_extent_reference",
        "reviewer_name",
        "review_date",
    ]:
        review[key] = ""
    write_table(review, cases / "doctor_review_to_fill.csv")
    policy = {
        "test": "6 highest Dice cases (one/view), 6 outside activation candidates (one/view, inside>=0.5 and outside mass>=0.6), 3 lowest Dice diagnostics (views1-3); unique patients",
        "val": "3 highest Dice diagnostic cases from views1,3,5; unique patients",
        "tie_break": "image_uid lexical order",
        "clinical_confirmation": "not available",
        "selection_is": "descriptive case selection, not unbiased performance sampling",
        "prohibited_claim": "outside activation proves lesion extension or complete visible lesion coverage",
    }
    write_json(cases / "selection_policy.json", policy)
    blocks = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>SynAP-Fib E 病例证据</title>",
        "<style>body{max-width:1500px;margin:32px auto;font-family:Arial,sans-serif;background:#f3f6f8;color:#17304c}article{background:white;margin:24px 0;padding:20px;border-radius:12px}img{width:100%}p{line-height:1.7}</style>",
        "<h1>SynAP-Fib E · 冻结 checkpoint 103 病例证据</h1>",
        "<p>同一原始 ROI / 原始医生弱框 / 冻结预测定位图。固定 0–1 色标，阈值 0.5。下列为按显式规则选择的描述性病例，不代表总体性能。医生解释与框外区域依据均待医生复核。</p>",
    ]
    for item in all_cases:
        rel = f"{item['split']}/{item['case_id']}/triptych.png"
        blocks.append(
            f"<article><h2>{html.escape(item['case_id'])} · {TITLES[item['selection_category']]}</h2><img src='{rel}' alt='原始ROI、医生标注、模型预测三联图'><p>医生复核：待填写。<a href='{item['split']}/{item['case_id']}/case.json'>病例字段</a></p></article>"
        )
    blocks.append("</html>")
    (cases / "index.html").write_text("\n".join(blocks), encoding="utf-8")
    print(f"Rendered {len(all_cases)} real cases; clinician review pending", flush=True)


if __name__ == "__main__":
    main()
