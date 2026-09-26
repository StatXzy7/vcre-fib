"""Compose the retained Figure 2 cases with labels beside intact image panels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pymupdf as pdf
from case_binding import verify_case_binding

PAPER = Path(os.environ.get("VCRE_PAPER_ROOT", str(Path(__file__).resolve().parents[2] / ".tmp/paper-generation"))).resolve()
SOURCE = PAPER / "figures/sources/fig06_cases_before_header_upgrade.pdf"
METADATA = PAPER / "results/selected_cases.json"
OUTPUT = PAPER / "figures/assets/fig06_cases.pdf"
CASE_IDS = ()  # Loaded in metadata order from the private case file.
WIDTH, HEIGHT = 396, 410
IMAGE_SIDE, ROW_HEIGHT, FIRST_ROW = 84, 91, 30
IMAGE_LEFTS = (133, 221, 309)
INK = (0.13, 0.19, 0.25)


def digest(path: Path) -> str:
    """Return a file digest without modifying its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_digests(document: pdf.Document) -> set[str]:
    """Fingerprint the embedded raster payloads, independent of placement."""
    return {
        hashlib.sha256(document.extract_image(item[0])["image"]).hexdigest()
        for item in document[0].get_images(full=True)
    }


def main() -> None:
    """Reflow case information and vector-embed all twelve original panels."""
    global CASE_IDS
    (PAPER / "audits").mkdir(parents=True, exist_ok=True)
    records = json.loads(METADATA.read_text(encoding="utf-8"))
    cases = [record for record in records if record["split"] == "test"]
    CASE_IDS = tuple(record["case_id"] for record in cases)
    if len(CASE_IDS) != 4 or len(set(CASE_IDS)) != 4:
        raise ValueError("Case order changed; inspect the panel bindings first.")
    verify_case_binding(SOURCE, METADATA, PAPER / "figures/sources/fig06_cases_binding.json", CASE_IDS)
    metadata_hash = digest(METADATA)
    with pdf.open(SOURCE) as source, pdf.open(SOURCE) as panels, pdf.open() as document:
        if len(source) != 1 or tuple(source[0].rect) != (0, 0, 396, 570):
            raise ValueError("Unexpected source figure geometry.")
        original_images = image_digests(source)
        # Old labels slightly overlap the panel boundaries. Remove vector text
        # from a disposable copy; keep raster pixels and annotation graphics.
        panels[0].add_redact_annot(panels[0].rect, fill=False)
        panels[0].apply_redactions(
            images=pdf.PDF_REDACT_IMAGE_NONE,
            graphics=pdf.PDF_REDACT_LINE_ART_NONE,
            text=pdf.PDF_REDACT_TEXT_REMOVE,
        )
        if panels[0].get_text().strip() or image_digests(panels) != original_images:
            raise ValueError("Panel text cleanup did not preserve the original images.")
        page = document.new_page(width=WIDTH, height=HEIGHT)
        page.draw_rect(
            pdf.Rect(0, 0, WIDTH, 27.5), color=None, fill=(0.95, 0.965, 0.975)
        )
        page.draw_line((0, 27.5), (WIDTH, 27.5), color=(0.67, 0.73, 0.78), width=0.5)
        headings = [(0, 127, "Case and prediction", 9.2)] + [
            (left, IMAGE_SIDE, title, 10.2)
            for left, title in zip(
                IMAGE_LEFTS,
                (
                    "Original\nultrasound",
                    "Clinician\nannotation",
                    "Model\nlocalization",
                ),
                strict=True,
            )
        ]
        for left, width, title, size in headings:
            top = 8 if "\n" not in title else 1
            remaining = page.insert_textbox(
                pdf.Rect(left, top, left + width, 27.5),
                title,
                fontname="hebo",
                fontsize=size,
                lineheight=1.02,
                color=INK,
                align=1,
            )
            if remaining < 0:
                raise ValueError(f"Header overflow: {title}")
        for index, case in enumerate(cases):
            top = FIRST_ROW + index * ROW_HEIGHT
            category = (
                "Correspondence"
                if case["category"] == "localization_correspondence"
                else "Extent expansion"
            )
            page.insert_text(
                (0, top + 8),
                f"({chr(97 + index)}) {category}",
                fontsize=8.2,
                fontname="hebo",
                color=INK,
            )
            score = (
                f"Score: reference {case['reference_score']:.1f} / "
                f"predicted {case['predicted_score']:.3f}"
            )
            if pdf.get_text_length(score, fontsize=7.2) > 127:
                raise ValueError(f"Score label overflow: {case['case_id']}")
            page.insert_text((0, top + 19), score, fontsize=7.2)
            if case["true_view_name"] == case["predicted_view_name"]:
                views = f"True / predicted view:\n{case['true_view_name']}"
            else:
                views = (
                    f"True view: {case['true_view_name']}\n"
                    f"Predicted view: {case['predicted_view_name']}"
                )
            remaining = page.insert_textbox(
                pdf.Rect(0, top + 24, 126, top + 88),
                views,
                fontsize=7.6,
                fontname="helv",
                lineheight=1.05,
            )
            if remaining < 0:
                raise ValueError(f"View label overflow: {case['case_id']}")
            for column, left in enumerate(IMAGE_LEFTS):
                source_left = 16 + column * 132
                source_top = 50 + index * 132
                page.show_pdf_page(
                    pdf.Rect(left, top, left + IMAGE_SIDE, top + IMAGE_SIDE),
                    panels,
                    0,
                    clip=pdf.Rect(
                        source_left, source_top, source_left + 100, source_top + 100
                    ),
                )
            if index < len(cases) - 1:
                page.draw_line(
                    (0, top + 88.5),
                    (WIDTH, top + 88.5),
                    color=(0.84, 0.86, 0.88),
                    width=0.3,
                )
        # Retain the exact original color scale and display-parameter legend.
        page.show_pdf_page(
            pdf.Rect(0, HEIGHT - 16, WIDTH, HEIGHT),
            source,
            0,
            clip=pdf.Rect(0, 554, 396, 570),
        )
        temporary = OUTPUT.with_name(OUTPUT.stem + ".pending.pdf")
        document.save(temporary, garbage=4, deflate=True)

    with pdf.open(temporary) as result:
        if image_digests(result) != original_images:
            raise ValueError("Embedded image payloads changed.")
        text = " ".join(result[0].get_text().split())
        if text.count("Score: reference") != 4:
            raise ValueError("Missing or duplicate case scores.")
        for case in cases:
            for field in ("true_view_name", "predicted_view_name"):
                if case[field] not in text:
                    raise ValueError(f"View name missing: {case['case_id']}")
        preview = OUTPUT.with_name(OUTPUT.stem + ".pending.png")
        result[0].get_pixmap(matrix=pdf.Matrix(3, 3)).save(preview)
    if digest(METADATA) != metadata_hash:
        raise ValueError("Case metadata changed during composition.")
    temporary.replace(OUTPUT)
    preview.replace(OUTPUT.with_suffix(".png"))
    report = {
        "status": "PASS",
        "source": str(SOURCE.relative_to(PAPER)),
        "source_sha256": digest(SOURCE),
        "case_metadata_sha256": metadata_hash,
        "output_sha256": digest(OUTPUT),
        "source_size_pt": [396, 570],
        "output_size_pt": [WIDTH, HEIGHT],
        "image_side_pt": IMAGE_SIDE,
        "column_heading_font_pt": 10.2,
        "case_view_font_pt": 7.6,
        "case_score_font_pt": 7.2,
        "case_ids": list(CASE_IDS),
        "retained_panels": 12,
        "embedded_image_payloads_unchanged": True,
        "original_vector_annotation_graphics_preserved": True,
        "legacy_label_fragments_removed_from_panel_copy": True,
        "complete_view_names_verified": True,
        "inference_calls": 0,
    }
    (PAPER / "audits/FIGURE2_COMPACT_LAYOUT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
