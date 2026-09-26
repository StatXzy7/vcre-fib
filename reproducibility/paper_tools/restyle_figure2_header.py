"""Restyle Figure 2's column headings while preserving its case body exactly.

The saved pre-edit PDF is the reproducible source. No result import or model
inference is performed. Rendered pixels below the header must remain identical.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pymupdf as pdf

PAPER = Path(os.environ.get("VCRE_PAPER_ROOT", str(Path(__file__).resolve().parents[2] / ".tmp/paper-generation"))).resolve()
SOURCE = PAPER / "figures/sources/fig06_cases_before_header_upgrade.pdf"
OUTPUT = PAPER / "figures/assets/fig06_cases.pdf"
HEADINGS = ("Original ultrasound", "Clinician annotation", "Model localization")
HEADER_BOTTOM = 18.0
FONT_SIZE = 10.2


def sha256(path: Path) -> str:
    """Return a file digest for the figure provenance record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Enlarge vector headings, validate the untouched body, and export."""
    (PAPER / "audits").mkdir(parents=True, exist_ok=True)
    with pdf.open(SOURCE) as document:
        if len(document) != 1 or tuple(document[0].rect) != (0, 0, 396, 570):
            raise ValueError("Unexpected source layout; inspect before restyling.")
        page = document[0]
        header = pdf.Rect(0, 0, page.rect.width, HEADER_BOTTOM)
        body = pdf.Rect(0, HEADER_BOTTOM, page.rect.width, page.rect.height)
        if page.get_text("text", clip=header).splitlines() != list(HEADINGS):
            raise ValueError("Source headings differ from the expected three columns.")
        body_before = page.get_pixmap(matrix=pdf.Matrix(3, 3), clip=body).samples
        body_text = page.get_text("text", clip=body)

        # Only the original vector heading text lies in this 18-point strip.
        page.add_redact_annot(header, fill=(1, 1, 1))
        page.apply_redactions(
            images=pdf.PDF_REDACT_IMAGE_NONE,
            graphics=pdf.PDF_REDACT_LINE_ART_NONE,
            text=pdf.PDF_REDACT_TEXT_REMOVE,
        )
        page.draw_rect(
            pdf.Rect(0, 0, 396, 17.5),
            color=None,
            fill=(0.95, 0.965, 0.975),
        )
        page.draw_line((0, 17.5), (396, 17.5), color=(0.67, 0.73, 0.78), width=0.5)
        for center, heading in zip((66, 198, 330), HEADINGS, strict=True):
            width = pdf.get_text_length(heading, fontname="hebo", fontsize=FONT_SIZE)
            if width > 124:
                raise ValueError(f"Heading exceeds its column: {heading}")
            page.insert_text(
                (center - width / 2, 12.5),
                heading,
                fontname="hebo",
                fontsize=FONT_SIZE,
                color=(0.13, 0.19, 0.25),
            )
        if page.get_pixmap(matrix=pdf.Matrix(3, 3), clip=body).samples != body_before:
            raise ValueError("Body pixels changed; refusing to publish the figure.")
        if page.get_text("text", clip=body) != body_text:
            raise ValueError("Case labels changed; refusing to publish the figure.")

        temporary = OUTPUT.with_name(OUTPUT.stem + ".pending.pdf")
        document.save(temporary, garbage=4, deflate=True)

    # Reopen the serialized PDF before replacing the current manuscript asset.
    with pdf.open(temporary) as document:
        page = document[0]
        if page.get_pixmap(matrix=pdf.Matrix(3, 3), clip=body).samples != body_before:
            raise ValueError("Serialized PDF changed body pixels.")
        spans = [
            span
            for block in page.get_text("dict", clip=header)["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
        ]
        if [span["text"] for span in spans] != list(HEADINGS):
            raise ValueError("Missing or duplicated column heading.")
        if any(abs(span["size"] - FONT_SIZE) > 0.001 for span in spans):
            raise ValueError("Unexpected heading font size.")
        preview = OUTPUT.with_name(OUTPUT.stem + ".pending.png")
        page.get_pixmap(matrix=pdf.Matrix(3, 3)).save(preview)

    temporary.replace(OUTPUT)
    preview.replace(OUTPUT.with_suffix(".png"))
    audit = {
        "status": "PASS",
        "source": str(SOURCE.relative_to(PAPER)),
        "source_sha256": sha256(SOURCE),
        "output": str(OUTPUT.relative_to(PAPER)),
        "output_sha256": sha256(OUTPUT),
        "old_heading_font_pt": 7,
        "new_heading_font_pt": FONT_SIZE,
        "heading_size_increase_percent": round((FONT_SIZE / 7 - 1) * 100, 2),
        "style": "Bold dark-slate headings, pale header band, fine lower rule.",
        "page_size_pt": [396, 570],
        "unchanged_body_from_y_pt": HEADER_BOTTOM,
        "body_pixel_comparison_scale": 3,
        "body_pixels_identical": True,
        "body_text_identical": True,
        "inference_calls": 0,
    }
    (PAPER / "audits/FIGURE2_HEADER_STYLE.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
