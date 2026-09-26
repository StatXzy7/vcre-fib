"""Local dependency for checking the three-panel publication plot layout."""
from __future__ import annotations

import json
from pathlib import Path


def require_matplotlib_panel_alignment(figure, *, json_out, overlay_svg,
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                       strict=True):
    """Check horizontal/vertical alignment in PDF points, retaining diagnostics."""
    width, height = figure.get_size_inches() * 72
    boxes = sorted((axis.get_position() for axis in figure.axes), key=lambda box: box.x0)
    if len(boxes) != 3:
        raise ValueError("This publication figure requires three axes")
    bottoms = [box.y0 * height for box in boxes]
    tops = [box.y1 * height for box in boxes]
    gutters = [(right.x0 - left.x1) * width for left, right in zip(boxes, boxes[1:])]
    aligned = (max(bottoms) - min(bottoms) <= tolerance_pt
               and max(tops) - min(tops) <= tolerance_pt
               and min(gutters) > 0
               and max(gutters) - min(gutters) <= gutter_tolerance_pt)
    report = {"status": "PASS" if aligned else "FAIL", "bottoms_pt": bottoms,
              "tops_pt": tops, "gutters_pt": gutters,
              "implementation": "public release three-panel layout check"}
    path = Path(json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rects = "".join(f'<rect x="{b.x0*width}" y="{(1-b.y1)*height}" width="{b.width*width}" height="{b.height*height}" fill="none" stroke="red"/>' for b in boxes)
    Path(overlay_svg).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">{rects}</svg>\n', encoding="utf-8")
    if strict and not aligned:
        raise ValueError("Publication plot panels are misaligned; inspect the retained diagnostic")
    return report
