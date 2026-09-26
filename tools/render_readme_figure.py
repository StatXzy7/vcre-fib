"""Render the committed Figure 1 PDF for GitHub's README without changing its content."""
from pathlib import Path

import pymupdf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "paper/figures/assets/fig01_architecture.pdf"
    target = root / "paper/figures/previews/fig01_architecture.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as document:
        if len(document) != 1:
            raise ValueError("Figure 1 must be a single-page PDF")
        document[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False).save(target)
    print(target.relative_to(root))


if __name__ == "__main__":
    main()
