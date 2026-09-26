"""Build a separately supplied private manuscript; no manuscript text is distributed."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh", "all"), default="all")
    parser.add_argument("--paper-root", type=Path, required=True, help="External directory containing the private LaTeX sources")
    args = parser.parse_args()
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise SystemExit("Install TeX Live with latexmk, pdfLaTeX, XeLaTeX and ctex.")
    for language in ("en", "zh"):
        if args.language not in (language, "all"):
            continue
        suffix = "_zh" if language == "zh" else ""
        engine = "-xelatex" if language == "zh" else "-pdf"
        for entry in ("main", "supplementary"):
            subprocess.run(
                [latexmk, engine, "-interaction=nonstopmode", "-halt-on-error", f"{entry}{suffix}.tex"],
                cwd=args.paper_root, check=True,
            )


if __name__ == "__main__":
    main()
