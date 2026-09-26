# Manuscript and publication assets

Run `python tools/build_paper.py --language all` from the repository root.
English uses pdfLaTeX; Chinese uses XeLaTeX. Both main manuscripts include their
appendix; standalone supplementary entry points are also retained.

The source text, numerical tables and selected figures are imported from the active
VCRE-Fib draft. Data-source reconciliation is documented in
[docs/PROVENANCE.md](../docs/PROVENANCE.md); a successful PDF build is not validation
of the underlying scientific claims or source/result correspondence.

Only composed publication figures are included. `figures/editable/Figure1.pptx`
is the editable architecture panel. Dataset images and case-to-patient mappings
are absent. Existing aggregate CSVs contain reported summary values only.
