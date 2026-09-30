# Repository maintenance

- Use English for primary documentation, issues, pull requests, commits, and new
  comments. Keep the Chinese README as an optional translation.
- Read and write text as UTF-8 without BOM.
- Keep the README concise: paper and code overview, citation, main figure, setup,
  usage, and license. Paper metadata must match the published arXiv record.
- The published VCRE-Fib citation is allowed in root `CITATION.cff` and
  `CITATION.bib`, and in the README. Preserve the pinned third-party citation;
  keep other bibliographies outside the checkout.
- Do not add manuscript text, result tables, clinical
  data, annotations, patient/image lists, per-example outputs, weights, or credentials.
- The main figure is the only public media asset. Render previews from its PDF.
- `code/` and `research/` implement the regional architecture. The distinct
  `reproducibility/delivered_checkpoint/` source must use a separate import root.
- Preserve source hashes and document packaging changes in provenance.
- Do not claim numerical reproduction from software tests.
- Train on train, select on validation, freeze, then evaluate test. Report image,
  patient-max, patient-median, and center metrics for formal experiments.
- Follow `CONTRIBUTING.md`: test, review, regenerate the inventory, stage explicit
  files, and run the final publication check before committing.
