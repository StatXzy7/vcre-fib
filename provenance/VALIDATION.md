# Release verification

Checks performed for the source release on 2026-09-26/27:

- Synthetic/model/loss/data-contract/evaluation/sampler and publication-guard tests:
  59 passed, 1 skipped locally; includes uppercase/private-file publication checks,
  exact staged-content matching, case-panel binding, imported-source completeness,
  and isolated pinned-baseline imports without data or weights.
- An export containing only Git-staged files passed the same 59 tests (1 skipped)
  and the exact publication inventory check without pre-existing temporary files.
- Before manuscript text was withdrawn from the public tree, all four private
  manuscript entry points compiled successfully with TeX Live 2026:
  English main 19 pages; Chinese main 17; supplementary 7 pages in each language.
  Existing duplicate PDF-destination warnings remain in supplementary builds.
- A Python wheel was built and checked for both `sfibai_b` and `synap_search`.
- The pinned SFibAI training module imported and all eight cited source hashes matched.
- The native train/val preparation code reproduced all three historical file hashes
  using the existing private Data V4 copy. Initial LF serialization mismatched the
  CSV hash; explicit CRLF serialization corrected it. Both annotation hashes matched.
  No clinical data or generated development manifests were copied into the public repo.
- Both the simple aggregate redraw and the original final-layout plot rendered all
  360 recorded rows without inference. The latter used a private derived 90-epoch
  input bundle and retained the four recorded selected epochs.
- Selected publication panels were visually inspected. Their PDFs contain no embedded
  attachments; the editable PPTX passed XML/external-link screening.
- File inventory/UTF-8/credential-pattern/private-artifact checks are implemented by
  `tools/check_release.py`; the final inventory is recorded beside this file.
- The first Linux CI run exposed a missing parent for pytest's temporary directory.
  The configured base directory now sits directly under the checkout. The source
  directory `third_party/SFibAI/src/sfibai/data` is included explicitly; the root
  dataset ignore no longer excludes Python packages of the same name.

The current release contains code and the main figure only. Manuscript text,
aggregate results, and project citation metadata are absent from the current tree.
The build and private-data checks above record local verification, not distributed
paper assets or numerical-reproduction claims. GitHub CI results are available in
the repository's Actions tab.

No model training, new test inference, new bootstrap, or numerical replication of
the paper's results was performed for this release. Read `docs/PROVENANCE.md` for
the unresolved source/result binding. CLI dry-runs and CPU checks do not certify
CUDA/DDP end-to-end execution on another machine.

Local software check environment: Python 3.10.20, torch 2.11.0+cu128,
torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, opencv-python 5.0.0.93,
scikit-learn 1.7.2, tensorboard 2.20.0, matplotlib 3.10.9, PyYAML 6.0.3,
optuna 4.9.0, Pillow 12.2.0, pytest 9.1.1. These are the release-check versions,
not a retroactive assertion about the historical training environment.
