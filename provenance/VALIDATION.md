# Release verification

Checks performed for the initial source release on 2026-09-26:

- Synthetic/model/loss/data-contract/evaluation/sampler and publication-guard tests:
  57 passed, 1 skipped; includes uppercase/private-file publication checks, exact
  staged-content matching, and source/metadata/order binding for case panels.
- All four manuscript entry points compiled successfully with TeX Live 2026:
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

No model training, new test inference, new bootstrap, or numerical replication of
the paper's results was performed for this release. Read `docs/PROVENANCE.md` for
the unresolved source/result binding. CLI dry-runs and CPU checks do not certify
CUDA/DDP end-to-end execution on another machine.

Local software check environment: Python 3.10.20, torch 2.11.0+cu128,
torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, opencv-python 5.0.0.93,
scikit-learn 1.7.2, tensorboard 2.20.0, matplotlib 3.10.9, PyYAML 6.0.3,
optuna 4.9.0, Pillow 12.2.0, pytest 9.1.1. These are the release-check versions,
not a retroactive assertion about the historical training environment.
