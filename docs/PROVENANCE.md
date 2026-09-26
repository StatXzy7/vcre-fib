# Source map and reproducibility boundary

This repository is a curated initial release from the active VCRE-Fib manuscript
snapshot of 2026-09-26. It has a new Git history. Private working directories,
experiment outputs, old paper drafts, and credentials were not imported.

## Source identities

| Public location | Source and intended use |
| --- | --- |
| `code/`, `research/main_experiment/`, `research/re_ablation/`, `research/autosearch/` | Source-only retrieval from the regional-method experiment deployment on 2026-09-26 |
| `reproducibility/delivered_checkpoint/` | `source/round_snapshot` in the corrected `SynAP-Fib_seed2026` delivery |
| `reproducibility/delivery_tools/` | Case selection, visualization, resource and evidence exporters shipped with that delivery |
| `third_party/SFibAI/` | Published baseline at commit `e5c2dc637c4a955fe305896644209d39a6a85fa6` |
| `paper/` | Active `iclr2027_re` manuscript; published aggregate rows preserved |

`provenance/IMPORT_MANIFEST.json` records original SHA-256 values for imported files,
release SHA-256 values, and whether packaging modified them. The separate
`RELEASE_MANIFEST.json` inventories every distributable file. A new release hash is
never presented as the original training-source hash.

## Source/result reconciliation remains open

The current manuscript associates its corrected full-model test row with checkpoint
SHA-256 `856ff4998e2007b731933d575684d4fa1a15a859cce58ef96d115d2261b14904`
and reported epoch 47. The accompanying delivery contains an `AE_COR_v2` source
snapshot whose constructor is `sfibai_b.model.build_model(arm="E", ...)` and whose
model defines position/lesion residual and gate modules. The regional implementation
uses `synap_search.re_model.build_re_model(...)`, view adapters, local evidence,
support pooling, and a probability mixture. These are distinct constructors and
state-dictionary layouts; they have not been silently aliased.

The delivery's training metadata records 120 epochs. The paper displays a 90-epoch
comparison prefix, while some full-model validation/trajectory/resource rows retain
the earlier regional-run source. The manuscript's existing values are preserved.
The release does not certify that its full-model test, validation, trajectory and
resource rows all belong to one implementation/checkpoint. Resolving this requires
an explicit, verified source/checkpoint/metric binding. No new experiment or result
replacement was performed as part of publication.

Current retrieved training sources are preserved with their actual model and loss
computations. A prior manuscript prose audit is not substituted for executable
source identity. Use the separate delivery archive to inspect the result-bearing
implementation. Do not expect the regional constructor to load the delivered
checkpoint, or either version to reproduce all paper numbers merely by renaming it.

## Packaging changes

- Replaced workstation/server default paths with `VCRE_WORKSPACE` or repository-relative paths.
- Added a portable launcher and an explicit `--epochs` argument (default 90, accepted
  21–120). The original regional runners hard-coded 120; historical amendments
  ended some runs at 90 externally. New executions get new source/config identities.
- Kept native data/count/hash checks, train/validation selection, frozen-checkpoint
  verification, and canonical test exports. Added an evaluation method identity check.
- Main summary generation reads the recorded horizon and does not implicitly run
  the historical optional bootstrap. The bootstrap implementation remains available.
- Added cross-platform paper build support, Chinese font fallback, publication
  filtering, source inventory and synthetic release tests.
- Preserved model and loss equations. The original research checkout, frozen source
  snapshot, metrics, predictions and checkpoints were not modified.

## Third-party material

SFibAI retains its Apache-2.0 license and original notices. The baseline source-lock
file verifies its cited public source files. ICLR style files and bundled LaTeX
packages retain their embedded licenses and authorship. No pretrained ImageNet
weights or other model binaries are redistributed.
