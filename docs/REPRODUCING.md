# Reproduction guide

Read [PROVENANCE.md](PROVENANCE.md) first. Software checks and manuscript builds
can run from this public repository. Full numerical reproduction additionally
requires authorized access to the private dataset, appropriate hardware, and
resolution of the corrected full-model source/result binding.

## Install and check without data

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_release.py
python tools/build_paper.py --language all
```

`sfibai_b` and `synap_search` retain their original package names for compatibility.
Do not put the delivered-checkpoint archive on the same Python path as `code/src`.

## Private workspace contract

Use an external directory, referred to below as `$VCRE_WORKSPACE`:

```text
private-workspace/
  data/schisto_2024_clean_v4/
    manifests/images.csv
    manifests/annotations.jsonl
    train/...  val/...  test/...
  development/                 # generated, private
  cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
  outputs/                     # generated, private
```

The CSV schema includes `image_uid`, `patient_uid`, `center_id`, `image_path`,
`position_norm` (1–6), `image_label_max` (0.0–3.5 in steps of 0.1), `split`, and
`image_hash_v2`. Box JSONL records use `image_uid`, `coordinate_frame=roi_crop`,
`lesion_label_float`, and normalized `bbox_crop_norm_x_min/y_min/x_max/y_max`.
Only maximum-score boxes form weak targets; score zero does not use weak-box loss.
Images and annotations remain private; this document defines fields, not records.

The original Data V4 identity is locked in the source: native train/val/test contain
83,722/20,880/4,107 images, 4,906/1,227/240 patients, and 35/33/4 centers.
Patients and identical images must not cross splits. The data preparation command
checks the original manifest hashes, derives native train/val inputs, and checks
their historical hashes. It retains a diagnostic and stops if they differ:

```bash
python tools/prepare_data.py --workspace "$VCRE_WORKSPACE"
```

The legacy inner-partition metadata is retained solely to reproduce the serialized
development manifest. Formal training consumes all native train and native val.
Using a different cohort requires a separate protocol and identity, not removal
of the original identity checks while claiming a paper reproduction.

Acquire torchvision's ResNet-50 ImageNet-1K V2 initialization through its official
weight loader and use `TORCH_HOME=$VCRE_WORKSPACE/cache/torch`. The baseline and
regional models then share `resnet50-11ad3fa6.pth`. These weights are not in Git.

## Regional-model recipe

Run on Linux with a supported CUDA GPU. Use the same GPU count for training and
the subsequent evaluation, since the execution environment is identity-bound.
Every output directory must be new unless an explicit strict resume is requested.

```bash
python tools/run_experiment.py train --method vcre --workspace "$VCRE_WORKSPACE" --output "$VCRE_WORKSPACE/outputs/vcre" --epochs 90 --gpus 1
python tools/run_experiment.py evaluate --method vcre --workspace "$VCRE_WORKSPACE" --output "$VCRE_WORKSPACE/outputs/vcre" --gpus 1
python tools/run_experiment.py resources --method vcre --workspace "$VCRE_WORKSPACE" --output "$VCRE_WORKSPACE/outputs/vcre" --gpus 1
```

Repeat with `--method sfibai`, `wo-view`, or `wo-weakloc` and separate outputs.
Use `--dry-run` to inspect a command without GPU work. `CUDA_VISIBLE_DEVICES`
controls the allocated devices. `--gpus` supports 1, 2, 4 or 8; all keep global batch
24 using the original unpadded sampler and global-loss gathering. Different GPU
counts may produce different floating-point trajectories.

Train uses only train; validation selects from epochs 21 through the declared cap
by lower R_final, then lower image COR, then earlier epoch. Freeze precedes test.
The evaluator exports per-image predictions, structured metrics, patient-max and
patient-median tables, center-weighted metrics, view and available weak-localization
metrics. Those outputs stay in the private workspace. Test access never fits model
parameters, epoch selection, thresholds or calibration.

R_final is `0.4 * image COR + 0.4 * patient-max COR + 0.2 * center-balanced COR`,
lower is better. Center aggregation uses square-root patient-count weights.

For a new two-method round, put outputs in `ROUND/SFIBAI` and `ROUND/SYNAP`, then
run `research/main_experiment/finalize.py --round-root ROUND` after both evaluations
and profiles complete. The deletion summary/launch scripts retain historical
reference-hash guards; use the portable per-method launcher for new executions.

## Result-bearing archive and case exports

`reproducibility/delivered_checkpoint/` includes the delivered model, preprocessing,
loss, training, prediction, evaluation, configuration, scripts and tests. It is a
separate historical implementation, with 120-epoch metadata and its own freeze
contracts. Set `VCRE_WORKSPACE` and use this archive's `src` only in an isolated
Python process. Its snapshot/provenance inputs and weights are private and are not
re-created or fabricated by this release.

`reproducibility/delivery_tools/` contains the evidence and case-generation code.
Set `VCRE_DELIVERY_ROOT` to your private delivery package, `VCRE_WORKSPACE` to the
private workspace, and `VCRE_FIGURE_FONT` to an installed font supporting Chinese.
These scripts consume the delivery's original private file schema. They are not
called by the paper build. Case selection needs private predictions, image manifests,
boxes and attention arrays; the public composed figure is sufficient for typesetting.

## Paper assets and tables

The canonical manuscript build uses the committed `.tex` tables and composed PDFs.
`paper/scripts/update_aligned_tables.py` regenerates the numerical table content
from published aggregate CSVs, but manual table layout edits in the committed paper
may differ; run it in a disposable checkout if preserving the exact layout matters.
`paper/scripts/plot_training.py` provides a simple redraw from the 360-row aggregate
CSV. The original final-layout plot, case importer, compact compositor and header
restyler are also included in `reproducibility/paper_tools/`. Set `VCRE_PAPER_ROOT`
to a private generation directory with the expected inputs; its default is the
ignored `.tmp/paper-generation`. The plot expects its original `--source` directory
containing per-method training CSVs and `MANIFEST.json`, plus the private source
bindings under the generation root. The importer takes a private delivery directory
and a private selection JSON with `main` (four case IDs), `supplement` (two IDs),
and optional `excluded` lists. IDs are external inputs, not public constants.
The importer writes a private binding containing the original panel PDF hash,
metadata hash and ordered main-case IDs. The compact compositor requires that
binding (`figures/sources/fig06_cases_binding.json`), `results/selected_cases.json`
and the original panel PDF; it rejects reordering, relabeling or source drift.
Intermediate case panels and bindings are not redistributed. Editable Figure 1 is
`paper/figures/editable/Figure1.pptx`.

Raw images and a patient mapping cannot be recovered from an authorized dataset
release in this repository, because no such release has been made here.
