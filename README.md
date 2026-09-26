# VCRE-Fib

Code for view-conditioned regional evidence learning in ultrasound liver fibrosis grading.

[![Release checks](https://github.com/StatXzy7/vcre-fib/actions/workflows/ci.yml/badge.svg)](https://github.com/StatXzy7/vcre-fib/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[![VCRE-Fib architecture: shared representation, view-specific regional evidence, and image-only predictions.](paper/figures/previews/fig01_architecture.png)](paper/figures/assets/fig01_architecture.pdf)

[Vector figure](paper/figures/assets/fig01_architecture.pdf) ·
[Editable PowerPoint](paper/figures/editable/Figure1.pptx) ·
[Usage guide](docs/REPRODUCING.md) · [中文说明](README.zh-CN.md)

## Quick start

Use Python 3.10 or later and a matching PyTorch/torchvision pair for your machine.

```bash
git clone https://github.com/StatXzy7/vcre-fib.git
cd vcre-fib
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_release.py
```

Tests use synthetic inputs and require no clinical data or model weights. Formal
training and evaluation use Linux, CUDA, and NCCL for multi-GPU execution.

## Code

| Directory | Contents |
| --- | --- |
| [code/src/sfibai_b](code/src/sfibai_b/) | Data processing, model components, losses, and metrics |
| [research/autosearch/src/synap_search](research/autosearch/src/synap_search/) | Regional model and training objective |
| [research/main_experiment](research/main_experiment/) | Training, checkpoint selection, evaluation, and resource profiling |
| [research/re_ablation](research/re_ablation/) | View and weak-localization deletion variants |
| [third_party/SFibAI](third_party/SFibAI/) | Pinned baseline source with its original license |
| [reproducibility](reproducibility/) | Historical implementation and visualization/export tools |
| [tools](tools/) | Portable entry points, data preparation, and release checks |
| [tests](tests/) | Synthetic/model/loss/data-contract tests |

The [usage guide](docs/REPRODUCING.md) describes installation, private input formats,
and experiment commands. Preview a command without starting training:

```bash
python tools/run_experiment.py train --method vcre --workspace /path/to/private-workspace --output /path/to/private-workspace/outputs/vcre --epochs 90 --gpus 1 --dry-run
```

## Availability

This release contains code and the main figure. Datasets, annotations, patient/image
lists, predictions, weights, raw logs, manuscript text, and result tables are not
distributed.

The regional model and the historical delivered-checkpoint implementation are
distinct. Their result/source correspondence remains unresolved; passing software
tests does not establish numerical reproduction. See [source provenance](docs/PROVENANCE.md)
and [software checks](provenance/VALIDATION.md).

## License and contributions

Original code uses [Apache-2.0](LICENSE). Third-party code retains its original
licenses and notices; see [NOTICE](NOTICE).

English is the default maintenance language. Read [CONTRIBUTING.md](CONTRIBUTING.md)
before making changes. Keep private inputs and experiment outputs outside this checkout.
