# VCRE-Fib

Code accompanying **VCRE-Fib: View-Conditioned Regional Evidence for Fine-Grained
Ultrasound Grading of Schistosoma japonicum-Associated Liver Fibrosis**.

[![arXiv](https://img.shields.io/badge/arXiv-2609.32840-b31b1b.svg)](https://arxiv.org/abs/2609.32840)
[![Release checks](https://github.com/StatXzy7/vcre-fib/actions/workflows/ci.yml/badge.svg)](https://github.com/StatXzy7/vcre-fib/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Ziyang Xu, Shuli An, Hao Zhou, Haitian Zhong, Tingting Wu, Tao Wang, Kun Yang,
Tieyong Zeng** · arXiv preprint, 2026.

[Paper](https://arxiv.org/abs/2609.32840) ·
[PDF](https://arxiv.org/pdf/2609.32840) ·
[HTML](https://arxiv.org/html/2609.32840v1) ·
[Citation](#citation) · [Usage guide](docs/REPRODUCING.md) · [中文说明](README.zh-CN.md)

## Overview

VCRE-Fib combines global image assessment with regional grading evidence conditioned
on the ultrasound acquisition view. View conditioning acts on local features before
spatial pooling; weak localization and learned support guide aggregation, and a
probability mixture combines the view-conditioned predictions.

Training uses grading labels, six-class acquisition-view labels, and weak lesion
boxes. Inference takes an image alone and returns a fine-grained fibrosis score,
an acquisition-view prediction, and a candidate abnormal-region map. The score
represents expert-assigned ultrasound appearance; localization is a weak spatial
prediction rather than exhaustive lesion segmentation. Study results and their
limitations are reported in the [paper](https://arxiv.org/abs/2609.32840).

## Main figure

[![VCRE-Fib architecture: shared representation, view-specific regional evidence, and image-only predictions.](paper/figures/previews/fig01_architecture.png)](paper/figures/assets/fig01_architecture.pdf)

**Figure 1.** View conditioning before spatial aggregation: (a) shared image
features, global context, view probabilities, weak localization, and learned
support; (b) six view-conditioned regional paths and their probability mixture;
(c) image-only outputs with a reference annotation and the paper's test summary.
Reference boxes are shown for comparison and are not inference inputs.

[Vector figure](paper/figures/assets/fig01_architecture.pdf) ·
[Editable PowerPoint](paper/figures/editable/Figure1.pptx)

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

This release contains code, paper citation metadata, and the main figure. Datasets,
annotations, patient/image lists, predictions, weights, raw logs, manuscript text,
and result tables are not distributed.

The regional model and the historical delivered-checkpoint implementation are
distinct. Their result/source correspondence remains unresolved; passing software
tests does not establish numerical reproduction. See [source provenance](docs/PROVENANCE.md)
and [software checks](provenance/VALIDATION.md).

## Citation

If you use this work, please cite the [arXiv preprint](https://arxiv.org/abs/2609.32840):

```bibtex
@misc{xu2026vcrefib,
  title         = {{VCRE-Fib}: View-Conditioned Regional Evidence for Fine-Grained Ultrasound Grading of {Schistosoma japonicum}-Associated Liver Fibrosis},
  author        = {Xu, Ziyang and An, Shuli and Zhou, Hao and Zhong, Haitian and Wu, Tingting and Wang, Tao and Yang, Kun and Zeng, Tieyong},
  year          = {2026},
  eprint        = {2609.32840},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2609.32840},
  url           = {https://arxiv.org/abs/2609.32840}
}
```

Download [BibTeX](CITATION.bib). [CITATION.cff](CITATION.cff) supplies GitHub's
"Cite this repository" metadata with the paper as the preferred citation.

## License and contributions

Original code uses [Apache-2.0](LICENSE). Third-party code retains its original
licenses and notices; see [NOTICE](NOTICE).

English is the default maintenance language. Read [CONTRIBUTING.md](CONTRIBUTING.md)
before making changes. Keep private inputs and experiment outputs outside this checkout.
