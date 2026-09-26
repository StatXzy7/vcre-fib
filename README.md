# VCRE-Fib

**View-Conditioned Regional Evidence for Fine-Grained Ultrasound Grading of
Schistosoma japonicum-Associated Liver Fibrosis**

Code, experiment entry points, bilingual manuscript sources, and selected publication
figures. Original code is licensed under **Apache-2.0**.

**Reproduction status:** manuscript build assets and the available implementation
sources are included. The clinical dataset and trained weights are not released.
The corrected full-model result delivery and the regional-method source have an
unresolved source/result binding; this release does **not** certify end-to-end
reproduction of every reported number. Read [the source map](docs/PROVENANCE.md)
before selecting an implementation or interpreting a reproduced result.

| Entry | Contents |
| --- | --- |
| [Model](research/autosearch/src/synap_search/re_model.py) | View-conditioned regional architecture and probability mixture |
| [Loss](research/autosearch/src/synap_search/re_training.py) | Grading, view, and weak-localization objectives |
| [Shared code](code/src/sfibai_b/) | Preprocessing, model components, losses, canonical metrics |
| [Experiments](research/main_experiment/) | Train, validation selection, freeze, test, resource profiling |
| [Deletion variants](research/re_ablation/) | w/o View and w/o Weak Localization |
| [Released SFibAI](third_party/SFibAI/) | Pinned baseline with its original license |
| [Delivered checkpoint source](reproducibility/delivered_checkpoint/) | Separate source archive associated with the corrected result delivery |
| [Case/evidence export](reproducibility/delivery_tools/) | Source scripts; private inputs are deliberately absent |
| [Paper](paper/) | English/Chinese main text and supplementary material |
| [Figures](paper/figures/) | Composed PDFs and editable Figure 1 PPTX |
| [Reproduction guide](docs/REPRODUCING.md) | Installation, data contract, commands, and scope |

## Quick start

Use Python 3.10 or later. Install a matching PyTorch/torchvision pair appropriate
for your machine, then install the repository:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_release.py
```

Tests use synthetic inputs and require no clinical data or pretrained download.
Linux with a CUDA GPU is required for the formal training/evaluation runners;
multi-GPU runs use NCCL. The release test environment is recorded in
[provenance/VALIDATION.md](provenance/VALIDATION.md).

## Build the paper

Install TeX Live with `latexmk`, pdfLaTeX, XeLaTeX, and `ctex`:

```bash
python tools/build_paper.py --language all
```

This produces `paper/main.pdf`, `paper/main_zh.pdf`, `paper/supplementary.pdf`,
and `paper/supplementary_zh.pdf`. Composed figures and already reported aggregate
tables are included, so the paper build needs neither images from the dataset
nor checkpoints. Chinese text uses Microsoft YaHei when available and Fandol
otherwise; font substitution can change pagination.

## Data and publication boundary

No raw dataset, patient/image list, annotations, per-example predictions, attention
arrays, checkpoints, secrets, or raw training logs are included. The CSV files under
`paper/results/` contain only aggregate values already reported in the manuscript.
The figures are the selected, composed publication panels; no standalone case
images or case-to-patient mapping is distributed.

`.gitignore` is a convenience filter. `tools/check_release.py` also examines tracked
files, rejects excluded artifacts even if force-added, and verifies a SHA-256
inventory. See [CONTRIBUTING.md](CONTRIBUTING.md) for updates.

## 中文说明

这是 VCRE-Fib 的独立开源维护仓库，采用 Apache-2.0。临床数据、标注、患者及图像清单、
逐图预测、注意力数组和模型权重暂不公开。仓库保留论文汇总表、已组合主图和中英文构建源文件。

当前源码与最新完整模型结果的对应关系仍有待核清，详见 [来源说明](docs/PROVENANCE.md)。
论文能够编译、软件测试通过，不代表所有实验数值已经独立复现。原研究目录及历史结果未被此仓库覆盖。
