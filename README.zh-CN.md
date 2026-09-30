# VCRE-Fib 中文说明

[English README](README.md) 是默认入口。仓库文档、Issue、Pull Request、提交信息和新增代码注释默认使用英文。

本仓库对应论文 **VCRE-Fib: View-Conditioned Regional Evidence for Fine-Grained Ultrasound Grading of Schistosoma japonicum-Associated Liver Fibrosis**。

作者：Ziyang Xu、Shuli An、Hao Zhou、Haitian Zhong、Tingting Wu、Tao Wang、Kun Yang、Tieyong Zeng。论文已于 2026 年发布为 arXiv 预印本。

[![arXiv](https://img.shields.io/badge/arXiv-2609.32840-b31b1b.svg)](https://arxiv.org/abs/2609.32840)

[论文页面](https://arxiv.org/abs/2609.32840) · [PDF](https://arxiv.org/pdf/2609.32840) · [HTML](https://arxiv.org/html/2609.32840v1) · [BibTeX](CITATION.bib) · [GitHub 引用元数据](CITATION.cff)

## 方法与主图

VCRE-Fib 将全局图像判断与采集视图条件化的区域分级证据结合。模型先在局部特征上引入视图条件，再通过弱定位与学习到的支持权重进行空间聚合，最后混合各视图的预测概率。

训练使用分级标签、六类采集视图标签和弱病灶框；推理仅需图像，同时输出细粒度纤维化评分、采集视图预测和候选异常区域图。评分目标是专家标注的超声外观，弱定位图不代表完整病灶分割。研究结果与证据限制见[论文](https://arxiv.org/abs/2609.32840)。

[![VCRE-Fib 主图](paper/figures/previews/fig01_architecture.png)](paper/figures/assets/fig01_architecture.pdf)

**图 1：空间聚合前的视图条件化。** (a) 共享图像特征、全局上下文、视图概率、弱定位与支持权重；(b) 六个视图条件化区域路径及概率混合；(c) 仅图像推理的输出、参考标注与论文中的 test 摘要。参考框仅用于展示比较，不是推理输入。

[矢量 PDF](paper/figures/assets/fig01_architecture.pdf) · [可编辑 PowerPoint](paper/figures/editable/Figure1.pptx)

## 使用与公开范围

仓库提供模型、数据处理、训练、评估、消融和可视化代码、论文引用信息，以及主图的 PDF、PNG 和可编辑 PPTX。原创代码采用 [Apache-2.0](LICENSE)，第三方代码保留原许可证。

临床数据、标注、患者及图像清单、逐图预测、权重、原始日志、论文正文和结果表暂不公开。

```bash
git clone https://github.com/StatXzy7/vcre-fib.git
cd vcre-fib
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_release.py
```

使用 Python 3.10 或更高版本。正式训练与评估需要 Linux 和 CUDA。安装、私有输入格式和运行命令见 [使用指南](docs/REPRODUCING.md)。

区域模型与历史交付源码是不同实现，其源码和结果对应关系仍需核清，详见 [来源说明](docs/PROVENANCE.md)。软件测试通过不代表实验数值已复现。

后续公开修改直接维护本仓库；私有输入与实验输出存放在仓库外。发布步骤见 [贡献指南](CONTRIBUTING.md)。

## 引用

使用本工作时请引用 [arXiv:2609.32840](https://arxiv.org/abs/2609.32840)。可直接下载 [CITATION.bib](CITATION.bib)，或复制 [English README 中的 BibTeX](README.md#citation)。[CITATION.cff](CITATION.cff) 将论文设为 GitHub 的首选引用。
