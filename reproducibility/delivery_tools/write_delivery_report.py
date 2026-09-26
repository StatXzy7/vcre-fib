"""Write test-first documentation and a relocatable SHA-256 file manifest."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from package_common import PACK, read_json, write_json, sha256, markdown_table


def main() -> None:
    validation = read_json(PACK / "verification/VALIDATION.json")
    if validation.get("status") != "PASS":
        raise ValueError("A passing evidence validation is required before delivery")
    report = [
        "# Test set 主结果 — SynAP-Fib E",
        "",
        "本包仅包含本地旧版 SynAP-Fib E（Legacy E），不包含 SFibAI A 基线或其他模型的权重/结果。不是当前 VCRE-Fib / Full-RE 模型。",
        "",
        "比较轮次 AE_COR_v2；seed 2026；训练 120 epochs；冻结 best checkpoint 为 epoch 103。test 为 4,107 图像 / 240 患者 / 4 中心。",
        "主终点 R_final = 0.4×image COR + 0.4×patient-max COR + 0.2×center-balanced patient-max COR，越低越好。此次仅交付单模型，跨模型排名与相对基线差值按作者最新要求不生成。",
        "",
        "## 数据来源与本次计算范围",
        "",
        "两套 val/test 的 metrics.json、predictions.csv.gz、attention NPZ 均逐字节来自 epoch 103 的已冻结 canonical evaluation，保留原始 SHA-256。本次无需重复推理或独立重算第二套 metrics。新导出的内容为患者分别聚合预测、混淆矩阵、中心权重表、视图 support、原始日志曲线、病例图，以及本机 GPU 资源实测。",
        "",
        "[完整性检查](verification/VALIDATION.json)覆盖来源哈希、指标字段、逐图/患者/中心对应、attention 对应、病例原图像素及冻结数据管线的 box 几何。",
        "",
    ]
    compact = [
        "n",
        "mae",
        "rmse",
        "accuracy_within_0_3",
        "accuracy_within_0_5",
        "grade_accuracy",
        "grade_macro_f1",
        "tmae",
        "severe_error_rate",
        "cor",
    ]
    for split in ["test", "val"]:
        m = read_json(PACK / f"results/{split}/metrics.json")
        title = (
            "Test 主结果"
            if split == "test"
            else "Validation：checkpoint 选择与训练诊断"
        )
        report += [
            f"## {title}",
            "",
            markdown_table(pd.read_csv(PACK / f"tables/{split}/main.csv")),
            "",
            "### 三层级分级指标",
            "",
            "accuracy、recall、frequency 与 error rate 均为 0–1 比例。",
            "",
            markdown_table(
                pd.DataFrame(
                    [
                        {"level": k, **{v: m[k][v] for v in compact}}
                        for k in ["image", "patient_max", "patient_median"]
                    ]
                )
            ),
            "",
            "### 各级 recall/support 与 COR 五项",
            "",
        ]
        detail_keys = [
            f"grade_f{g}_{suffix}" for g in range(4) for suffix in ["recall", "support"]
        ]
        detail_keys += [
            "cor_continuous",
            "cor_error_gt_0_3",
            "cor_error_gt_0_5",
            "cor_stage_distance",
            "cor_severe_stage",
        ]
        detail = pd.DataFrame(
            [
                {
                    "metric": key,
                    **{
                        level: m[level][key]
                        for level in ["image", "patient_max", "patient_median"]
                    },
                }
                for key in detail_keys
            ]
        )
        report += [
            markdown_table(detail),
            "",
            "### Image-level 概率指标",
            "",
            markdown_table(pd.read_csv(PACK / f"tables/{split}/probabilities.csv")),
            "",
            "### 视图与弱定位",
            "",
            markdown_table(pd.read_csv(PACK / f"tables/{split}/spatial.csv")),
            "",
            markdown_table(pd.read_csv(PACK / f"results/{split}/view_classes.csv")),
            "",
            "### 中心结果",
            "",
            markdown_table(pd.read_csv(PACK / f"results/{split}/centers.csv")),
            "",
            f"[4×4 分级混淆矩阵](results/{split}/grade_confusion.png) · [6×6 视图混淆矩阵](results/{split}/view_confusion.png)",
            "",
        ]
    resource = read_json(PACK / "resources/resources.json")
    report += [
        "## 资源实测",
        "",
        markdown_table(pd.read_csv(PACK / "resources/resources.csv")),
        "",
        f"GPU：{resource['gpu']}；输入 512×512；eager；FP16 autocast，保持旧版 lesion head FP32 例外。每 batch 50 次预热、200 次 CUDA-event 同步计时；延迟为整个 batch 的模型前向时间，不含读图和预处理。显存为 PyTorch peak allocated，含模型、输入及输出。",
        "GMACs 只统计 Conv2d/Linear 乘加；GFLOPs=2×GMACs，不包含激活、归一化、逐元素操作和池化。参数量为参数元素数，未把 BatchNorm buffer 当参数。",
        "[原始计时](resources/raw_latency_samples.csv) · [环境与口径](resources/resources.json)",
        "",
        "## 训练轨迹和模型来源",
        "",
        "- [最佳权重](model/checkpoints/best.pt)：epoch 103、model-only。",
        "- [末轮权重](model/checkpoints/last.pt)：epoch 120、完整恢复状态，仅归档。",
        "- [120 轮原始 history](training/history.csv)、[训练与验证曲线](training/training_curves.png)、[矢量版曲线](training/training_curves.pdf)。",
        "- training/tensorboard 保存原始 event；training/val_epochs 保存 120 套逐轮完整指标和紧凑预测。",
        "- training/last_checkpoint_diagnostics 保留 last 的历史 val/test bundle，未用于主表。",
        "- [选择记录](training/selection.json)：仅 epoch 21–120 有资格；val R_final → val image COR → 较早 epoch。best val R_final=0.15407460400768486。",
        "- provenance 保存 task、初始化审计、RUN_COMPLETE、文件来源、数据指纹及原始 FP32 lesion 修订。source/round_snapshot 是未改动的原训练代码和协议；其中保留原 A–E 接口不表示本包包含其他模型结果。",
        "- history 记载 153 次 AMP skipped steps；保留原始训练健康信息，不把归档描述为无跳步训练。",
        "",
        "## 病例图与医生复核",
        "",
        "[打开病例浏览页](cases/index.html)：15 个 test + 3 个 val，均为原始超声 ROI｜医生原始弱框｜冻结模型预测。每例保存原图、原始 attention、固定阈值 mask、标注、三联 PNG/PDF、分值和真实/预测视图名称。",
        "6 个 test 定位对应示例、6 个 test 框外激活候选、3 个 test 低重叠诊断示例，以及 3 个 val 补充诊断。选择规则与 tie-break 全部保存在 cases/selection_policy.json。病例为描述性挑选，不是总体性能的随机样本。",
        "所有图使用相同视野/缩放规则、固定 0–1 attention 色标和 0.5 轮廓阈值，不作逐图 min-max 归一化。原始 ROI 保持比例；attention 从 32×32 双线性映射到 ROI，阈值 mask 在 32×32 计算后最近邻映射。展示使用保存的 float16 attention；原 canonical 弱定位指标在保存前用 float32 attention 计算，阈值邻近的量化差异不用于重写原始指标。",
        "**医生解释、扩展区域的影像依据、完整可见病灶范围参考均按作者确认留为待医生复核。** [填写表](cases/doctor_review_to_fill.csv)。已有弱框只支持局部弱标注一致性，框外激活不是已证实的更广泛病灶。",
        "",
        "## 指标定义",
        "",
        "TMAE=mean(max(abs(error)-0.5,0))；severe error=mean(abs(error)>1.0)。四级阈值0.5/1.5/2.5，边界进入高一级。Patient-max/median 对真值和预测值分别聚合。COR 第五项是跨至少两级错误，与 severe error 的分数误差>1.0 不同。",
        "概率 AUROC/AUPRC 仅 image-level；AUPRC 使用 average precision，缺乏有效正负类时为 null/N/A。ECE 为 10 个等宽区间。常规四级混淆矩阵来自连续评分分级；概率 ECE 使用概率 argmax，二者定义不同。",
        "弱框有效图像计数是 valid_box_count，不是矩形总数。只对有效非空框且真实 label_bin>0 的图像评价，排除恰为0.0的评分，不排除整个F0。",
        "",
        "## 复现与边界",
        "",
        "本包是本地研究证据，保留逐图预测和私有标注追溯；没有上传。全部 Data V4 图像仍在原数据目录，未复制整套训练数据；所选病例原图已随包保存。",
        "本次未训练、未新增 seed、未运行 bootstrap/p 值、未运行路径关闭或贡献积分、未调用自动统计 finalize.py。所有训练/指标来源文件保持不变。",
        "tools/ 保存本次打包、病例、资源和校验脚本；脚本默认定位原工作区。读取结果不依赖原目录；重新校验原始来源或重建病例需要原数据路径。",
        "[SHA-256 文件清单](MANIFEST.json) · [源文件拷贝对应](provenance/copied_sources.json)",
        "",
    ]
    (PACK / "README.md").write_text("\n".join(report), encoding="utf-8")
    checklist = {
        "model": "SynAP-Fib E",
        "seed": 2026,
        "best_epoch": 103,
        "status": "COMPUTATIONAL_EVIDENCE_COMPLETE_CLINICIAN_REVIEW_PENDING",
        "completed": [
            "frozen best/last weights",
            "val/test canonical metrics and predictions",
            "image/patient-max/patient-median metrics",
            "patient prediction exports",
            "center tables",
            "grade/view confusion and view names/support",
            "image probability metrics",
            "all val/test raw attention maps",
            "18 real case triptychs",
            "GPU resource measurements",
            "120-epoch history/TensorBoard/validation predictions",
            "frozen code/config/source hashes",
        ],
        "pending_by_author_instruction": [
            "doctor interpretation",
            "extended region imaging basis",
            "full visible lesion reference if claimed",
        ],
        "excluded_by_latest_author_scope": [
            "SFibAI A baseline",
            "cross-model ranking/deltas",
        ],
        "not_run": [
            "training",
            "additional seed",
            "formal re-inference",
            "bootstrap",
            "p values",
            "path closure",
            "contribution integration",
        ],
    }
    write_json(PACK / "DELIVERY_STATUS.json", checklist)
    inventory: list[dict[str, Any]] = [
        {
            "path": str(p.relative_to(PACK)).replace("\\", "/"),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in sorted(PACK.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts
    ]
    write_json(
        PACK / "MANIFEST.json",
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "file_count": len(inventory),
            "total_bytes": sum(e["bytes"] for e in inventory),
            "excludes": ["MANIFEST.json itself", "__pycache__"],
            "files": inventory,
        },
    )
    require_text = (PACK / "README.md").read_text(encoding="utf-8")
    if "待医生复核" not in require_text or "\ufffd" in require_text:
        raise ValueError("Chinese encoding verification failed")
    print(
        f"Wrote report and manifest: {len(inventory)} files, {sum(e['bytes'] for e in inventory)} bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
