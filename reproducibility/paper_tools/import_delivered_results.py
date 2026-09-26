"""Import aggregate exports and compose original case panels; never run inference.

Run with the delivery directory as the only argument. Patient-level exports,
original images, and checkpoint files are not copied into the paper tree.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pymupdf as pdf

PAPER = Path(os.environ.get("VCRE_PAPER_ROOT", str(Path(__file__).resolve().parents[2] / ".tmp/paper-generation"))).resolve()
ROUND = "zgc_main_seed2026_20260925"
CASES = []  # Supplied through the private selection JSON.


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def table(name, label, en, zh, headers, rows, spec=None):
    spec = spec or ("l" + "r" * (len(headers) - 1))
    text = ("% Generated from verified delivered aggregate exports.\n"
            "\\begin{table}[!htbp]\n\\centering\\small\n"
            f"\\caption{{\\lang{{{en}}}{{{zh}}}}}\\label{{{label}}}\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\fitresulttable{%\n"
            f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n\\toprule\n")
    text += " & ".join(headers) + " \\\\\n\\midrule\n"
    text += "\n".join(" & ".join(map(str, r)) + " \\\\" for r in rows)
    text += "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    write(PAPER / "tables" / name, text)


def f(value, digits=6):
    return f"{float(value):.{digits}f}"


def pct(value):
    return f"{100 * float(value):.2f}"


def case_figure(delivery, ids, output, compact=False, full_rows=False):
    # Embed the three exact PDF panel regions, retaining the original raster,
    # clinician boxes and model contours. Only page composition is changed.
    width, rowheight = 396, 87 if not compact else 65
    size = 83 if not compact else 61
    left = 133 if not compact else 197
    font = 7.5 if not compact else 6.2
    if full_rows:
        rowheight, size, left = 132, 100, 16
    doc = pdf.open()
    page = doc.new_page(width=width, height=22 + rowheight * len(ids) + 20)
    for j, title in enumerate(["Original ultrasound", "Clinician annotation", "Model localization"]):
        x = left + j * (132 if full_rows else size + 3)
        result = page.insert_textbox(pdf.Rect(x, 2, x + size, 20), title,
                                     fontname="hebo", fontsize=7 if not compact else 6.1, align=1)
        if result < 0:
            raise ValueError(f"Column heading does not fit: {title}")
    if not full_rows:
        page.insert_text((0, 10), "Scores and acquisition views", fontname="hebo", fontsize=8)
    metadata = []
    for i, case_id in enumerate(ids):
        meta = read_json(delivery / "cases" / case_id / "metadata.json")
        metadata.append({k: v for k, v in meta.items() if k != "image_uid"})
        top = 22 + i * rowheight
        with pdf.open(delivery / "cases" / case_id / "triptych.pdf") as source:
            panels = [x for x in source[0].get_image_info() if x["width"] > 100]
            assert len(panels) == 3
            for j, panel in enumerate(panels):
                x = left + j * (132 if full_rows else size + 3)
                image_top = top + 28 if full_rows else top
                page.show_pdf_page(pdf.Rect(x, image_top, x + size, image_top + size), source, 0,
                                   clip=pdf.Rect(panel["bbox"]))
        letter = chr(97 + i)
        group = "Correspondence" if meta["category"] == "localization_correspondence" else "Extent expansion"
        header = f"({letter}) {meta['split'].upper()} | {group}"
        score = f"Score: reference {meta['reference_score']:.1f} / predicted {meta['predicted_score']:.3f}"
        if full_rows:
            page.insert_text((0, top + 7), header + "  |  " + score, fontsize=8, fontname="hebo")
        else:
            page.insert_text((0, top + 7), header, fontsize=font, fontname="hebo")
            page.insert_text((0, top + 17), score, fontsize=font - .3)
        text = f"True view: {meta['true_view_name']}\nPredicted view: {meta['predicted_view_name']}"
        box = pdf.Rect(0, top + 9, width, top + 28) if full_rows else pdf.Rect(0, top + 21, left - 5, top + size)
        result = page.insert_textbox(box, text, fontsize=8 if full_rows else font,
                                     lineheight=1.02, fontname="helv")
        if result < 0:
            raise ValueError(f"Case label overflow: {case_id}, {result}")
        if i < len(ids) - 1:
            page.draw_line((0, top + rowheight - 2), (width, top + rowheight - 2),
                           color=(.80, .80, .80), width=.25)
    # Shared color scale taken directly from the delivered first case.
    y = page.rect.height - 13
    page.insert_text((0, y + 7), "Localization: fixed 0-1 scale; alpha 0.55; cyan contour = 0.5", fontsize=7)
    with pdf.open(delivery / "cases" / ids[0] / "triptych.pdf") as source:
        colorbar = [x for x in source[0].get_image_info() if x["width"] <= 100]
        assert len(colorbar) == 1
        page.show_pdf_page(pdf.Rect(305, y, 372, y + 5), source, 0,
                           clip=pdf.Rect(colorbar[0]["bbox"]), rotate=270)
    page.insert_text((296, y + 6), "0", fontsize=7)
    page.insert_text((376, y + 6), "1", fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output, garbage=4, deflate=True)
    page.get_pixmap(matrix=pdf.Matrix(2, 2)).save(output.with_suffix(".png"))
    doc.close()
    return metadata


def main():
    global CASES
    if len(sys.argv) != 3:
        raise SystemExit("Usage: import_delivered_results.py PRIVATE_DELIVERY PRIVATE_SELECTION_JSON")
    selection = read_json(Path(sys.argv[2]))
    CASES = selection["main"] + selection["supplement"]
    if len(selection["main"]) != 4 or len(selection["supplement"]) != 2 or len(set(CASES)) != 6:
        raise ValueError("Supply four main and two supplementary case IDs")
    if any(not isinstance(value, str) or "/" in value or "\\" in value or value in {".", ".."} for value in CASES):
        raise ValueError("Case IDs must be simple relative directory names")
    for folder in ("tables", "figures/assets", "audits"):
        (PAPER / folder).mkdir(parents=True, exist_ok=True)
    delivery = Path(sys.argv[1]).resolve()
    manifest = read_json(delivery / "MANIFEST.json")
    for relative, record in manifest.items():
        path = (delivery / relative).resolve()
        if not path.is_relative_to(delivery) or digest(path) != record["sha256"]:
            raise ValueError(f"Delivery hash mismatch: {relative}")
    audit = read_json(delivery / "AUDIT_PASS.json")
    assert audit["status"] == "PASS"
    results = PAPER / "results"
    results.mkdir(exist_ok=True)
    for split in ("test", "val"):
        for source in (delivery / "tables" / split).glob("*.csv"):
            destination = results / split / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
    aliases = {"primary": "primary", "image_metrics": "image_metrics",
               "patient_metrics": "patient_metrics", "auxiliary_metrics": "spatial_metrics"}
    for target, source in aliases.items():
        shutil.copyfile(delivery / "tables/test" / f"{source}.csv", results / f"{target}.csv")
    shutil.copyfile(delivery / "resources/resources.csv", results / "resources.csv")
    metrics = {method: read_json(delivery / method / "test/metrics.json") for method in ("SFIBAI", "SYNAP")}
    primary = sorted(read_csv(results / "primary.csv"), key=lambda row: int(row["rank"]))
    assert all(r["round"] == ROUND and r["frozen"] == "True" for r in primary)
    assert all((int(r["images"]), int(r["patients"]), int(r["centers"])) == (4107, 240, 4) for r in primary)
    names = {"SFIBAI": "SFibAI", "SYNAP": "SynAP-Fib"}
    table("tab01_main.tex", "tab:main",
          "Test set primary results, seed 2026: 4,107 images, 240 patients, four centers. All checkpoints were frozen after validation selection. COR and the prespecified composite endpoint are lower-is-better.",
          "Test set 主结果，seed 2026：4,107 图、240 患者、4 中心。全部 checkpoint 经 validation 选择后冻结。COR 与预定复合主终点均越低越好。",
          ["Rank", "Method", r"$R_{\mathrm{final}}$", "Image COR", "Max COR", "Median COR", "Center COR"],
          [[r["rank"], r["model"]] + [f(r[k]) for k in ("R_final", "image_COR", "patient_max_COR", "patient_median_COR", "center_COR")] for r in primary], "rlrrrrr")
    heads = ["Method", "$N$", "MAE", r"Acc.$\pm.3$", r"Acc.$\pm.5$", "4-grade acc.", "TMAE", "Severe", "COR"]
    keys = ["n", "mae", "accuracy_within_0_3", "accuracy_within_0_5", "grade_accuracy", "tmae", "severe_error_rate", "cor"]
    def metric_values(m):
        return [str(m[k]) if k == "n" else (pct(m[k]) if k in keys[2:5] + ["severe_error_rate"] else f(m[k])) for k in keys]
    table("tab05_image.tex", "tab:image", "Image-level test metrics. Accuracy and severe-error entries are percentages; severe error is absolute score error above 1.0. TMAE is mean excess error above 0.5. COR is clinical objective risk.",
          "图像级 test 指标。准确率与严重错误率以百分数表示；严重错误为分数绝对误差大于 1.0，TMAE 为超过 0.5 的平均超额误差，COR 为临床目标风险。",
          heads, [[names[k]] + metric_values(m["image"]) for k, m in metrics.items()])
    table("tab06_patient.tex", "tab:patient", "Patient-level test metrics, 240 patients per row. Reference and predicted scores are aggregated separately. Accuracy and severe-error entries are percentages. Center-balanced COR is reported in Table~\\ref{tab:main}.",
          "患者级 test 指标，每行 240 名患者。参考与预测分别聚合。准确率与严重错误率以百分数表示；中心平衡 COR 见表~\\ref{tab:main}。",
          ["Method", "Aggregation"] + heads[2:],
          [[names[k], agg.replace("patient_", "").title()] + metric_values(m[agg])[1:] for agg in ("patient_max", "patient_median") for k, m in metrics.items()], "llrrrrrrr")
    m = metrics["SYNAP"]
    table("tab07_auxiliary.tex", "tab:auxiliary", "Test spatial outputs. SFibAI has neither head (N/A). Valid-box count is the number of images with a valid weak target, not the number of drawn rectangles. Agreement does not establish lesion-contour accuracy.",
          "Test 空间输出。SFibAI 没有相应预测头（N/A）。有效弱框计数按具有有效弱目标的图像计数，并非矩形框个数。一致性不等同于病灶轮廓准确性。",
          ["Method", "$N$ views", r"View acc. (\%)", "Macro-F1", "$N$ valid", "Dice@.5", "IoU@.5"],
          [["SFibAI"] + [r"\na"] * 6, ["SynAP-Fib", m["position"]["n"], pct(m["position"]["accuracy"]), f(m["position"]["macro_f1"]), m["lesion"]["valid_box_count"], f(m["lesion"]["dice_at_0_5"]), f(m["lesion"]["iou_at_0_5"])]])
    table("tab04_comparison.tex", "tab:comparison", "Single-seed test differences from SFibAI. Differences use unrounded canonical values; relative difference divides by the baseline. The baseline defines zero.",
          "相对 SFibAI 的单 seed test 差异。差值由未取整 canonical 数值计算；相对差除以基线数值，基线差异为零。",
          ["Method", "Seed", "Rank", r"$R_{\mathrm{final}}$", r"$\Delta$ absolute", r"$\Delta$ relative (\%)"],
          [[r["model"], r["seed"], r["rank"], f(r["R_final"]), f(r["delta"]), f(r["relative_delta_percent"], 3)] for r in primary])
    resource = read_csv(results / "resources.csv")
    table("tab09_resources.tex", "tab:resources", "Matched-device inference cost on an A100-SXM4-80GB, FP32, TF32 disabled, 512 by 512 inputs. Latency is milliseconds per batch; memory is peak allocated MiB. GMACs count Conv2d/Linear only (two FLOPs per MAC).",
          "A100-SXM4-80GB 上的同条件推理成本，FP32、关闭 TF32、输入 512 乘 512。延迟为每批毫秒，显存为峰值 allocated MiB。GMACs 仅统计 Conv2d/Linear，每 MAC 计两个 FLOPs。",
          ["Method", "Batch", "Parameters", "GMACs", "Median ms", "P95 ms", "MiB"],
          [[names[r["method"]], r["batch_size"], f"{int(r['parameters']):,}"] + [f(r[k], 3) for k in ("gmacs_per_image", "latency_median_ms", "latency_p95_ms", "peak_allocated_mib")] for r in resource])
    val = {r["model"]: r for r in read_csv(results / "val/primary.csv")}
    checkpoints = []
    bindings = {"round": ROUND, "delivery_manifest_sha256": digest(delivery / "MANIFEST.json"), "verified_files": len(manifest), "methods": {}}
    for method in metrics:
        freeze = read_json(delivery / method / "CHECKPOINT_FREEZE.json")
        identity = read_json(delivery / method / "IDENTITY.json")
        v = val[names[method]]
        checkpoints.append([names[method], audit["methods"][method]["training_epochs"], freeze["epoch"], f(v["R_final"]), f(v["image_COR"]), freeze["checkpoint_sha256"][:12], "Frozen"])
        bindings["methods"][method] = {"freeze": freeze, "actual_training_epochs": audit["methods"][method]["training_epochs"], "identity_sha256": digest(delivery / method / "IDENTITY.json"), "identity": identity,
            "metrics_sha256": {split: digest(delivery / method / split / "canonical_metrics.json") for split in ("val", "test")}}
    table("tab10_freeze.tex", "tab:freeze", "Checkpoint selection on validation (20,880 images, 1,227 patients, 33 centers), seed 2026. SHA-256 prefixes identify the frozen weights; complete hashes are retained in the source bindings. Budget is the actual number of training epochs.",
          "Validation 选点（20,880 图、1,227 患者、33 中心），seed 2026。SHA-256 前缀标识冻结权重，完整哈希保存在来源绑定中。预算为实际训练 epoch 数。",
          ["Method", "Budget", "Best epoch", r"Val $R_{\mathrm{final}}$", "Val Image COR", "SHA-256 prefix", "Status"], checkpoints, "lrrrrll")
    with (results / "checkpoints.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "actual_epochs", "selected_epoch", "val_r_final", "val_image_cor", "checkpoint_sha256_prefix", "status"])
        writer.writerows(checkpoints)
    views = read_csv(results / "test/view_classes.csv")
    table("tab13_views.tex", "tab:views", "Six acquisition views and SynAP-Fib test recall. The selected main-paper cases cover four true views. Names are the delivered, author-confirmed labels.",
          "六类采集视图及 SynAP-Fib test 召回率。正文病例覆盖四种真实视图，名称沿用已交付且经作者确认的标签。",
          ["View", "English name", "$N$", r"Recall (\%)"],
          [[r["label"], r["name"], r["support"], pct(r["recall"])] for r in views], "lp{.64\\linewidth}rr")
    extra = []
    for key in ("rmse", "grade_macro_f1", "grade_macro_auroc", "grade_macro_auprc", "grade_brier", "grade_ece"):
        extra.append([{"rmse": "RMSE", "grade_macro_f1": "Grade macro-F1", "grade_macro_auroc": "Macro AUROC", "grade_macro_auprc": "Macro AUPRC", "grade_brier": "Brier score", "grade_ece": "ECE"}[key]] + [f(m["image"][key]) for m in metrics.values()])
    table("tab14_additional.tex", "tab:additional", "Additional image-level test metrics from the canonical export; no recalibration was fitted.", "Canonical 导出的补充图像级 test 指标，未重新拟合校准。", ["Metric", "SFibAI", "SynAP-Fib"], extra)
    centers = metrics["SYNAP"]["center_balanced_patient_max"]["center_patient_counts"]
    table("tab15_centers.tex", "tab:centers", "Patient-max COR by test center. Every test center occurs in training; this table describes center heterogeneity, not unseen-center generalization.", "Test 各中心 patient-max COR。全部 test 中心出现在训练中，该表描述中心间差异，不代表未见中心泛化。", ["Center", "$N$ patients", "SFibAI", "SynAP-Fib"],
          [[c.replace("_", r"\_"), n] + [f(m["center_balanced_patient_max"]["per_center_cor"][c]) for m in metrics.values()] for c, n in centers.items()])
    case_metadata = case_figure(delivery, CASES[:4], PAPER / "figures/assets/fig06_cases.pdf", full_rows=True)
    case_metadata += case_figure(delivery, CASES[4:], PAPER / "figures/assets/fig06_validation_cases.pdf", full_rows=True)
    write(results / "selected_cases.json", json.dumps(case_metadata, ensure_ascii=False, indent=2))
    source_panels = PAPER / "figures/sources/fig06_cases_before_header_upgrade.pdf"
    source_panels.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAPER / "figures/assets/fig06_cases.pdf", source_panels)
    write(PAPER / "figures/sources/fig06_cases_binding.json", json.dumps({
        "source_pdf_sha256": digest(source_panels),
        "metadata_sha256": digest(results / "selected_cases.json"),
        "case_ids": CASES[:4],
    }, indent=2))
    for base in ("training_curves", "synap_loss_components"):
        shutil.copyfile(delivery / "figures" / f"{base}.pdf", PAPER / "figures/assets" / f"{base}.pdf")
    for method in metrics:
        shutil.copyfile(delivery / method / "training_curves.csv", results / f"training_{method.lower()}.csv")
    write(results / "source_bindings.json", json.dumps(bindings, ensure_ascii=False, indent=2))
    write(PAPER / "audits/RESULT_IMPORT.json", json.dumps({"status": "PASS", "round": ROUND,
        "verified_delivery_files": len(manifest), "canonical_evaluator_calls": 0, "new_training": False,
        "main_case_ids": CASES[:4], "supplement_case_ids": CASES[4:],
        "excluded_at_author_request": selection.get("excluded", []),
        "layout": {"width_pt": 396, "main_image_side_pt": 100, "main_label_font_pt": 8}}, indent=2))
    print(f"Verified {len(manifest)} source files; imported canonical aggregates; composed 4/2-case layouts.")


if __name__ == "__main__":
    main()
