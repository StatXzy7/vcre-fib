"""Render Figure S2 from verified training exports, capped at epoch 90."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(os.environ.get("VCRE_PAPER_ROOT", str(Path(__file__).resolve().parents[2] / ".tmp/paper-generation"))).resolve()
CAP = 90
METHODS = ("SFIBAI", "RE_WO_VIEW", "RE_WO_WEAKLOC", "SYNAP")
LABELS = ("SFibAI", "w/o View", "w/o Weak Loc.", "VCRE-Fib")
COLORS = ("#606060", "#238C9B", "#B38337", "#5554A0")
STYLES = ("--", "-.", ":", "-")
FIELDS = ("train_total", "val_R_final", "val_image_COR")


def sha256(path: Path) -> str:
    """Hash a small source or figure file without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_curves(source: Path) -> tuple[dict[str, list[dict[str, str]]], dict]:
    """Require four complete prefixes and agreement with frozen selection."""
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
    bindings = json.loads(
        (ROOT / "results/aligned_ablation_20260926/source_bindings.json").read_text(
            encoding="utf-8"
        )
    )
    curves, provenance = {}, {}
    for method in METHODS:
        relative = f"{method}/training_curves.csv"
        path = source / relative
        expected = manifest[relative]
        if (
            sha256(path) != expected["sha256"]
            or path.stat().st_size != expected["bytes"]
        ):
            raise ValueError(f"Source manifest mismatch: {method}")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(r["epoch"]) for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Noncontinuous or duplicated epochs: {method}")
        selected = [r for r in rows if int(r["epoch"]) <= CAP]
        if [int(r["epoch"]) for r in selected] != list(range(1, CAP + 1)):
            raise ValueError(f"Incomplete 90-epoch prefix: {method}")
        for row in selected:
            if any(not math.isfinite(float(row[k])) for k in FIELDS):
                raise ValueError(f"Missing/nonfinite plotted value: {method}")
            if float(row["train_total"]) <= 0:
                raise ValueError("Log-scale loss must be positive")
        best = min(
            (float(r["val_R_final"]), float(r["val_image_COR"]), int(r["epoch"]))
            for r in selected
            if int(r["epoch"]) >= 21
        )
        frozen = bindings["records"][method]["freeze"]
        if best[2] != frozen["epoch"] or any(
            abs(best[i] - frozen["selection_key"][i]) > 1e-12 for i in (0, 1)
        ):
            raise ValueError(f"Frozen checkpoint selection mismatch: {method}")
        curves[method] = selected
        provenance[method] = {
            "source_csv_sha256": expected["sha256"],
            "source_rows": len(rows),
            "plotted_rows": len(selected),
            "excluded_epochs": [int(r["epoch"]) for r in rows if int(r["epoch"]) > CAP],
            "selected_epoch": best[2],
            "selection_key": best,
            "checkpoint_sha256": frozen["checkpoint_sha256"],
        }
    return curves, provenance


def main() -> None:
    """Validate, export 360 unsmoothed rows, and draw three shared panels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    curves, provenance = load_curves(args.source)
    assets = ROOT / "figures/assets"
    assets.mkdir(parents=True, exist_ok=True)
    audits = ROOT / "audits"
    output = ROOT / "results/training_90_four_models"
    output.mkdir(exist_ok=True)
    csv_path = output / "training_curves_90.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "model",
                "epoch",
                "comparison_cap",
                *FIELDS,
                "selected_checkpoint",
            ],
        )
        writer.writeheader()
        for method, label in zip(METHODS, LABELS):
            for row in curves[method]:
                writer.writerow(
                    {
                        "method": method,
                        "model": label,
                        "epoch": row["epoch"],
                        "comparison_cap": CAP,
                        **{k: row[k] for k in FIELDS},
                        "selected_checkpoint": int(row["epoch"])
                        == provenance[method]["selected_epoch"],
                    }
                )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.55))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.75, wspace=0.55)
    titles = ("Training", "Selection", "Image diagnostic")
    ylabels = ("Total loss (log scale)", "Validation R_final", "Validation image COR")
    for index, (ax, field, title, ylabel) in enumerate(
        zip(axes, FIELDS, titles, ylabels)
    ):
        for method, label, color, style in zip(METHODS, LABELS, COLORS, STYLES):
            rows = curves[method]
            epochs = [int(r["epoch"]) for r in rows]
            ys = [float(r[field]) for r in rows]
            ax.plot(
                epochs,
                ys,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.35 if method == "SYNAP" else 1.05,
            )
            best_epoch = provenance[method]["selected_epoch"]
            if index > 0:
                ax.plot(
                    best_epoch,
                    ys[best_epoch - 1],
                    "o",
                    color=color,
                    markersize=4,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=5,
                )
        if index == 0:
            ax.set_yscale("log")
        ax.set(xlim=(1, CAP), xlabel="Epoch", ylabel=ylabel)
        ax.set_xticks([1, 30, 60, 90])
        ax.set_title(title, pad=10)
        ax.text(
            -0.28,
            1.08,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
        ax.grid(axis="y", color="#DADADA", linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=0.9,
    )
    fig.text(
        0.52,
        0.025,
        "All methods: 90-epoch cap  |  Dots: validation-selected checkpoints",
        ha="center",
        fontsize=7,
    )
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(audits / "FIGURE_S2_ALIGNMENT.json"),
        overlay_svg=str(audits / "FIGURE_S2_ALIGNMENT.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    for extension in ("pdf", "svg"):
        fig.savefig(assets / f"training_curves.{extension}", facecolor="white")
    fig.savefig(assets / "training_curves.png", dpi=600, facecolor="white")
    plt.close(fig)
    report = {
        "status": "PASS",
        "comparison_cap": CAP,
        "selection_epochs": [21, CAP],
        "methods": provenance,
        "plotted_rows": 360,
        "smoothing": "none",
        "loss_axis": "logarithmic; recorded total loss includes each model's retained auxiliary objectives",
        "exclusion_rule": "Only epochs 1-90, as requested; raw sources preserved.",
        "seeds": [2026],
        "uncertainty": "Single run per method; no interval or statistical test.",
        "source_manifest_sha256": sha256(args.source / "MANIFEST.json"),
        "source_data_sha256": sha256(csv_path),
        "figure_pdf_sha256": sha256(assets / "training_curves.pdf"),
        "inference_calls": 0,
        "training_calls": 0,
    }
    (audits / "FIGURE_S2_CURVES.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": 360,
                "cap": CAP,
                "selected_epochs": {
                    m: provenance[m]["selected_epoch"] for m in METHODS
                },
            }
        )
    )


if __name__ == "__main__":
    main()
