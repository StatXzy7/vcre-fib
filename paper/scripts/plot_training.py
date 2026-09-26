"""Redraw the published training comparison from aggregate values only."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
METHODS = ("SFIBAI", "RE_WO_VIEW", "RE_WO_WEAKLOC", "SYNAP")
LABELS = ("SFibAI", "w/o View", "w/o Weak Loc.", "VCRE-Fib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER.parent / ".tmp/training_curves.pdf")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (PAPER / "results/training_90_four_models/training_curves_90.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    fields = ("train_total", "val_R_final", "val_image_COR")
    if len(records) != 360 or {row["method"] for row in records} != set(METHODS):
        raise ValueError("Expected four methods and 360 aggregate records")
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for method, label in zip(METHODS, LABELS):
        rows = [row for row in records if row["method"] == method]
        if [int(row["epoch"]) for row in rows] != list(range(1, 91)):
            raise ValueError("Missing, duplicate or out-of-order epochs")
        selected = [row for row in rows if row["selected_checkpoint"].lower() == "true"]
        if len(selected) != 1:
            raise ValueError("Expected one published selected checkpoint")
        for axis, field in zip(axes, fields):
            values = [float(row[field]) for row in rows]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("Invalid aggregate trajectory")
            line, = axis.plot(range(1, 91), values, label=label, linewidth=1.1)
            axis.scatter([int(selected[0]["epoch"])], [float(selected[0][field])], color=line.get_color(), s=18)
    for axis, title in zip(axes, ("Recorded total loss", "Validation R_final", "Validation image COR")):
        axis.set(title=title, xlabel="Epoch", xlim=(1, 90))
        axis.grid(alpha=.2)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)
    print(f"Rendered 360 published aggregate records: {args.output}")


if __name__ == "__main__":
    main()
