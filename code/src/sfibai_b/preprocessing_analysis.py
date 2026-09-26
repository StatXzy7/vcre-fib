from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "image_uid",
    "split",
    "center_id",
    "crop_image_width",
    "crop_image_height",
}


def per_image_geometry(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest geometry columns are missing: {missing}")
    width = pd.to_numeric(manifest["crop_image_width"], errors="raise").astype(float)
    height = pd.to_numeric(manifest["crop_image_height"], errors="raise").astype(float)
    if (width <= 0).any() or (height <= 0).any():
        raise ValueError("Image width and height must be positive")
    side = np.maximum(width.to_numpy(), height.to_numpy())
    return pd.DataFrame(
        {
            "image_uid": manifest["image_uid"].astype(str),
            "split": manifest["split"].astype(str),
            "center_id": manifest["center_id"].astype(str),
            "width": width.astype(int),
            "height": height.astype(int),
            "aspect_ratio_width_over_height": width / height,
            "stretch_anisotropy": np.maximum(width / height, height / width),
            "letterbox_padding_fraction": 1.0
            - (width.to_numpy() * height.to_numpy()) / np.square(side),
        }
    )


def _summary(group: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"images": int(len(group))}
    for column in (
        "aspect_ratio_width_over_height",
        "stretch_anisotropy",
        "letterbox_padding_fraction",
    ):
        values = group[column].to_numpy(dtype=np.float64)
        output[column] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=0)),
            "min": float(values.min()),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    return output


def write_preprocessing_analysis(
    manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path, encoding="utf-8")
    geometry = per_image_geometry(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(
        output / "per_image_geometry.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )
    payload: dict[str, Any] = {
        "all": _summary(geometry),
        "by_split": {
            str(split): _summary(group)
            for split, group in geometry.groupby("split", sort=True)
        },
        "by_split_and_center": {
            f"{split}/{center}": _summary(group)
            for (split, center), group in geometry.groupby(
                ["split", "center_id"], sort=True
            )
        },
    }
    (output / "geometry_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for split, group in geometry.groupby("split", sort=True):
        axes[0].hist(
            group["aspect_ratio_width_over_height"],
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=str(split),
        )
        axes[1].hist(
            group["letterbox_padding_fraction"],
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=str(split),
        )
    axes[0].set(title="Data V4 aspect ratio", xlabel="width / height", ylabel="density")
    axes[1].set(
        title="Letterbox padding fraction",
        xlabel="padding area / square area",
        ylabel="density",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output / "geometry_distributions.png", dpi=180)
    plt.close(figure)
    return payload
