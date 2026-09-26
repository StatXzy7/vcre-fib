"""Derive locked train/validation manifests from an authorized private Data V4 copy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "research/autosearch/src")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    from sfibai_b.data import validate_data_v4
    from synap_search.data import inner_partition
    from synap_search.io import digest, sha256, write_json

    workspace = args.workspace.resolve()
    root = workspace / "data/schisto_2024_clean_v4"
    target = workspace / "development"
    if target.exists():
        raise FileExistsError("Refuse to overwrite development inputs")
    manifest, annotations = root / "manifests/images.csv", root / "manifests/annotations.jsonl"
    validate_data_v4(root, manifest, annotations)
    frame = pd.read_csv(manifest, dtype=str, encoding="utf-8")
    columns = ["image_uid", "patient_uid", "center_id", "image_path", "position_norm", "image_label_max", "split", "image_hash_v2"]
    if frame[columns].isna().any().any() or (frame.groupby("image_hash_v2").split.nunique() > 1).any():
        raise ValueError("Missing metadata or cross-split duplicate images")
    grouped: dict[str, list[dict]] = {}
    with annotations.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            grouped.setdefault(item["image_uid"], []).append(item)
    dev = frame.loc[frame.split != "test", columns].copy()
    train = inner_partition(dev.loc[dev.split == "train"], 31001, .2)
    dev = dev.merge(train[["image_uid", "inner_split"]], on="image_uid", how="left")
    dev["inner_split"] = dev.inner_split.fillna("native_val")
    dev["parent_image_uid"] = dev.image_uid
    dev["original_split"] = dev.split
    dev["recipe_version"] = "v1"
    dev["source_annotation_sha256"] = dev.image_uid.map(lambda uid: digest(grouped.get(uid, [])))
    target.mkdir(parents=True)
    # The historical Windows manifest used CRLF; preserve its bytes on Linux too.
    dev.to_csv(target / "images.csv", index=False, encoding="utf-8", lineterminator="\r\n")
    fields = ["image_uid", "coordinate_frame", "lesion_label_float"] + [f"bbox_crop_norm_{x}" for x in ("x_min", "y_min", "x_max", "y_max")]
    for split in ("train", "val"):
        with (target / f"native_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for uid in dev.loc[dev.split == split].image_uid:
                for item in grouped.get(uid, []):
                    stream.write(json.dumps({key: item[key] for key in fields}) + "\n")
    expected = {
        "images.csv": "d3a3d04b9adb947b6c3905a2d18d256f6a5ca28586970dc1910881786f48d7cb",
        "native_train.jsonl": "531347f450bb8473274cd5bcd7875586a05ec01d69ddcd2d9aa0a7cf6d874c68",
        "native_val.jsonl": "4e61622f2064582a29a6ce7032e32d82328365ba599688a41c7497326093bd22",
    }
    actual = {name: sha256(target / name) for name in expected}
    write_json(target / "PREPARATION.json", {"expected": expected, "actual": actual, "matches": actual == expected})
    if actual != expected:
        raise ValueError("Derived inputs differ from the locked run; inspect the retained receipt before training")
    print("PASS: native train/val manifests match the locked hashes; no training or inference.")


if __name__ == "__main__":
    main()
