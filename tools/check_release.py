"""Audit tracked and unignored files before publication; clinical inputs are forbidden."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "provenance/RELEASE_MANIFEST.json"
TEXT = {".py", ".md", ".toml", ".yml", ".yaml", ".tex", ".sty", ".bst", ".bib", ".csv", ".json", ".cff", ".txt"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".parquet", ".h5", ".hdf5", ".dcm", ".nii", ".jsonl", ".pem", ".key", ".zip", ".gz"}
FORBIDDEN_DIRECTORIES = {"development", "outputs", "runs", "checkpoints", "research-private", ".ssh", ".aws", ".git", "__pycache__"}
MEDIA = {
    "paper/figures/assets/fig01_architecture.pdf",
    "paper/figures/assets/fig06_cases.pdf",
    "paper/figures/assets/fig06_validation_cases.pdf",
    "paper/figures/assets/training_curves.pdf",
    "paper/figures/assets/synap_loss_components.pdf",
    "paper/figures/editable/Figure1.pptx",
}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub credential": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "API credential": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "cloud credential": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "private workstation": re.compile(r"(?:[A-Z]:[/\\](?:PhD|Users)[/\\]|/online" r"1/|/home/" r"dataset-local/)"),
    "private case identifier": re.compile(r"\b(?:test|val)-[0-9a-f]{12}\b"),
}


def candidates(root: Path = ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=",
         "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, check=True, capture_output=True,
    )
    return sorted(set(x.decode("utf-8") for x in result.stdout.split(b"\0") if x))


def audit(path: str, payload: bytes) -> list[str]:
    relative = Path(path)
    suffix = relative.suffix.lower()
    failures: list[str] = []
    parts = {part.lower() for part in relative.parts}
    if suffix in FORBIDDEN_SUFFIXES or parts & FORBIDDEN_DIRECTORIES or relative.parts[0].lower() == "data":
        failures.append("forbidden private artifact")
    if relative.name.startswith(".env") or relative.name in {"images.csv", "annotations.jsonl"}:
        failures.append("private input or credential file")
    if suffix in {".pdf", ".pptx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svg"} and path not in MEDIA:
        failures.append("media not on the reviewed publication allowlist")
    if len(payload) > 5 * 1024 * 1024:
        failures.append("unexpected file larger than 5 MiB")
    is_text = suffix in TEXT or relative.name in {"LICENSE", "NOTICE", ".gitignore", ".gitattributes"}
    if not is_text and path not in MEDIA:
        failures.append("file type outside source/publication allowlist")
    if is_text:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return failures + ["non-UTF-8 text"]
        if text.startswith("\ufeff"):
            failures.append("UTF-8 BOM")
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(name)
        if suffix == ".json" and path not in {"provenance/IMPORT_MANIFEST.json", "provenance/SFIBAI_SOURCE_SNAPSHOT.json", MANIFEST}:
            failures.append("JSON outside source-inventory allowlist")
        if suffix == ".csv":
            allowed = path.startswith("paper/results/aligned_ablation_20260926/") or path == "paper/results/training_90_four_models/training_curves_90.csv"
            if not allowed:
                failures.append("CSV outside aggregate-paper allowlist")
            columns = next(csv.reader(io.StringIO(text)), [])
            if set(columns) & {"patient_uid", "image_uid", "image_path", "patient_id", "image_id"}:
                failures.append("patient/image-level table")
    if suffix == ".pptx":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.namelist():
                if member.endswith((".xml", ".rels")):
                    content = archive.read(member).decode("utf-8")
                    for name, pattern in PATTERNS.items():
                        if pattern.search(content):
                            failures.append(f"{name} in presentation XML")
                    if 'TargetMode="External"' in content:
                        failures.append("external presentation relationship")
    return failures


def index_mismatches(payloads: dict[str, bytes], root: Path = ROOT) -> list[str]:
    """Require the exact audited bytes to be the bytes Git will commit."""
    result = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=root, check=True, capture_output=True)
    indexed = {}
    errors = []
    for row in result.stdout.split(b"\0"):
        if not row:
            continue
        header, raw_path = row.split(b"\t", 1)
        mode, digest, stage = header.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if stage != "0" or mode not in {"100644", "100755"}:
            errors.append(f"{path}: unsupported index entry")
        indexed[path] = digest
    for path in sorted(set(payloads) | set(indexed)):
        payload, expected = payloads.get(path), indexed.get(path)
        if payload is None or expected is None:
            errors.append(f"{path}: staged file inventory differs")
            continue
        algorithm = "sha1" if len(expected) == 40 else "sha256"
        actual = hashlib.new(algorithm, b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
        if actual != expected:
            errors.append(f"{path}: staged bytes differ from audited working tree")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    inventory: dict[str, dict[str, int | str]] = {}
    payloads: dict[str, bytes] = {}
    errors: list[str] = []
    for relative in candidates():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            errors.append(f"{relative}: missing file or unsafe link")
            continue
        payload = path.read_bytes()
        payloads[relative] = payload
        errors += [f"{relative}: {error}" for error in audit(relative, payload)]
        if relative != MANIFEST:
            inventory[relative] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    if not args.write_manifest:
        errors += index_mismatches(payloads)
    if errors:
        raise SystemExit("Release audit FAILED:\n" + "\n".join(errors))
    target = ROOT / MANIFEST
    if args.write_manifest:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema_version": 1, "files": inventory}, indent=2) + "\n", encoding="utf-8")
        print(f"DRAFT: {len(inventory)} working-tree files audited. Stage the reviewed files and manifest, then run this tool without --write-manifest.")
    else:
        expected = json.loads(target.read_text(encoding="utf-8"))["files"]
        changed = sorted(path for path in set(expected) | set(inventory) if expected.get(path) != inventory.get(path))
        if changed:
            raise SystemExit("Release inventory differs; review before updating:\n" + "\n".join(changed))
        print(f"PASS: {len(inventory)} files audited; exact staged bytes and publication inventory verified.")


if __name__ == "__main__":
    main()
