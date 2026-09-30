"""Publication boundary tests use synthetic bytes only."""
import importlib.util
import json
from pathlib import Path
import subprocess

SPEC = importlib.util.spec_from_file_location("release_guard", Path(__file__).resolve().parents[1] / "tools/check_release.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def test_raw_predictions_and_weights_rejected_even_if_force_added():
    assert guard.audit("weights/model.pt", b"synthetic")
    assert guard.audit("paper/results/aligned_ablation_20260926/test/extra.csv", b"patient_uid,pred_score\nsynthetic,1\n")


def test_unreviewed_images_rejected():
    assert guard.audit("paper/figures/assets/extra.png", b"synthetic")


def test_manuscript_results_and_unreviewed_bibliographies_are_not_released():
    assert guard.audit("paper/results/aligned_ablation_20260926/test/primary.csv", b"method,R_final\nExample,0.2\n")
    assert guard.audit("paper/main.tex", b"Synthetic manuscript")
    assert guard.audit("paper/references.bib", b"Synthetic bibliography")
    assert guard.audit("docs/CITATION.bib", b"Synthetic bibliography")
    assert guard.audit("docs/CITATION.cff", b"cff-version: 1.2.0")
    assert guard.audit("paper/CITATION.cff", b"cff-version: 1.2.0")
    assert guard.audit("citation.cff", b"cff-version: 1.2.0")
    assert guard.audit("CITATION.BIB", b"Synthetic bibliography")


def test_published_citation_files_allowed_with_private_content_checks():
    assert guard.audit("CITATION.cff", b"cff-version: 1.2.0\n") == []
    assert guard.audit("CITATION.bib", b"@misc{synthetic, year={2026}}\n") == []
    assert guard.audit("third_party/SFibAI/CITATION.cff", b"cff-version: 1.2.0\n") == []
    for path in guard.CITATION_FILES:
        assert guard.audit(path, b"-----BEGIN " b"PRIVATE KEY-----")
        assert guard.audit(path, b"test-" b"012345abcdef")
        assert guard.audit(path, b"\xff")


def test_private_data_and_uppercase_suffixes_fail_closed():
    assert guard.audit("data/labels.json", b'{"patient_uid":"synthetic","label":1}')
    assert guard.audit("data/scan.nii", b"synthetic")
    assert guard.audit("docs/patients.CSV", b"patient_uid,pred_score\nsynthetic,1\n")
    assert guard.audit("docs/labels.JSON", b'{"patient_uid":"synthetic"}')


def test_unstaged_sanitization_cannot_hide_staged_bytes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "record.txt"
    path.write_bytes(b"synthetic private staged content")
    subprocess.run(["git", "-c", "core.excludesFile=", "-C", str(tmp_path), "add", "record.txt"], check=True)
    path.write_bytes(b"clean working tree")
    assert guard.index_mismatches({"record.txt": path.read_bytes()}, tmp_path)
    subprocess.run(["git", "-c", "core.excludesFile=", "-C", str(tmp_path), "add", "record.txt"], check=True)
    assert guard.index_mismatches({"record.txt": path.read_bytes()}, tmp_path) == []


def test_ignored_imported_source_is_not_a_complete_release(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = "third_party/example/data/dataset.py"
    (tmp_path / source).parent.mkdir(parents=True)
    (tmp_path / source).write_text("# Synthetic dataset loader\n", encoding="utf-8")
    manifest = tmp_path / guard.IMPORT_MANIFEST
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"files": [{"path": source}]}), encoding="utf-8")
    (tmp_path / ".gitignore").write_text("data/\n", encoding="utf-8")
    payloads = {name: (tmp_path / name).read_bytes() for name in guard.candidates(tmp_path)}
    assert guard.missing_imports(payloads)
    (tmp_path / ".gitignore").write_text("/data/\n", encoding="utf-8")
    payloads = {name: (tmp_path / name).read_bytes() for name in guard.candidates(tmp_path)}
    assert guard.missing_imports(payloads) == []
