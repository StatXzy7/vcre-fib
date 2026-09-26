"""Publication boundary tests use synthetic bytes only."""
import importlib.util
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


def test_aggregate_paper_table_allowed():
    assert guard.audit("paper/results/aligned_ablation_20260926/test/primary.csv", b"method,R_final\nExample,0.2\n") == []


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
