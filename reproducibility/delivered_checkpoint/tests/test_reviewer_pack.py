from __future__ import annotations

from sfibai_b.reviewer_pack import reviewer_pack_entries


def test_reviewer_pack_index_references_evidence_without_copying_it(tmp_path) -> None:
    evidence = tmp_path / "final_ranking" / "test_ranking.json"
    evidence.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "seed_2026" / "A" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    entries = reviewer_pack_entries(tmp_path)

    assert "final_ranking/test_ranking.json" in entries
    assert "seed_2026/A/checkpoints/best.pt" in entries
