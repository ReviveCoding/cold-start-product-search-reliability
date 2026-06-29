from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from product_search.candidate_handoff import (
    aggregate_fingerprint,
    diff_from_baseline,
    source_file_hashes,
    validate_handoff,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_fingerprint_excludes_generated_and_self_referential_files(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "generated.json").write_text("{}", encoding="utf-8")
    (tmp_path / "release_candidate_handoff.json").write_text("{}", encoding="utf-8")
    hashes = source_file_hashes(tmp_path)
    assert hashes == {"src/module.py": _sha(tmp_path / "src" / "module.py")}
    assert len(aggregate_fingerprint(hashes)) == 64


def test_diff_checksum_is_deterministic_and_classifies_changes():
    baseline = {"same": "a", "changed": "b", "deleted": "c"}
    candidate = {"same": "a", "changed": "d", "added": "e"}
    first, first_checksum = diff_from_baseline(baseline, candidate)
    second, second_checksum = diff_from_baseline(baseline, candidate)
    assert first == second
    assert first_checksum == second_checksum
    assert {(row["path"], row["status"]) for row in first} == {
        ("added", "added"),
        ("changed", "modified"),
        ("deleted", "deleted"),
    }


def test_validate_handoff_checks_artifact_hashes_and_candidate_boundary(tmp_path: Path):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"wheel")
    dependency = tmp_path / "pyproject.toml"
    dependency.write_text("[project]\n", encoding="utf-8")
    handoff = tmp_path / "release_candidate_handoff.json"
    payload = {
        "schema_version": "1.0",
        "project_name": "example",
        "candidate_version": "1.0-candidate.1",
        "created_at_utc": "2026-06-18T00:00:00Z",
        "qualification_status": "RELEASE_CANDIDATE_NOT_RELEASE_QUALIFIED",
        "source": {},
        "dependency_manifests": [
            {"path": "pyproject.toml", "sha256": _sha(dependency)}
        ],
        "required_entrypoints": {},
        "metrics": {},
        "tests_and_command_evidence": [{"exit_code": 0}],
        "build_artifacts": [{"path": "artifact.whl", "sha256": _sha(artifact)}],
        "known_limitations": [],
        "unresolved_items": [],
        "evidence_level": {},
        "next_qualification_gates": [],
    }
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_handoff(tmp_path, handoff)
    assert result["status"] == "PASS"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_handoff(tmp_path, handoff)


def test_validate_handoff_can_preflight_source_only_before_dist_build(tmp_path: Path):
    dependency = tmp_path / "pyproject.toml"
    dependency.write_text("[project]\n", encoding="utf-8")
    handoff = tmp_path / "release_candidate_handoff.json"
    payload = {
        "schema_version": "1.0",
        "project_name": "example",
        "candidate_version": "1.0-candidate.1",
        "created_at_utc": "2026-06-18T00:00:00Z",
        "qualification_status": "RELEASE_CANDIDATE_NOT_RELEASE_QUALIFIED",
        "source": {},
        "dependency_manifests": [
            {"path": "pyproject.toml", "sha256": _sha(dependency)}
        ],
        "required_entrypoints": {},
        "metrics": {},
        "tests_and_command_evidence": [{"exit_code": 0}],
        "build_artifacts": [
            {"path": "dist/missing.whl", "sha256": "0" * 64},
            {"path": "dist/missing.tar.gz", "sha256": "1" * 64},
        ],
        "known_limitations": [],
        "unresolved_items": [],
        "evidence_level": {},
        "next_qualification_gates": [],
    }
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"dist[\\/]+missing\.whl"):
        validate_handoff(tmp_path, handoff)
    result = validate_handoff(
        tmp_path, handoff, allow_missing_build_artifacts=True
    )
    assert result["status"] == "PASS"
    assert result["verified_records"] == 1
    assert result["strict_build_artifacts_verified"] is False
    assert result["missing_build_artifacts"] == [
        "dist/missing.whl",
        "dist/missing.tar.gz",
    ]
