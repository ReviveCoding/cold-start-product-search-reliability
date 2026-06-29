from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from product_search.release_packaging import build_deterministic_zip


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_zip_is_byte_reproducible_and_clean(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "module.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "bad.pyc").write_bytes(b"bad")
    (root / "artifacts" / "smoke").mkdir(parents=True)
    (root / "artifacts" / "smoke" / "generated.json").write_text("{}", encoding="utf-8")

    first = build_deterministic_zip(root, tmp_path / "first.zip", root_name="release")
    second = build_deterministic_zip(root, tmp_path / "second.zip", root_name="release")
    assert _sha(first) == _sha(second)
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert "release/README.md" in names
    assert all("__pycache__" not in name for name in names)
    assert all("artifacts/smoke" not in name for name in names)


def test_enhanced_release_zip_includes_generated_artifacts(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "artifacts" / "smoke").mkdir(parents=True)
    (root / "artifacts" / "smoke" / "release.json").write_text("{}", encoding="utf-8")
    output = build_deterministic_zip(
        root,
        tmp_path / "enhanced.zip",
        include_generated_artifacts=True,
        root_name="release",
    )
    with zipfile.ZipFile(output) as archive:
        assert "release/artifacts/smoke/release.json" in archive.namelist()


def test_source_release_zip_includes_candidate_handoff(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "release_candidate_handoff.json").write_text("{}\n", encoding="utf-8")
    output = build_deterministic_zip(root, tmp_path / "source.zip", root_name="release")
    with zipfile.ZipFile(output) as archive:
        assert "release/release_candidate_handoff.json" in archive.namelist()
