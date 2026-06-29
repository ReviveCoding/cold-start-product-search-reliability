from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from product_search.release_store import (
    force_unlock,
    publication_lock,
    publish_release,
    read_pointer,
    resolve_current_release,
    rollback_release,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path, name: str, fingerprint: str, payload: str) -> Path:
    bundle = root / name
    bundle.mkdir()
    (bundle / "payload.txt").write_text(payload, encoding="utf-8")
    manifest = {
        "release_status": "LAUNCH",
        "model_fingerprint": fingerprint,
        "artifact_hashes": {"payload.txt": _sha(bundle / "payload.txt")},
        "serving_artifact_hashes": {"payload.txt": _sha(bundle / "payload.txt")},
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


def test_publish_and_rollback_use_atomic_generation_pointer(tmp_path: Path):
    release_root = tmp_path / "store"
    first_source = _bundle(tmp_path, "first", "a" * 64, "first")
    second_source = _bundle(tmp_path, "second", "b" * 64, "second")

    first = publish_release(first_source, release_root, generation="g1")
    assert first.generation == "g1"
    assert first.previous_generation is None
    assert resolve_current_release(release_root).name == "g1"

    second = publish_release(second_source, release_root, generation="g2")
    assert second.previous_generation == "g1"
    assert read_pointer(release_root).generation == "g2"
    assert (resolve_current_release(release_root) / "payload.txt").read_text() == "second"

    rolled_back = rollback_release(release_root)
    assert rolled_back.generation == "g1"
    assert rolled_back.previous_generation == "g2"
    assert resolve_current_release(release_root).name == "g1"


def test_publication_lock_rejects_concurrent_writer(tmp_path: Path):
    with publication_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another release operation"):
            with publication_lock(tmp_path):
                pass
    assert not (tmp_path / ".publish.lock").exists()


def test_force_unlock_removes_abandoned_lock(tmp_path: Path):
    lock = tmp_path / ".publish.lock"
    lock.mkdir()
    (lock / "owner.json").write_text("{}", encoding="utf-8")
    assert force_unlock(tmp_path)
    assert not lock.exists()
    assert not force_unlock(tmp_path)


def test_publish_rejects_symlink_and_unsafe_generation(tmp_path: Path):
    source = _bundle(tmp_path, "source", "c" * 64, "safe")
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    try:
        os.symlink(target, source / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(RuntimeError, match="must not contain symlinks"):
        publish_release(source, tmp_path / "store", generation="g1")
    (source / "link.txt").unlink()
    with pytest.raises(ValueError, match="safe path component"):
        publish_release(source, tmp_path / "store", generation="../escape")


def test_rollback_revalidates_target_before_pointer_swap(tmp_path: Path):
    release_root = tmp_path / "store"
    first_source = _bundle(tmp_path, "first", "d" * 64, "first")
    second_source = _bundle(tmp_path, "second", "e" * 64, "second")
    publish_release(first_source, release_root, generation="g1")
    publish_release(second_source, release_root, generation="g2")
    (release_root / "generations" / "g1" / "payload.txt").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="hash"):
        rollback_release(release_root)
    assert read_pointer(release_root).generation == "g2"


def test_generation_metadata_tampering_blocks_resolution_and_rollback(tmp_path: Path):
    release_root = tmp_path / "store"
    first_source = _bundle(tmp_path, "first", "f" * 64, "first")
    second_source = _bundle(tmp_path, "second", "1" * 64, "second")
    publish_release(first_source, release_root, generation="g1")
    publish_release(second_source, release_root, generation="g2")

    metadata_path = release_root / "generations" / "g1" / "generation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_manifest_sha256"] = "0" * 64
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="manifest hash"):
        rollback_release(release_root, generation="g1")
    assert read_pointer(release_root).generation == "g2"

    current_metadata_path = release_root / "generations" / "g2" / "generation.json"
    current_metadata = json.loads(current_metadata_path.read_text(encoding="utf-8"))
    current_metadata["generation"] = "wrong"
    current_metadata_path.write_text(
        json.dumps(current_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="directory name"):
        resolve_current_release(release_root)


def test_default_generation_ids_do_not_collide_for_same_artifact(tmp_path: Path):
    release_root = tmp_path / "store"
    source = _bundle(tmp_path, "source", "2" * 64, "same")
    first = publish_release(source, release_root)
    second = publish_release(source, release_root)
    assert first.generation != second.generation


def test_publish_copies_only_verified_serving_closure(tmp_path: Path):
    source = _bundle(tmp_path, "source", "3" * 64, "payload")
    (source / "unverified.log").write_text("must not deploy", encoding="utf-8")
    release_root = tmp_path / "store"
    publish_release(source, release_root, generation="g1")
    deployed = resolve_current_release(release_root)
    assert (deployed / "payload.txt").is_file()
    assert (deployed / "manifest.json").is_file()
    assert (deployed / "generation.json").is_file()
    assert not (deployed / "unverified.log").exists()
    assert {path.name for path in deployed.iterdir()} == {
        "payload.txt",
        "manifest.json",
        "generation.json",
    }


def test_post_promotion_validation_failure_removes_orphan_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import product_search.release_store as release_store

    source = _bundle(tmp_path, "source", "4" * 64, "payload")
    release_root = tmp_path / "store"

    def reject_promoted_generation(path: Path):
        if path.name == "g1":
            raise RuntimeError("injected post-promotion metadata failure")
        return None

    monkeypatch.setattr(release_store, "read_generation_metadata", reject_promoted_generation)
    with pytest.raises(RuntimeError, match="post-promotion"):
        publish_release(source, release_root, generation="g1")
    assert not (release_root / "generations" / "g1").exists()
    assert not (release_root / "current.json").exists()
