from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from product_search.provenance import verify_artifact_hashes


POINTER_SCHEMA_VERSION = "1.0"
_GENERATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ReleasePointer:
    generation: str
    previous_generation: str | None
    model_fingerprint: str
    published_at: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "schema_version": POINTER_SCHEMA_VERSION,
            "generation": self.generation,
            "previous_generation": self.previous_generation,
            "model_fingerprint": self.model_fingerprint,
            "published_at": self.published_at,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read release metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Release metadata must be a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_tree(root: Path) -> None:
    """Flush a staged immutable generation before its atomic directory rename."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("r+b" if os.name == "nt" else "rb") as handle:
                os.fsync(handle.fileno())
    if os.name != "nt":
        for path in sorted(
            (entry for entry in root.rglob("*") if entry.is_dir()),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            _fsync_directory(path)
        _fsync_directory(root)


def _safe_generation(generation: str) -> str:
    if not _GENERATION_PATTERN.fullmatch(generation):
        raise ValueError("generation must be one safe path component")
    return generation


def _generation_dir(root: Path, generation: str) -> Path:
    safe = _safe_generation(generation)
    generations = (root / "generations").resolve()
    candidate = (generations / safe).resolve()
    if candidate.parent != generations:
        raise ValueError("generation escapes the release store")
    return candidate


def _default_generation(manifest: dict) -> str:
    fingerprint = str(manifest.get("model_fingerprint", ""))
    if len(fingerprint) != 64:
        raise RuntimeError("Release manifest does not contain a valid model fingerprint")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{fingerprint[:12]}"


def _validate_source_tree(source: Path) -> None:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("artifact source must be a directory")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Release bundle must not contain symlinks: {path}")
        if not path.is_file() and not path.is_dir():
            raise RuntimeError(f"Release bundle contains an unsupported entry: {path}")


def _hash_contract(manifest: dict, field: str) -> dict[str, str]:
    hashes = manifest.get(field)
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f"Release manifest does not contain {field}")
    return {str(path): str(digest) for path, digest in hashes.items()}


def _copy_deployment_closure(source: Path, destination: Path, manifest: dict) -> None:
    """Copy only the manifest and hash-verified files needed by the online scorer."""
    hashes = _hash_contract(manifest, "serving_artifact_hashes")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    source_root = source.resolve(strict=True)
    destination_root = destination.resolve(strict=True)
    for relative in sorted(hashes):
        rel = Path(relative)
        if rel.is_absolute() or any(part in {"", ".."} for part in rel.parts):
            raise RuntimeError(f"Unsafe serving artifact path: {relative!r}")
        source_path = (source_root / rel).resolve(strict=True)
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(f"Serving artifact is not a regular file: {relative!r}")
        if not source_path.is_relative_to(source_root):
            raise RuntimeError(f"Serving artifact escapes source bundle: {relative!r}")
        destination_path = (destination_root / rel).resolve(strict=False)
        if not destination_path.is_relative_to(destination_root):
            raise RuntimeError(f"Serving artifact escapes deployment bundle: {relative!r}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def validate_release_bundle(
    artifact_dir: Path,
    *,
    hash_field: str = "artifact_hashes",
    runtime_validator: Callable[[Path], None] | None = None,
) -> dict:
    artifact_dir = artifact_dir.resolve(strict=True)
    _validate_source_tree(artifact_dir)
    manifest = _read_json(artifact_dir / "manifest.json")
    if str(manifest.get("release_status", "")).upper() != "LAUNCH":
        raise RuntimeError("Only a LAUNCH artifact can be published")
    hashes = _hash_contract(manifest, hash_field)
    verify_artifact_hashes(artifact_dir, hashes)
    if runtime_validator is not None:
        runtime_validator(artifact_dir)
    return manifest


@contextlib.contextmanager
def publication_lock(root: Path) -> Iterator[None]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_dir = root / ".publish.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        owner = lock_dir / "owner.json"
        detail = owner.read_text(encoding="utf-8") if owner.exists() else "unknown owner"
        raise RuntimeError(f"Another release operation is active: {detail.strip()}") from exc
    try:
        _atomic_json(
            lock_dir / "owner.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": _utc_now(),
            },
        )
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def force_unlock(root: Path) -> bool:
    lock_dir = root.resolve() / ".publish.lock"
    if not lock_dir.exists():
        return False
    shutil.rmtree(lock_dir)
    return True


def read_pointer(root: Path) -> ReleasePointer:
    root = root.resolve()
    payload = _read_json(root / "current.json")
    if str(payload.get("schema_version")) != POINTER_SCHEMA_VERSION:
        raise RuntimeError("Unsupported release pointer schema")
    generation = _safe_generation(str(payload.get("generation", "")))
    previous = payload.get("previous_generation")
    if previous is not None:
        previous = _safe_generation(str(previous))
    fingerprint = str(payload.get("model_fingerprint", ""))
    if len(fingerprint) != 64:
        raise RuntimeError("Release pointer contains an invalid model fingerprint")
    published_at = str(payload.get("published_at", ""))
    return ReleasePointer(generation, previous, fingerprint, published_at)


def read_generation_metadata(artifact_dir: Path) -> dict | None:
    artifact_dir = artifact_dir.resolve(strict=True)
    metadata_path = artifact_dir / "generation.json"
    if not metadata_path.exists():
        return None
    metadata = _read_json(metadata_path)
    generation = _safe_generation(str(metadata.get("generation", "")))
    if generation != artifact_dir.name:
        raise RuntimeError("Generation metadata differs from its directory name")
    manifest_path = artifact_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    fingerprint = str(metadata.get("model_fingerprint", ""))
    if fingerprint != str(manifest.get("model_fingerprint", "")):
        raise RuntimeError("Generation metadata fingerprint differs from its manifest")
    source_hash = str(metadata.get("source_manifest_sha256", ""))
    if len(source_hash) != 64 or source_hash != _sha256(manifest_path):
        raise RuntimeError("Generation metadata manifest hash is missing or inconsistent")
    if not str(metadata.get("published_at", "")):
        raise RuntimeError("Generation metadata does not contain a publication time")
    return metadata


def resolve_current_release(root: Path) -> Path:
    root = root.resolve(strict=True)
    pointer = read_pointer(root)
    generation = _generation_dir(root, pointer.generation)
    if not generation.is_dir():
        raise RuntimeError(f"Published release generation is missing: {pointer.generation}")
    manifest = _read_json(generation / "manifest.json")
    if manifest.get("model_fingerprint") != pointer.model_fingerprint:
        raise RuntimeError("Release pointer fingerprint differs from generation manifest")
    metadata = read_generation_metadata(generation)
    if metadata is None:
        raise RuntimeError("Published generation metadata is missing")
    if metadata.get("generation") != pointer.generation:
        raise RuntimeError("Release pointer differs from generation metadata")
    return generation


def publish_release(
    source: Path,
    root: Path,
    *,
    generation: str | None = None,
    runtime_validator: Callable[[Path], None] | None = None,
) -> ReleasePointer:
    source = source.resolve(strict=True)
    root = root.resolve()
    manifest = validate_release_bundle(source, runtime_validator=runtime_validator)
    generation = _safe_generation(generation or _default_generation(manifest))
    fingerprint = str(manifest["model_fingerprint"])
    with publication_lock(root):
        generations = root / "generations"
        staging_root = root / ".staging"
        generations.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        destination = _generation_dir(root, generation)
        if destination.exists():
            raise RuntimeError(f"Release generation already exists: {generation}")
        staging = staging_root / f"{generation}-{uuid.uuid4().hex}"
        promoted = False
        pointer_written = False
        try:
            _copy_deployment_closure(source, staging, manifest)
            copied_manifest = validate_release_bundle(
                staging,
                hash_field="serving_artifact_hashes",
                runtime_validator=runtime_validator,
            )
            if copied_manifest.get("model_fingerprint") != fingerprint:
                raise RuntimeError("Copied release fingerprint changed during publication")
            _atomic_json(
                staging / "generation.json",
                {
                    "generation": generation,
                    "model_fingerprint": fingerprint,
                    "published_at": _utc_now(),
                    "source_manifest_sha256": _sha256(source / "manifest.json"),
                },
            )
            _sync_tree(staging)
            os.replace(staging, destination)
            promoted = True
            _fsync_directory(generations)
            metadata = read_generation_metadata(destination)
            if metadata is None or metadata.get("generation") != generation:
                raise RuntimeError("Published generation metadata could not be verified")
            previous: str | None = None
            if (root / "current.json").exists():
                previous = read_pointer(root).generation
            pointer = ReleasePointer(
                generation=generation,
                previous_generation=previous,
                model_fingerprint=fingerprint,
                published_at=_utc_now(),
            )
            _atomic_json(root / "current.json", pointer.as_dict())
            pointer_written = True
            return pointer
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if promoted and not pointer_written and destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
                _fsync_directory(generations)


def rollback_release(
    root: Path,
    *,
    generation: str | None = None,
    runtime_validator: Callable[[Path], None] | None = None,
) -> ReleasePointer:
    root = root.resolve(strict=True)
    with publication_lock(root):
        current = read_pointer(root)
        target_name = generation or current.previous_generation
        if target_name is None:
            raise RuntimeError("No previous release generation is recorded")
        target_name = _safe_generation(target_name)
        if target_name == current.generation:
            raise RuntimeError("Rollback target is already current")
        target = _generation_dir(root, target_name)
        manifest = validate_release_bundle(
            target,
            hash_field="serving_artifact_hashes",
            runtime_validator=runtime_validator,
        )
        metadata = read_generation_metadata(target)
        if metadata is None:
            raise RuntimeError("Rollback target generation metadata is missing")
        if metadata.get("model_fingerprint") != manifest.get("model_fingerprint"):
            raise RuntimeError("Rollback target metadata differs from its manifest")
        pointer = ReleasePointer(
            generation=target_name,
            previous_generation=current.generation,
            model_fingerprint=str(manifest["model_fingerprint"]),
            published_at=_utc_now(),
        )
        _atomic_json(root / "current.json", pointer.as_dict())
        return pointer
