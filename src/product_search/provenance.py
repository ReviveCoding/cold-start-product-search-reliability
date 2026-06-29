from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd


ARTIFACT_SCHEMA_VERSION = "6.0"
RUNTIME_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "fastapi",
    "uvicorn",
    "pydantic",
    "PyYAML",
    "joblib",
)
_DISTRIBUTION_CANDIDATES: dict[str, tuple[str, ...]] = {
    "xgboost": ("xgboost-cpu", "xgboost"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(start: Path | None = None) -> str:
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        git_dir = directory / ".git"
        head = git_dir / "HEAD"
        if not head.exists():
            continue
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            ref_path = git_dir / value[5:]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed = git_dir / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#"):
                        commit, ref = line.split(" ", 1)
                        if ref == value[5:]:
                            return commit
            return "unresolved-ref"
        return value
    return "unavailable"


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def atomic_write_json(path: Path, payload: Any) -> None:
    temp = _atomic_path(path)
    try:
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    temp = _atomic_path(path)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = _atomic_path(path)
    try:
        frame.to_csv(temp, index=False, lineterminator="\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_joblib_dump(path: Path, value: Any) -> None:
    temp = _atomic_path(path)
    try:
        joblib.dump(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _safe_artifact_path(root: Path, relative: str, *, require_exists: bool = True) -> Path:
    """Resolve a manifest path without allowing traversal or symlink escapes."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("Artifact path must be a non-empty relative string")
    rel = Path(relative)
    if rel.is_absolute() or any(part in {"..", ""} for part in rel.parts):
        raise ValueError(f"Unsafe artifact path: {relative!r}")
    root_resolved = root.resolve(strict=True)
    current = root_resolved
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Artifact path must not contain symlinks: {relative!r}")
    resolved = current.resolve(strict=False)
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise ValueError(f"Artifact path escapes root: {relative!r}")
    if require_exists:
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        if not resolved.is_file():
            raise ValueError(f"Artifact path is not a regular file: {relative!r}")
    return resolved


def artifact_hashes(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        if relative in hashes:
            raise ValueError(f"Duplicate artifact path in manifest input: {relative}")
        path = _safe_artifact_path(root, relative)
        hashes[relative] = sha256_file(path)
    return hashes


def verify_artifact_hashes(root: Path, hashes: dict[str, str]) -> None:
    failures: list[str] = []
    for relative, expected in hashes.items():
        try:
            path = _safe_artifact_path(root, relative)
        except (ValueError, FileNotFoundError) as exc:
            failures.append(f"unsafe-or-missing:{relative}:{type(exc).__name__}")
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            failures.append(f"invalid-hash:{relative}")
        elif sha256_file(path) != expected:
            failures.append(f"hash-mismatch:{relative}")
    if failures:
        raise RuntimeError(f"Artifact integrity validation failed: {failures}")


def runtime_environment(packages: tuple[str, ...] = RUNTIME_PACKAGES) -> dict[str, Any]:
    package_records: dict[str, dict[str, str]] = {}
    for package in packages:
        candidates = _DISTRIBUTION_CANDIDATES.get(package, (package,))
        record = {"distribution": candidates[0], "version": "not-installed"}
        for distribution in candidates:
            try:
                record = {"distribution": distribution, "version": metadata.version(distribution)}
                break
            except metadata.PackageNotFoundError:
                continue
        package_records[package] = record
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": package_records,
    }


def package_versions(packages: tuple[str, ...] = RUNTIME_PACKAGES) -> dict[str, str]:
    environment = runtime_environment(packages)
    return {
        name: str(record["version"])
        for name, record in environment["packages"].items()
    }


def write_artifact_requirements(path: Path, environment: dict[str, Any]) -> None:
    lines = [
        "# Exact runtime distributions used to create this artifact bundle.",
        f"# Python {environment['python']} ({environment['python_implementation']})",
    ]
    records = environment.get("packages", {})
    for name in sorted(records):
        record = records[name]
        version = str(record.get("version", "not-installed"))
        distribution = str(record.get("distribution", name))
        if version != "not-installed":
            lines.append(f"{distribution}=={version}")
    atomic_write_text(path, "\n".join(lines) + "\n")



def package_source_sha256(package_root: Path | None = None) -> str:
    """Hash importable package source independently of checkout location and timestamps."""
    root = (package_root or Path(__file__).resolve().parent).resolve(strict=True)
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise RuntimeError(f"No Python source files found under package root: {root}")
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Package source must not contain symlinks: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def serving_config_sha256(config: dict[str, Any]) -> str:
    """Hash only configuration values that can change online scoring behavior."""
    ranking = config.get("ranking", {})
    payload = {
        "config_schema_version": config.get("config_schema_version"),
        "retrieval": config.get("retrieval"),
        "ranking": {
            "semantic_weight": ranking.get("semantic_weight"),
            "behavior_weight": ranking.get("behavior_weight"),
        },
        "qrsbt": config.get("qrsbt"),
    }
    if any(value is None for value in payload.values()):
        raise ValueError("Serving configuration is incomplete")
    if any(value is None for value in payload["ranking"].values()):
        raise ValueError("Serving ranking configuration is incomplete")
    return canonical_json_sha256(payload)


def canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def model_fingerprint(
    *,
    config_sha256: str,
    artifact_hashes_value: dict[str, str],
    serving_code_sha256: str,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
) -> str:
    if len(config_sha256) != 64 or len(serving_code_sha256) != 64:
        raise ValueError("Model fingerprint inputs must be SHA-256 digests")
    payload = {
        "artifact_schema_version": schema_version,
        "serving_config_sha256": config_sha256,
        "serving_code_sha256": serving_code_sha256,
        "artifact_hashes": dict(sorted(artifact_hashes_value.items())),
    }
    return canonical_json_sha256(payload)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    if hasattr(os, "uname"):
        uname = os.uname()
        platform_value = f"{uname.sysname}-{uname.machine}-{uname.release}"
    else:
        platform_value = f"{platform.system()}-{platform.machine()}"
    base = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform_value,
        "git_commit": git_commit(path.parent),
    }
    base.update(payload)
    atomic_write_json(path, base)
