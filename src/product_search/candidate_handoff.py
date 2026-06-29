from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


HANDOFF_SCHEMA_VERSION = "1.0"
SOURCE_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".hypothesis",
    "htmlcov",
    "build",
    "dist",
    "artifacts",
    "releases",
    ".coverage",
}
SOURCE_EXCLUDED_FILES = {"release_candidate_handoff.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included_source_file(relative: Path) -> bool:
    if relative.name in SOURCE_EXCLUDED_FILES:
        return False
    return not any(
        part in SOURCE_EXCLUDED_PARTS
        or part.endswith(".egg-info")
        or part.endswith(".pyc")
        for part in relative.parts
    )


def source_file_hashes(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if _included_source_file(relative):
            hashes[relative.as_posix()] = sha256_file(path)
    return hashes


def aggregate_fingerprint(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, file_digest in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def diff_from_baseline(
    baseline: dict[str, str], candidate: dict[str, str]
) -> tuple[list[dict[str, str]], str]:
    changes: list[dict[str, str]] = []
    for name in sorted(set(baseline) | set(candidate)):
        old = baseline.get(name)
        new = candidate.get(name)
        if old == new:
            continue
        status = "added" if old is None else "deleted" if new is None else "modified"
        changes.append(
            {
                "path": name,
                "status": status,
                "before_sha256": old or "",
                "after_sha256": new or "",
            }
        )
    encoded = json.dumps(changes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return changes, hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _relative_or_absolute(root: Path, path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def artifact_record(root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Artifact must be a regular file: {resolved}")
    return {
        "kind": kind,
        "path": _relative_or_absolute(root, resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def collect_command_evidence(evidence_dir: Path) -> list[dict[str, Any]]:
    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory does not exist: {evidence_dir}")
    records: list[dict[str, Any]] = []
    for exit_path in sorted(evidence_dir.glob("*.exit_code.txt")):
        name = exit_path.name.removesuffix(".exit_code.txt")
        command_path = evidence_dir / f"{name}.command.txt"
        stdout_path = evidence_dir / f"{name}.stdout.txt"
        stderr_path = evidence_dir / f"{name}.stderr.txt"
        records.append(
            {
                "name": name,
                "command": command_path.read_text(encoding="utf-8").strip()
                if command_path.exists()
                else "",
                "exit_code": int(exit_path.read_text(encoding="utf-8").strip()),
                "stdout_tail": stdout_path.read_text(encoding="utf-8", errors="replace")[-1200:]
                if stdout_path.exists()
                else "",
                "stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace")[-1200:]
                if stderr_path.exists()
                else "",
                "evidence_level": "E2",
            }
        )
    if not records:
        raise ValueError(f"No command evidence found in {evidence_dir}")
    return records


def installed_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def build_handoff(
    *,
    root: Path,
    output: Path,
    candidate_version: str,
    source_commit: str | None,
    baseline_hashes_path: Path,
    evidence_dir: Path,
    release_manifest_path: Path,
    build_artifacts: list[tuple[str, Path]],
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    output = output.resolve()
    baseline_hashes = _read_json(baseline_hashes_path)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in baseline_hashes.items()):
        raise ValueError("Baseline file hashes must be a string-to-string mapping")
    candidate_hashes = source_file_hashes(root)
    changes, diff_checksum = diff_from_baseline(baseline_hashes, candidate_hashes)
    upstream = _read_json(release_manifest_path)
    artifact_dir = root / "artifacts" / "smoke"
    metrics = _read_json(artifact_dir / "metrics.json")
    release_decision = _read_json(artifact_dir / "release_decision.json")
    runtime_manifest = _read_json(artifact_dir / "manifest.json")
    dynamic = _read_json(artifact_dir / "dynamic_summary.json")
    ope = _read_json(artifact_dir / "ope_metrics.json")
    serving = _read_json(artifact_dir / "serving_benchmark.json")
    commands = collect_command_evidence(evidence_dir)
    dependencies = [
        artifact_record(root, root / "pyproject.toml", kind="dependency_manifest"),
        artifact_record(root, root / "requirements.txt", kind="dependency_manifest"),
        artifact_record(
            root, root / "constraints" / "validated.txt", kind="validated_constraints"
        ),
    ]
    artifacts = [artifact_record(root, path, kind=kind) for kind, path in build_artifacts]
    payload: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "project_name": "cold-start-product-search-reliability",
        "candidate_version": candidate_version,
        "package_version": upstream.get("version", "0.6.0"),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "qualification_status": "RELEASE_CANDIDATE_NOT_RELEASE_QUALIFIED",
        "source": {
            "supplied_source_commit": source_commit,
            "vcs_metadata_available": (root / ".git").exists(),
            "dirty_tree": None,
            "dirty_tree_reason": (
                "Git metadata was not present in the supplied archive; file-level before/after hashes "
                "and a deterministic diff checksum are recorded instead."
            ),
            "baseline_source_fingerprint": aggregate_fingerprint(baseline_hashes),
            "candidate_source_fingerprint": aggregate_fingerprint(candidate_hashes),
            "source_fingerprint_scope": (
                "Repository files excluding Git metadata, caches, build/dist/artifacts/releases, "
                "egg-info, bytecode, and release_candidate_handoff.json to avoid self-reference."
            ),
            "diff_checksum_sha256": diff_checksum,
            "changed_files": changes,
        },
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "os": platform.platform(),
        },
        "dependency_manifests": dependencies,
        "validated_dependency_versions": runtime_manifest.get("package_versions", {}),
        "supported_environment_claims": [
            "Python 3.11 through 3.13 by package metadata",
            "Ubuntu and Windows GitHub Actions workflows are configured",
            "CPU execution is required and locally validated",
            "GPU and Qwen/FAISS paths are optional and not candidate-gating",
        ],
        "required_entrypoints": {
            "pipeline": "python scripts/run_full_pipeline.py --config configs/smoke.yaml",
            "tests": "python scripts/run_tests.py",
            "integration": "python scripts/integration_validation.py --config configs/smoke.yaml --skip-pipeline",
            "replay": "python scripts/reproducibility_check.py --config configs/smoke.yaml",
            "api": "PRODUCT_SEARCH_ARTIFACT_DIR=artifacts/smoke python scripts/serve.py",
            "build": "python scripts/build_release.py --output-dir dist",
            "release_publish": "python scripts/manage_release.py publish --source artifacts/smoke --release-root releases",
        },
        "dataset_and_fixtures": {
            "candidate_gate_data": "Deterministic synthetic/small canonical product-search fixture",
            "claim_boundary": (
                "Correctness, integration, and offline research evidence only; not production traffic "
                "or an online A/B-test result."
            ),
        },
        "metrics": {
            "baseline": {
                "overall_ndcg_at_10": metrics["base_ndcg_at_10"],
                "cold_ndcg_at_10": metrics["cold_ndcg_at_10_base"],
                "warm_ndcg_at_10": metrics["warm_ndcg_at_10_base"],
                "dynamic_utility": dynamic["base_utility"],
            },
            "final": {
                "overall_ndcg_at_10": metrics["final_ndcg_at_10"],
                "cold_ndcg_at_10": metrics["cold_ndcg_at_10_final"],
                "warm_ndcg_at_10": metrics["warm_ndcg_at_10_final"],
                "cold_lift_ci_low": metrics["cold_ndcg_lift_ci_low"],
                "cold_lift_ci_high": metrics["cold_ndcg_lift_ci_high"],
                "behavior_brier": metrics["behavior_brier"],
                "behavior_ece": metrics["behavior_ece"],
                "future_behavior_roc_auc": metrics["future_behavior_roc_auc"],
                "dynamic_utility": dynamic["qrsbt_utility"],
                "dynamic_p10_utility_delta": dynamic[
                    "p10_scenario_replication_utility_delta"
                ],
                "ope_dr_abs_error": ope["dr_abs_error"],
                "ope_effective_sample_size": ope["effective_sample_size"],
                "serving_p95_ms": serving["p95_latency_ms"],
                "serving_fallbacks": serving["fallbacks"],
            },
            "gates": {
                "release_status": release_decision["status"],
                "passed": sum(bool(value) for value in release_decision["gates"].values()),
                "total": len(release_decision["gates"]),
                "failed": release_decision["failed_gates"],
                "regression_tolerance": (
                    "Frozen 29-gate policy in configs/smoke.yaml; no threshold or metric definition "
                    "was changed during the handoff round."
                ),
            },
        },
        "tests_and_command_evidence": commands,
        "build_artifacts": artifacts,
        "known_limitations": [
            "Synthetic/small canonical evidence does not establish production business lift.",
            "Running workers do not hot-reload current.json; publish and rollback require rolling restart.",
            "Metrics and admission limits are process-local in multi-worker serving.",
            "Local release-store atomicity assumes one filesystem and is not distributed consensus.",
            "Docker daemon, native Windows, hosted GitHub workflows, and GitHub attestations were not executed in this environment.",
            "GPU/Qwen/FAISS and full public-data scale paths were not candidate-gating validations.",
        ],
        "unresolved_items": [
            {
                "severity": "Medium",
                "item": "Hosted GitHub Actions and artifact attestation execution",
                "reason": "Requires a pushed exact commit and GitHub-hosted release context.",
            },
            {
                "severity": "Medium",
                "item": "Docker image build and health check",
                "reason": "No Docker daemon is available in the current execution environment.",
            },
            {
                "severity": "Medium",
                "item": "Native Windows runtime execution",
                "reason": "No PowerShell/Windows runner is available in the current environment.",
            },
        ],
        "evidence_level": {
            "maximum_achieved": "E3",
            "E0": "Repository and artifact inspection completed.",
            "E1": "Canonical commands and gates documented.",
            "E2": "Tests, pipeline, service, replay, and build executed in current workspace.",
            "E3": "Clean virtual-environment installation and execution completed.",
            "E4": "Not achieved; exact-commit GitHub-hosted execution remains for qualification.",
        },
        "next_qualification_gates": [
            "Push the exact candidate snapshot to GitHub and require all hosted CI jobs to pass.",
            "Run the release workflow and verify GitHub artifact attestations for wheel and sdist.",
            "Build and health-check the Docker image on an Ubuntu Docker host.",
            "Run the canonical commands on native Windows Python 3.11.",
            "Confirm repository protection rules and dependency-review enforcement.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _resolve_record_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_handoff(
    root: Path,
    handoff_path: Path,
    *,
    allow_missing_build_artifacts: bool = False,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    payload = _read_json(handoff_path.resolve(strict=True))
    required = {
        "schema_version",
        "project_name",
        "candidate_version",
        "created_at_utc",
        "qualification_status",
        "source",
        "dependency_manifests",
        "required_entrypoints",
        "metrics",
        "tests_and_command_evidence",
        "build_artifacts",
        "known_limitations",
        "unresolved_items",
        "evidence_level",
        "next_qualification_gates",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Handoff is missing required fields: {missing}")
    if payload["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise ValueError("Unsupported handoff schema version")
    if payload["qualification_status"] == "RELEASE_QUALIFIED":
        raise ValueError("Candidate handoff must not claim RELEASE_QUALIFIED")
    dependency_records = list(payload["dependency_manifests"])
    build_records = list(payload["build_artifacts"])
    verified_records = 0
    missing_build_records: list[str] = []
    for record in dependency_records + build_records:
        path = _resolve_record_path(root, str(record["path"]))
        if not path.is_file():
            if allow_missing_build_artifacts and record in build_records:
                missing_build_records.append(str(record["path"]))
                continue
            raise FileNotFoundError(f"Handoff path is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"Handoff checksum mismatch: {path}")
        verified_records += 1
    if not payload["tests_and_command_evidence"]:
        raise ValueError("Handoff contains no command evidence")
    if any(int(record["exit_code"]) != 0 for record in payload["tests_and_command_evidence"]):
        raise ValueError("Handoff contains a failed candidate-gating command")
    return {
        "status": "PASS",
        "candidate_version": payload["candidate_version"],
        "verified_records": verified_records,
        "missing_build_artifacts": missing_build_records,
        "strict_build_artifacts_verified": not missing_build_records,
        "command_evidence": len(payload["tests_and_command_evidence"]),
        "handoff_sha256": sha256_file(handoff_path),
    }
