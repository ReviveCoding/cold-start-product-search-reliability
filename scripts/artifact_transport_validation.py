# ruff: noqa: E402
"""Validate canonical artifact transport and canonical-runtime serving.

This script intentionally never publishes or rolls back a release. Release
promotion remains LAUNCH-only in integration_validation.py and release_store.py.
"""

from __future__ import annotations

from _bootstrap import bootstrap_src

bootstrap_src()

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from product_search.provenance import (
    ARTIFACT_SCHEMA_VERSION,
    model_fingerprint,
    package_source_sha256,
    serving_config_sha256,
    verify_artifact_hashes,
)


def fail(message: str) -> None:
    raise SystemExit(message)


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        fail(f"Expected JSON object: {path}")
    return payload


def resolve_output(config_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(cfg["output_dir"])
    if not output.is_absolute():
        output = (root / output).resolve()
    return output


def validate_transport(output: Path, *, allow_nonlaunch: bool) -> dict:
    stage_path = output / "release_stage_metadata.json"
    manifest_path = output / "manifest.json"
    decision_path = output / "release_decision.json"

    for path in (stage_path, manifest_path, decision_path):
        if not path.is_file():
            fail(f"Required artifact file is missing: {path.name}")

    stage = read_json(stage_path)
    manifest = read_json(manifest_path)
    decision = read_json(decision_path)

    if stage.get("stage_status") != "complete":
        fail("Release stage is not complete")
    if manifest.get("stage_status") != "complete":
        fail("Artifact manifest is not complete")
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        fail("Artifact schema version differs from the installed validator")

    release_status = str(manifest.get("release_status", "")).upper()
    decision_status = str(decision.get("status", "")).upper()
    if release_status not in {"LAUNCH", "HOLD"}:
        fail(f"Unexpected release status: {release_status!r}")
    if release_status != decision_status:
        fail("Manifest and release-decision statuses differ")
    if release_status != "LAUNCH" and not allow_nonlaunch:
        fail("HOLD artifact requires --allow-nonlaunch")

    artifact_hashes = manifest.get("artifact_hashes")
    serving_hashes = manifest.get("serving_artifact_hashes")
    model_hashes = manifest.get("model_artifact_hashes")

    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        fail("Manifest has no artifact integrity hashes")
    if not isinstance(serving_hashes, dict) or not serving_hashes:
        fail("Manifest has no serving artifact integrity hashes")
    if serving_hashes != model_hashes:
        fail("Serving and model artifact hash contracts differ")
    if any(
        artifact_hashes.get(path) != digest
        for path, digest in serving_hashes.items()
    ):
        fail("Serving hashes are not a subset of artifact hashes")

    verify_artifact_hashes(output, artifact_hashes)
    verify_artifact_hashes(output, serving_hashes)

    code_hash = str(manifest.get("serving_code_sha256", ""))
    if code_hash != package_source_sha256():
        fail("Artifact serving-code fingerprint differs from current source")

    config_hash = str(manifest.get("serving_config_sha256", ""))
    expected_config_hash = serving_config_sha256(dict(manifest.get("config", {})))
    if config_hash != expected_config_hash:
        fail("Artifact serving-config fingerprint is inconsistent")

    fingerprint = str(manifest.get("model_fingerprint", ""))
    expected_fingerprint = model_fingerprint(
        config_sha256=config_hash,
        artifact_hashes_value=model_hashes,
        serving_code_sha256=code_hash,
    )
    if fingerprint != expected_fingerprint:
        fail("Artifact model fingerprint cannot be reproduced")

    return {
        "release_status": release_status,
        "model_fingerprint": fingerprint,
        "verified_artifact_hashes": len(artifact_hashes),
        "verified_serving_hashes": len(serving_hashes),
    }


def validate_canonical_serving(output: Path, *, release_status: str) -> dict:
    if release_status != "LAUNCH":
        os.environ["PRODUCT_SEARCH_ALLOW_NONLAUNCH"] = "1"

    os.environ["PRODUCT_SEARCH_ARTIFACT_DIR"] = str(output)
    os.environ["PRODUCT_SEARCH_VERIFY_ARTIFACTS"] = "1"

    from fastapi.testclient import TestClient

    from product_search.serving.app import SearchService, create_app, get_service

    service = SearchService(output, verify_hashes=True)
    queries = [
        "wireless headphones",
        "trail running shoes",
        "business laptop",
    ]
    direct = [service.search(query, 5) for query in queries]

    if any(response.fallback_used for response in direct):
        fail("Canonical serving used a fallback")
    if any(len(response.results) != 5 for response in direct):
        fail("Canonical serving returned an unexpected result count")
    if any(
        len({item.product_id for item in response.results}) != 5
        for response in direct
    ):
        fail("Canonical serving returned duplicate products")

    get_service.cache_clear()
    application = create_app(preload_artifacts=True)

    with TestClient(application) as client:
        ready = client.get("/ready")
        ready.raise_for_status()
        api = client.post("/search", json={"query": queries[0], "k": 5})
        api.raise_for_status()
        batch = client.post("/batch_search", json={"queries": queries, "k": 5})
        batch.raise_for_status()

    direct_ids = [item.product_id for item in direct[0].results]
    api_ids = [item["product_id"] for item in api.json()["results"]]
    if direct_ids != api_ids:
        fail("Canonical direct/API ranking IDs differ")
    if any(row["fallback_used"] for row in batch.json()["responses"]):
        fail("Canonical batch API used a fallback")

    return {
        "canonical_direct_api_parity": "PASS",
        "canonical_batch_full_ranker": "PASS",
        "model_version": ready.json().get("model_version"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("canonical-serving", "transport"),
    )
    parser.add_argument("--allow-nonlaunch", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    output = resolve_output(config_path)
    result = validate_transport(output, allow_nonlaunch=args.allow_nonlaunch)
    result.update({"status": "PASS", "mode": args.mode})

    if args.mode == "canonical-serving":
        result.update(
            validate_canonical_serving(
                output,
                release_status=result["release_status"],
            )
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
