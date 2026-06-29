"""Cross-platform clean-checkout integration validation.

The validator regenerates all smoke artifacts, verifies integrity hashes and stage contracts, loads the
same scoring service used by FastAPI, and compares direct and HTTP results.
"""
from __future__ import annotations

from _bootstrap import bootstrap_src

bootstrap_src()

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from product_search.provenance import (
    model_fingerprint,
    package_source_sha256,
    serving_config_sha256,
    verify_artifact_hashes,
)
from product_search.serving.app import SearchService, create_app, get_service
from product_search.release_store import (
    publish_release,
    read_pointer,
    resolve_current_release,
    rollback_release,
)

def _run_uvicorn_validator(
    root: Path,
    env: dict[str, str],
    output: Path,
    *arguments: str,
) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "uvicorn_validation.py"),
            "--artifact-dir",
            str(output),
            *arguments,
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Real Uvicorn validation failed:\n"
            + (completed.stdout + completed.stderr)[-4000:]
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit("Real Uvicorn validation did not emit valid JSON") from exc
    if payload.get("status") != "PASS":
        raise SystemExit("Real Uvicorn validation did not pass")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--skip-pipeline", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(cfg["output_dir"])
    if not output.is_absolute():
        output = (root / output).resolve()

    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = None
    if not args.skip_pipeline:
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_full_pipeline.py"),
                "--config",
                str(config_path),
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=True,
        )

    required = [
        "metrics.json",
        "dynamic_summary.json",
        "ope_metrics.json",
        "release_decision.json",
        "serving_benchmark.json",
        "manifest.json",
        "artifact_requirements.txt",
        "ranked_test.csv",
        "behavior_model.joblib",
        "release_catalog.csv",
        "lambdamart_model.json",
        "lambdamart_metadata.json",
        "product_behavior_snapshot.csv",
        "behavior_future_audit_features.csv",
        "reports/future_audit_report.md",
        "reports/claim_boundaries.md",
        "reports/serving_benchmark.md",
        "policy_sensitivity/policy_sensitivity.csv",
        "policy_sensitivity/policy_sensitivity.json",
        "policy_sensitivity/policy_sensitivity.md",
    ]
    missing = [relative for relative in required if not (output / relative).exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage_status") != "complete":
        raise SystemExit("Final manifest is not marked complete")
    verify_artifact_hashes(output, manifest["artifact_hashes"])
    serving_hashes = manifest.get("serving_artifact_hashes")
    model_hashes = manifest.get("model_artifact_hashes")
    if not isinstance(serving_hashes, dict) or serving_hashes != model_hashes:
        raise SystemExit("Serving and model artifact hash contracts are missing or inconsistent")
    if any(manifest["artifact_hashes"].get(path) != digest for path, digest in serving_hashes.items()):
        raise SystemExit("Serving artifact hashes are not a valid subset of release hashes")
    verify_artifact_hashes(output, serving_hashes)
    fingerprint = str(manifest.get("model_fingerprint", ""))
    if len(fingerprint) != 64:
        raise SystemExit("Manifest model fingerprint is missing or malformed")
    serving_code_hash = str(manifest.get("serving_code_sha256", ""))
    if serving_code_hash != package_source_sha256():
        raise SystemExit("Manifest serving-code fingerprint differs from installed package source")
    scoring_config_hash = str(manifest.get("serving_config_sha256", ""))
    if scoring_config_hash != serving_config_sha256(dict(manifest.get("config", {}))):
        raise SystemExit("Manifest serving-config fingerprint is missing or inconsistent")
    expected_fingerprint = model_fingerprint(
        config_sha256=scoring_config_hash,
        artifact_hashes_value=model_hashes,
        serving_code_sha256=serving_code_hash,
    )
    if fingerprint != expected_fingerprint:
        raise SystemExit("Manifest model fingerprint cannot be reproduced")
    advanced_argv = manifest.get("advanced_challenger_argv")
    if not isinstance(advanced_argv, list) or not advanced_argv:
        raise SystemExit("Portable advanced challenger argv is missing")
    forbidden_roots = ("/mnt/", "/tmp/", "C:\\", "D:\\")
    if any(str(value).startswith(forbidden_roots) for value in advanced_argv):
        raise SystemExit("Advanced challenger command contains a machine-specific absolute path")
    if any("project9_v050_work" in str(value) for value in advanced_argv):
        raise SystemExit("Advanced challenger command contains a checkout-specific path")
    release = json.loads((output / "release_decision.json").read_text(encoding="utf-8"))
    if release["status"] == "HOLD":
        raise SystemExit(f"Smoke release is HOLD: {release['failed_gates']}")
    if release["status"] != manifest["release_status"]:
        raise SystemExit("Release status differs between decision and manifest")
    serving_benchmark = json.loads((output / "serving_benchmark.json").read_text(encoding="utf-8"))
    if serving_benchmark["fallbacks"] != 0 or not serving_benchmark["result_contract_pass"]:
        raise SystemExit("Serving benchmark failed its result or fallback contract")
    if not release["gates"].get("serving_latency_p95", False):
        raise SystemExit("Serving p95 latency release gate did not pass")
    if not release["gates"].get("policy_sensitivity", False):
        raise SystemExit("Policy sensitivity release gate did not pass")
    sensitivity = json.loads(
        (output / "policy_sensitivity" / "policy_sensitivity.json").read_text(
            encoding="utf-8"
        )
    )
    if sensitivity.get("status") != "PASS":
        raise SystemExit("Policy sensitivity artifact is not PASS")

    candidates = pd.read_csv(output / "ranked_test.csv")
    if candidates.empty or candidates.duplicated(["query_id", "product_id"]).any():
        raise SystemExit("Candidate artifact is empty or contains duplicate query-product pairs")
    score_columns = ["base_score", "final_score", "semantic_rank_score", "behavior_score"]
    if candidates[score_columns].isna().any().any():
        raise SystemExit("Candidate score artifact contains nulls")

    release_catalog = pd.read_csv(output / "release_catalog.csv")
    test_block = int(manifest["test_block"])
    if (release_catalog["launch_block"].astype(int) > test_block).any():
        raise SystemExit("Release catalog contains post-cutoff products")
    future = pd.read_csv(output / "behavior_future_audit_features.csv")
    future_block = int(manifest["future_audit_block"])
    if set(future["time_block"].astype(int)) != {future_block}:
        raise SystemExit("Future audit artifact does not contain exactly the frozen audit block")
    if not set(future["product_id"].astype(int)).issubset(
        set(release_catalog["product_id"].astype(int))
    ):
        raise SystemExit("Future audit contains products outside the release-time catalog")
    if int(manifest.get("future_audit_rows", -1)) != len(future):
        raise SystemExit("Future audit row count differs between artifact and manifest")
    if manifest.get("historical_retriever_policy") != (
        "refit_on_catalog_available_at_each_scoring_block"
    ):
        raise SystemExit("Manifest does not declare strict temporal retriever refitting")
    if int(manifest.get("dynamic_replications", 0)) < 2:
        raise SystemExit("Release manifest does not contain multi-replication simulation evidence")

    service = SearchService(output)
    queries = [
        "wireless headphones",
        "trail running shoes",
        "business laptop",
    ]
    direct = [service.search(query, 5) for query in queries]
    if any(response.fallback_used for response in direct):
        raise SystemExit("Full-ranker API path unexpectedly used a fallback")
    if any(len(response.results) != 5 for response in direct):
        raise SystemExit("Direct service returned an unexpected number of results")
    if any(len({item.product_id for item in response.results}) != 5 for response in direct):
        raise SystemExit("Direct service returned duplicate products")

    os.environ["PRODUCT_SEARCH_ARTIFACT_DIR"] = str(output)
    os.environ["PRODUCT_SEARCH_VERIFY_ARTIFACTS"] = "1"
    get_service.cache_clear()
    application = create_app(preload_artifacts=True)
    with TestClient(application) as client:
        live = client.get("/live")
        live.raise_for_status()
        ready = client.get("/ready")
        ready.raise_for_status()
        health = client.get("/health")
        health.raise_for_status()
        api_single = client.post("/search", json={"query": queries[0], "k": 5})
        api_single.raise_for_status()
        api_batch = client.post("/batch_search", json={"queries": queries, "k": 5})
        api_batch.raise_for_status()
        metrics_response = client.get("/metrics")
        metrics_response.raise_for_status()
        manifest_response = client.get("/model_manifest")
        manifest_response.raise_for_status()

    if ready.json().get("model_version") != fingerprint[:12]:
        raise SystemExit("Readiness model version differs from manifest fingerprint")
    if health.json() != ready.json():
        raise SystemExit("Health compatibility endpoint differs from readiness endpoint")
    metrics_payload = metrics_response.json()
    if int(metrics_payload.get("requests_total", 0)) < len(queries) + 1:
        raise SystemExit("API metrics did not record all single and batch searches")
    if int(metrics_payload.get("fallbacks_total", -1)) != 0:
        raise SystemExit("API metrics recorded an unexpected fallback")
    if int(metrics_payload.get("source_counts", {}).get("full_ranker", 0)) < len(queries) + 1:
        raise SystemExit("API source metrics did not record the full-ranker path")
    if manifest_response.json().get("model_version") != fingerprint[:12]:
        raise SystemExit("Model manifest endpoint exposes a different model version")

    api_ids = [item["product_id"] for item in api_single.json()["results"]]
    direct_ids = [item.product_id for item in direct[0].results]
    if api_ids != direct_ids:
        raise SystemExit("Direct service and FastAPI ranking results differ")
    batch_payload = api_batch.json()["responses"]
    if len(batch_payload) != len(queries) or any(row["fallback_used"] for row in batch_payload):
        raise SystemExit("Batch API path failed or used an unexpected fallback")

    uvicorn_result = _run_uvicorn_validator(
        root,
        env,
        output,
        "--requests",
        "16",
        "--concurrency",
        "4",
        "--workers",
        "1",
    )
    multiworker_result = _run_uvicorn_validator(
        root,
        env,
        output,
        "--requests",
        "24",
        "--concurrency",
        "8",
        "--workers",
        "2",
        "--max-concurrency",
        "4",
    )
    overload_result = _run_uvicorn_validator(
        root,
        env,
        output,
        "--requests",
        "16",
        "--concurrency",
        "8",
        "--workers",
        "1",
        "--max-concurrency",
        "1",
        "--admission-timeout-ms",
        "0",
        "--expect-overload",
    )

    with tempfile.TemporaryDirectory(prefix="product-search-release-store-") as temp_dir:
        release_root = Path(temp_dir)
        def validator(path: Path) -> None:
            SearchService(path, verify_hashes=True)
        first_pointer = publish_release(
            output,
            release_root,
            generation="smoke-a",
            runtime_validator=validator,
        )
        second_pointer = publish_release(
            output,
            release_root,
            generation="smoke-b",
            runtime_validator=validator,
        )
        if first_pointer.generation != "smoke-a" or second_pointer.previous_generation != "smoke-a":
            raise SystemExit("Atomic release publication did not preserve generation history")
        deployed = release_root / "generations" / "smoke-b"
        deployed_files = {
            path.relative_to(deployed).as_posix()
            for path in deployed.rglob("*")
            if path.is_file()
        }
        expected_deployed_files = {
            "manifest.json",
            "generation.json",
            *manifest["serving_artifact_hashes"].keys(),
        }
        if deployed_files != expected_deployed_files:
            raise SystemExit("Published generation is not the minimal verified serving closure")

        pointer_before_failure = read_pointer(release_root)
        def reject_publish(path: Path) -> None:
            del path
            raise RuntimeError("injected runtime validation failure")
        try:
            publish_release(
                output,
                release_root,
                generation="smoke-failed",
                runtime_validator=reject_publish,
            )
        except RuntimeError as exc:
            if "injected runtime validation failure" not in str(exc):
                raise
        else:
            raise SystemExit("Injected failed publication unexpectedly succeeded")
        if read_pointer(release_root) != pointer_before_failure:
            raise SystemExit("Failed publication changed the active release pointer")
        if (release_root / "generations" / "smoke-failed").exists():
            raise SystemExit("Failed publication left a visible immutable generation")

        rollback_pointer = rollback_release(
            release_root,
            runtime_validator=validator,
        )
        if rollback_pointer.generation != "smoke-a":
            raise SystemExit("Release rollback did not restore the previous generation")
        if read_pointer(release_root).generation != "smoke-a":
            raise SystemExit("Release pointer differs after rollback")
        resolved = resolve_current_release(release_root)
        pointer_service = SearchService(resolved, verify_hashes=True)
        pointer_response = pointer_service.search(queries[0], 5)
        if pointer_response.fallback_used or len(pointer_response.results) != 5:
            raise SystemExit("Pointer-resolved release did not serve the full ranking path")

    result = {
        "status": release["status"],
        "output_dir": str(output),
        "required_artifacts": len(required),
        "verified_hashes": len(manifest["artifact_hashes"]),
        "verified_serving_hashes": len(serving_hashes),
        "candidate_rows": int(len(candidates)),
        "direct_queries": len(direct),
        "api_top_ids_match": True,
        "api_fallbacks": 0,
        "serving_p95_latency_ms": serving_benchmark["p95_latency_ms"],
        "uvicorn_requests": uvicorn_result["requests"],
        "uvicorn_concurrency": uvicorn_result["concurrency"],
        "uvicorn_throughput_requests_per_second": uvicorn_result[
            "throughput_requests_per_second"
        ],
        "uvicorn_multiworker_requests": multiworker_result["requests"],
        "uvicorn_workers": multiworker_result["workers"],
        "uvicorn_multiworker_model_consistency": "PASS",
        "uvicorn_overload_accepted": overload_result["accepted"],
        "uvicorn_overload_rejections": overload_result["overload_rejections"],
        "uvicorn_overload_http_503": "PASS",
        "atomic_publish": "PASS",
        "minimal_serving_closure": "PASS",
        "failed_publish_pointer_stable": "PASS",
        "generation_metadata_integrity": "PASS",
        "rollback": "PASS",
        "pointer_resolved_serving": "PASS",
        "stdout_tail": completed.stdout[-500:] if completed is not None else "pipeline skipped",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
    # All validation outputs are complete. Avoid rare native XGBoost/BLAS teardown stalls after
    # the validator has loaded the released artifacts in the same process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
