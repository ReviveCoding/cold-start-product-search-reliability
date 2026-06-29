from __future__ import annotations

from _bootstrap import bootstrap_src

bootstrap_src()

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from product_search.config import load_config
from product_search.evaluation.sensitivity import (
    evaluate_policy_sensitivity,
    sensitivity_markdown,
)
from product_search.ope.estimators import estimate_ope, generate_known_propensity_lab
from product_search.policy.release import evaluate_release
from product_search.provenance import (
    artifact_hashes,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_sha256,
    atomic_write_text,
    model_fingerprint,
    package_source_sha256,
    runtime_environment,
    serving_config_sha256,
    sha256_file,
    write_artifact_requirements,
    write_manifest,
)
from product_search.serving.app import SearchService


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_reports(
    out: Path,
    metrics: dict,
    ope: dict,
    dynamic: dict,
    release: dict,
    serving: dict,
) -> None:
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        reports / "benchmark_summary.md",
        "# Smoke Benchmark Summary\n\n"
        + "\n".join(f"- **{k}**: {_format_value(v)}" for k, v in metrics.items())
        + "\n",
    )
    atomic_write_text(
        reports / "future_audit_report.md",
        "# Untouched Future-Block Audit\n\n"
        "This report evaluates the frozen release-time catalog and model on the final observed "
        "time block. Products introduced after the release cutoff are intentionally excluded, so "
        "this is a temporal stability audit rather than a new-product launch benchmark.\n\n"
        + "\n".join(
            f"- **{k}**: {_format_value(v)}"
            for k, v in metrics.items()
            if k.startswith("future_") or k.startswith("relation_eligibility_")
        )
        + "\n",
    )
    atomic_write_text(
        reports / "ope_validity_report.md",
        "# Known-Propensity OPE Validation\n\n"
        "This controlled laboratory has known logging propensities and true simulated policy "
        "value. It is not a claim of online business uplift.\n\n"
        + "\n".join(f"- **{k}**: {_format_value(v)}" for k, v in ope.items())
        + "\n",
    )
    atomic_write_text(
        reports / "dynamic_feedback_report.md",
        "# Dynamic Feedback Simulation\n\n"
        "Results are proxy outcomes under conservative, neutral, and exploratory simulated user "
        "models.\n\n"
        + "\n".join(f"- **{k}**: {_format_value(v)}" for k, v in dynamic.items())
        + "\n",
    )
    atomic_write_text(
        reports / "release_decision.md",
        "# Release Decision\n\n"
        + f"## {release['status']}\n\n"
        + "\n".join(
            f"- **{k}**: {'PASS' if v else 'FAIL'}" for k, v in release["gates"].items()
        )
        + "\n\n## Diagnostics\n\n"
        + "\n".join(
            f"- **{k}**: {_format_value(v)}"
            for k, v in release["diagnostics"].items()
        )
        + "\n",
    )
    atomic_write_text(
        reports / "serving_benchmark.md",
        "# Serving Path Benchmark\n\n"
        "The benchmark loads the immutable artifact bundle and executes the same full scoring path "
        "used by FastAPI after a warm-up query. Latency is a local CPU smoke measurement, not a "
        "production service-level objective.\n\n"
        + "\n".join(f"- **{k}**: {_format_value(v)}" for k, v in serving.items())
        + "\n",
    )
    atomic_write_text(
        reports / "advanced_challenger_plan.md",
        "# Advanced Challenger Plan\n\n"
        "The reproducible CPU release path uses a temporally calibrated classical behavior model. "
        "The PyTorch DCN challenger consumes the same exported feature contract in an isolated "
        "process and uses CUDA when available. Qwen3 and FAISS remain opt-in full-data components.\n",
    )
    atomic_write_text(
        reports / "claim_boundaries.md",
        "# Claim Boundaries\n\n"
        "- `first-observed` and `zero-history` are observable log states, not verified catalog launch dates.\n"
        "- Logged-data prediction metrics are observational, not causal policy-value estimates.\n"
        "- OPE validation uses a known-propensity semi-synthetic laboratory.\n"
        "- Dynamic marketplace outcomes are simulated proxies, not revenue or online A/B-test lift.\n"
        "- The smoke relation model recovers deterministic synthetic rules; its score is not a real-world annotation benchmark.\n"
        "- The untouched future-block audit evaluates only the frozen release-time catalog; it does not evaluate products launched after the cutoff.\n"
        "- Qwen3, FAISS, and GPU paths are implemented but require opt-in dependencies and external model/data downloads.\n",
    )




def benchmark_serving_path(out: Path, *, max_queries: int = 24, k: int = 10) -> dict:
    service = SearchService(out, allow_nonlaunch=True)
    queries = pd.read_csv(out / "data" / "queries.csv")["query"].astype(str).drop_duplicates()
    query_list = queries.head(max_queries).tolist()
    if not query_list:
        raise RuntimeError("Serving benchmark has no queries")
    service.search(query_list[0], k)  # warm-up model and sparse/vector operations
    durations: list[float] = []
    fallbacks = 0
    contract_pass = True
    for query in query_list:
        started = time.perf_counter()
        response = service.search(query, k)
        durations.append((time.perf_counter() - started) * 1000.0)
        fallbacks += int(response.fallback_used)
        ids = [result.product_id for result in response.results]
        scores = [result.score for result in response.results]
        contract_pass = contract_pass and (
            len(ids) == k
            and len(ids) == len(set(ids))
            and all(scores[index] >= scores[index + 1] for index in range(len(scores) - 1))
            and all(np.isfinite(scores))
        )
    values = np.asarray(durations, dtype=float)
    return {
        "queries": int(len(values)),
        "k": int(k),
        "mean_latency_ms": float(values.mean()),
        "p50_latency_ms": float(np.quantile(values, 0.50)),
        "p95_latency_ms": float(np.quantile(values, 0.95)),
        "p99_latency_ms": float(np.quantile(values, 0.99)),
        "throughput_queries_per_second": float(1000.0 / max(values.mean(), 1e-9)),
        "fallbacks": int(fallbacks),
        "result_contract_pass": bool(contract_pass),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--orchestrator-runtime", type=float, default=0.0)
    args = parser.parse_args()

    ope = estimate_ope(generate_known_propensity_lab(seed=args.seed, n=args.rows), seed=args.seed)
    if args.output:
        atomic_write_json(Path(args.output), ope)

    if not (args.config and args.output_dir):
        print(json.dumps(ope, indent=2, sort_keys=True))
        return

    config = load_config(args.config)
    out = Path(args.output_dir)
    required_stages = {
        "model": out / "model_stage_metadata.json",
        "dynamic": out / "dynamic_stage_metadata.json",
        "metrics": out / "metrics.json",
        "dynamic_summary": out / "dynamic_summary.json",
    }
    missing = [name for name, path in required_stages.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Required upstream artifacts are missing: {missing}")
    model_meta = json.loads(required_stages["model"].read_text(encoding="utf-8"))
    dynamic_meta = json.loads(required_stages["dynamic"].read_text(encoding="utf-8"))
    if model_meta.get("stage_status") != "complete" or dynamic_meta.get("stage_status") != "complete":
        raise RuntimeError("An upstream stage is not marked complete")

    metrics = json.loads(required_stages["metrics"].read_text(encoding="utf-8"))
    dynamic = json.loads(required_stages["dynamic_summary"].read_text(encoding="utf-8"))
    atomic_write_json(out / "ope_metrics.json", ope)

    sensitivity_frame, sensitivity = evaluate_policy_sensitivity(
        pd.read_csv(out / "ranked_test.csv"),
        config.raw,
    )
    sensitivity_dir = out / "policy_sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(sensitivity_dir / "policy_sensitivity.csv", sensitivity_frame)
    atomic_write_json(sensitivity_dir / "policy_sensitivity.json", sensitivity)
    atomic_write_text(
        sensitivity_dir / "policy_sensitivity.md",
        sensitivity_markdown(
            sensitivity_frame, float(sensitivity["selected_max_boost"])
        ),
    )

    environment = runtime_environment()
    write_artifact_requirements(out / "artifact_requirements.txt", environment)

    serving_model_paths = [
        "release_catalog.csv",
        "bm25.joblib",
        "dense.joblib",
        "lambdamart_model.json",
        "lambdamart_metadata.json",
        "behavior_model.joblib",
        "qrsbt.joblib",
        "product_behavior_snapshot.csv",
        "behavior_features.json",
    ]
    evidence_paths = [
        "data/products.csv",
        "data/queries.csv",
        "data/relevance.csv",
        "data/interactions.csv",
        "behavior_future_audit_features.csv",
        "metrics.json",
        "dynamic_summary.json",
        "ope_metrics.json",
        "artifact_requirements.txt",
    ]
    model_paths = [*serving_model_paths, *evidence_paths]
    model_hashes = artifact_hashes(out, serving_model_paths)
    config_file_sha256 = sha256_file(config.source_path)
    model_config = {
        key: value for key, value in config.raw.items() if key != "output_dir"
    }
    config_sha256 = canonical_json_sha256(model_config)
    scoring_config_sha256 = serving_config_sha256(config.raw)
    serving_code_hash = package_source_sha256()
    fingerprint = model_fingerprint(
        config_sha256=scoring_config_sha256,
        artifact_hashes_value=model_hashes,
        serving_code_sha256=serving_code_hash,
    )
    repository_root = Path(__file__).resolve().parents[1]
    configured_output = Path(str(config.raw["output_dir"]))
    artifact_prefix = (
        configured_output.as_posix()
        if not configured_output.is_absolute()
        else "${ARTIFACT_DIR}"
    )
    try:
        config_argument = config.source_path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        config_argument = "${CONFIG_PATH}"
    advanced_argv = [
        "python",
        "scripts/train_dcn.py",
        "--train",
        f"{artifact_prefix}/behavior_train_features.csv",
        "--validation",
        f"{artifact_prefix}/behavior_validation_features.csv",
        "--test",
        f"{artifact_prefix}/behavior_test_features.csv",
        "--candidates",
        f"{artifact_prefix}/ranked_test.csv",
        "--features",
        f"{artifact_prefix}/behavior_features.json",
        "--output-dir",
        f"{artifact_prefix}/dcn_challenger",
        "--config",
        config_argument,
    ]
    base_payload = {
        "seed": config.seed,
        "mode": config.mode,
        "config": config.raw,
        "config_sha256": config_sha256,
        "serving_config_sha256": scoring_config_sha256,
        "config_file_sha256": config_file_sha256,
        "serving_code_sha256": serving_code_hash,
        "runtime_seconds": float(args.orchestrator_runtime),
        "model_stage_runtime_seconds": float(model_meta["runtime_seconds"]),
        "behavior_champion": model_meta["behavior_champion"],
        "n_products": model_meta["n_products"],
        "release_catalog_products": model_meta["release_catalog_products"],
        "n_interactions": model_meta["n_interactions"],
        "train_rows": model_meta["train_rows"],
        "validation_rows": model_meta["validation_rows"],
        "test_rows": model_meta["test_rows"],
        "future_audit_rows": model_meta["future_audit_rows"],
        "candidate_rows": model_meta["candidate_rows"],
        "validation_block": model_meta["validation_block"],
        "test_block": model_meta["test_block"],
        "future_audit_block": model_meta["future_audit_block"],
        "retriever_catalog_cutoff_block": model_meta["retriever_catalog_cutoff_block"],
        "historical_retriever_policy": model_meta["historical_retriever_policy"],
        "dynamic_replications": int(config.raw["simulation"]["replications"]),
        "stage_status": "complete",
        "orchestration": "isolated_model_dynamic_ope_release_stages",
        "package_versions": {
            name: record["version"] for name, record in environment["packages"].items()
        },
        "package_distributions": {
            name: record["distribution"] for name, record in environment["packages"].items()
        },
        "python_implementation": environment["python_implementation"],
        "model_fingerprint": fingerprint,
        "model_artifact_hashes": model_hashes,
        "serving_artifact_hashes": model_hashes,
        "advanced_challenger_argv": advanced_argv,
        "advanced_challenger_command": shlex.join(advanced_argv),
    }
    provisional_hashes = artifact_hashes(out, model_paths)
    write_manifest(
        out / "manifest.json",
        {**base_payload, "release_status": "PENDING", "artifact_hashes": provisional_hashes},
    )
    serving = benchmark_serving_path(out)
    atomic_write_json(out / "serving_benchmark.json", serving)

    release = evaluate_release(
        metrics,
        ope,
        dynamic,
        config.raw["release"],
        system=serving,
        policy_sensitivity=sensitivity,
    )
    atomic_write_json(out / "release_decision.json", release)
    write_reports(out, metrics, ope, dynamic, release, serving)

    critical_paths = [
        *model_paths,
        "serving_benchmark.json",
        "release_decision.json",
        "reports/benchmark_summary.md",
        "reports/future_audit_report.md",
        "reports/ope_validity_report.md",
        "reports/dynamic_feedback_report.md",
        "reports/serving_benchmark.md",
        "reports/release_decision.md",
        "reports/claim_boundaries.md",
        "policy_sensitivity/policy_sensitivity.csv",
        "policy_sensitivity/policy_sensitivity.json",
        "policy_sensitivity/policy_sensitivity.md",
    ]
    hashes = artifact_hashes(out, critical_paths)
    write_manifest(
        out / "manifest.json",
        {
            **base_payload,
            "release_status": release["status"],
            "artifact_hashes": hashes,
            "serving_benchmark": serving,
        },
    )
    atomic_write_json(
        out / "release_stage_metadata.json",
        {
            "stage_status": "complete",
            "release_status": release["status"],
            "ope_rows": int(args.rows),
            "artifact_count": int(len(hashes)),
            "policy_sensitivity_status": sensitivity["status"],
        },
    )
    print(json.dumps({"release": release, "ope": ope}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
    if os.getenv("PRODUCT_SEARCH_HARD_EXIT", "0") == "1":
        # The isolated release CLI has already atomically written every artifact and its completion
        # marker. Bypass rare native BLAS/OpenMP interpreter-teardown stalls in chained jobs.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
