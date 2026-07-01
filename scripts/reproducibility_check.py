"""Independent replay with byte-exact artifacts and semantic retriever checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

DETERMINISTIC_FILES = (
    "metrics.json",
    "data/products.csv",
    "data/queries.csv",
    "data/relevance.csv",
    "data/interactions.csv",
    "ranked_test.csv",
    "release_catalog.csv",
    "behavior_future_audit_features.csv",
    "product_behavior_snapshot.csv",
    "behavior_features.json",
    "lambdamart_metadata.json",
    "bm25.joblib",
    "dense.joblib",
    "lambdamart_model.json",
    "behavior_model.joblib",
    "qrsbt.joblib",
)
SEMANTIC_RETRIEVER_FILES = frozenset({"bm25.joblib", "dense.joblib"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_text(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _update_json(digest: "hashlib._Hash", value: object) -> None:
    _update_text(
        digest,
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ),
    )


def _update_array(digest: "hashlib._Hash", label: str, value: object) -> None:
    array = np.asarray(value)
    _update_text(digest, label)
    _update_json(
        digest,
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        },
    )

    if array.dtype.kind in {"O", "U", "S"}:
        _update_json(digest, array.tolist())
        return

    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.tobytes(order="C"))


def _update_mapping(
    digest: "hashlib._Hash",
    label: str,
    mapping: dict[object, object],
) -> None:
    _update_text(digest, label)
    _update_json(
        digest,
        [
            (str(key), int(value))
            for key, value in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        ],
    )


def _query_texts(reference: Path) -> list[str]:
    queries = pd.read_csv(reference / "data" / "queries.csv")
    candidates = ("query", "query_text", "text")
    column = next((name for name in candidates if name in queries.columns), None)
    if column is None:
        raise RuntimeError(
            "Could not find a text column in data/queries.csv; "
            f"columns={list(queries.columns)}"
        )

    values = [str(value) for value in queries[column].fillna("").tolist()]
    if not values:
        raise RuntimeError("data/queries.csv is empty")

    return values


def _update_query_behavior(
    digest: "hashlib._Hash",
    model: object,
    queries: list[str],
    score_column: str,
) -> None:
    product_count = len(model.product_ids_)

    for query in queries:
        _update_text(digest, query)
        scores = np.asarray(model.score(query), dtype=np.float64)
        _update_array(digest, "scores", scores)

        ranked = model.search(query, k=product_count)
        _update_json(
            digest,
            {
                "product_ids": [str(value) for value in ranked["product_id"].tolist()],
                "scores": [
                    float(value)
                    for value in ranked[score_column].to_numpy(dtype=np.float64)
                ],
            },
        )


def _bm25_semantic_fingerprint(model: object, queries: list[str]) -> str:
    digest = hashlib.sha256()
    _update_text(digest, "bm25-semantic-v1")
    _update_array(digest, "product_ids", model.product_ids_)
    _update_json(
        digest,
        {
            "k1": float(model.k1),
            "b": float(model.b),
            "avgdl": float(model.avgdl_),
            "token_pattern": str(model.vectorizer_.token_pattern),
            "lowercase": bool(model.vectorizer_.lowercase),
        },
    )
    _update_mapping(digest, "vocabulary", model.vectorizer_.vocabulary_)
    _update_array(digest, "doc_len", model.doc_len_)
    _update_array(digest, "idf", model.idf_)

    counts = model.counts_.tocsr()
    _update_json(digest, {"counts_shape": list(counts.shape)})
    _update_array(digest, "counts_data", counts.data)
    _update_array(digest, "counts_indices", counts.indices)
    _update_array(digest, "counts_indptr", counts.indptr)
    _update_query_behavior(digest, model, queries, "bm25_score")
    return digest.hexdigest()


def _canonical_dense_arrays(model: object) -> tuple[np.ndarray, np.ndarray | None]:
    embeddings = np.asarray(model.embeddings_, dtype=np.float64).copy()
    if model.svd_ is None:
        return embeddings, None

    components = np.asarray(model.svd_.components_, dtype=np.float64).copy()
    for index, row in enumerate(components):
        if row.size == 0:
            continue
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            components[index] *= -1.0
            embeddings[:, index] *= -1.0
    return embeddings, components


def _dense_semantic_fingerprint(model: object, queries: list[str]) -> str:
    digest = hashlib.sha256()
    _update_text(digest, "dense-semantic-v1")
    _update_array(digest, "product_ids", model.product_ids_)
    _update_json(
        digest,
        {
            "dimension": int(model.dimension),
            "seed": int(model.seed),
            "embedding_mode": str(model.embedding_mode_),
            "ngram_range": list(model.vectorizer_.ngram_range),
            "min_df": model.vectorizer_.min_df,
            "sublinear_tf": bool(model.vectorizer_.sublinear_tf),
        },
    )
    _update_mapping(digest, "vocabulary", model.vectorizer_.vocabulary_)
    _update_array(digest, "tfidf_idf", model.vectorizer_.idf_)

    embeddings, components = _canonical_dense_arrays(model)
    _update_array(digest, "embeddings", embeddings)

    if model.svd_ is None:
        _update_text(digest, "svd:none")
    else:
        _update_array(digest, "svd_components", components)
        _update_array(digest, "svd_singular_values", model.svd_.singular_values_)
        _update_array(
            digest,
            "svd_explained_variance",
            model.svd_.explained_variance_,
        )
        _update_array(
            digest,
            "svd_explained_variance_ratio",
            model.svd_.explained_variance_ratio_,
        )

    _update_query_behavior(digest, model, queries, "dense_score")
    return digest.hexdigest()


def _semantic_retriever_audit(reference: Path, replay: Path) -> dict[str, dict[str, object]]:
    queries = _query_texts(reference)
    definitions = {
        "bm25.joblib": _bm25_semantic_fingerprint,
        "dense.joblib": _dense_semantic_fingerprint,
    }
    audit: dict[str, dict[str, object]] = {}

    for relative, fingerprint in definitions.items():
        reference_path = reference / relative
        replay_path = replay / relative
        reference_model = joblib.load(reference_path)
        replay_model = joblib.load(replay_path)
        reference_fingerprint = fingerprint(reference_model, queries)
        replay_fingerprint = fingerprint(replay_model, queries)
        audit[relative] = {
            "raw_serialization_equal": _sha256(reference_path)
            == _sha256(replay_path),
            "semantic_fingerprint_equal": reference_fingerprint
            == replay_fingerprint,
            "reference_semantic_fingerprint": reference_fingerprint,
            "replay_semantic_fingerprint": replay_fingerprint,
            "probe_query_count": len(queries),
        }

    return audit


def _run_model_stage(root: Path, config: Path, timeout_seconds: float) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get(
        "PYTHONPATH",
        "",
    )
    env["PRODUCT_SEARCH_HARD_STAGE_EXIT"] = "1"
    env.setdefault("PYTHONHASHSEED", "0")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_model_stage.py"),
            "--config",
            str(config),
        ],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
        start_new_session=(os.name != "nt"),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Independent reproducibility stage failed for {config.name}:\n"
            + completed.stderr[-4000:]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--stage-timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    supplied = Path(args.config)
    config_path = (
        supplied.resolve()
        if supplied.is_absolute()
        else (Path.cwd() / supplied).resolve()
    )
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise RuntimeError("Reproducibility configuration root must be a mapping")

    configured_output = Path(str(base["output_dir"]))
    reference = (
        configured_output.resolve()
        if configured_output.is_absolute()
        else (root / configured_output).resolve()
    )
    metadata_path = reference / "model_stage_metadata.json"
    manifest_path = reference / "manifest.json"
    if not metadata_path.exists() or not manifest_path.exists():
        raise RuntimeError(
            "A completed reference run is required; run the smoke or "
            "integration pipeline first"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        metadata.get("stage_status") != "complete"
        or manifest.get("stage_status") != "complete"
    ):
        raise RuntimeError("Configured reference artifact is not marked complete")
    if manifest.get("config_file_sha256") != _sha256(config_path):
        raise RuntimeError(
            "Configured reference artifact was produced from a different config file"
        )

    missing_reference = [
        relative
        for relative in DETERMINISTIC_FILES
        if not (reference / relative).is_file()
    ]
    if missing_reference:
        raise RuntimeError(f"Reference artifact is incomplete: {missing_reference}")

    replay = root / "artifacts" / "repro_check"
    replay_config = root / "artifacts" / "repro_check.yaml"
    shutil.rmtree(replay, ignore_errors=True)

    run_config = dict(base)
    run_config["output_dir"] = str(replay.relative_to(root))
    replay_config.parent.mkdir(parents=True, exist_ok=True)
    replay_config.write_text(
        yaml.safe_dump(run_config, sort_keys=False),
        encoding="utf-8",
    )

    try:
        _run_model_stage(root, replay_config, args.stage_timeout_seconds)

        raw_mismatches = [
            relative
            for relative in DETERMINISTIC_FILES
            if _sha256(reference / relative) != _sha256(replay / relative)
        ]
        semantic_audit = _semantic_retriever_audit(reference, replay)
        failures = [
            relative
            for relative in raw_mismatches
            if relative not in SEMANTIC_RETRIEVER_FILES
        ]
        failures.extend(
            relative
            for relative, result in semantic_audit.items()
            if not bool(result["semantic_fingerprint_equal"])
        )

        result = {
            "status": "PASS" if not failures else "FAIL",
            "model_runs": 2,
            "reference_run": str(reference),
            "independent_replay_runs": 1,
            "compared_files": len(DETERMINISTIC_FILES),
            "byte_exact_files": len(DETERMINISTIC_FILES)
            - len(SEMANTIC_RETRIEVER_FILES),
            "dynamic_and_ope_determinism": "covered_by_unit_tests",
            "raw_serialization_mismatches": raw_mismatches,
            "retriever_semantic_audit": semantic_audit,
            "failures": sorted(set(failures)),
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if not failures else 1
    finally:
        if not args.keep:
            shutil.rmtree(replay, ignore_errors=True)
        replay_config.unlink(missing_ok=True)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
