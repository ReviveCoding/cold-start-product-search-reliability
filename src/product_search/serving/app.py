from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from product_search.pipeline import BASE_RANK_FEATURES, BEHAVIOR_FEATURES
from product_search.policy.gate import GateConfig, apply_coverage_overreach_gate
from product_search.ranking.lambdamart import LambdaMARTRanker
from product_search.retrieval.hybrid import hybrid_retrieve
from product_search.provenance import (
    ARTIFACT_SCHEMA_VERSION,
    model_fingerprint,
    package_source_sha256,
    package_versions,
    serving_config_sha256,
    verify_artifact_hashes,
)
from product_search.release_store import read_generation_metadata, resolve_current_release


LOGGER = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class BatchSearchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=50)
    k: int = Field(default=10, ge=1, le=50)

    @field_validator("queries")
    @classmethod
    def queries_must_not_contain_blank_values(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.split()).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("queries must not contain blank values")
        if any(len(value) > 300 for value in normalized):
            raise ValueError("each query must contain at most 300 characters")
        return normalized


class SearchResult(BaseModel):
    product_id: int
    title: str
    score: float
    bm25_score: float
    dense_score: float
    semantic_score: float
    behavior_score: float
    source: str
    gate_action: str
    gate_reason: str
    cold_start: bool
    qrsbt_confidence: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    fallback_used: bool
    fallback_reason: str | None = None
    model_version: str


class BatchSearchResponse(BaseModel):
    responses: list[SearchResponse]


class ServiceOverloadedError(RuntimeError):
    """Raised when the bounded serving admission queue is saturated."""


class SearchService:
    """Load a versioned artifact bundle and reuse the offline scoring path online.

    The service intentionally loads all artifacts once.  It verifies every hash before model
    deserialization and then applies retrieval, LambdaMART, calibrated behavior scoring, Q-RSBT,
    and the bounded coverage-overreach policy used by offline evaluation.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        verify_hashes: bool = True,
        allow_nonlaunch: bool = False,
        strict_environment: bool | None = None,
    ):
        self.artifact_dir = artifact_dir.resolve()
        self.manifest = _read_json(self.artifact_dir / "manifest.json")
        schema = str(self.manifest.get("artifact_schema_version", ""))
        if schema != ARTIFACT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported artifact schema {schema!r}; expected {ARTIFACT_SCHEMA_VERSION!r}"
            )
        release_status = str(self.manifest.get("release_status", "unknown")).upper()
        allow_env = os.getenv("PRODUCT_SEARCH_ALLOW_NONLAUNCH", "0") == "1"
        if release_status != "LAUNCH" and not (allow_nonlaunch or allow_env):
            raise RuntimeError(
                f"Refusing to serve artifact with release_status={release_status!r}; "
                "set PRODUCT_SEARCH_ALLOW_NONLAUNCH=1 only for controlled validation"
            )
        full_hashes = self.manifest.get("artifact_hashes")
        serving_hashes = self.manifest.get("serving_artifact_hashes")
        if verify_hashes:
            if not isinstance(serving_hashes, dict) or not serving_hashes:
                raise RuntimeError("Manifest does not contain serving artifact integrity hashes")
            verify_artifact_hashes(self.artifact_dir, serving_hashes)

        # Package compatibility must be checked before any pickle/joblib artifact is loaded.
        # Otherwise an incompatible runtime can fail inside unpickling with an opaque error before
        # the service can report the actual environment mismatch.
        self._validate_environment_contract(strict_environment)

        recorded_code_hash = str(self.manifest.get("serving_code_sha256", ""))
        actual_code_hash = package_source_sha256()
        if len(recorded_code_hash) != 64 or recorded_code_hash != actual_code_hash:
            raise RuntimeError(
                "Runtime serving code differs from artifact manifest; install the exact project "
                "source or wheel used to create the artifact bundle"
            )
        recorded_serving_config_hash = str(self.manifest.get("serving_config_sha256", ""))
        try:
            actual_serving_config_hash = serving_config_sha256(
                dict(self.manifest.get("config", {}))
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Manifest serving configuration is incomplete") from exc
        if (
            len(recorded_serving_config_hash) != 64
            or recorded_serving_config_hash != actual_serving_config_hash
        ):
            raise RuntimeError("Manifest serving configuration hash is missing or inconsistent")

        fingerprint = str(self.manifest.get("model_fingerprint", ""))
        model_hashes = self.manifest.get("model_artifact_hashes", {})
        if not isinstance(model_hashes, dict) or not model_hashes:
            raise RuntimeError("Manifest does not contain model artifact hashes")
        if model_hashes != serving_hashes:
            raise RuntimeError("Model artifact hashes differ from serving artifact hashes")
        if isinstance(full_hashes, dict):
            inconsistent = {
                path: value
                for path, value in serving_hashes.items()
                if full_hashes.get(path) != value
            }
            if inconsistent:
                raise RuntimeError("Serving artifact hashes are inconsistent with release hashes")
        expected_fingerprint = model_fingerprint(
            config_sha256=recorded_serving_config_hash,
            artifact_hashes_value=model_hashes,
            serving_code_sha256=recorded_code_hash,
        )
        if len(fingerprint) != 64 or fingerprint != expected_fingerprint:
            raise RuntimeError("Manifest model fingerprint is missing or inconsistent")
        self.model_version = fingerprint[:12]
        generation_payload = read_generation_metadata(self.artifact_dir)
        self.release_generation = (
            str(generation_payload["generation"]) if generation_payload is not None else None
        )

        self.products = pd.read_csv(self.artifact_dir / "release_catalog.csv")
        self.behavior_snapshot = pd.read_csv(self.artifact_dir / "product_behavior_snapshot.csv")
        self.bm25 = joblib.load(self.artifact_dir / "bm25.joblib")
        self.dense = joblib.load(self.artifact_dir / "dense.joblib")
        self.lambdamart = LambdaMARTRanker.load(
            self.artifact_dir / "lambdamart_model.json",
            self.artifact_dir / "lambdamart_metadata.json",
        )
        self.behavior_model = joblib.load(self.artifact_dir / "behavior_model.joblib")
        self.qrsbt = joblib.load(self.artifact_dir / "qrsbt.joblib")
        self.qrsbt.clear_runtime_cache()
        self._validate_runtime_contract()
        self.product_index = self.products.set_index("product_id", drop=False)
        self.snapshot_index = self.behavior_snapshot.set_index("product_id", drop=False)
        self.config = dict(self.manifest["config"])
        self.gate_config = _gate_config_from_manifest(self.config)
        self.cutoff_block = int(self.manifest.get("test_block", 0))
        self.candidate_k = int(self.config["retrieval"]["candidate_k"])
        self.max_result_k = int(self.config["retrieval"]["final_k"])
        self.behavior_reference = self.behavior_snapshot.copy()
        self.behavior_reference["time_block"] = self.cutoff_block
        self.prepared_behavior_reference = self.qrsbt.prepare_behavior_reference(
            self.behavior_reference
        )
        self._lock = threading.Lock()
        self._request_count = 0
        self._fallback_count = 0
        self._error_count = 0
        self._last_fallback_reason: str | None = None
        self._latencies_ms: deque[float] = deque(maxlen=1000)
        self._source_counts: Counter[str] = Counter()
        try:
            self.max_concurrency = int(os.getenv("PRODUCT_SEARCH_MAX_CONCURRENCY", "8"))
            self.admission_timeout_ms = float(
                os.getenv("PRODUCT_SEARCH_ADMISSION_TIMEOUT_MS", "50")
            )
        except ValueError as exc:
            raise RuntimeError("Serving concurrency settings must be numeric") from exc
        if self.max_concurrency < 1 or self.admission_timeout_ms < 0:
            raise RuntimeError("Serving concurrency settings are outside valid bounds")
        self._admission = threading.BoundedSemaphore(self.max_concurrency)
        self._active_requests = 0
        self._overload_rejections = 0
        try:
            self.test_admission_hold_ms = float(
                os.getenv("PRODUCT_SEARCH_TEST_ADMISSION_HOLD_MS", "0")
            )
        except ValueError as exc:
            raise RuntimeError("PRODUCT_SEARCH_TEST_ADMISSION_HOLD_MS must be numeric") from exc
        if self.test_admission_hold_ms < 0:
            raise RuntimeError("PRODUCT_SEARCH_TEST_ADMISSION_HOLD_MS must be nonnegative")

    def _validate_environment_contract(self, strict_environment: bool | None) -> None:
        strict = (
            os.getenv("PRODUCT_SEARCH_STRICT_ENV", "1") == "1"
            if strict_environment is None
            else strict_environment
        )
        if not strict:
            return
        expected_python = str(self.manifest.get("python", ""))
        actual_python = platform.python_version()
        expected_mm = ".".join(expected_python.split(".")[:2])
        actual_mm = ".".join(actual_python.split(".")[:2])
        expected_impl = str(self.manifest.get("python_implementation", ""))
        actual_impl = platform.python_implementation()
        if expected_mm != actual_mm or (expected_impl and expected_impl != actual_impl):
            raise RuntimeError(
                "Runtime Python differs from artifact manifest: "
                f"expected={expected_python}/{expected_impl}, actual={actual_python}/{actual_impl}; "
                "install the environment recorded in artifact_requirements.txt"
            )
        expected = self.manifest.get("package_versions", {})
        if not isinstance(expected, dict) or not expected:
            raise RuntimeError("Manifest does not contain runtime package versions")
        actual = package_versions(tuple(expected))
        mismatches = {
            name: {"expected": version, "actual": actual.get(name)}
            for name, version in expected.items()
            if version not in {None, "not-installed"} and actual.get(name) != version
        }
        if mismatches:
            raise RuntimeError(
                "Runtime package versions differ from artifact manifest: "
                f"{mismatches}; install the versions in artifact_requirements.txt"
            )

    def _validate_runtime_contract(self) -> None:
        feature_path = self.artifact_dir / "behavior_features.json"
        exported_features = _read_json(feature_path)
        if exported_features != BEHAVIOR_FEATURES:
            raise RuntimeError("Exported behavior feature contract differs from package contract")
        if list(getattr(self.behavior_model, "features_", [])) != exported_features:
            raise RuntimeError("Behavior model feature order differs from exported contract")
        if list(getattr(self.lambdamart, "features_", [])) != BASE_RANK_FEATURES:
            raise RuntimeError("LambdaMART feature order differs from package contract")

        catalog_ids = set(self.products.product_id.astype(int))
        if self.products.product_id.duplicated().any():
            raise RuntimeError("Product catalog contains duplicate product IDs")
        for name, retriever in (("bm25", self.bm25), ("dense", self.dense)):
            ids = list(map(int, getattr(retriever, "product_ids_", [])))
            if len(ids) != len(set(ids)) or set(ids) != catalog_ids:
                raise RuntimeError(f"{name} index product IDs do not match the catalog")
        required_snapshot = {
            "product_id",
            "prior_impressions",
            "smoothed_ctr",
            "smoothed_purchase_rate",
        }
        missing_snapshot = sorted(required_snapshot - set(self.behavior_snapshot.columns))
        if missing_snapshot:
            raise RuntimeError(f"Behavior snapshot missing columns: {missing_snapshot}")
        snapshot_ids = self.behavior_snapshot.product_id.astype(int)
        if snapshot_ids.duplicated().any() or set(snapshot_ids) != catalog_ids:
            raise RuntimeError("Behavior snapshot product IDs do not match the catalog")

        qrsbt_products = getattr(self.qrsbt, "products_", None)
        if qrsbt_products is None:
            raise RuntimeError("Q-RSBT artifact does not contain a product contract")
        if "product_id" in qrsbt_products.columns:
            qrsbt_ids = set(qrsbt_products["product_id"].astype(int))
        else:
            qrsbt_ids = set(pd.Index(qrsbt_products.index).astype(int))
        if qrsbt_ids != catalog_ids:
            raise RuntimeError("Q-RSBT product contract does not match the catalog")

    @contextmanager
    def _admit(self):
        acquired = self._admission.acquire(timeout=self.admission_timeout_ms / 1000.0)
        if not acquired:
            with self._lock:
                self._overload_rejections += 1
            raise ServiceOverloadedError("search capacity is temporarily saturated")
        with self._lock:
            self._active_requests += 1
        if self.test_admission_hold_ms > 0:
            time.sleep(self.test_admission_hold_ms / 1000.0)
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
            self._admission.release()

    def _record(self, fallback_reason: str | None, source: str, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            self._source_counts[source] += 1
            self._latencies_ms.append(float(latency_ms))
            if fallback_reason is not None:
                self._fallback_count += 1
                self._last_fallback_reason = fallback_reason

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def metrics(self) -> dict[str, int | float | str | None | dict[str, int]]:
        with self._lock:
            total = self._request_count
            fallbacks = self._fallback_count
            latency = np.asarray(tuple(self._latencies_ms), dtype=float)
            return {
                "model_version": self.model_version,
                "release_generation": getattr(self, "release_generation", None),
                "requests_total": total,
                "active_requests": getattr(self, "_active_requests", 0),
                "max_concurrency": getattr(self, "max_concurrency", 0),
                "overload_rejections_total": getattr(self, "_overload_rejections", 0),
                "errors_total": self._error_count,
                "fallbacks_total": fallbacks,
                "fallback_rate": float(fallbacks / total) if total else 0.0,
                "last_fallback_reason": self._last_fallback_reason,
                "latency_p50_ms": float(np.quantile(latency, 0.50)) if len(latency) else 0.0,
                "latency_p95_ms": float(np.quantile(latency, 0.95)) if len(latency) else 0.0,
                "source_counts": dict(self._source_counts),
            }

    def search(self, query: str, k: int) -> SearchResponse:
        started = time.perf_counter()
        normalized = " ".join(query.split()).strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        if len(normalized) > 300:
            raise ValueError("query must contain at most 300 characters")
        if not 1 <= int(k) <= self.max_result_k:
            raise ValueError(
                f"k must be between 1 and the released top-k contract ({self.max_result_k})"
            )
        with self._admit():
            fallback_reason: str | None = None
            try:
                ranked = self._rank_full_pipeline(normalized)
                source = "full_ranker"
            except Exception as exc:  # operational fallback is intentionally broad
                fallback_reason = f"full_ranker:{type(exc).__name__}"
                try:
                    ranked = self._hybrid_retrieval(normalized, max(k, 20))
                    ranked["score"] = ranked["retrieval_score"]
                    source = "hybrid_fallback"
                except Exception as dense_exc:
                    fallback_reason += f";hybrid:{type(dense_exc).__name__}"
                    ranked = self._bm25_retrieval(normalized, k)
                    ranked["score"] = ranked["bm25_score"]
                    ranked["dense_score"] = 0.0
                    ranked["semantic_rank_score"] = ranked["score"]
                    ranked["behavior_score"] = 0.0
                    ranked["gate_action"] = "FALLBACK"
                    ranked["gate_reason"] = "bm25_only"
                    ranked["zero_history"] = 0
                    ranked["qrsbt_confidence"] = 0.0
                    ranked["candidate_source"] = "bm25"
                    source = "bm25_fallback"

            ranked = ranked.sort_values(["score", "product_id"], ascending=[False, True]).head(k)
            self._record(fallback_reason, source, (time.perf_counter() - started) * 1000.0)
            results = []
            for row in ranked.itertuples(index=False):
                results.append(
                    SearchResult(
                        product_id=int(row.product_id),
                        title=str(row.title),
                        score=float(row.score),
                        bm25_score=float(getattr(row, "bm25_score", 0.0)),
                        dense_score=float(getattr(row, "dense_score", 0.0)),
                        semantic_score=float(getattr(row, "semantic_rank_score", 0.0)),
                        behavior_score=float(getattr(row, "behavior_score", 0.0)),
                        source=str(getattr(row, "candidate_source", source)),
                        gate_action=str(getattr(row, "gate_action", "FALLBACK")),
                        gate_reason=str(getattr(row, "gate_reason", source)),
                        cold_start=bool(getattr(row, "zero_history", 0)),
                        qrsbt_confidence=float(getattr(row, "qrsbt_confidence", 0.0)),
                    )
                )
            return SearchResponse(
                query=normalized,
                results=results,
                fallback_used=fallback_reason is not None,
                fallback_reason=fallback_reason,
                model_version=self.model_version,
            )

    def _bm25_retrieval(self, query: str, k: int) -> pd.DataFrame:
        scores = np.asarray(self.bm25.score(query), dtype=float)
        ids = np.asarray(self.bm25.product_ids_, dtype=int)
        if len(scores) != len(ids):
            raise RuntimeError("BM25 fallback score and product-ID lengths differ")
        available = self.products.launch_block.astype(int).to_numpy() <= self.cutoff_block
        catalog_ids = self.products.product_id.astype(int).to_numpy()
        availability = dict(zip(catalog_ids, available, strict=True))
        valid = np.fromiter(
            (bool(availability.get(int(pid), False)) for pid in ids),
            dtype=bool,
            count=len(ids),
        )
        order = np.argsort(np.where(valid, -scores, np.inf), kind="stable")
        order = [int(pos) for pos in order if valid[pos]][:k]
        frame = pd.DataFrame(
            {
                "product_id": ids[order],
                "bm25_score": scores[order],
            }
        )
        return frame.merge(
            self.products[["product_id", "title"]],
            on="product_id",
            how="left",
            validate="one_to_one",
        )

    def _hybrid_retrieval(self, query: str, candidate_k: int) -> pd.DataFrame:
        frame = hybrid_retrieve(
            query,
            products=self.products,
            bm25=self.bm25,
            dense=self.dense,
            candidate_k=candidate_k,
            cutoff_block=self.cutoff_block,
        )
        return frame.merge(
            self.products[
                [
                    "product_id",
                    "title",
                    "brand",
                    "category",
                    "attribute",
                    "model",
                    "price",
                    "quality",
                    "launch_block",
                ]
            ],
            on="product_id",
            how="left",
            validate="one_to_one",
        )

    def _rank_full_pipeline(self, query: str) -> pd.DataFrame:
        frame = self._hybrid_retrieval(query, self.candidate_k)
        if frame.empty:
            raise RuntimeError("retrieval returned no candidates")
        top = frame.sort_values(["retrieval_score", "product_id"], ascending=[False, True]).iloc[0]
        query_category = str(top.category)
        target_product_id = int(top.product_id)
        query_intent = _infer_intent(query, top)
        query_id = -int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:4], "big") - 1

        frame = frame.merge(
            self.behavior_snapshot,
            on="product_id",
            how="left",
            validate="one_to_one",
        )
        frame["time_block"] = self.cutoff_block
        frame["user_id"] = -1
        frame["query_id"] = query_id
        frame["query_text"] = query
        frame["query_intent"] = query_intent
        frame["query_category"] = query_category
        frame["query_anchor_product_id"] = target_product_id
        frame["position"] = 0
        frame["clicked"] = 0
        frame["purchased"] = 0
        frame["logging_propensity"] = 1.0
        frame["price_log"] = np.log1p(frame.price.astype(float))
        frame = self.qrsbt.transfer(frame, prepared_reference=self.prepared_behavior_reference)
        _require_features(frame, BASE_RANK_FEATURES, "semantic ranker")
        frame["semantic_rank_score"] = self.lambdamart.predict(frame)
        _require_features(frame, BEHAVIOR_FEATURES, "behavior ranker")
        frame["behavior_score"] = self.behavior_model.predict_proba(frame)[:, 1]
        frame = apply_coverage_overreach_gate(frame, self.gate_config)
        frame["score"] = frame["final_score"]
        return frame


def _require_features(frame: pd.DataFrame, features: list[str], stage: str) -> None:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise RuntimeError(f"{stage} missing features: {missing}")


def _infer_intent(query: str, top_product: pd.Series) -> str:
    tokens = set(query.lower().replace("_", " ").split())
    model = str(top_product.model).lower()
    brand = str(top_product.brand).lower()
    attribute_tokens = set(str(top_product.attribute).lower().split())
    category_tokens = set(str(top_product.category).lower().replace("_", " ").split())
    if model not in {"", "unknown", "nan"} and model in query.lower():
        return "exact_model"
    if tokens & attribute_tokens:
        return "attribute"
    if brand not in {"", "unknown", "nan"} and brand in query.lower() and tokens & category_tokens:
        return "brand_category"
    if tokens & {"case", "cover", "charger", "strap", "replacement"}:
        return "accessory"
    return "generic"


def _gate_config_from_manifest(config: dict) -> GateConfig:
    qcfg = config["qrsbt"]
    rcfg = config["ranking"]
    return GateConfig(
        semantic_threshold=float(qcfg["semantic_threshold"]),
        confidence_threshold=float(qcfg["confidence_threshold"]),
        compatibility_threshold=float(qcfg["compatibility_threshold"]),
        irrelevant_risk_threshold=float(qcfg["irrelevant_risk_threshold"]),
        min_support=int(qcfg["min_support"]),
        max_boost=float(qcfg["max_boost"]),
        semantic_weight=float(rcfg["semantic_weight"]),
        behavior_weight=float(rcfg["behavior_weight"]),
        top_k=int(config["retrieval"]["final_k"]),
        promotion_window=int(qcfg["promotion_window"]),
        max_promotions_per_query=int(qcfg["max_promotions_per_query"]),
        promotion_mode=str(qcfg.get("promotion_mode", "in_window")),
    )


@lru_cache(maxsize=1)
def get_service() -> SearchService:
    configured_artifact = os.getenv("PRODUCT_SEARCH_ARTIFACT_DIR")
    release_root = os.getenv("PRODUCT_SEARCH_RELEASE_ROOT")
    if configured_artifact:
        artifact_dir = Path(configured_artifact)
    elif release_root:
        artifact_dir = resolve_current_release(Path(release_root))
    else:
        artifact_dir = Path("artifacts/smoke")
    if not artifact_dir.exists():
        raise RuntimeError(f"Artifact directory does not exist: {artifact_dir}")
    verify = os.getenv("PRODUCT_SEARCH_VERIFY_ARTIFACTS", "1") != "0"
    return SearchService(artifact_dir, verify_hashes=verify)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search_service = get_service()
    yield
    app.state.search_service = None


def service_dependency(request: Request) -> SearchService:
    service = getattr(request.app.state, "search_service", None)
    return service if service is not None else get_service()


ServiceDependency = Annotated[SearchService, Depends(service_dependency)]


def create_app(*, preload_artifacts: bool = True) -> FastAPI:
    application = FastAPI(
        title="Cold-Start Product Search Reliability API",
        version="0.6.0",
        lifespan=lifespan if preload_artifacts else None,
    )

    @application.get("/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/ready")
    @application.get("/health")
    def health(service: ServiceDependency) -> dict[str, str]:
        payload = {
            "status": "ok",
            "model_version": service.model_version,
            "release_status": str(service.manifest.get("release_status", "unknown")),
        }
        generation = getattr(service, "release_generation", None)
        if generation is not None:
            payload["release_generation"] = generation
        return payload

    @application.get("/model_manifest")
    def model_manifest(service: ServiceDependency) -> dict:
        if os.getenv("PRODUCT_SEARCH_EXPOSE_FULL_MANIFEST", "0") == "1":
            manifest_payload = service.manifest
        else:
            public_fields = (
                "artifact_schema_version",
                "release_status",
                "behavior_champion",
                "release_catalog_products",
                "test_block",
                "future_audit_block",
                "model_fingerprint",
            )
            manifest_payload = {
                key: service.manifest[key] for key in public_fields if key in service.manifest
            }
        return {
            "model_version": service.model_version,
            "release_generation": getattr(service, "release_generation", None),
            "manifest": manifest_payload,
        }

    @application.get("/metrics")
    def metrics(service: ServiceDependency) -> dict:
        return service.metrics()

    @application.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest, service: ServiceDependency) -> SearchResponse:
        try:
            return service.search(request.query, request.k)
        except ServiceOverloadedError as exc:
            raise HTTPException(status_code=503, detail="search capacity unavailable") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            service.record_error()
            LOGGER.exception("Search request failed")
            raise HTTPException(status_code=500, detail="search failed") from exc

    @application.post("/batch_search", response_model=BatchSearchResponse)
    def batch_search(
        request: BatchSearchRequest, service: ServiceDependency
    ) -> BatchSearchResponse:
        try:
            return BatchSearchResponse(
                responses=[service.search(query, request.k) for query in request.queries]
            )
        except ServiceOverloadedError as exc:
            raise HTTPException(status_code=503, detail="search capacity unavailable") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            service.record_error()
            LOGGER.exception("Batch search request failed")
            raise HTTPException(status_code=500, detail="batch search failed") from exc

    return application


app = create_app()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
