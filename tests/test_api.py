from fastapi.testclient import TestClient

from product_search.serving.app import (
    SearchService,
    SearchResponse,
    SearchResult,
    create_app,
    service_dependency,
)


class _FakeService:
    model_version = "test-version"
    manifest = {
        "artifact_schema_version": "6.0",
        "release_status": "LAUNCH",
        "behavior_champion": "test",
        "model_fingerprint": "a" * 64,
        "package_versions": {"secret-package": "9.9"},
        "advanced_challenger_command": "secret internal command",
    }

    def search(self, query: str, k: int) -> SearchResponse:
        return SearchResponse(
            query=query,
            results=[
                SearchResult(
                    product_id=index,
                    title=f"Product {index}",
                    score=1.0 - index / 100,
                    bm25_score=0.5,
                    dense_score=0.6,
                    semantic_score=0.7,
                    behavior_score=0.4,
                    source="full_ranker",
                    gate_action="NATIVE",
                    gate_reason="native_history",
                    cold_start=False,
                    qrsbt_confidence=0.0,
                )
                for index in range(k)
            ],
            fallback_used=False,
            model_version=self.model_version,
        )

    def metrics(self):
        return {"requests_total": 0, "fallbacks_total": 0, "fallback_rate": 0.0}

    def record_error(self):
        return None


def _client() -> TestClient:
    app = create_app(preload_artifacts=False)
    app.dependency_overrides[service_dependency] = lambda: _FakeService()
    return TestClient(app)


def test_api_routes_and_contracts():
    with _client() as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/health").status_code == 200
        single = client.post("/search", json={"query": "wireless headphones", "k": 3})
        assert single.status_code == 200
        assert len(single.json()["results"]) == 3
        batch = client.post(
            "/batch_search",
            json={"queries": ["wireless headphones", "trail shoes"], "k": 2},
        )
        assert batch.status_code == 200
        assert len(batch.json()["responses"]) == 2
        manifest_response = client.get("/model_manifest")
        assert manifest_response.status_code == 200
        public_manifest = manifest_response.json()["manifest"]
        assert public_manifest["release_status"] == "LAUNCH"
        assert "package_versions" not in public_manifest
        assert "advanced_challenger_command" not in public_manifest
        assert client.get("/metrics").status_code == 200


def test_api_rejects_blank_query():
    with _client() as client:
        response = client.post("/search", json={"query": "", "k": 3})
        assert response.status_code == 422


def test_api_rejects_whitespace_only_queries():
    with _client() as client:
        assert client.post("/search", json={"query": "   ", "k": 3}).status_code == 422
        assert (
            client.post("/batch_search", json={"queries": ["valid", "   "], "k": 3}).status_code
            == 422
        )


def test_api_does_not_expose_internal_exception_details():
    class _FailingService(_FakeService):
        def search(self, query: str, k: int) -> SearchResponse:
            raise RuntimeError("secret local artifact path")

    app = create_app(preload_artifacts=False)
    app.dependency_overrides[service_dependency] = lambda: _FailingService()
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "headphones", "k": 3})
    assert response.status_code == 500
    assert response.json()["detail"] == "search failed"
    assert "secret" not in response.text


def test_service_rejects_k_beyond_released_top_k_before_scoring():
    service = SearchService.__new__(SearchService)
    service.max_result_k = 10
    try:
        service.search("headphones", 11)
    except ValueError as exc:
        assert "released top-k contract" in str(exc)
    else:
        raise AssertionError("Expected oversized k to be rejected")


def test_bm25_fallback_respects_availability_cutoff():
    import numpy as np
    import pandas as pd

    class _BM25:
        product_ids_ = np.array([1, 2], dtype=int)

        def score(self, query: str):
            return np.array([0.1, 10.0], dtype=float)

    service = SearchService.__new__(SearchService)
    service.bm25 = _BM25()
    service.cutoff_block = 2
    service.products = pd.DataFrame(
        {
            "product_id": [1, 2],
            "title": ["available", "future"],
            "launch_block": [1, 3],
        }
    )
    result = service._bm25_retrieval("anything", 2)
    assert result.product_id.tolist() == [1]


def test_batch_api_rejects_oversized_individual_query():
    with _client() as client:
        response = client.post(
            "/batch_search",
            json={"queries": ["x" * 301], "k": 3},
        )
    assert response.status_code == 422


def test_direct_service_rejects_oversized_query_before_scoring():
    service = SearchService.__new__(SearchService)
    service.max_result_k = 10
    try:
        service.search("x" * 301, 3)
    except ValueError as exc:
        assert "at most 300" in str(exc)
    else:
        raise AssertionError("Expected oversized query to be rejected")


def test_model_manifest_full_payload_requires_explicit_debug_opt_in(monkeypatch):
    monkeypatch.setenv("PRODUCT_SEARCH_EXPOSE_FULL_MANIFEST", "1")
    with _client() as client:
        payload = client.get("/model_manifest").json()["manifest"]
    assert payload["package_versions"] == {"secret-package": "9.9"}
    assert payload["advanced_challenger_command"] == "secret internal command"


def test_get_service_resolves_atomic_release_pointer(monkeypatch, tmp_path):
    import product_search.serving.app as serving_app

    generation = tmp_path / "generations" / "g1"
    generation.mkdir(parents=True)
    captured = {}

    class _ResolvedService:
        def __init__(self, artifact_dir, *, verify_hashes=True):
            captured["artifact_dir"] = artifact_dir
            captured["verify_hashes"] = verify_hashes

    monkeypatch.delenv("PRODUCT_SEARCH_ARTIFACT_DIR", raising=False)
    monkeypatch.setenv("PRODUCT_SEARCH_RELEASE_ROOT", str(tmp_path))
    monkeypatch.setattr(serving_app, "resolve_current_release", lambda root: generation)
    monkeypatch.setattr(serving_app, "SearchService", _ResolvedService)
    serving_app.get_service.cache_clear()
    try:
        service = serving_app.get_service()
        assert isinstance(service, _ResolvedService)
        assert captured["artifact_dir"] == generation
        assert captured["verify_hashes"] is True
    finally:
        serving_app.get_service.cache_clear()


def test_admission_control_rejects_saturated_capacity():
    import threading

    from product_search.serving.app import SearchService, ServiceOverloadedError

    service = SearchService.__new__(SearchService)
    service.max_concurrency = 1
    service.admission_timeout_ms = 0.0
    service._admission = threading.BoundedSemaphore(1)
    service._lock = threading.Lock()
    service._active_requests = 0
    service._overload_rejections = 0
    service.test_admission_hold_ms = 0.0

    with service._admit():
        assert service._active_requests == 1
        try:
            with service._admit():
                pass
        except ServiceOverloadedError:
            pass
        else:
            raise AssertionError("Expected saturated admission to be rejected")
    assert service._active_requests == 0
    assert service._overload_rejections == 1


def test_api_maps_admission_rejection_to_503():
    from product_search.serving.app import ServiceOverloadedError

    class _OverloadedService(_FakeService):
        def search(self, query: str, k: int):
            raise ServiceOverloadedError("saturated")

    app = create_app(preload_artifacts=False)
    app.dependency_overrides[service_dependency] = lambda: _OverloadedService()
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "headphones", "k": 3})
    assert response.status_code == 503
    assert response.json()["detail"] == "search capacity unavailable"
