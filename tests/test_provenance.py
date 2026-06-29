from pathlib import Path

import pytest

from product_search.provenance import artifact_hashes, verify_artifact_hashes


def test_artifact_integrity_detects_tampering(tmp_path: Path):
    path = tmp_path / "artifact.txt"
    path.write_text("original", encoding="utf-8")
    hashes = artifact_hashes(tmp_path, ["artifact.txt"])
    verify_artifact_hashes(tmp_path, hashes)
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash-mismatch"):
        verify_artifact_hashes(tmp_path, hashes)


def test_nonlaunch_artifact_is_rejected_by_default(tmp_path: Path):
    import json

    from product_search.provenance import ARTIFACT_SCHEMA_VERSION
    from product_search.serving.app import SearchService

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "release_status": "ITERATE",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Refusing to serve"):
        SearchService(tmp_path, verify_hashes=False)


def test_environment_mismatch_is_rejected_before_joblib_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json
    import platform

    import product_search.serving.app as serving_app
    from product_search.provenance import ARTIFACT_SCHEMA_VERSION

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "release_status": "LAUNCH",
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "package_versions": {"scikit-learn": "999.0"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        serving_app,
        "package_versions",
        lambda names: {"scikit-learn": "1.0"},
    )
    load_called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("joblib.load must not run before environment validation")

    monkeypatch.setattr(serving_app.joblib, "load", fail_if_loaded)
    with pytest.raises(RuntimeError, match="Runtime package versions differ"):
        serving_app.SearchService(
            tmp_path,
            verify_hashes=False,
            strict_environment=True,
        )
    assert not load_called


def test_artifact_integrity_rejects_parent_traversal(tmp_path: Path):
    outside = tmp_path.parent / "outside-artifact.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe-or-missing"):
        verify_artifact_hashes(tmp_path, {"../outside-artifact.txt": "0" * 64})


def test_artifact_hashing_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        artifact_hashes(tmp_path, ["link.txt"])


def test_model_fingerprint_ignores_runtime_metadata_and_tracks_model_hashes():
    from product_search.provenance import model_fingerprint

    hashes = {"model.json": "a" * 64, "catalog.csv": "b" * 64}
    first = model_fingerprint(
        config_sha256="c" * 64,
        artifact_hashes_value=hashes,
        serving_code_sha256="e" * 64,
    )
    second = model_fingerprint(
        config_sha256="c" * 64,
        artifact_hashes_value=dict(reversed(list(hashes.items()))),
        serving_code_sha256="e" * 64,
    )
    changed = model_fingerprint(
        config_sha256="c" * 64,
        artifact_hashes_value={**hashes, "model.json": "d" * 64},
        serving_code_sha256="e" * 64,
    )
    code_changed = model_fingerprint(
        config_sha256="c" * 64,
        artifact_hashes_value=hashes,
        serving_code_sha256="f" * 64,
    )
    assert first == second
    assert first != changed
    assert first != code_changed


def test_python_mismatch_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json

    import product_search.serving.app as serving_app
    from product_search.provenance import ARTIFACT_SCHEMA_VERSION

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "release_status": "LAUNCH",
                "python": "0.0.0",
                "python_implementation": "CPython",
                "package_versions": {"scikit-learn": "1.0"},
            }
        ),
        encoding="utf-8",
    )
    load_called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("model deserialization must not run before Python validation")

    monkeypatch.setattr(serving_app.joblib, "load", fail_if_loaded)
    monkeypatch.setattr(serving_app.pd, "read_csv", fail_if_loaded)
    with pytest.raises(RuntimeError, match="Runtime Python differs"):
        serving_app.SearchService(
            tmp_path,
            verify_hashes=False,
            strict_environment=True,
        )
    assert not load_called


def test_fingerprint_mismatch_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json
    import platform

    import product_search.serving.app as serving_app
    from product_search.provenance import ARTIFACT_SCHEMA_VERSION

    model_hashes = {"bm25.joblib": "a" * 64}
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "release_status": "LAUNCH",
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "package_versions": {"scikit-learn": "1.0"},
                "artifact_hashes": model_hashes,
                "serving_artifact_hashes": model_hashes,
                "model_artifact_hashes": model_hashes,
                "config": {
                    "config_schema_version": "6.0",
                    "retrieval": {"candidate_k": 50, "final_k": 10},
                    "ranking": {"semantic_weight": 0.65, "behavior_weight": 0.35},
                    "qrsbt": {"max_boost": 0.015},
                },
                "config_sha256": "a" * 64,
                "serving_config_sha256": "c" * 64,
                "serving_code_sha256": "e" * 64,
                "model_fingerprint": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        serving_app,
        "package_versions",
        lambda names: {"scikit-learn": "1.0"},
    )
    monkeypatch.setattr(serving_app, "package_source_sha256", lambda: "e" * 64)
    monkeypatch.setattr(serving_app, "serving_config_sha256", lambda config: "c" * 64)
    load_called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("model deserialization must not run before fingerprint validation")

    monkeypatch.setattr(serving_app.joblib, "load", fail_if_loaded)
    monkeypatch.setattr(serving_app.pd, "read_csv", fail_if_loaded)
    with pytest.raises(RuntimeError, match="model fingerprint"):
        serving_app.SearchService(
            tmp_path,
            verify_hashes=False,
            strict_environment=True,
        )
    assert not load_called


def test_runtime_metrics_track_sources_latency_fallbacks_and_errors():
    import threading
    from collections import Counter, deque

    from product_search.serving.app import SearchService

    service = SearchService.__new__(SearchService)
    service.model_version = "fingerprint"
    service._lock = threading.Lock()
    service._request_count = 0
    service._fallback_count = 0
    service._error_count = 0
    service._last_fallback_reason = None
    service._latencies_ms = deque(maxlen=1000)
    service._source_counts = Counter()

    service._record(None, "full_ranker", 10.0)
    service._record("full_ranker:RuntimeError", "hybrid_fallback", 30.0)
    service.record_error()
    metrics = service.metrics()

    assert metrics["requests_total"] == 2
    assert metrics["fallbacks_total"] == 1
    assert metrics["errors_total"] == 1
    assert metrics["fallback_rate"] == pytest.approx(0.5)
    assert metrics["latency_p50_ms"] == pytest.approx(20.0)
    assert metrics["latency_p95_ms"] == pytest.approx(29.0)
    assert metrics["source_counts"] == {"full_ranker": 1, "hybrid_fallback": 1}


def test_serving_config_hash_ignores_release_and_simulation_only_changes():
    from copy import deepcopy

    import yaml

    from product_search.provenance import serving_config_sha256

    config = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    changed = deepcopy(config)
    changed["release"]["max_serving_p95_ms"] = 999.0
    changed["simulation"]["replications"] = 99
    assert serving_config_sha256(config) == serving_config_sha256(changed)

    changed["qrsbt"]["max_boost"] = 0.01
    assert serving_config_sha256(config) != serving_config_sha256(changed)


def test_serving_code_mismatch_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json
    import platform

    import product_search.serving.app as serving_app
    from product_search.provenance import ARTIFACT_SCHEMA_VERSION

    model_hashes = {"bm25.joblib": "a" * 64}
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "release_status": "LAUNCH",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "package_versions": {"scikit-learn": "1.0"},
        "artifact_hashes": model_hashes,
        "serving_artifact_hashes": model_hashes,
        "model_artifact_hashes": model_hashes,
        "serving_code_sha256": "e" * 64,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        serving_app, "package_versions", lambda names: {"scikit-learn": "1.0"}
    )
    monkeypatch.setattr(serving_app, "package_source_sha256", lambda: "f" * 64)
    load_called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("model deserialization must not run before code validation")

    monkeypatch.setattr(serving_app.joblib, "load", fail_if_loaded)
    monkeypatch.setattr(serving_app.pd, "read_csv", fail_if_loaded)
    with pytest.raises(RuntimeError, match="serving code differs"):
        serving_app.SearchService(
            tmp_path, verify_hashes=False, strict_environment=True
        )
    assert not load_called
