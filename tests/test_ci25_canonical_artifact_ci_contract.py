from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_artifact_ci_contract() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    transport_path = ROOT / "scripts" / "artifact_transport_validation.py"
    transport = transport_path.read_text(encoding="utf-8")
    integration = (ROOT / "scripts" / "integration_validation.py").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    ast.parse(transport, filename=str(transport_path))
    workflow_payload = yaml.safe_load(workflow)
    assert set(workflow_payload["jobs"]) == {
        "quality",
        "canonical-smoke",
        "portable-artifact-smoke",
        "docker",
    }

    canonical = workflow.split("  canonical-smoke:", 1)[1]
    canonical = canonical.split("  portable-artifact-smoke:", 1)[0]
    assert "Build canonical smoke artifact" in canonical
    assert "artifact_transport_validation.py" in canonical
    assert "--mode canonical-serving --allow-nonlaunch" in canonical
    assert "integration_validation.py" not in canonical

    portable = workflow.split("  portable-artifact-smoke:", 1)[1]
    portable = portable.split("  docker:", 1)[0]
    assert "actions/download-artifact@" in portable
    assert "artifact_transport_validation.py" in portable
    assert "--mode transport --allow-nonlaunch" in portable
    assert "run_full_pipeline.py" not in portable
    assert "reproducibility_check.py" not in portable
    assert "integration_validation.py" not in portable

    assert 'choices=("canonical-serving", "transport")' in transport
    assert 'if args.mode == "canonical-serving":' in transport
    assert "publish_release" not in transport
    assert "rollback_release" not in transport
    assert "SearchService" in transport
    assert "create_app" in transport

    assert "--allow-hold" not in integration
    assert "PRODUCT_SEARCH_ALLOW_NONLAUNCH" not in integration

    docker_job = workflow.split("  docker:", 1)[1]
    assert "PRODUCT_SEARCH_ALLOW_NONLAUNCH=1" in docker_job
    assert "COPY --chown=appuser:appuser artifacts/smoke ./artifacts/smoke" in dockerfile
    assert "RUN python scripts/run_full_pipeline.py" not in dockerfile
    assert "artifacts/*" in dockerignore
    assert "!artifacts/smoke/**" in dockerignore

def test_ci25_replay_uses_semantic_retriever_contract() -> None:
    replay = (ROOT / "scripts" / "reproducibility_check.py").read_text(
        encoding="utf-8"
    )

    assert "SEMANTIC_RETRIEVER_FILES" in replay
    assert "_semantic_retriever_audit" in replay
    assert "_bm25_semantic_fingerprint" in replay
    assert "_dense_semantic_fingerprint" in replay
    assert "raw_serialization_mismatches" in replay
    assert "filecmp" not in replay
