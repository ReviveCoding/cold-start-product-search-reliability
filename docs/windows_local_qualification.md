# Windows Local Qualification

## Final status

**CONDITIONALLY QUALIFIED â€” Windows local P00-P14 chain PASS**

- Run ID: p11-cli-sync-v1
- Final stage: P14_FINALIZE
- Final controller state: PASSED
- Release decision: LAUNCH
- Strict handoff: PASS
- Independent wheel smoke: PASS

## Verified scope

- constrained dependencies and editable import-origin validation
- lint, compile, and full test runner
- synthetic smoke pipeline and release decision
- integration validation and deterministic replay
- policy sensitivity evaluation
- strict inference and Uvicorn service smoke
- reproducible build and strict handoff validation
- isolated wheel clean-install, pip check, and strict search smoke

The independent wheel smoke imported the installed package from:

C:\\p9b\\p14ws\\p11-cli-sync-v1\\Lib\\site-packages\\product_search\\__init__.py

It returned 5 search results with fallback_used = False.

## Explicitly unqualified scope

- GitHub-hosted workflow execution and artifact attestation
- Docker build and container health validation
- hosted Windows runner parity
- optional GPU/Qwen/FAISS paths
- external-scale public benchmark validation
- production A/B test, CTR, conversion, or revenue lift