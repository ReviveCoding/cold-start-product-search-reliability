# Evidence-Grounded Improvement Report

## Scope and verdict

This report covers the candidate-handoff round performed on the supplied
`cold-start-product-search-reliability` v0.6.0 enhanced archive. The round used
`MODIFY_AND_RUN` / locally available full-tool execution and an `EXTENDED`
candidate-validation profile. The result is a **release candidate handoff**, not a
claim of final hosted production qualification.

The supplied archive had no `.git` directory. Before modifying it, the process
recorded 165 file hashes and aggregate source fingerprint
`80ef73377249f4cbaaf0f8abe3fd3e436ae9dc5b06c254ede58f6b435ff8b7d1`.
The supplied source and enhanced ZIP checksums matched the uploaded release
manifest.

## Frozen candidate gates

No model metric, split, threshold, or release gate was changed in this round.
The frozen gate remained the existing 29-gate policy in `configs/smoke.yaml`.
The primary quality direction remained higher overall/cold NDCG and dynamic
utility, with warm-ranking and exposure regressions constrained by the existing
point and confidence-interval non-inferiority gates.

Required candidate entrypoints:

- `python scripts/run_full_pipeline.py --config configs/smoke.yaml`
- `python scripts/run_tests.py`
- `python scripts/integration_validation.py --config configs/smoke.yaml --skip-pipeline`
- `python scripts/reproducibility_check.py --config configs/smoke.yaml`
- `python scripts/verify_build_reproducibility.py`
- `PRODUCT_SEARCH_ARTIFACT_DIR=artifacts/smoke python scripts/serve.py`

## Phase 0 repository and pipeline map

| Stage | Implementation | Input contract | Output contract | Failure behavior | Tests/evidence |
|---|---|---|---|---|---|
| Input and config | `config.py`, `data/contracts.py`, `data/bundle.py`, adapters | Canonical `products`, `queries`, `relevance`, `interactions`; schema 6.0 config | Validated temporal bundle | Explicit schema/key/time errors | Config, data, bundle, adapter tests |
| Temporal preprocessing | `pipeline.py`, `features.py` | Validated bundle and scoring blocks | Train/calibration/test/future frames and cutoff-safe features | Leakage and cutoff assertions fail closed | Temporal leakage/context tests |
| Retrieval | `retrieval/bm25.py`, `dense.py`, `hybrid.py`, `candidates.py` | Query and block-local catalog | Ranked candidate union | Empty/malformed inputs produce explicit errors or controlled fallback | Retrieval, dense, hybrid, candidate tests |
| Ranking and behavior | `ranking/lambdamart.py`, `behavior.py` | Candidate features and behavior frames | Native JSON ranker and calibrated behavior scores | Model/version/hash checks before runtime load | Persistence, provenance, pipeline tests |
| Q-RSBT policy | `substitutes/qrsbt.py`, `policy/gate.py` | Semantic candidates, substitute support, uncertainty | Bounded final rank/boost decision | Support/risk/movement guardrails block unsafe transfer | Q-RSBT and policy tests |
| Evaluation | `evaluation/metrics.py`, `sensitivity.py`, `simulation/dynamic.py`, `ope/estimators.py` | Baseline/final rankings and logged/simulated outcomes | Static, future, dynamic, calibration, OPE and sensitivity evidence | Undefined/invalid contracts raise rather than silently coerce | Metrics, dynamic, OPE, sensitivity tests |
| Release decision | `policy/release.py`, pipeline orchestration | Frozen metrics and thresholds | `release_decision.json`, LAUNCH/HOLD | Any failed required gate prevents LAUNCH | Release tests and full pipeline |
| Artifact publication | `provenance.py`, `release_store.py` | LAUNCH artifact and serving hash closure | Immutable generation and atomic `current.json` | Hash/runtime validation before pointer change; failed promotion cleanup | Provenance and release-store tests |
| Runtime loading and API | `serving/app.py`, `scripts/serve.py` | Verified artifact directory or release pointer | Search/batch API, readiness, process-local metrics | Strict load, bounded admission, HTTP 503 overload | API and real Uvicorn integration |
| Packaging and handoff | `release_build.py`, `release_packaging.py`, `candidate_handoff.py` | Source snapshot, command evidence, artifacts | Deterministic wheel/sdist/ZIP and verified candidate handoff | Missing paths, failed commands, or checksum mismatch fail validation | Build, packaging and handoff tests |

## Baseline evidence before modification

The unchanged archive was installed into a clean Python 3.13 virtual
environment with `constraints/validated.txt`.

| Check | Baseline result |
|---|---:|
| Dependency installation / `pip check` | PASS |
| Test collection | PASS |
| Full tests | 109 passed, 25 isolated modules |
| Core coverage | 73% |
| Ruff / compileall | PASS |
| Integration | PASS |
| Full pipeline | LAUNCH, 29/29 gates |
| Independent replay | PASS, 16 compared files |
| Reproducible wheel/sdist | PASS |
| Real Uvicorn single/multi-worker and overload | PASS |

## Prioritized gap analysis

| Priority | Severity | Gap | Evidence | Resolution |
|---:|---|---|---|---|
| 16.0 | High | Required `release_candidate_handoff.json` did not exist | Archive inspection | Added deterministic evidence-grounded generator and checksum validator |
| 12.0 | High | Required `docs/improvement_report.md` and `docs/known_limitations.md` did not exist | Archive inspection | Added explicit candidate reports matching the handoff contract |
| 8.0 | Medium | Handoff paths and hashes could otherwise be copied manually without validation | No existing handoff schema/test | Added validation that fails on missing/tampered dependency or build artifacts and failed command evidence |
| 4.0 | Medium | Source archive inclusion of the handoff was not regression-tested | Packaging tests covered ordinary files/artifacts only | Added source-ZIP handoff inclusion test |

No executable Critical issue or additional model/retrieval/runtime High issue was
found during this round. The existing project already covered failure-oriented
publication, corrupted artifacts, concurrency, overload, replay and dependency
drift.

## Improvement round

### Problem and root cause

The repository had comprehensive release evidence but no canonical machine-readable
candidate handoff matching the requested schema. A manually assembled manifest
would be vulnerable to stale paths, stale checksums, hidden failed commands and
accidental `RELEASE_QUALIFIED` wording.

### Minimal implementation

Added:

- `src/product_search/candidate_handoff.py`
- `scripts/generate_candidate_handoff.py`
- `scripts/validate_candidate_handoff.py`
- `tests/test_candidate_handoff.py`
- candidate-handoff packaging regression coverage
- `docs/improvement_report.md`
- `docs/known_limitations.md`
- root `release_candidate_handoff.json` generated after final evidence collection

The source fingerprint intentionally excludes the generated handoff and generated
artifacts to avoid self-referential hashing. The handoff receives its own external
SHA-256 sidecar in the final delivery.

### Acceptance criteria

- Existing frozen 29 gates remain unchanged and pass.
- All existing tests plus new handoff tests pass.
- Handoff contains required fields and does not claim `RELEASE_QUALIFIED`.
- Every dependency/build-artifact path in the handoff exists and matches SHA-256.
- Every candidate-gating command embedded in the handoff has exit code 0.
- Source ZIP includes the handoff and remains byte-reproducible.

## Final candidate evidence

| Check | Candidate result |
|---|---:|
| Test collection | 113 tests |
| Full tests | 113 passed, 26 isolated modules |
| Core coverage | 73% |
| Ruff / compileall / pip check | PASS |
| End-to-end pipeline | LAUNCH, 29/29 gates |
| Real Uvicorn integration | Single-worker, two-worker consistency and overload 503 PASS |
| Independent replay | PASS, 16 files compared |
| Wheel/sdist reproducibility | PASS, byte-identical |
| Candidate handoff checksum/path validation | PASS |

The four additional tests cover source-fingerprint scope, deterministic file-level diff classification, checksum failure behavior, and source-ZIP inclusion of the candidate handoff.

## Scientific and regression assessment

This round did not change data, model features, split logic, labels, thresholds or
metric definitions. Therefore the expected model metrics are identical to the
validated v0.6.0 evidence. The final run must still independently satisfy all 29
frozen gates, including positive cold lift confidence interval, warm
non-inferiority, calibration, future behavior, lower-tail dynamic utility, OPE
validity, serving fallback and artifact integrity gates.

## Stop condition

After the handoff implementation and validation, two re-analysis passes identified
only external/platform qualification items or low-ROI optional features. No
Critical issue remains, no executable High issue remains, and further local feature
work would add complexity without improving the candidate gate. Candidate
improvement therefore stops here; hosted operational qualification remains a
separate phase.

## Final-loop source-only handoff preflight improvement

A final usability gap remained: source ZIPs intentionally exclude `dist/`, while strict `release_candidate_handoff.json` validation checks wheel and sdist records. The validation CLI now supports `--allow-missing-build-artifacts` for source-only preflight. This mode still verifies dependency manifests and command evidence, reports missing build artifacts explicitly, and leaves final release validation strict by default.
