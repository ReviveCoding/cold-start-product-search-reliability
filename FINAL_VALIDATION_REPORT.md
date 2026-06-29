# Final Validation Report v0.6.0

## Validation summary

| Check | Result |
|---|---:|
| Release decision | **LAUNCH** |
| Release gates | **29 / 29** |
| Automated tests | **113 passed** |
| Isolated modules | **25 / 25** |
| Core coverage | **73%** |
| Ruff / compileall / pip check | PASS |
| Full staged pipeline | PASS |
| Static / future / dynamic / OPE evidence | PASS |
| Direct service / FastAPI parity | PASS |
| Real one-worker Uvicorn | PASS |
| Real two-worker Uvicorn | PASS |
| Real HTTP overload 503 | PASS |
| Atomic publish / failed-publish stability / rollback | PASS |
| Minimal serving closure | PASS |
| Independent replay | PASS |
| Deterministic package builds | PASS |

## Scientific evidence

| Metric | Result |
|---|---:|
| Base / final NDCG@10 | 0.89104 / **0.89226** |
| Cold NDCG@10 | 0.45735 / **0.46931** |
| Cold lift CI | **[0.00577, 0.02028]** |
| Warm NDCG@10 | 0.66619 / 0.65729 |
| Calibrated Brier / ECE | **0.13112 / 0.02499** |
| Future ROC-AUC / Brier / ECE | **0.77405 / 0.13805 / 0.03346** |
| Relation eligibility Brier / ECE | **0.000060 / 0.000528** |
| Dynamic utility | 3,450.32 / **3,534.44** |
| Worst-scenario mean utility delta | **+20.44** |
| Lower-tail utility delta | **+6.20** |
| DR absolute error / ESS | **0.01139 / 4,049** |

## Pipeline connectivity

1. canonical/synthetic contract validation;
2. strict temporal split;
3. block-local BM25 and dense fit;
4. frozen release catalog;
5. cutoff-safe behavior snapshot;
6. hybrid candidates and temporal anchors;
7. LambdaMART, calibrated behavior, and Q-RSBT;
8. bounded intervention and static evaluation;
9. untouched future audit;
10. multi-replication simulator;
11. OPE validity laboratory;
12. serving benchmark and 29 release gates;
13. manifest, source/config/model fingerprints, and hashes;
14. real ASGI requests and overload behavior;
15. immutable generation publication, pointer resolution, injected failure, and rollback;
16. independent replay and deterministic package build.

## Operational protections

- validated dependency constraints plus environment and fingerprint checks before Python-object deserialization;
- native XGBoost JSON for LambdaMART;
- complete-release and serving-closure hash contracts;
- immutable generation directories and atomic pointer replacement;
- no pointer change or orphan generation on injected publication failure;
- rollback revalidation;
- bounded per-worker admission and retryable 503 overload response;
- one-worker exact metrics and two-worker model consistency validation;
- sanitized manifest endpoint;
- process-isolated pipeline and test shards;
- full-SHA GitHub action pins, dependency review, Dependabot, and release attestation workflow configuration.

## Explicit limitations

- active workers do not hot-reload `current.json`; publish/rollback requires rolling restart;
- metrics and admission limits are process-local, so total capacity is worker count times per-worker capacity;
- local atomicity assumes one filesystem and does not provide distributed consensus;
- hosted GitHub Actions and attestations are configured but not executed locally;
- Docker, native Windows, full public datasets, GPU adaptation, and live traffic are not claimed unless separately executed.
