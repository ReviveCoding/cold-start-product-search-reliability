# Known Limitations and Qualification Boundary

## Release blockers

There are no known locally executable Critical or High release-candidate blockers.
This statement applies only to the candidate gate and does not mean the project is
fully production-qualified.

## Required next qualification work

| Severity | Item | Current evidence | Operational impact | Required resolution |
|---|---|---|---|---|
| Medium | Exact-snapshot GitHub-hosted CI | Workflows inspected locally; E4 not executed | Cross-runner and repository-policy behavior is not yet proven | Push the exact candidate snapshot and require every hosted job to pass |
| Medium | GitHub artifact attestations | Release workflow configured; no hosted release run | Hosted build provenance is not yet available to consumers | Publish a release and verify wheel/sdist attestations |
| Medium | Docker runtime | Dockerfile and CI job exist; no local daemon | Image construction and container health are not locally proven | Build image, start container and exercise `/live`, `/ready`, `/search` |
| Medium | Native Windows | Windows workflow configured; no Windows host in this environment | Native path/process behavior remains E4-only | Run canonical setup, tests, pipeline and API on Windows Python 3.11 |

## Scientific and data limitations

- Candidate metrics use deterministic synthetic/small canonical data. They validate
  correctness and integration, not production demand, causal business lift or live
  user behavior.
- `launch_block` may represent first observed time rather than true commercial
  launch time for adapted public data.
- The OPE and dynamic simulator are controlled laboratories; conclusions depend on
  logging-policy support and simulation assumptions.
- Full KuaiSearch/ESCI-scale execution, multiple real-domain datasets and online A/B
  testing were not part of the candidate gate.
- Warm NDCG is lower than baseline, although it remains inside the frozen point and
  confidence-interval non-inferiority gates. This tradeoff must remain visible.

## Runtime and deployment limitations

- Workers resolve `current.json` at startup and do not hot-reload. Publish or
  rollback requires a rolling process restart.
- Admission capacity and `/metrics` are process-local; aggregate multi-worker
  capacity and monitoring require an external proxy/metrics backend.
- Atomic publication assumes a single local filesystem. It is not a distributed
  lock, remote object-store transaction or consensus protocol.
- Automatic generation retention, remote signing-key management and Kubernetes
  rollout are intentionally not implemented because they require operator-specific
  infrastructure and policy.
- The local server has bounded admission and integrity checks, but no bundled auth,
  TLS termination, rate-limiting proxy or centralized audit sink.

## Advanced-path limitations

- GPU/DCN code is optional; the candidate gate does not claim a current CUDA/GPU
  benchmark.
- Qwen and FAISS paths are opt-in and were not downloaded or scale-qualified.
- Million-item retrieval memory, throughput and tail latency remain unmeasured.

## Workarounds

- Use `constraints/validated.txt` for every candidate-validation install.
- Use `scripts/manage_release.py` and rolling restarts for local publish/rollback.
- Run `scripts/validate_candidate_handoff.py` before distributing the candidate.
- Treat the source ZIP, enhanced ZIP, wheel, sdist and handoff SHA sidecars as one
  evidence set; do not detach claims from their checksums.

## Intentionally not implemented

- Runtime hot reload
- Distributed release coordination
- Automatic artifact retention policy
- Proprietary data connectors or paid APIs
- Production authentication and network perimeter
- Claims of RELEASE QUALIFIED status

## Source-only handoff validation order

The source ZIP excludes `dist/` by design. Use `scripts/validate_candidate_handoff.py --allow-missing-build-artifacts` before the wheel/sdist are built, then rerun strict validation without the flag after `scripts/build_release.py --output-dir dist`. This keeps source archives clean without weakening final artifact checksum validation.
