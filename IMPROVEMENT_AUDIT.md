# Improvement Audit v0.6.0

This audit begins from the validated v0.5.0 baseline and records only newly discovered material gaps.

## Retained strengths

The prior version already provided strict temporal retrieval, Q-RSBT, calibration, untouched future evaluation, multi-replication simulation, OPE, full-ranker API parity, stable model persistence, source-code fingerprinting, deterministic packages, and clean-checkout validation. The v0.6.0 loop therefore concentrated on deployment operations and supply-chain boundaries.

## Loop 1: immutable generation publication and rollback

| Gap | Risk | Resolution | Evidence |
|---|---|---|---|
| A validated directory had no atomic promotion mechanism | Partially copied bundle or mixed model versions | Stage a minimal serving closure, fsync it, atomically rename the generation, then atomically replace `current.json` | Publish and pointer-resolution tests |
| No first-class rollback | Slow or unsafe recovery | Current pointer records the previous generation; rollback revalidates hashes, metadata, fingerprint, and runtime before pointer replacement | Publish A -> B -> rollback A integration |
| Concurrent publishers could race | Lost update or corrupt pointer | Atomic publish-lock directory with owner metadata | Concurrent writer rejection test |
| Failure after directory promotion but before pointer update could leave an orphan | Unreferenced deployment accumulation | Remove promoted destination and fsync parent unless pointer write completed | Injected post-promotion failure test |
| Full experiment directory was copied to serving store | Unverified logs and training frames in runtime unit | Copy only manifest plus `serving_artifact_hashes` closure | Exact deployed-file-set integration test |
| Generation metadata was outside the serving contract | Pointer or generation identity could be altered | Cross-check directory name, fingerprint, source manifest hash, and publication time | Metadata tamper tests |

## Loop 2: bounded serving and real ASGI validation

| Gap | Risk | Resolution | Evidence |
|---|---|---|---|
| CPU-heavy searches were accepted without a bound | Thread-pool saturation and latency collapse | Per-worker bounded semaphore with configurable wait timeout | Unit saturation test |
| Overload behavior was not part of HTTP contract | Clients receive slow 500/timeouts rather than retryable response | Map capacity rejection to HTTP 503 and record rejection metrics | Unit and real-Uvicorn overload tests |
| TestClient did not prove actual ASGI process behavior | Startup, worker, socket, and shutdown bugs could remain | Launch real Uvicorn subprocesses and issue concurrent HTTP requests | One-worker and two-worker validation |
| Multi-worker model consistency was unverified | Workers could load different generations | Compare model version returned by every accepted response | 24-request, 2-worker replay |
| Uvicorn stdout used an unread pipe | Child could block if log buffer filled | Redirect to a temporary file and include tail on failure | Real-process validator |
| Readiness and metrics semantics were unclear | Operator misinterpretation | Separate `/live`; readiness requires loaded model; metrics documented as process-local | Endpoint tests and docs |

## Loop 3: release-store durability and failure semantics

- files are flushed before generation rename;
- staged directories and parent directories are fsynced on POSIX;
- pointer JSON is written to a temporary file, fsynced, and replaced;
- publication failure never changes the active pointer;
- validation failure before or after promotion leaves no visible failed generation;
- rollback target is fully revalidated before pointer replacement;
- symlinks, path traversal, unsafe generation names, and unsupported file types are rejected.

The guarantee is scoped to a local same-filesystem store. Network filesystems and distributed consensus are outside this implementation.

## Loop 4: test-process stability

| Gap | Risk | Resolution | Evidence |
|---|---|---|---|
| Native XGBoost/BLAS state could stall between modules or during interpreter teardown | CI timeout despite completed assertions | Run each test module in an isolated process, detect terminal completion marker, clean up its process tree, and combine coverage | 113 tests, 73% core coverage, exit 0 |
| Runner originally waited only for process exit | Completed shard could wait for native teardown timeout | Completion-marker-aware termination | Full suite runtime is environment-dependent and recorded in command evidence |
| Partial service objects lacked new metrics attributes | Backward-incompatible unit constructions | Metrics use safe defaults for uninitialized optional counters | Regression tests |

## Loop 5: supply-chain hardening

- GitHub Actions references are pinned to verified full-length commit SHAs;
- checkout credentials are not persisted;
- dependency review is configured on pull requests with a high-severity threshold;
- Dependabot monitors pip and Actions dependencies;
- release workflow verifies tag equals `v<project.version>`;
- release assets cannot be overwritten with `--clobber`;
- wheel, sdist, and checksums are covered by GitHub artifact-attestation configuration.

These workflows are configured and locally inspected. Hosted dependency review and attestations require repository execution and are not claimed as completed.

## Loop 6: dependency-range drift versus scientific reproducibility

A source-only install resolved XGBoost CPU 3.3.0 while all other direct dependencies matched the validated environment. The pipeline still executed, but release changed from LAUNCH 29/29 to HOLD 24/29: warm non-inferiority, lower-tail dynamic utility, false warm-up, and sensitivity gates failed. This demonstrated that semantic version compatibility did not guarantee reproducible scientific evidence.

Resolution:

- constrain XGBoost to `>=3.2,<3.3`;
- add `constraints/validated.txt` for the complete direct validation stack;
- use the constraints file in CI, Docker, Make, requirements, and README commands;
- retain strict package-version checks in generated artifacts;
- require the complete validation chain before refreshing constraints.

## Loop 7: structure and runtime-boundary optimization

- canonical Python entry points replace duplicated shell logic;
- release store contains only the online scoring closure;
- machine-specific commands are excluded from manifests;
- public model manifest is sanitized by default;
- release management is isolated in `release_store.py` and `manage_release.py`;
- unused legacy reproducibility helper was removed;
- generated caches, builds, egg-info, artifacts, and Git metadata are excluded from source archives;
- model, release-evidence, deployment-generation, and GitHub-distribution integrity are treated as separate contracts.

## Stop condition

The remaining meaningful work requires an external system or data source rather than additional smoke architecture:

1. hosted GitHub workflow and attestation execution;
2. Docker-daemon image build and health check;
3. native Windows execution;
4. full KuaiSearch and ESCI datasets;
5. Qwen3/FAISS GPU and scale benchmarks;
6. distributed release coordination, remote object storage, or Kubernetes rollout;
7. real logging propensities and online experimentation.

Generation retention, automatic hot reload, remote signing-key management, and distributed locking are intentionally not added because they require an operator-specific deployment platform. Further local abstraction would now add more complexity than evidence.
