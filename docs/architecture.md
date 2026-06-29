# Architecture v0.6.0

```text
query at scoring block t
  -> validation and admission control
  -> retrievers fit on catalog available at t
  -> hybrid candidate union and anchor
  -> LambdaMART semantic score
  -> calibrated behavioral score
  -> global Q-RSBT transfer
  -> bounded reliability gate
  -> top-K response and audit metrics
```

## Release flow

```text
validated LAUNCH artifact
  -> verify complete research-release hashes
  -> stage manifest + minimal serving closure
  -> verify serving hashes and load runtime
  -> write generation metadata
  -> fsync staged files/directories
  -> atomic generation rename
  -> verify promoted generation
  -> atomic current.json replacement
  -> rolling worker restart
```

Rollback revalidates the target generation before replacing the pointer. A failed publication never changes the active pointer; failures before pointer replacement also remove a promoted orphan generation.

## Serving process model

Each Uvicorn worker loads one immutable generation and owns:

- one model bundle;
- one bounded admission semaphore;
- one request/latency/fallback metric set.

Metrics are process-local. The release pointer is resolved at process startup; hot reload is intentionally not implemented. Use rolling replacement after publish or rollback.

## Integrity layers

1. configuration schema;
2. Python and package versions;
3. serving source-code and config hashes;
4. model fingerprint and scoring artifact hashes;
5. complete research-release hashes;
6. immutable generation metadata and pointer;
7. GitHub-distribution build provenance when hosted workflow runs.
