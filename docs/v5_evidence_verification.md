# V5 Evidence Verification

The V5 closeout package is self-contained under [`../evidence/v5/`](../evidence/v5/).

## Verify the archived package

From the repository root:

```bash
python scripts/verify_v5_closeout.py
```

A successful run emits:

```text
V5_CLOSEOUT_EVIDENCE_VERIFIED
```

The verifier checks:

1. Every package file recorded in `evidence/v5/FILE_MANIFEST.csv`, excluding the manifest itself by design.
2. SHA-256 and byte-count integrity for every recorded artifact.
3. The final closure state in `evidence/v5/EVIDENCE_MANIFEST.json`.
4. The cleanup receipt's no-go decision: V5-J is not justified, the source baseline is retained, and no V5 serving model, threshold, calibration, or confirmation artifact was created.

## Manifest semantics

- `SOURCE_FILE_MANIFEST.csv` maps canonical archive-time source paths to repository-contained destinations and SHA-256 values.
- `FILE_MANIFEST.csv` is the repository package integrity manifest. It excludes itself because a file cannot contain a stable checksum of its own final bytes.
- `EVIDENCE_MANIFEST.json` records package scope and final policy boundary.
- `CLEANUP_RECEIPT.json` records checksum verification before external V5 roots and the redundant candidate worktree were removed.

The package is synthetic offline evidence. It does not establish production traffic impact or authorize a personalized promotion policy.
