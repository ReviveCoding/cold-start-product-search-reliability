# Windows Local Qualification Evidence

- **Run ID:** p11-cli-sync-v1
- **Final status:** LOCAL_PIPELINE_COMPLETED
- **Qualification boundary:** CONDITIONALLY QUALIFIED

This directory contains compact, tracked evidence for the final Windows local P00-P14 qualification run.

- local_qualification_summary.json: consolidated result summary
elease_decision.json: 29 release-gate decision and metrics
- policy_sensitivity.json: selected Q-RSBT policy evidence
- strict_handoff_validation.json: strict wheel/sdist validation
- wheel_strict_smoke.json: independent installed-wheel smoke result
- stage_markers.json: completed controller stages
- controller_identity.json: controller, runtime, source, and artifact identities
- sha256sums.txt: checksums for this evidence bundle

## Claim boundary

Evidence is from deterministic synthetic/small-canonical offline evaluation and a Windows local qualification run. It does not establish production A/B-test lift, live CTR/conversion impact, GitHub-hosted workflow success, hosted artifact attestation, Docker readiness, or external-scale benchmark generalization.

## Key result

Cold NDCG@10 improved by 0.00820 with a bootstrap confidence interval of [0.00239, 0.01698]. Overall NDCG@10 delta was 0.00137, irrelevant exposure delta was 0.00000, and the frozen release decision was **LAUNCH**.
