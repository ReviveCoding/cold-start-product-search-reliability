# Results

## Evaluation scope

The project was evaluated using deterministic synthetic/small-canonical cold-start search data. The repository supports offline and local-qualification claims only.

| Metric | Result |
|---|---:|
| Cold NDCG@10 lift | 0.00820 |
| Cold lift 95% CI | [0.00239, 0.01698] |
| Overall NDCG@10 delta | 0.00137 |
| Warm NDCG@10 delta | -0.00587 |
| Irrelevant exposure delta | 0.00000 |
| Dynamic worst-scenario utility delta | 8.24667 |
| Dynamic p10 utility delta | 0.00000 |
| Future behavior ROC-AUC | 0.77404 |
| Serving p95 latency | 72.30 ms |
| Serving fallbacks | 0 |
| Release gates | 29 / 29 PASS |
| Selected max boost | 0.01275 |

## Interpretation

The selected Q-RSBT policy improved cold-item ranking while preserving overall ranking quality and holding irrelevant-exposure increase at zero. Warm ranking had a bounded negative trade-off, so this repository does not claim universal ranking improvement across all item segments.

The lower-tail dynamic gate passed at zero. The supported claim is lower-tail non-degradation under the frozen release contract, not material lower-tail uplift.

See evidence/windows-local-p11-cli-sync-v1/ for machine-readable evidence.