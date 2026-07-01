# V5 Selective-Promotion Closeout

## Final decision

```text
V5I4_PAIRED_CONTRAST_STABILITY_INSUFFICIENT
V5I4_COMPLETION_CONTRACT_RESOLVED_BASELINE_RETAINED

V5-J paired-contrast predictive viability: NOT_JUSTIFIED
Source baseline retained: True
No-promotion serving policy authorized: True
Final serving model trained: False
Serving threshold selected: False
Calibration seeds executed: []
Confirmation seeds executed: []
```

V5 evaluated whether fully observed synthetic action effects and recovered pre-action runtime state could justify an unseen-seed, personalized rank-placement policy for cold-start promotion. The answer is **no**. The source baseline is retained and a personalized promotion policy is not authorized.

This is a reliability-first no-go conclusion. It does **not** claim that every placement action has the same outcome. It concludes that the available pre-action state did not provide enough stable, safety-compatible evidence to justify a new serving policy.

## Claim boundary

- **Evaluation setting:** synthetic, fully observed, offline simulator evidence.
- **Action space:** `PLACE_AT_1`, `PLACE_AT_2`, `PLACE_AT_3`, `PLACE_AT_5`, and `PLACE_AT_10`.
- **Not established:** production traffic, online experimentation, causal sales lift, or deployment authorization.
- **Interpretation:** this extension does not reclassify the repository's existing baseline pipeline validation.

## Evidence chain

| Gate | Result |
|---|---|
| V5-D direct action-effect corpus | `720 action labels`, `144 contexts`, `18 training seeds`, and `108,000` daily rows |
| V5-E label feasibility | Action utility variation in `144` of `144` contexts; strict-safe-improving action in `42` contexts |
| V5-F | `V5F_MULTIHEAD_RUNTIME_SIGNAL_INSUFFICIENT` |
| V5-G | `V5G_RUNTIME_STATE_REINSTRUMENTATION_REQUIRED` |
| V5-G1 | `144/144` exact context matches, `720` recovered state-action rows, and outcome/oracle leakage exclusion passed |
| V5-H | `V5H_RICH_PREACTION_RUNTIME_SIGNAL_INSUFFICIENT` under seed-disjoint model viability evaluation |
| V5-I.3 | `V5I3_POLICY_DELTA_UTILITY_RECONCILED` using `qrsbt_gate - base` and `mean_scenario_replication_five_day_sum_policy_delta` |
| V5-I.4 | `V5I4_PAIRED_CONTRAST_STABILITY_INSUFFICIENT` |
| V5-I.4.1 | `V5I4_COMPLETION_CONTRACT_RESOLVED_BASELINE_RETAINED`; key-aligned reconstruction exact with max absolute error `0.0000000000000175` |

## Utility-label reconstruction

The frozen V5-D action-utility label was reconstructed from the daily table after retaining `policy` in the observation grain:

```text
five_day_policy_delta
  = sum_day [utility(qrsbt_gate) - utility(base)]

mean_scenario_replication_utility_delta
  = mean over 15 scenario-replication blocks of five_day_policy_delta
```

V5-I.3 identified the unique reconstruction formula. V5-I.4.1 verified the target after aligning all `720` action records by `(seed, proposal_index, action)`, avoiding invalid comparison of independently ordered arrays.

## Paired-contrast stability result

`PLACE_AT_1` was an analytical comparison anchor, **not** a newly authorized serving policy.

| Quantity | Value |
|---|---:|
| Contexts whose retrospective oracle action differed from `PLACE_AT_1` | `90` |
| Total oracle gap vs. `PLACE_AT_1` | `73.7333` |
| Stable oracle gap | `20.5600` |
| Stable oracle-gap share | `27.88%` |
| Strict-safe stable oracle gap | `8.9733` |
| Strict-safe stable oracle-gap share | `12.17%` |
| Required stable oracle-gap share | `50%` |
| Stable utility contexts / seeds | `13 / 11` |
| Stable safe contexts / seeds | `6 / 6` |

Neither the utility-only gate nor the strict safety-compatible gate passed. V5-J paired-contrast predictive modeling was therefore not justified.

## Operational outcome

```text
Keep the source baseline.
Do not release a personalized promotion policy.
Do not train a final V5 serving model.
Do not select a V5 serving threshold.
Do not execute V5 calibration or confirmation.
```

## Reproducibility

The canonical package is in [`evidence/v5/`](../evidence/v5/). It contains the archived run outputs, audited input snapshot, environment capture, manifests, and cleanup receipt.

Run:

```bash
python scripts/verify_v5_closeout.py
```

The verifier checks all package hashes recorded in `evidence/v5/FILE_MANIFEST.csv` and verifies the final no-go closure in `evidence/v5/CLEANUP_RECEIPT.json`.

See [V5 evidence verification](v5_evidence_verification.md).
