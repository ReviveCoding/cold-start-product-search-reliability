# V5-F Seed-Disjoint Multi-Head Viability Audit

- Status: `V5F_MULTIHEAD_RUNTIME_SIGNAL_INSUFFICIENT`
- Outer protocol: 6-fold GroupKFold grouped by seed.
- Input rows: 720 action labels across 18 seeds.
- Utility MAE improvement: -0.005879
- Utility top-1 action hit gain: -0.041667
- Utility regret reduction: -0.074167
- Qualified risk heads: []

This audit emits only out-of-fold diagnostics. It does not persist a final serving model, select a threshold, calibrate probabilities, or use confirmation seeds.
