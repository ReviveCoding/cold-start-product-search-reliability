from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


v5d_decision_path = Path(sys.argv[1]).resolve()
v5e_decision_path = Path(sys.argv[2]).resolve()
v5g1_decision_path = Path(sys.argv[3]).resolve()
v5h_decision_path = Path(sys.argv[4]).resolve()
v5i3_decision_path = Path(sys.argv[5]).resolve()
v5i4_decision_path = Path(sys.argv[6]).resolve()
output_root = Path(sys.argv[7]).resolve()
expected_head = sys.argv[8]

ACTION_KEYS = (
    "seed",
    "proposal_index",
    "action",
)
CONTEXT_KEYS = (
    "seed",
    "proposal_index",
)
TARGET_UTILITY_COLUMN = (
    "mean_scenario_replication_utility_delta"
)
RECONSTRUCTED_UTILITY_COLUMN = (
    "reconstructed_mean_utility_delta"
)
TOLERANCE = 1e-8
EXPECTED_STATUS = (
    "V5I4_PAIRED_CONTRAST_STABILITY_INSUFFICIENT"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def validate_decision(
    decision: dict[str, Any],
    *,
    label: str,
    expected_status: str,
) -> None:
    if decision.get("status") != expected_status:
        raise RuntimeError(
            f"{label}: unexpected status {decision.get('status')}"
        )

    if decision.get("baseline_commit") != expected_head:
        raise RuntimeError(
            f"{label}: baseline commit mismatch."
        )

    for key in (
        "calibration_seeds_executed",
        "confirmation_seeds_executed",
    ):
        if decision.get(key) not in (None, [], False):
            raise RuntimeError(
                f"{label}: {key} indicates execution."
            )


def normalize_integer(
    series: pd.Series,
    *,
    label: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{label}: non-finite values."
        )

    rounded = np.rint(values)

    if not np.allclose(values, rounded):
        raise RuntimeError(
            f"{label}: non-integral values."
        )

    return pd.Series(
        rounded.astype(np.int64),
        index=series.index,
    )


def assert_unique(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    label: str,
) -> None:
    duplicated = frame.loc[
        frame.duplicated(keys, keep=False),
        keys,
    ]

    if not duplicated.empty:
        raise RuntimeError(
            f"{label} duplicate keys: "
            f"{duplicated.head(10).to_dict(orient='records')}"
        )


def close_enough(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{label} mismatch: actual={actual}, expected={expected}"
        )


v5d = read_json(v5d_decision_path)
v5e = read_json(v5e_decision_path)
v5g1 = read_json(v5g1_decision_path)
v5h = read_json(v5h_decision_path)
v5i3 = read_json(v5i3_decision_path)
v5i4 = read_json(v5i4_decision_path)

validate_decision(
    v5d,
    label="V5-D",
    expected_status=(
        "V5D_FINAL_TRAINING_DIRECT_ACTION_EFFECT_CORPUS_COMPLETE"
    ),
)
validate_decision(
    v5e,
    label="V5-E",
    expected_status=(
        "V5E_LABEL_FEASIBILITY_AND_FEATURE_PROVENANCE_AUDIT_COMPLETE"
    ),
)
validate_decision(
    v5g1,
    label="V5-G1",
    expected_status=(
        "V5G1_PREACTION_RUNTIME_STATE_RECOVERED_AND_CONTRACT_READY"
    ),
)
validate_decision(
    v5h,
    label="V5-H",
    expected_status=(
        "V5H_RICH_PREACTION_RUNTIME_SIGNAL_INSUFFICIENT"
    ),
)
validate_decision(
    v5i3,
    label="V5-I.3",
    expected_status=(
        "V5I3_POLICY_DELTA_UTILITY_RECONCILED"
    ),
)
validate_decision(
    v5i4,
    label="V5-I.4",
    expected_status=EXPECTED_STATUS,
)

if int(v5d.get("action_count", -1)) != 720:
    raise RuntimeError("V5-D action count mismatch.")

if int(v5d.get("proposal_count", -1)) != 144:
    raise RuntimeError("V5-D proposal count mismatch.")

if int(v5g1.get("context_count", -1)) != 144:
    raise RuntimeError("V5-G1 context count mismatch.")

if int(v5g1.get("state_action_row_count", -1)) != 720:
    raise RuntimeError("V5-G1 state-action count mismatch.")

if v5g1.get("forbidden_outcome_or_oracle_inputs") is not True:
    raise RuntimeError(
        "V5-G1 leakage-exclusion contract failed."
    )

if int(
    v5i3.get(
        "exact_reconciliation_candidate_count",
        -1,
    )
) != 1:
    raise RuntimeError(
        "V5-I.3 exact policy-delta reconstruction mismatch."
    )

if v5i4.get("next_gate") != (
    "Close V5 selective-promotion optimization with the source "
    "baseline retained. The fully observed simulator does not "
    "provide enough stable, safe paired action contrast to justify "
    "another predictive model search."
):
    raise RuntimeError(
        "V5-I.4 did not preserve the expected closure gate."
    )

effects_path = Path(v5d["action_effects_path"]).resolve()
artifacts = v5i4.get("artifacts", {})

reconstructed_path = Path(
    artifacts["reconstructed_action_utility_path"]
).resolve()
block_coverage_path = Path(
    artifacts["action_contrast_block_coverage_path"]
).resolve()
contrasts_path = Path(
    artifacts["paired_context_action_contrasts_path"]
).resolve()
oracle_path = Path(
    artifacts["oracle_gap_stability_decomposition_path"]
).resolve()

for path in (
    effects_path,
    reconstructed_path,
    block_coverage_path,
    contrasts_path,
    oracle_path,
):
    if not path.is_file():
        raise RuntimeError(f"Missing V5-I.4 artifact: {path}")

effects = pd.read_csv(effects_path)
reconstructed = pd.read_csv(reconstructed_path)
block_coverage = pd.read_csv(block_coverage_path)
contrasts = pd.read_csv(contrasts_path)
oracle = pd.read_csv(oracle_path)

for frame, label in (
    (effects, "action effects"),
    (reconstructed, "reconstructed utility"),
):
    missing = sorted(
        set(ACTION_KEYS) - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label} missing action keys: {missing}"
        )

for frame, label in (
    (effects, "action effects"),
    (reconstructed, "reconstructed utility"),
    (contrasts, "paired contrasts"),
    (oracle, "oracle decomposition"),
):
    for column in ("seed", "proposal_index"):
        if column in frame.columns:
            frame[column] = normalize_integer(
                frame[column],
                label=f"{label} {column}",
            )

assert_unique(
    effects,
    keys=list(ACTION_KEYS),
    label="action effects",
)
assert_unique(
    reconstructed,
    keys=list(ACTION_KEYS),
    label="reconstructed utility",
)

if len(effects) != 720 or len(reconstructed) != 720:
    raise RuntimeError(
        "Action utility reconstruction cardinality is not 720."
    )

if TARGET_UTILITY_COLUMN not in effects.columns:
    raise RuntimeError(
        "Action effects lack frozen utility target."
    )

if RECONSTRUCTED_UTILITY_COLUMN not in reconstructed.columns:
    raise RuntimeError(
        "V5-I.4 reconstruction lacks reconstructed utility."
    )

effects[TARGET_UTILITY_COLUMN] = pd.to_numeric(
    effects[TARGET_UTILITY_COLUMN],
    errors="raise",
)
reconstructed[RECONSTRUCTED_UTILITY_COLUMN] = pd.to_numeric(
    reconstructed[RECONSTRUCTED_UTILITY_COLUMN],
    errors="raise",
)

aligned = effects.loc[
    :,
    [
        *ACTION_KEYS,
        TARGET_UTILITY_COLUMN,
    ],
].merge(
    reconstructed.loc[
        :,
        [
            *ACTION_KEYS,
            RECONSTRUCTED_UTILITY_COLUMN,
        ],
    ],
    on=list(ACTION_KEYS),
    how="inner",
    validate="one_to_one",
)

if len(aligned) != 720:
    raise RuntimeError(
        "Key-aligned reconstruction merge did not cover 720 actions."
    )

aligned["absolute_error"] = np.abs(
    aligned[TARGET_UTILITY_COLUMN].to_numpy(dtype=float)
    - aligned[RECONSTRUCTED_UTILITY_COLUMN].to_numpy(
        dtype=float
    )
)

key_aligned_max_abs_error = float(
    aligned["absolute_error"].max()
)
key_aligned_mean_abs_error = float(
    aligned["absolute_error"].mean()
)

if key_aligned_max_abs_error > TOLERANCE:
    raise RuntimeError(
        "V5-I.4 reconstruction fails under a key-aligned check: "
        f"max_abs_error={key_aligned_max_abs_error}"
    )

required_coverage = {
    "seed",
    "proposal_index",
    "alternative_action",
    "paired_block_count",
    "scenario_count",
    "replication_count",
}

missing_coverage = sorted(
    required_coverage - set(block_coverage.columns)
)

if missing_coverage:
    raise RuntimeError(
        "Block coverage missing columns: "
        f"{missing_coverage}"
    )

if len(block_coverage) != 576:
    raise RuntimeError(
        "Expected 576 action-contrast coverage rows."
    )

if not block_coverage["paired_block_count"].eq(15).all():
    raise RuntimeError(
        "At least one action contrast lacks 15 paired blocks."
    )

if not block_coverage["scenario_count"].eq(3).all():
    raise RuntimeError(
        "At least one action contrast lacks 3 scenarios."
    )

if not block_coverage["replication_count"].eq(5).all():
    raise RuntimeError(
        "At least one action contrast lacks 5 replications."
    )

required_contrasts = {
    "seed",
    "proposal_index",
    "alternative_action",
    "strict_paired_utility_advantage",
    "strict_safe_paired_advantage",
}

missing_contrasts = sorted(
    required_contrasts - set(contrasts.columns)
)

if missing_contrasts:
    raise RuntimeError(
        "Paired contrast artifact missing columns: "
        f"{missing_contrasts}"
    )

if len(contrasts) != 576:
    raise RuntimeError(
        "Expected 576 paired contrast rows."
    )

contrast_counts = (
    contrasts.groupby(
        "alternative_action",
        as_index=False,
    )
    .agg(
        contrast_count=("proposal_index", "size"),
    )
)

if not contrast_counts["contrast_count"].eq(144).all():
    raise RuntimeError(
        "Every alternative action must have 144 contrasts."
    )

utility_stable_contexts = (
    contrasts.loc[
        contrasts["strict_paired_utility_advantage"].astype(bool),
        list(CONTEXT_KEYS),
    ]
    .drop_duplicates()
)

safe_stable_contexts = (
    contrasts.loc[
        contrasts["strict_safe_paired_advantage"].astype(bool),
        list(CONTEXT_KEYS),
    ]
    .drop_duplicates()
)

utility_stable_context_count = int(len(utility_stable_contexts))
utility_stable_seed_count = int(
    utility_stable_contexts["seed"].nunique()
)
safe_stable_context_count = int(len(safe_stable_contexts))
safe_stable_seed_count = int(
    safe_stable_contexts["seed"].nunique()
)

required_oracle = {
    "seed",
    "proposal_index",
    "oracle_differs_from_place_at_1",
    "oracle_gap_vs_place_at_1",
    "stable_oracle_gap",
    "strict_safe_stable_oracle_gap",
    "unstable_oracle_gap",
}

missing_oracle = sorted(
    required_oracle - set(oracle.columns)
)

if missing_oracle:
    raise RuntimeError(
        "Oracle decomposition missing columns: "
        f"{missing_oracle}"
    )

if len(oracle) != 144:
    raise RuntimeError(
        "Expected 144 oracle contexts."
    )

oracle_non_place_at_1_context_count = int(
    oracle["oracle_differs_from_place_at_1"]
    .astype(bool)
    .sum()
)
total_oracle_gap = float(
    oracle["oracle_gap_vs_place_at_1"].sum()
)
stable_oracle_gap = float(
    oracle["stable_oracle_gap"].sum()
)
strict_safe_stable_oracle_gap = float(
    oracle["strict_safe_stable_oracle_gap"].sum()
)
unstable_oracle_gap = float(
    oracle["unstable_oracle_gap"].sum()
)

stable_oracle_gap_share = (
    stable_oracle_gap / total_oracle_gap
    if total_oracle_gap > 0.0
    else 0.0
)
strict_safe_stable_oracle_gap_share = (
    strict_safe_stable_oracle_gap / total_oracle_gap
    if total_oracle_gap > 0.0
    else 0.0
)

gate = v5i4["label_stability_gate"]
oracle_decision = v5i4["oracle_gap_decomposition"]

expected_gate = {
    "utility_stable_context_count": utility_stable_context_count,
    "utility_stable_seed_count": utility_stable_seed_count,
    "safe_stable_context_count": safe_stable_context_count,
    "safe_stable_seed_count": safe_stable_seed_count,
}

for key, value in expected_gate.items():
    if int(gate[key]) != value:
        raise RuntimeError(
            f"V5-I.4 {key} mismatch: decision={gate[key]}, "
            f"artifact={value}"
        )

close_enough(
    float(oracle_decision["total_oracle_gap_vs_place_at_1"]),
    total_oracle_gap,
    label="total_oracle_gap",
)
close_enough(
    float(oracle_decision["stable_oracle_gap"]),
    stable_oracle_gap,
    label="stable_oracle_gap",
)
close_enough(
    float(oracle_decision["strict_safe_stable_oracle_gap"]),
    strict_safe_stable_oracle_gap,
    label="strict_safe_stable_oracle_gap",
)
close_enough(
    float(oracle_decision["unstable_oracle_gap"]),
    unstable_oracle_gap,
    label="unstable_oracle_gap",
)
close_enough(
    float(oracle_decision["stable_oracle_gap_share"]),
    stable_oracle_gap_share,
    label="stable_oracle_gap_share",
)
close_enough(
    float(
        oracle_decision[
            "strict_safe_stable_oracle_gap_share"
        ]
    ),
    strict_safe_stable_oracle_gap_share,
    label="strict_safe_stable_oracle_gap_share",
)

utility_signal = bool(
    utility_stable_context_count
    >= int(gate["minimum_stable_contexts"])
    and utility_stable_seed_count
    >= int(gate["minimum_stable_seeds"])
    and stable_oracle_gap_share
    >= float(gate["minimum_stable_oracle_gap_share"])
)

safe_signal = bool(
    safe_stable_context_count
    >= int(gate["minimum_stable_contexts"])
    and safe_stable_seed_count
    >= int(gate["minimum_stable_seeds"])
    and strict_safe_stable_oracle_gap_share
    >= float(gate["minimum_stable_oracle_gap_share"])
)

if bool(gate["utility_label_signal"]) != utility_signal:
    raise RuntimeError(
        "V5-I.4 utility label-signal decision mismatch."
    )

if bool(gate["safe_label_signal"]) != safe_signal:
    raise RuntimeError(
        "V5-I.4 safe label-signal decision mismatch."
    )

if utility_signal or safe_signal:
    raise RuntimeError(
        "V5-I.4 closure status conflicts with recomputed gates."
    )

reported_positional_error = float(
    v5i4["policy_delta_label_contract"][
        "utility_reconstruction_max_abs_error"
    ]
)

resolution = {
    "status": (
        "V5I4_COMPLETION_CONTRACT_RESOLVED_"
        "BASELINE_RETAINED"
    ),
    "baseline_commit": expected_head,
    "v5d_decision_path": str(v5d_decision_path),
    "v5d_decision_sha256": sha256(v5d_decision_path),
    "v5e_decision_path": str(v5e_decision_path),
    "v5e_decision_sha256": sha256(v5e_decision_path),
    "v5g1_decision_path": str(v5g1_decision_path),
    "v5g1_decision_sha256": sha256(v5g1_decision_path),
    "v5h_decision_path": str(v5h_decision_path),
    "v5h_decision_sha256": sha256(v5h_decision_path),
    "v5i3_decision_path": str(v5i3_decision_path),
    "v5i3_decision_sha256": sha256(v5i3_decision_path),
    "v5i4_decision_path": str(v5i4_decision_path),
    "v5i4_decision_sha256": sha256(v5i4_decision_path),
    "original_v5i4_status": v5i4["status"],
    "original_metadata_field_under_review": {
        "field": (
            "policy_delta_label_contract."
            "utility_reconstruction_max_abs_error"
        ),
        "reported_value": reported_positional_error,
        "interpretation": (
            "The original decision metadata used a positional "
            "array subtraction after independently ordered action "
            "tables. It is not a valid reconstruction diagnostic "
            "unless action-key order is first aligned."
        ),
    },
    "key_aligned_reconstruction_contract": {
        "action_key_columns": list(ACTION_KEYS),
        "matched_action_count": int(len(aligned)),
        "key_aligned_max_abs_error": key_aligned_max_abs_error,
        "key_aligned_mean_abs_error": key_aligned_mean_abs_error,
        "tolerance": TOLERANCE,
        "exact": True,
    },
    "paired_block_integrity": {
        "contrast_row_count": int(len(contrasts)),
        "coverage_row_count": int(len(block_coverage)),
        "contexts_per_alternative_action": 144,
        "paired_blocks_per_context_action": 15,
        "scenarios_per_context_action": 3,
        "replications_per_scenario": 5,
    },
    "recomputed_label_stability_gate": {
        "utility_stable_context_count": utility_stable_context_count,
        "utility_stable_seed_count": utility_stable_seed_count,
        "safe_stable_context_count": safe_stable_context_count,
        "safe_stable_seed_count": safe_stable_seed_count,
        "utility_label_signal": utility_signal,
        "safe_label_signal": safe_signal,
    },
    "recomputed_oracle_gap_decomposition": {
        "oracle_non_place_at_1_context_count": (
            oracle_non_place_at_1_context_count
        ),
        "total_oracle_gap_vs_place_at_1": total_oracle_gap,
        "stable_oracle_gap": stable_oracle_gap,
        "stable_oracle_gap_share": stable_oracle_gap_share,
        "strict_safe_stable_oracle_gap": (
            strict_safe_stable_oracle_gap
        ),
        "strict_safe_stable_oracle_gap_share": (
            strict_safe_stable_oracle_gap_share
        ),
        "unstable_oracle_gap": unstable_oracle_gap,
    },
    "closure": {
        "v5j_paired_contrast_predictive_model_viability": (
            "NOT_JUSTIFIED"
        ),
        "source_baseline_retained": True,
        "no_promotion_serving_policy_authorized": True,
        "final_serving_model_trained": False,
        "threshold_selected": False,
        "calibration_seeds_executed": [],
        "confirmation_seeds_executed": [],
    },
    "source_modified": False,
    "config_modified": False,
    "v5d_corpus_rerun": False,
    "model_trained": False,
    "final_serving_model_trained": False,
    "threshold_selected": False,
    "calibration_seeds_executed": [],
    "confirmation_seeds_executed": [],
    "commit_created": False,
    "push_performed": False,
    "next_gate": (
        "Close V5 selective-promotion optimization. Preserve the "
        "source baseline. Do not execute V5-J predictive modeling, "
        "threshold selection, calibration, or confirmation."
    ),
}

output_root.mkdir(parents=True, exist_ok=True)

aligned_path = (
    output_root
    / "v5i4_key_aligned_utility_reconstruction.csv"
)
resolution_path = (
    output_root
    / "v5i4_completion_contract_resolution.json"
)

aligned.to_csv(aligned_path, index=False)
write_json(resolution_path, resolution)

print("===== V5-I.4.1 COMPLETION CONTRACT RESOLUTION =====")
print(
    json.dumps(
        {
            "resolution": resolution,
            "key_aligned_reconstruction_preview": (
                aligned.head(10).to_dict(
                    orient="records"
                )
            ),
        },
        indent=2,
        sort_keys=True,
    )
)